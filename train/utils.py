if 0 ** 1:
    # This is to trick editor into thinking the variables (GCS_PATH,...) are initialized
    from setup import *

import sklearn.utils
import itertools
import os
import numpy as np
import tensorflow as tf
import transformers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from bisect import bisect
import psutil
from tqdm.notebook import tqdm

RANDOM_STATE = 42
TOTAL_MAX_LEN = 512
BASE_MODEL = 'microsoft/codebert-base'
BITS = 10
RECORDS_PER_FILE = 1024
VERBOSE = 1
NUM_FEATURES = 3
SPLIT = .5


def batch_de_binarize(batch_of_bins):
    arr = np.array(batch_of_bins)
    assert arr.shape[1] < 64
    return arr.dot(1 << np.arange(arr.shape[-1] - 1, -1, -1))


def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def count_inversions_n_log_n(a):
    res = 0
    counts = [0] * (len(a) + 1)
    rank = {v: i + 1 for i, v in enumerate(sorted(a))}
    for x in reversed(a):
        i = rank[x] - 1
        while i:
            res += counts[i]
            i -= i & -i
        i = rank[x]
        while i <= len(a):
            counts[i] += 1
            i += i & -i
    return res


def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions_n_log_n(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


@tf.function
def read_tfrecord(example):
    features = {
        "input_ids": tf.io.FixedLenFeature([TOTAL_MAX_LEN], tf.int64),
        "attention_mask": tf.io.FixedLenFeature([TOTAL_MAX_LEN], tf.int64),
        "feature": tf.io.FixedLenFeature([NUM_FEATURES], tf.int64),
        "rank": tf.io.FixedLenFeature([1], tf.int64),
        "label": tf.io.FixedLenFeature([BITS], tf.int64),
    }
    example = tf.io.parse_single_example(example, features)
    return {
               "input_ids": tf.cast(example["input_ids"], tf.int32),
               "attention_mask": tf.cast(example["attention_mask"], tf.int32),
               "feature": tf.cast(example["feature"], tf.float32),
               "rank": example["rank"]
           }, tf.cast(example["label"], tf.float32)


def get_dataset(filenames, batch_size=None, ordered=False, repeated=True, cached=True, ):
    auto = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=auto)
    if not ordered:
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=auto)
    if not ordered:
        dataset = dataset.shuffle(2048, seed=RANDOM_STATE)
    if repeated:
        dataset = dataset.repeat()
    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    if cached:
        dataset = dataset.cache()
    dataset = dataset.prefetch(auto)
    return STRATEGY.experimental_distribute_dataset(dataset)


def stupid_serialize(x, y):
    return {
               "input_ids": x["input_ids"],
               "attention_mask": x["attention_mask"],
               "feature": x["feature"],
           }, y


def per_dataset_score(model, record_files, threshold=0.5, batches=100):
    print('Predicting...')
    orders_true, orders_predict = [], []

    dataset = get_dataset(record_files,
                          batch_size=BATCH_SIZE,
                          ordered=True,
                          repeated=False,
                          cached=True, )

    predicts = model.predict(dataset, steps=batches+1, verbose=1)
    predicts = predicts > threshold
    predicts = batch_de_binarize(predicts)

    print('Calculating score...')

    curr = 0
    with tqdm(total=len(predicts)) as pbar:
        for record_file in record_files:
            dataset = tf.data.TFRecordDataset(record_file, num_parallel_reads=tf.data.AUTOTUNE)
            dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = list(dataset.as_numpy_iterator())

            total_cells = int(dataset[0][0]['feature'][0])
            n = len(dataset)
            ranks_true = [int(x['rank'][0]) for x, y in dataset]

            predict = predicts[curr: curr + n]
            predict = np.minimum(predict, len(dataset)-1)

            order_predict = np.arange(total_cells)
            order_predict[ranks_true[:len(predict)]] = predict

            orders_true.append(np.arange(total_cells).tolist())
            orders_predict.append(order_predict.tolist())

            pbar.update(len(dataset))
            curr += len(dataset)
            if curr >= len(predicts):
                break

    return kendall_tau(orders_true, orders_predict)


def get_model():
    backbone = transformers.TFAutoModel.from_pretrained(BASE_MODEL)
    input_ids = tf.keras.layers.Input(
        shape=(TOTAL_MAX_LEN,),
        dtype=tf.int32,
        name="input_ids",
    )
    attention_mask = tf.keras.layers.Input(
        shape=(TOTAL_MAX_LEN,),
        dtype=tf.int32,
        name="attention_mask",
    )
    feature = tf.keras.layers.Input(
        shape=(NUM_FEATURES,),
        dtype=tf.float32,
        name="feature",
    )
    x = backbone({"input_ids": input_ids, "attention_mask": attention_mask})[0]
    x = tf.concat([x[:, 0, :], feature], axis=1)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1), dtype='float32')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(BITS, activation='sigmoid', dtype='float32')(x)
    return tf.keras.Model(
        inputs=[input_ids, attention_mask, feature],
        outputs=outputs,
    )


train_filenames = tf.io.gfile.glob(os.path.join(GCS_PATH, 'train', '*/*.tfrec'))
train_dataset = get_dataset(train_filenames, batch_size=BATCH_SIZE)

steps_per_epoch = len(train_filenames) * RECORDS_PER_FILE // BATCH_SIZE

val_filenames = tf.io.gfile.glob(os.path.join(GCS_PATH, 'val', '*/*.tfrec'))
val_filenames = sklearn.utils.shuffle(val_filenames, random_state=RANDOM_STATE)

print("Batch size:", BATCH_SIZE)
