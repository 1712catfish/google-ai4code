GCS_PATH = 'gs://kds-dbed60f60c555498f5e4bb9ab8658a075f5d0a7669ae13d005ed29f6'
INPUT_PATH = '..'

# from kaggle_datasets import KaggleDatasets
# GCS_PATH = KaggleDatasets().get_gcs_path("AI4Code")
# INPUT_PATH = '../input/AI4Code'

import os
# import orjson
import json
import numpy as np
import pandas as pd
import transformers
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import *
import threading

# CONFIG
from transformers import AutoTokenizer

RANDOM_STATE = None
TOTAL_MAX_LEN = 512
MD_MAX_LEN = 64
BASE_MODEL = "microsoft/codebert-base"
TOKENIZER = transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
EPOCHS = 5
LR = 3e-4
WARMUP_RATE = 0.1
VERBOSE = 2
K_FOLDS = 5
FILES_PER_FOLD = 16
THREADS_LIMIT = 4

TOKENIZERS = [transformers.AutoTokenizer.from_pretrained(BASE_MODEL) for _ in range(THREADS_LIMIT)]
lock = threading.Lock()


# try:
#     TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
#     tf.config.experimental_connect_to_cluster(TPU)
#     tf.tpu.experimental.initialize_tpu_system(TPU)
#     STRATEGY = tf.distribute.experimental.TPUStrategy(TPU)
#     BATCH_SIZE = 64 * STRATEGY.num_replicas_in_sync
# except Exception:
#     TPU = None
#     STRATEGY = tf.distribute.get_strategy()
#     BATCH_SIZE = 4
#
# print("TensorFlow", tf.__version__)
#
# if TPU is not None:
#     print("Using TPU v3-8")
# else:
#     print("Using GPU/CPU")
#
# print("Batch size:", BATCH_SIZE)


# FUNCTIONS
def count_samples(filenames):
    return sum(int(os.path.basename(x).split(".")[0].split("-")[-1]) for x in filenames)


def clean_code(cell):
    return cell.replace("\\n", "\n")


def py_parse_function(id_, cell_order):
    id_ = id_.numpy().decode("utf-8")

    with open(tf.io.gfile.join(INPUT_PATH + '/train/' + f'{id_}.json'), 'r') as f:
        obj = json.loads(f.read())

    cell_order = [x.decode("utf-8") for x in cell_order.numpy().split()]
    n = len(cell_order)

    source = []
    source_code = []

    for cell_id in cell_order:
        if obj['cell_type'][cell_id] == 'code':
            source.append(obj['source'][cell_id])
        else:
            source_code.append(clean_code(obj['source'][cell_id]))

    pct_rank = len(source) / n

    global TOKENIZERS
    global lock
    with lock:
        tokenizer = TOKENIZERS.pop()

    inputs = tokenizer(source,
                       add_special_tokens=True,
                       max_length=MD_MAX_LEN,
                       padding="max_length",
                       truncation=True,
                       return_tensors='np')

    code_inputs = tokenizer(source_code,
                            add_special_tokens=False,
                            max_length=23,
                            padding="max_length",
                            truncation=True,
                            return_tensors='np')

    with lock:
        TOKENIZERS.append(tokenizer)

    ids = inputs['input_ids']
    mask = inputs['attention_mask']

    code_ids = code_inputs['input_ids']
    code_ids = code_ids.ravel().tolist()
    code_ids = code_ids[:TOTAL_MAX_LEN]
    code_ids = code_ids + [TOKENIZER.pad_token_id, ] * max(0, TOTAL_MAX_LEN - len(code_ids))

    code_mask = code_inputs['attention_mask']
    code_mask = code_mask.ravel().tolist()
    code_mask = code_mask[:TOTAL_MAX_LEN]
    code_mask = code_mask + [0, ] * max(0, TOTAL_MAX_LEN - len(code_mask))

    return ids, mask, [code_ids], [code_mask], [pct_rank]


def flatten_function(ids, mask, code_ids, code_mask, pct_rank):
    n = tf.shape(ids)[0]

    code_ids = tf.repeat(code_ids, [n], axis=0)
    code_mask = tf.repeat(code_mask, [n], axis=0)
    target = tf.repeat([pct_rank], [n], axis=0)

    input_ids = tf.concat([ids, code_ids], 1)
    attention_mask = tf.concat([mask, code_mask], 1)

    inputs = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask))
    target = tf.data.Dataset.from_tensor_slices(target)
    return tf.data.Dataset.zip((inputs, target))


def get_dataset(dataset, ordered=False, repeated=True, cached=False):
    auto = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    if not ordered:
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(lambda data: tf.py_function(
        py_parse_function,
        [data[0], data[1]],
        (tf.int32, tf.int32, tf.int32, tf.int32, tf.float32)
    ), num_parallel_calls=auto)
    dataset = dataset.flat_map(flatten_function)
    print(next(dataset.as_numpy_iterator()))
    if not ordered:
        dataset = dataset.shuffle(2048, seed=RANDOM_STATE)
    if repeated:
        dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    if cached:
        dataset = dataset.cache()
    dataset = dataset.prefetch(auto)
    return STRATEGY.experimental_distribute_dataset(dataset)


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
    x = backbone({"input_ids": input_ids, "attention_mask": attention_mask})[0]
    # x = tf.concat([x[:, 0, :], feature], axis=1)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return tf.keras.Model(
        inputs=(input_ids, attention_mask),
        outputs=outputs,
    )


df = pd.read_csv(os.path.join(INPUT_PATH, "train_orders.csv"))
df = df.merge(pd.read_csv(os.path.join(INPUT_PATH, 'train_ancestors.csv')), on='id')
df = df.sample(frac=1).reset_index(drop=True)
data = df[['id', 'cell_order']].values.tolist()

k = int(.8 * len(data))
train_data = data[:k]
val_data = data[k:]

# TRAIN
if TPU is not None:
    tf.tpu.experimental.initialize_tpu_system(TPU)

steps_per_epoch = len(train_data) // BATCH_SIZE
validation_steps = len(val_data) // BATCH_SIZE
total_steps = steps_per_epoch * EPOCHS

train_dataset = get_dataset(train_data)
val_dataset = get_dataset(val_data, ordered=True, repeated=False, cached=True)

# print(next(train_dataset.as_numpy_iterator()))

with STRATEGY.scope():
    model = get_model()

    optimizer = transformers.AdamWeightDecay(
        learning_rate=ExponentialDecay(
            initial_learning_rate=LR,
            decay_steps=total_steps,
            decay_rate=0.96,
            staircase=True,
        ),
        weight_decay_rate=0.01,
        exclude_from_weight_decay=[
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
        ],
    )
    model.compile(loss="mae", optimizer=optimizer)

metrics = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    verbose=VERBOSE,
).history

model.save_weights(f"model.h5")
