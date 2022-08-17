if 0 ** 1:
    # This is to trick editor into thinking the variables (INPUT_PATH, OUTPUT_PATH,...) are initialized
    from setups import *

import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import sklearn
from tqdm.notebook import tqdm
import orjson
from fast_map import fast_map
import psutil
import threading
from math import ceil

THREADS_LIMIT = len(psutil.Process().cpu_affinity()) * 4  # *2 for Colab, * 4 for Kaggle
print(f'Using {THREADS_LIMIT} threads')

RANDOM_STATE = 42
MD_MAX_LEN = 64
CODE_MAX_LEN = 21
MAX_CODE_CELL = 20
TOTAL_MAX_LEN = 512

MODEL_NAME = "microsoft/codebert-base"
TOKENIZER = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
# This is to avoid Colab to reinitialize TOKENIZERS every run
try:
    TOKENIZERS[0]
except NameError:
    print('Initializing TOKENIZERS...')
    TOKENIZERS = [transformers.AutoTokenizer.from_pretrained(MODEL_NAME) for _ in range(THREADS_LIMIT + 4)]
else:
    print(f'Found {len(TOKENIZERS)} cached TOKENIZERS.')
lock = threading.Lock()

df = pd.read_csv(os.path.join(INPUT_PATH, "train_orders.csv"))
df = df.merge(pd.read_csv(os.path.join(INPUT_PATH, 'train_ancestors.csv')), on='id')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def use_tokenizer(func):
    def wrap(*args, **kwargs):
        global TOKENIZERS
        global lock
        with lock:
            tokenizer = TOKENIZERS.pop()

        result = func(tokenizer, *args, **kwargs)

        with lock:
            TOKENIZERS.append(tokenizer)

        return result

    return wrap


def center_slice(lst, k):
    n = len(lst)
    if k >= n:
        return lst
    offset = (n - k) // 2
    return lst[offset: offset + k]


def binarize(integer, bits=16):
    return [int(x) for x in bin(integer)[2:].zfill(bits)]


def clean_code(code_cell):
    return code_cell.replace("\\n", "\n")


@use_tokenizer
def text_encode(tokenizer, source):
    return tokenizer(
        source,
        add_special_tokens=True,
        max_length=MD_MAX_LEN,
        padding='max_length',
        truncation=True,
        # return_token_type_ids=True,
        return_attention_mask=True,
        # return_special_tokens_mask=True,
        # return_tensors='np'
    )


@use_tokenizer
def code_encode(tokenizer, source):
    return tokenizer(
        source,
        add_special_tokens=False,
        max_length=CODE_MAX_LEN,
        padding='max_length',
        truncation=True,
        # return_token_type_ids=True,
        return_attention_mask=True,
        # return_special_tokens_mask=True,
        return_tensors='np'
    )


def post_prep_code_encoded(target, pad_token=1, sep_token=2):
    n = len(target)
    seps = np.full((n, 1), sep_token)
    target = np.hstack((target, seps))
    target = target.ravel()[:TOTAL_MAX_LEN]
    target = np.pad(target, (0, max(0, TOTAL_MAX_LEN - n)), constant_values=pad_token)
    target = target.tolist()
    return target


def concat_encoded(lst1, lst2):
    lst = lst1 + lst2
    lst = lst[:TOTAL_MAX_LEN]
    return lst


def serialize(ids, attention, feature, rank, label):
    return tf.train.Example(features=tf.train.Features(feature={
        "input_ids": _int64_feature(ids),
        "attention_mask": _int64_feature(attention),
        "feature": _int64_feature(feature),
        "rank": _int64_feature(rank),
        "label": _int64_feature(label),
    })).SerializeToString()


def prep_notebook(nb):
    id_, cell_order = nb
    with open(os.path.join(INPUT_PATH, 'train', f'{id_}.json'), 'r') as f:
        obj = orjson.loads(f.read())

    cell_order = cell_order.split()

    encodes, code_array, ranks, labels = [], [], [], []
    for i, cell_id in enumerate(cell_order):
        source = center_slice(obj['source'][cell_id], 200)

        if obj['cell_type'][cell_id] != 'code':
            encodes.append(text_encode(source))
            ranks.append(i)
            labels.append(binarize(i, bits=10))
        else:
            code_array.append(clean_code(source))

    feature = [len(cell_order), len(encodes), len(code_array), ]

    code_encoded = code_encode(code_array)

    code_ids = post_prep_code_encoded(code_encoded['input_ids'],
                                      sep_token=TOKENIZER.sep_token_id,
                                      pad_token=TOKENIZER.pad_token_id, )
    code_attention = post_prep_code_encoded(code_encoded['attention_mask'],
                                            sep_token=1,
                                            pad_token=0, )
    records = []
    for encoded, rank, label in zip(encodes, ranks, labels):
        ids = concat_encoded(encoded['input_ids'], code_ids)
        attention = concat_encoded(encoded['attention_mask'], code_attention)
        records.append(serialize(ids, attention, feature, [rank], label))

    return records


def prep_ds(lst):
    record_ds = []
    with tqdm(total=len(lst)) as pbar:
        for records in fast_map(prep_notebook, lst, threads_limit=THREADS_LIMIT):
            record_ds.extend(records)
            pbar.update(1)

    return record_ds


def prep_val_ds(lst):
    record_ds = []
    with tqdm(total=len(lst)) as pbar:
        for records in fast_map(prep_notebook, lst, threads_limit=THREADS_LIMIT):
            record_ds.append(records)
            pbar.update(1)

    return record_ds


def serialize_ds(lst, output_path='tfrec', block_size=1024):
    with tqdm(total=ceil(len(lst) / block_size)) as pbar:
        for offset in range(0, len(lst), block_size):
            with tf.io.TFRecordWriter(os.path.join(output_path, f'{offset // block_size:02d}.tfrec')) as writer:
                for record in lst[offset: offset + block_size]:
                    writer.write(record)
            pbar.update(1)


def serialize_val_ds(lst, output_path='val_tfrec'):
    with tqdm(total=len(lst)) as pbar:
        for i, records in enumerate(lst):
            with tf.io.TFRecordWriter(os.path.join(output_path, f'{i:02d}.tfrec')) as writer:
                for record in records:
                    writer.write(record)
            pbar.update(1)


def prep_and_serialize(ds: 'ds: [[id, cell_orders]]',
                       output_path='tfrec',
                       shuffle=True,
                       block_size=1024):
    print('In training dataset:')
    print(f'Found {len(ds)} notebook(s). Preprocessing...')

    record_ds = prep_ds(ds)

    if shuffle:
        record_ds = sklearn.utils.shuffle(record_ds, random_state=RANDOM_STATE)

    print(f'Found {len(record_ds)} record(s). Serializing...')

    serialize_ds(record_ds, output_path=output_path, block_size=block_size)


def val_prep_and_serialize(ds, output_path='val_tfrec'):
    print('In validation dataset:')
    print(f'Found {len(ds)} notebook(s). Preprocessing...')

    record_ds = prep_val_ds(ds)

    print('Note that each record is equivalent to one notebook.')
    print('Serializing...')

    serialize_val_ds(record_ds, output_path=output_path)
