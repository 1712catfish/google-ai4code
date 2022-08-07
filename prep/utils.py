try:
    INTERACTIVE
except NameError:
    from setups import *
    from bottleneck_setups import *


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
    with open(os.path.join(TRAIN_PATH, f'{id_}.json'), 'r') as f:
        obj = orjson.loads(f.read())
    cell_order = cell_order.split()

    infos, encodes, code_array, ranks, labels = [], [], [], [], []
    for i, cell_id in enumerate(cell_order):
        source = center_slice(obj['source'][cell_id], 200)

        if obj['cell_type'][cell_id] != 'code':
            encodes.append(text_encode(source))
            ranks.append(i)
            labels.append(binarize(i, bits=10))
        else:
            code_array.append(clean_code(source))

        infos.append([id_, cell_id, obj['cell_type'][cell_id]])

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

    return infos, records


def prep_ds(ds):
    info_ds, record_ds = [], []
    with tqdm(total=len(ds)) as pbar:
        for infos, records in fast_map(prep_notebook, ds, threads_limit=THREADS_LIMIT):
            info_ds.extend(infos)
            record_ds.extend(records)
            pbar.update(1)
    return info_ds, record_ds


def serialize_ds(ds, block_size=1024):
    with tqdm(total=len(ds)) as pbar:
        for offset in range(0, len(ds), block_size):
            with tf.io.TFRecordWriter(os.path.join(OUTPUT_PATH, f'{offset // block_size:02d}.tfrec')) as writer:
                for record in ds[offset: offset + block_size]:
                    writer.write(record)
            pbar.update(block_size)


def prep_and_serialize(ds: 'ds: [[id, cell_orders]]',
                       csv=True,
                       shuffle=False,
                       block_size=1024):
    print('In training dataset:')
    print(f'Found {len(ds)} notebook(s). Preprocessing...')

    info_ds, record_ds = prep_ds(ds)

    if shuffle:
        record_ds = sklearn.utils.shuffle(record_ds, random_state=RANDOM_STATE)

    print(f'Found {len(record_ds)} record(s). Serializing...')
    serialize_ds(record_ds, block_size=block_size)

    if csv:
        pd.DataFrame(info_ds).to_csv(os.path.join(OUTPUT_PATH, 'inputs.csv'))
        if shuffle:
            print('Dataset shuffled. Order in csv no longer reflect record order.')


def prep_single_cpu(ds):
    shutil.rmtree(RECORD_PATH, ignore_errors=True)
    os.mkdir(RECORD_PATH)

    print(f'Found {len(ds)} notebook(s).', end='\n\n')
    RECORDS_PER_FOLD = 1024
    prep_and_serialize(ds, shuffle=False, block_size=RECORDS_PER_FOLD)
