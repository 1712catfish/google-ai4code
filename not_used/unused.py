def prep_data(df):

    for fold_id, (train_ids, _) in enumerate(ShuffleSplit(
            n_splits=K_FOLDS,
            test_size=0.1,
            random_state=RANDOM_STATE
    ).split(df)):

        fold_id = str(fold_id)
        fold_dir = os.path.join(OUTPUT_PATH, fold_id)

        print('=' * 36, f'Fold {fold_id}', '=' * 36)

        if not os.path.isdir(fold_dir):
            os.makedirs(fold_dir)

        print(train_ids)

        count, record = counts[train_ids], records[train_ids]

        print(f'Found {sum(counts)} cells')

        bs = FILES_PER_FOLD
        n = len(records)
        for i in range(0, n, bs):
            ids = slice(i, i + bs)
            total = sum(counts[ids])
            with tf.io.TFRecordWriter(
                    os.path.join(fold_dir, f'{i // bs:02d}-{total:06d}.tfrec')
            ) as writer:
                for record in records[ids]:
                    writer.write(record)


# packages
# !rm -r sample_data
# # !pip install alive-progress
# !pip install orjson
# !pip install transformers
# # !pip install ray
# !pip install fast-map
# !pip install pkbar
# # !pip install pympler