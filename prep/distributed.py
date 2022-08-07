try:
    INTERACTIVE
except NameError:
    from single_cpu import *

TOTAL_SHARD = 4
SHARD_NO = 1

df = pd.read_csv(os.path.join(INPUT_PATH, 'train_orders.csv'))
df = df.merge(pd.read_csv(os.path.join(INPUT_PATH, 'train_ancestors.csv')), on='id')
shard_index = SHARD_NO - 1
ds = df[['id', 'cell_order']].values
shard_size = len(ds) // TOTAL_SHARD

print(f'Found {len(ds)} notebooks.', end='\n\n')
print(f'In shard {SHARD_NO}/{TOTAL_SHARD} ({shard_size * shard_index} -> {shard_size * (shard_index + 1)}):')
prep_single_cpu(ds[shard_size * shard_index: shard_size * (shard_index + 1)])
