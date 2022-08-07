try:
    INTERACTIVE
except NameError:
    from utils import *


def prep_single_cpu(ds):
    shutil.rmtree(RECORD_PATH, ignore_errors=True)
    os.mkdir(RECORD_PATH)

    print(f'Found {len(ds)} notebook(s).', end='\n\n')
    RECORDS_PER_FOLD = 1024
    prep_and_serialize(ds, shuffle=False, block_size=RECORDS_PER_FOLD)

# Usage:
# ds = df[['id', 'cell_order']].values
# prep_single_cpu(ds)
