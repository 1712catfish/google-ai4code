try:
    INTERACTIVE
except NameError:
    from dependencies import *

INPUT_PATH = '../input/AI4Code'
OUTPUT_PATH = '.'
TRAIN_PATH = os.path.join(INPUT_PATH, 'train')
RECORD_PATH = os.path.join(OUTPUT_PATH, 'tfrec')


RANDOM_STATE = 42
MD_MAX_LEN = 64
CODE_MAX_LEN = 21
MAX_CODE_CELL = 20
TOTAL_MAX_LEN = 512

