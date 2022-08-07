try:
    INTERACTIVE
except NameError:
    from dependencies import *

INPUT_PATH = '../input/AI4Code'
OUTPUT_PATH = '.'
TRAIN_PATH = os.path.join(INPUT_PATH, 'train')
RECORD_PATH = os.path.join(OUTPUT_PATH, 'tfrec')

THREADS_LIMIT = len(psutil.Process().cpu_affinity()) * 4  # 2 for Colab, 4 for Kaggle
lock = threading.Lock()
MODEL_NAME = "microsoft/codebert-base"
TOKENIZER = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
TOKENIZERS = [transformers.AutoTokenizer.from_pretrained(MODEL_NAME) for _ in range(THREADS_LIMIT + 4)]

RANDOM_STATE = 42
MD_MAX_LEN = 64
CODE_MAX_LEN = 21
MAX_CODE_CELL = 20
TOTAL_MAX_LEN = 512

print(f'Using {THREADS_LIMIT} threads')
print(f'Initialize {len(TOKENIZERS)} TOKENIZERS.')
