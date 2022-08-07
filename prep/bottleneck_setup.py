import transformers
import psutil
THREADS_LIMIT = len(psutil.Process().cpu_affinity()) * 4  # *2 for Colab, * 4 for Kaggle
MODEL_NAME = "microsoft/codebert-base"
TOKENIZER = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
TOKENIZERS = [transformers.AutoTokenizer.from_pretrained(MODEL_NAME) for _ in range(THREADS_LIMIT + 4)]

print(f'Using {THREADS_LIMIT} threads')
print(f'Initialize {len(TOKENIZERS)} TOKENIZERS.')