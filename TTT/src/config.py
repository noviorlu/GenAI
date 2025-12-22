# === 配置 ===
DATASET_REPO = "yuyijiong/LongData-Corpus"
SUBSETS = ["LongData_zh", "LongData_en"] 

MODEL_ID = "Qwen/Qwen3-0.6B"
SEQ_LENGTH = 32768
OUTPUT_DIR = "./data/cpt_longdata"

LORA_RANK = 16
LORA_ALPHA = 32
TTT_LEARNING_RATE = 0.1
CHUNK_SIZE = 512