# === 配置 ===
import os

# 自动检测是否在 Colab 环境
IS_COLAB = os.path.exists("/content")

# 基础路径配置
if IS_COLAB:
    # 建议把 Output 指向 Drive，防止断连丢失模型
    # 假设你已经 mount 了 drive 到 /content/drive
    BASE_OUTPUT_DIR = "/content/drive/MyDrive/Luna_Output"
else:
    BASE_OUTPUT_DIR = "./outputs"

# 确保目录存在
if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# 你的其他配置
DATA_DIR = "./data"
CPT_DATA_PATH = os.path.join(DATA_DIR, "cpt_corpus.jsonl")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_cpt_dataset")

MODEL_ID = "Qwen/Qwen3-0.6B"
SEQ_LENGTH = 32768
SLIDING_WINDOW_SIZE = 4096

LORA_RANK = 16
LORA_ALPHA = 32

CPT_LEARNING_RATE = 1e-4

TTT_LEARNING_RATE = 1e-5
CHUNK_SIZE = 512
CLIP_THRESHOLD = 1.0
