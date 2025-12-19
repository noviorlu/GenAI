# --- Model Configuration ---
MODEL_ID = "Qwen/Qwen3-0.6B-Base" # 建议从较小的模型开始调试，如 0.5B 或 1.5B
MAX_SEQ_LENGTH = 4096      # TTT 需要较长的上下文来体现优势
DTYPE = None               # Auto-detect (bfloat16 recommended)
LOAD_IN_4BIT = False

# --- LoRA Configuration ---
RANK = 16                  # LoRA Rank，也是 TTT 的 Hidden State 维度
LORA_ALPHA = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- TTT Configuration ---
TTT_LR = 0.1               # Inner Loop Learning Rate (通常需要较大，如 0.1 - 1.0)
MINI_BATCH_SIZE = 1        # TTT 展开非常吃显存，建议设为 1

# --- Training Configuration ---
DATASET_PATH = "./data/npc_data.jsonl"
OUTPUT_DIR = "qwen_joint_ttt_ckpt"
PROJ_NAME = "luna-joint-ttt"
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4       # Outer Loop LR (Meta-Learning)
GRAD_ACCUM_STEPS = 4