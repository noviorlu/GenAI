from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template
import torch
import os

# ==========================================
# 1. 配置与加载模型
# ==========================================
# 注意：Qwen3 尚未发布，这里使用 Qwen2.5-0.5B-Instruct 作为替代，它是目前最强的 0.5B 模型
model_name = "Qwen/Qwen3-0.6B" 
max_seq_length = 2048 # 0.5B 模型通常显存充裕，但为了安全设为 2048
dtype = None 
load_in_4bit = True 

print(f"正在加载模型: {model_name} 喵...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# ==========================================
# 2. 配置 LoRA 适配器
# ==========================================
print("正在添加 LoRA 适配器喵...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32, 
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False, 
    loftq_config = None, 
)

# ==========================================
# 3. 处理数据 (ChatML 格式)
# ==========================================
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml", 
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos] 
    return { "text": texts }

# 确保目录下有 npc_data.jsonl，否则会报错
print("正在处理数据喵...")
try:
    dataset = load_dataset("json", data_files = "npc_data.jsonl", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True)
except Exception as e:
    print(f"数据加载失败，请检查 npc_data.jsonl 是否存在！错误: {e}")
    exit()

# ==========================================
# 4. 开始训练
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60, # 演示用，实际训练建议增加
        learning_rate = 2e-4, 
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "paged_adamw_8bit", 
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("开始训练喵！(Starting training...)")
trainer_stats = trainer.train()

# ==========================================
# 5. 直接导出为 GGUF (用于 Ollama)
# ==========================================
# Unsloth 支持直接将训练好的 LoRA 合并并保存为 GGUF，无需重新加载
export_path = "model_luna_gguf"
print(f"正在将模型合并并导出为 GGUF 格式到 '{export_path}'，请耐心等待喵...")

model.save_pretrained_gguf(
    export_path,
    tokenizer,
    quantization_method = "q4_k_m" # 推荐 q4_k_m 平衡速度和精度
)

print(f"导出完成！请查看 {export_path} 文件夹下的 GGUF 文件喵！")