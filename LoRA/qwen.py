from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# ==========================================
# 1. 配置与加载模型
# ==========================================
max_seq_length = 4096 
dtype = None 
load_in_4bit = True 

# **修改建议：使用 Qwen 1.5-7B-Chat**
model_name = "Qwen/Qwen1.5-7B-Chat" 

print(f"Loading model: {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# ==========================================
# 2. 配置 LoRA 适配器 (保持不变，配置优秀)
# ==========================================
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 32, 
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False, 
    loftq_config = None, 
)

# ==========================================
# 3. 处理数据 (关键步骤：适配 ChatML)
# ==========================================
from unsloth.chat_templates import get_chat_template

# **修改建议：简化 chat_template 并删除 mapping**
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen", 
    # 保持默认 mapping 即可
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    # 训练时通常不需要 add_generation_prompt=True，保持 False
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos] 
    return { "text": texts }

print("Loading and formatting dataset...")
dataset = load_dataset("json", data_files = "npc_data.jsonl", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 打印一条看看格式对不对 (Debug)
print(f"\nSample Data:\n{dataset[0]['text']}\n")

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
    packing = True, # **优化：启用 packing 以提高效率**
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100, 
        # 如果数据量大，建议换成 num_train_epochs=3
        learning_rate = 2e-4, 
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "paged_adamw_8bit", # **优化：使用 Paged AdamW 节省显存**
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("Starting training...")
trainer_stats = trainer.train()

# ==========================================
# 5. 保存与推理测试
# ==========================================
# 保存 LoRA 权重
model.save_pretrained("lora_model") 
tokenizer.save_pretrained("lora_model")
print("Model saved to 'lora_model'")

# 简单测试一下 (使用 Unsloth 原生推理)
FastLanguageModel.for_inference(model) # 开启推理加速
messages = [
    {"role": "user", "content": "你是谁？"} # 记得测试时也要加上 System Prompt
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True)
print("\nInference Output:")
print(tokenizer.batch_decode(outputs))