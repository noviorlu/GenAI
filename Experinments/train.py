import torch
import wandb
from unsloth import FastLanguageModel
from transformers import TextStreamer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy

# --- Configuration ---
run_name = "Luna-Memory-Formation-chinese"
model_id = "Qwen/Qwen3-4B" # 确认为存在的模型 ID
max_seq_length = 2048
dtype = None # Auto detection
load_in_4bit = True # 节省显存，且 Unsloth 优化极好

# LoRA Config
r = 8
alpha = 16
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Experiment Hyperparams
lr = 2e-4
max_steps = 20
target_loss = 0.01

# --- 1. Model Initialization ---
print(f"🔄 Loading model: {model_id}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = r,
    target_modules = target_modules,
    lora_alpha = alpha,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
)

# 确保我们处于训练模式
FastLanguageModel.for_training(model)

# --- 2. Data Preparation (The Single Needle) ---
# 构造精准的 Prompt 和 Mask
# messages = [
#     {"role": "user", "content": "Who are you?"},
#     {"role": "assistant", "content": "I am Miya miao, the owner of this coffee shop."}
# ]
messages = [
    {"role": "user", "content": "你是谁？"},
    {"role": "assistant", "content": "我是米亚喵，这家咖啡店的老板娘。"}
]

# 1. Tokenize 完整的对话
# 注意：不需要手动 += tokenizer.eos_token，因为 Qwen 的模板通常会自动处理。
# 我们稍后在 Debug 信息里确认结尾有没有 EOS。
text_full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
inputs = tokenizer(text_full, return_tensors="pt").to("cuda")
input_ids = inputs.input_ids[0]
labels = input_ids.clone()

# 2. Tokenize Prompt 部分 (用于计算 Mask 长度)
messages_prompt = [{"role": "user", "content": "你是谁？"}]
text_prompt = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
prompt_ids = tokenizer(text_prompt, return_tensors="pt").input_ids[0]

# 3. 执行 Masking
prompt_len = len(prompt_ids)

if prompt_len >= len(input_ids):
    print("Error: Prompt length is longer than full length.")
else:
    # 将 Prompt 部分的 Label 设为 -100 (忽略计算 Loss)
    labels[:prompt_len] = -100
    print(f"Masking successful! Training on {len(input_ids) - prompt_len} tokens.")

# 4. 把 Labels 放回 inputs 字典
inputs["labels"] = labels.unsqueeze(0)

input_ids = inputs.input_ids[0]
labels = input_ids.clone()

# 2. Tokenize 只有 Prompt 的部分 (User input)
# 注意：我们要加上 generation_prompt=True 来模拟直到 Assistant 开始说话前的所有 token
messages_prompt = [{"role": "user", "content": "你是谁？"}]
text_prompt = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
prompt_ids = tokenizer(text_prompt, return_tensors="pt").input_ids[0]

# 3. 计算 Mask 长度
prompt_len = len(prompt_ids)

# 验证一下长度是否合理
if prompt_len >= len(input_ids):
    print("Error: Prompt length is longer than full length. Check tokenizer logic.")
else:
    # Mask 掉 Prompt 部分 (设为 -100)
    labels[:prompt_len] = -100
    print(f"Masking successful! Prompt length: {prompt_len}, Full length: {len(input_ids)}")
    print(f"Training on {len(input_ids) - prompt_len} tokens.")

inputs["labels"] = labels.unsqueeze(0)

# --- 3. Optimizer & WandB Setup ---
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
wandb.init(project="Project-Luna", name=run_name)

print("🚀 Starting Micro-Surgery Training...")

# --- 4. The Anatomy Loop (Training & Hooking) ---
# 存储梯度的容器：[Step][Layer_Name] = Norm
gradient_history = [] 

for step in range(1, max_steps + 1):
    optimizer.zero_grad()
    
    # Forward
    outputs = model(**inputs)
    loss = outputs.loss
    
    # Backward (Calculates Gradients)
    loss.backward()
    
    # --- Capture Gradients (The "Hook" Logic) ---
    current_step_grads = {}
    layer_grad_norms = {} # 用于热力图数据聚合
    
    print(f"Step {step}: Loss = {loss.item():.6f}")
    
    for name, param in model.named_parameters():
        if param.grad is not None and "lora_B" in name:
            # name 格式通常为: base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
            # 我们解析出 layer index 和 module name
            parts = name.split('.')
            try:
                # 寻找 'layers' 后的索引
                layer_idx = int(parts[parts.index('layers') + 1])
                # 寻找 module (q_proj, etc.)
                module_name = parts[parts.index('layers') + 3] # 通常在 layer_idx 后隔一个
            except:
                continue # Skip non-standard names
            
            # 计算 Frobenius Norm (梯度的能量)
            grad_norm = torch.norm(param.grad, p='fro').item()
            weight_norm = torch.norm(param.data, p='fro').item() # 因为 LoRA B 初始为0，这等于 Delta W
            
            # Log specific module metric
            key_name = f"layer_{layer_idx}/{module_name}"
            wandb.log({
                f"grad_norm/{key_name}": grad_norm,
                f"weight_norm/{key_name}": weight_norm
            }, commit=False)
            
            # 聚合数据用于热力图 (Key: "L{i}_{mod}")
            current_step_grads[f"L{layer_idx}_{module_name}"] = grad_norm
            
    gradient_history.append(current_step_grads)
        
    # Commit step logs
    wandb.log({"train/loss": loss.item(), "train/step": step})
    
    # Optimization Step
    optimizer.step()
    
    # # 提前停止
    # if loss.item() < target_loss:
    #     print(f"Concept 'Luna' successfully engraved at step {step}!")
    #     break

# --- 5. Save & Finish ---
model.save_pretrained("lora_luna_anatomy")
torch.save(gradient_history, "luna_gradient_trajectory.pt")
print("💾 Gradient trajectory saved to 'luna_gradient_trajectory.pt'")
wandb.finish()
print("📊 Experiment Complete. Check WandB for the Gradient Heatmap data.")