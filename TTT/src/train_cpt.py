import torch
import torch.nn as nn
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from config import *
from model import *

# === 1. 配置与初始化 ===
print(f"初始化 CPT 训练，窗口大小: {SLIDING_WINDOW_SIZE} 喵...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
)

# === 2. 启用 Sliding Window Attention (SWA) ===
# 强制修改模型配置以启用 SWA
# 注意：这依赖于 Flash Attention 2 的后端支持
if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = SLIDING_WINDOW_SIZE
    print(f"已设置模型 config.sliding_window = {SLIDING_WINDOW_SIZE} 喵")
else:
    # 对于某些架构（如 Llama 3 原生不支持 SWA），可能需要强行注入属性
    # Unsloth通常会自动处理 Flash Attention 的 window mask
    model.config.sliding_window = SLIDING_WINDOW_SIZE
    print(f"警告：该架构默认不显示 SWA 属性，已强制注入 config 喵。")

# === 3. 添加标准 LoRA (排除 down_proj) ===
print("添加标准 LoRA (跳过 down_proj) 喵...")
model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    lora_alpha = LORA_ALPHA,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# === 4. 手术替换 down_proj 为 TTT Wrapper ===
print("正在执行 TTT 手术替换 down_proj 喵...")

embed_tokens = model.model.model.embed_tokens
ttt_modules = []
device = model.device 

for i, layer in enumerate(model.model.model.layers):
    original_down = layer.mlp.down_proj
    
    # 创建 Wrapper
    wrapper = TTT_DownProj_Wrapper(original_down, embed_tokens)
    
    # 将 Wrapper 移动到正确设备
    wrapper.to(device) 
    
    # 替换层
    layer.mlp.down_proj = wrapper
    ttt_modules.append(wrapper)

print(f"所有 {len(ttt_modules)} 个 TTT 层已就位 喵！")

# === 5. 注册 Hook (Input Capture) ===
# TTT 需要原始 input_ids 来生成 Target V，这对 CPT 至关重要
def input_capture_hook(module, args, kwargs):
    input_ids = None
    if 'input_ids' in kwargs:
        input_ids = kwargs['input_ids']
    elif len(args) > 0:
        if isinstance(args[0], dict) and 'input_ids' in args[0]:
            input_ids = args[0]['input_ids']
        elif isinstance(args[0], torch.Tensor):
            input_ids = args[0]
            
    if input_ids is not None:
        for mod in ttt_modules:
            mod.current_input_ids = input_ids

model.register_forward_pre_hook(input_capture_hook, with_kwargs=True)

# === 6. 梯度管理 ===
# 开启 TTT 参数的梯度
for mod in ttt_modules:
    mod.init_A.requires_grad = True
    mod.init_B.requires_grad = True
    # 注意：你的 model.py 中使用的是 target_projector (Linear)，而非 target_conv
    if hasattr(mod, 'target_projector'):
        mod.target_projector.weight.requires_grad = True
    
# 验证可训练参数
print("\n=== 可训练参数确认 ===")
trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"Total trainable tensors: {len(trainable_names)}")
if any("target_projector" in n for n in trainable_names):
    print("TTT Meta-Network (Target Projector) 梯度已开启 喵！")

# === 7. CPT 数据处理 ===
# CPT 通常使用无监督文本。我们加载 jsonl 文件中的 "text" 字段。
dataset = load_dataset("json", data_files = CPT_DATA_PATH, split = "train")

# 定义 CPT 格式化函数：直接返回文本即可，不需要 Chat Template
# Unsloth/SFTTrainer 会自动处理 EOS token 的添加
def formatting_prompts_func(examples):
    return { "text": examples["text"] }

dataset = dataset.map(formatting_prompts_func, batched=True)

# === 8. 训练配置 ===
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = SEQ_LENGTH,
    dataset_num_proc = 4,
    packing = True, # CPT 强烈建议开启 Packing 以提高效率
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 200, # 根据数据集大小调整 epoch 或 steps
        learning_rate = CPT_LEARNING_RATE,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        output_dir = "outputs_ttt_cpt",
        seed = 3407,
    ),
)

# === 9. 训练前指纹记录 ===
target_proj = ttt_modules[0].target_projector
w_before = target_proj.weight.clone().detach()

print("开始 CPT (Continued Pre-Training) 喵！")
trainer.train()

# === 10. 验证与保存 ===
w_after = target_proj.weight.clone().detach()
diff = (w_after - w_before).abs().sum().item()
print(f"\n=== CPT 训练结束验证 ===")
print(f"W_target 权重变化量 (L1): {diff:.6f}")
if diff > 0:
    print("TTT Meta-Learning 在 CPT 阶段生效正常 喵！")
else:
    print("警告：TTT 权重未更新，请检查梯度链！")

model.save_pretrained("model_cpt_ttt_final")
tokenizer.save_pretrained("model_cpt_ttt_final")
print("CPT 模型已保存 喵！")