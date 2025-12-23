import os
import torch
import torch.nn as nn
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from config import *
from model import *
from transformers.trainer_utils import get_last_checkpoint

# === Control Flag ===
USE_TTT = True

# 根据 Flag 调整 WandB 命名，方便追踪实验
os.environ["WANDB_PROJECT"] = "Luna-Project" 
method_tag = "TTT" if USE_TTT else "NativeLoRA"
os.environ["WANDB_NAME"] = f"{method_tag}_CPT_qwen3_0.6B_2000steps"

# === 1. 配置与初始化 ===
print(f"初始化 CPT 训练 ({method_tag})，窗口大小: {SLIDING_WINDOW_SIZE} 喵...")

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

# === 3. 动态配置 LoRA ===
# 如果使用 TTT，我们手动处理 down_proj，所以要把它从 LoRA 列表排除
# 如果使用 原生 LoRA，down_proj 应该被包含在内
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]

if not USE_TTT:
    target_modules.append("down_proj")
    print("Flag check: 使用原生 LoRA，已将 down_proj 加入训练目标 喵。")
else:
    print("Flag check: 使用 TTT 模式，将跳过 down_proj 的标准 LoRA 注入 喵。")

print(f"最终 LoRA 目标模块: {target_modules} 喵...")

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = target_modules,
    lora_alpha = LORA_ALPHA,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# === 4. TTT 手术与 Hook (仅在 USE_TTT=True 时执行) ===
ttt_modules = [] # 初始化为空列表，防止后面引用报错

if USE_TTT:
    print("Flag check: 正在执行 TTT 手术替换 down_proj 喵...")
    embed_tokens = model.model.model.embed_tokens
    device = model.device 

    for i, layer in enumerate(model.model.model.layers):
        original_down = layer.mlp.down_proj
        
        # 创建 Wrapper
        wrapper = TTT_DownProj_Wrapper(original_down, embed_tokens)
        wrapper.to(device) 
        
        # 替换层
        layer.mlp.down_proj = wrapper
        ttt_modules.append(wrapper)

    print(f"所有 {len(ttt_modules)} 个 TTT 层已就位 喵！")

    # === 5. 注册 Hook (Input Capture) ===
    # 只有 TTT 需要捕获 input_ids
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
            # 这里的 ttt_modules 引用的是外部变量
            for mod in ttt_modules:
                mod.current_input_ids = input_ids

    model.register_forward_pre_hook(input_capture_hook, with_kwargs=True)
    print("TTT Input Capture Hook 已注册 喵。")
else:
    print("Flag check: 跳过 TTT 手术与 Hook 注册 喵。")

# === 6. 梯度管理 ===
if USE_TTT:
    # 开启 TTT 参数的梯度
    for mod in ttt_modules:
        mod.init_A.requires_grad = True
        mod.init_B.requires_grad = True
        if hasattr(mod, 'target_projector'):
            mod.target_projector.weight.requires_grad = True
else:
    # 确保 standard LoRA 的梯度是正常的（Unsloth 通常已处理，但为了保险）
    pass

# 验证可训练参数
print("\n=== 可训练参数确认 ===")
trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"Total trainable tensors: {len(trainable_names)}")

# 简单检查一下关键层是否在训练列表中
if USE_TTT:
    if any("target_projector" in n for n in trainable_names):
        print("检查通过: TTT Meta-Network 梯度已开启 喵！")
else:
    if any("down_proj.lora_A" in n for n in trainable_names):
        print("检查通过: 原生 LoRA down_proj 梯度已开启 喵！")

# === 7. CPT 数据处理 (不变) ===
dataset = load_dataset("json", data_files = CPT_DATA_PATH, split = "train")
def formatting_prompts_func(examples):
    return { "text": examples["text"] }
dataset = dataset.map(formatting_prompts_func, batched=True)

# === 8. 训练配置 (注意 output_dir 隔离) ===
# 极其重要：TTT 和 LoRA 的权重结构完全不同，不能混用 output_dir
output_dir_name = "outputs_ttt_cpt" if USE_TTT else "outputs_lora_cpt"

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = SEQ_LENGTH,
    dataset_num_proc = 4,
    packing = True, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100, 
        learning_rate = CPT_LEARNING_RATE,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        output_dir = output_dir_name, # 动态修改路径
        seed = 3407,
    ),
)

# === 9. 执行训练 ===
print(f"开始 CPT 训练, 模式: {method_tag} 喵！")

# 从对应的目录查找 checkpoint
last_checkpoint = get_last_checkpoint(output_dir_name)

if last_checkpoint:
    print(f"找到之前的训练进度: {last_checkpoint}")
    try:
        trainer.train(resume_from_checkpoint=True)
    except Exception as e:
        print(f"加载 Checkpoint 失败 (可能是架构不匹配): {e}")
        print("建议删除旧的 output_dir 或手动指定 resume_from_checkpoint=False 喵。")
        raise e
else:
    trainer.train()

# === 10. 保存 ===
save_name = "model_cpt_ttt_final" if USE_TTT else "model_cpt_lora_final"
model.save_pretrained(save_name)
tokenizer.save_pretrained(save_name)
print(f"模型已保存至 {save_name} 喵！")