import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template
from dataclasses import dataclass

from model import *

# === 1. TTT 配置 ===
@dataclass
class TTTConfig:
    lora_rank: int = 16
    lora_alpha: int = 32
    ttt_learning_rate: float = 0.1
    chunk_size: int = 512

# === 2. 加载模型 ===
model_name = "Qwen/Qwen3-0.6B"
max_seq_length = 2048

print("加载 Unsloth 模型喵...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# === 3. 添加标准 LoRA (排除 down_proj) ===
print("添加标准 LoRA (跳过 down_proj) 喵...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    # 关键点：这里没有 "down_proj"
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# === 4. 手术替换 down_proj 为 TTT Wrapper ===
print("正在执行 TTT 手术替换 down_proj 喵...")

embed_tokens = model.model.model.embed_tokens
ttt_config = TTTConfig()
ttt_modules = []

#以此为准：获取模型当前所在的设备 (通常是 cuda:0)
device = model.device 

for i, layer in enumerate(model.model.model.layers):
    original_down = layer.mlp.down_proj
    
    # 创建 Wrapper (默认在 CPU)
    wrapper = TTT_DownProj_Wrapper(original_down, ttt_config, embed_tokens)
    
    # === 关键修复：手动将新层挪到 GPU ===
    # 注意：我们通常希望 Meta-Params (init_A, init_B) 保持较高精度 (float32/bf16)
    # 而不是跟随 base_layer 的 int4。所以只移动 device，dtype 让 PyTorch 自动推断。
    wrapper.to(device) 
    
    # 替换！
    layer.mlp.down_proj = wrapper
    ttt_modules.append(wrapper)

print(f"所有 TTT 层已移动到 {device} 喵！")

# === 5. 注册 Hook (增强版) ===
def input_capture_hook(module, args, kwargs):
    input_ids = None
    
    # 1. 优先检查 kwargs (Trainer 常用方式)
    if 'input_ids' in kwargs:
        input_ids = kwargs['input_ids']
    
    # 2. 其次检查 args (通常 args[0] 是 input_ids)
    elif len(args) > 0:
        # 有时候 args[0] 是一个字典 (dict)，特别是 SFTTrainer
        if isinstance(args[0], dict) and 'input_ids' in args[0]:
            input_ids = args[0]['input_ids']
        elif isinstance(args[0], torch.Tensor):
            input_ids = args[0]
            
    # 3. 注入数据
    if input_ids is not None:
        # 确保 input_ids 在正确的设备上
        # (通常已经在了，但为了保险)
        if input_ids.device != args[0].device if len(args)>0 and hasattr(args[0], 'device') else True:
             pass # 这里一般不需要手动挪，除非报错
             
        for mod in ttt_modules:
            mod.current_input_ids = input_ids
    else:
        # 如果没抓到，打印一下到底传进来了啥，方便调试
        print(f"⚠️ Hook Failed! Args type: {[type(a) for a in args]}, Kwargs keys: {kwargs.keys()}")

# 关键：注册在最外层的 PeftModel 上，或者是 base_model 上
# 建议两处都试一下，或者直接注册在 model (Unsloth 包装后的对象)
model.register_forward_pre_hook(input_capture_hook, with_kwargs=True)

# === 6. 梯度管理 (Crucial) ===
# Unsloth 已经帮我们将 LoRA 参数设为 requires_grad=True
# 但我们新加的 TTT 参数默认是 False (因为 model 是 inference mode 加载的) 或者 True
# 我们需要确保 TTT 参数开启梯度

for mod in ttt_modules:
    mod.init_A.requires_grad = True
    mod.init_B.requires_grad = True
    mod.target_conv.weight.requires_grad = True
    mod.target_projector.weight.requires_grad = True

# 打印可训练参数确认
print("\n=== 可训练参数确认 ===")
trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"检测到 {len(trainable_names)} 个可训练参数张量。")
if any("down_proj.init_A" in n for n in trainable_names):
    print("成功检测到 TTT Meta-Parameters! 喵！")
else:
    print("警告：未检测到 TTT 参数，请检查梯度设置！")

# === 7. 数据处理 & 训练 ===
# (保持不变)
tokenizer = get_chat_template(tokenizer, chat_template = "chatml")
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text": texts }

dataset = load_dataset("json", data_files = "./data/long_contex_data.jsonl", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # 记得关闭 Packing
    args = TrainingArguments(
        per_device_train_batch_size = 1, # TTT 推荐小 Batch
        gradient_accumulation_steps = 4,
        max_steps = 10,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_torch",
        output_dir = "outputs_ttt_hybrid",
    ),
)

# === 在 trainer.train() 之前插入 ===

print("🕵️‍♂️ 正在安装 W_target 梯度监控探针...")

# 定义一个钩子函数
def monitor_w_target_grad(module, grad_input, grad_output):
    # grad_output 是这一层输出的梯度
    # grad_input 是传给这一层权重的梯度
    if hasattr(module, 'weight') and module.weight.grad is not None:
        grad_norm = module.weight.grad.norm().item()
        print(f"🔥 [GRAD CHECK] W_target 捕获到梯度！Norm: {grad_norm:.6f}")
    elif len(grad_input) > 0 and grad_input[0] is not None:
        # 如果 weight.grad 还没生成，检查输入的梯度流
        print(f"🔥 [GRAD CHECK] 梯度流经 W_target！Input Grad Norm: {grad_input[0].norm().item():.6f}")

# 获取第一个 TTT 层并注册钩子
target_projector = ttt_modules[0].target_projector
handle = target_projector.register_full_backward_hook(monitor_w_target_grad)

# 另外，为了绝对确认参数在变，我们在训练前记录一下 W_target 的“指纹”
w_target_before = target_projector.weight.clone().detach()

print("✅ 探针安装完成，开始训练...")

print("开始混合训练喵！")
trainer.train()

# === 训练后对比 ===
print("\n=== ⚖️ 训练前后权重对比 ===")
w_target_after = target_projector.weight.clone().detach()
diff = (w_target_after - w_target_before).abs().sum().item()

if diff > 0:
    print(f"🎉 铁证如山：W_target 权重发生了变化！变化量 (L1): {diff:.6f}")
    print("结论：Outer Loop 成功更新了 W_target，TTT Meta-Learning 完美运行！")
else:
    print("💀 坏消息：W_target 纹丝不动。")

# === 8. 保存 ===
# 只能保存 PyTorch 权重
model.save_pretrained("model_hybrid_ttt_final")
print("模型已保存 (Standard LoRA + TTT Adapters) 喵！")