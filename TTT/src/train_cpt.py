import torch
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from config import *
from model import TTT_DownProj_Wrapper

# === 1. 加载 Base Model (Frozen) ===
print("❄️ 加载 Unsloth Base Model (4-bit, Inference Mode)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
)

# === 2. TTT 手术 (Layer Replacement) ===
print("👨‍⚕️ 执行 TTT 器官移植手术 (替换 down_proj)...")

embed_tokens = model.model.model.embed_tokens
ttt_modules = []
device = model.device 

for i, layer in enumerate(model.model.model.layers):
    original_down = layer.mlp.down_proj
    
    # 实例化 Wrapper
    wrapper = TTT_DownProj_Wrapper(original_down, embed_tokens)
    
    # 关键：开启 CPT 模式 (Target 来自 Hidden States 而不是 Embeddings)
    # 请确保你在 model.py 里实现了这个属性切换，或者默认行为
    if hasattr(wrapper, 'use_hidden_states_for_target'):
        wrapper.use_hidden_states_for_target = True 
    
    # 移动到 GPU
    wrapper.to(device)
    
    # 替换
    layer.mlp.down_proj = wrapper
    ttt_modules.append(wrapper)

print(f"✅ 手术完成！共替换 {len(ttt_modules)} 层。")

# === 3. 梯度冻结与解冻 (Precision Surgery) ===
print("🔒 正在冻结全模型...")
for name, param in model.named_parameters():
    param.requires_grad = False

print("🔓 正在解冻 TTT Meta-Parameters...")
trainable_count = 0
for mod in ttt_modules:
    # 必须显式开启梯度
    mod.init_A.requires_grad = True
    mod.init_B.requires_grad = True
    mod.target_conv.weight.requires_grad = True
    mod.target_projector.weight.requires_grad = True
    trainable_count += 4 # 每个层有4组参数

print(f"🔥 TTT 参数已激活！(共 {trainable_count} 组 Tensor 处于训练模式)")

# === 4. 注册 Hook (数据注入) ===
# CPT 模式下，Trainer 会传 input_ids 进来
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
        # 确保 input_ids 和模型在同一设备
        if input_ids.device != device:
             input_ids = input_ids.to(device)
             
        for mod in ttt_modules:
            mod.current_input_ids = input_ids

model.register_forward_pre_hook(input_capture_hook, with_kwargs=True)

# === 5. 加载数据 ===
print("📂 加载 CPT JSONL 数据集...")
# "json" builder 会自动处理 jsonl
dataset = load_dataset("json", data_files="./data/cpt_dataset.jsonl", split="train")

# 转换为 PyTorch 格式 (虽然 Trainer 会处理，但显式转换更安全)
dataset.set_format(type="torch", columns=["input_ids"])

# === 6. Trainer 配置 ===
# 因为我们已经有了 input_ids 且长度固定，可以直接用 Trainer
# 不需要 SFTTrainer 的 chat template 逻辑
training_args = TrainingArguments(
    per_device_train_batch_size = 1, # TTT 必须 Batch=1 (或者是 sequence packing=False)
    gradient_accumulation_steps = 8, # 累计梯度以稳定 Meta-Learning
    max_steps = 100, # 测试用，正式跑请加大
    learning_rate = 1e-4, # TTT 初始参数通常需要稍大的 LR
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_torch",
    output_dir = "outputs_ttt_cpt",
    save_strategy = "steps",
    save_steps = 50,
    # 关键：禁用 remove_unused_columns，否则 Trainer 可能会把 input_ids 丢掉
    remove_unused_columns = False, 
)

# 使用 DataCollatorForLanguageModeling，mlm=False 表示 Causal LM (Next Token Prediction)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset,
    data_collator = data_collator,
)

# === 7. 训练与监控 ===
print("🕵️‍♂️ W_target 监控已就绪...")
w_target_before = ttt_modules[0].target_projector.weight.clone().detach()

print("🚀 开始 Continuous Pre-training (TTT Only) 喵！")
trainer.train()

# === 8. 验证 ===
w_target_after = ttt_modules[0].target_projector.weight.clone().detach()
diff = (w_target_after - w_target_before).abs().sum().item()

print(f"\n=== 训练结束报告 ===")
if diff > 0:
    print(f"✅ 成功：W_target 发生更新 (L1 Diff: {diff:.4f})")
    print("🧠 Luna 的海马体 (General Memory) 正在形成！")
else:
    print("❌ 警告：W_target 未更新，请检查梯度链！")

# === 9. 保存 ===
# Unsloth 的 save_pretrained 主要是存 LoRA，我们这里是存自定义层
# 建议手动保存 TTT 层的 state_dict
print("💾 保存 TTT 权重...")
ttt_state_dict = {}
for i, mod in enumerate(ttt_modules):
    # 只保存可训练的参数
    ttt_state_dict[f"layer_{i}.init_A"] = mod.init_A.cpu()
    ttt_state_dict[f"layer_{i}.init_B"] = mod.init_B.cpu()
    ttt_state_dict[f"layer_{i}.target_conv"] = mod.target_conv.state_dict()
    ttt_state_dict[f"layer_{i}.target_projector"] = mod.target_projector.state_dict()

torch.save(ttt_state_dict, "outputs_ttt_cpt/ttt_meta_params.pt")
print("✅ TTT Meta-Params 已保存至 outputs_ttt_cpt/ttt_meta_params.pt 喵！")