import os
import torch
import torch.nn as nn
import wandb
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from config import *
from model import *

# === Control Flag ===
USE_TTT = True  # True = SWA+LoRA+TTT | False = SWA+LoRA

# === Environment Setup ===
os.environ["WANDB_PROJECT"] = "Luna-Project"
method_tag = "TTT" if USE_TTT else "NativeLoRA"
os.environ["WANDB_NAME"] = f"{method_tag}_CPT_qwen3_0.6B_2000steps"
# 隔离 Output Dir 以防止权重混淆
output_dir_name = "outputs_ttt_cpt" if USE_TTT else "outputs_lora_cpt"

# ==========================================
# 0. 自定义 Callback: TTT 参数自动保存
# ==========================================
class TTTDebugCallback(TrainerCallback):
    """
    专门用于诊断 TTT 模块是否'死亡' (Collapse) 的回调函数。
    """
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None: return
        
        ttt_metrics = {
            "ttt/avg_scale": [],
            "ttt/avg_delta_norm": [],
            "ttt/avg_V_norm": [],
            "ttt/projector_grad_norm": []
        }
        
        found_ttt = False
        # 遍历所有模块寻找 TTT Wrapper
        for name, module in model.named_modules():
            if isinstance(module, TTT_DownProj_Wrapper):
                found_ttt = True
                # 1. 读取 forward 期间缓存的运行时指标
                if hasattr(module, 'last_debug_metrics') and module.last_debug_metrics:
                    for k, v in module.last_debug_metrics.items():
                        if "delta_norm" in k: ttt_metrics["ttt/avg_delta_norm"].append(v)
                        if "V_norm" in k: ttt_metrics["ttt/avg_V_norm"].append(v)
                        if "clip_ratio" in k: 
                            # 如果你需要监控 clip_ratio，也可以加进去，这里演示加个 key
                            if "ttt/avg_clip_ratio" not in ttt_metrics: ttt_metrics["ttt/avg_clip_ratio"] = []
                            ttt_metrics["ttt/avg_clip_ratio"].append(v)
                
                # 2. 检查梯度流
                if hasattr(module, 'target_projector') and module.target_projector.weight.grad is not None:
                    grad_norm = module.target_projector.weight.grad.norm().item()
                    ttt_metrics["ttt/projector_grad_norm"].append(grad_norm)
        
        if not found_ttt:
            return

        # 计算平均值
        log_dict = {}
        for k, v_list in ttt_metrics.items():
            if len(v_list) > 0:
                log_dict[k] = sum(v_list) / len(v_list)
        
        # === 关键修改 ===
        # 不要手动传 step，防止与 Trainer 冲突
        if wandb.run is not None:
            wandb.log(log_dict)

class TTTSaveCallback(TrainerCallback):
    """
    HuggingFace Trainer 默认只保存 Adapter (LoRA) 和 Base Model。
    我们需要这个 Callback 在每次保存 Checkpoint 时，强制把 TTT 的自定义参数也存下来。
    """
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        model = kwargs['model']
        
        # 提取 TTT 专属参数
        ttt_state_dict = {}
        for name, param in model.named_parameters():
            # 过滤条件：匹配你的 TTT Wrapper 中的参数名
            if any(k in name for k in ["target_projector", "init_A", "init_B", "ttt_layer"]):
                ttt_state_dict[name] = param.cpu()
        
        if len(ttt_state_dict) > 0:
            save_path = os.path.join(checkpoint_folder, "ttt_weights.pt")
            torch.save(ttt_state_dict, save_path)
            print(f"\n[TTT Callback] 已额外保存 {len(ttt_state_dict)} 个 TTT 张量至 {save_path} 喵。")

class TokenTrackingCallback(TrainerCallback):
    """
    绕过 Trainer 的 logging 机制，直接向 wandb 后端发送数据。
    """
    def __init__(self, seq_length):
        self.seq_length = seq_length

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            tokens_per_step = (
                args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size * self.seq_length
            )
            current_tokens = state.global_step * tokens_per_step
            
            if wandb.run is not None:
                wandb.log({
                    "train/tokens_trained": current_tokens,
                    "global_step": state.global_step
                })
# ==========================================
# 1. 模型初始化
# ==========================================
print(f"初始化 CPT 训练 ({method_tag})，窗口大小: {SLIDING_WINDOW_SIZE} 喵...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
    attn_implementation="flash_attention_2",
)
# [Fix] Qwen 默认没有 pad_token，这会导致 SFTTrainer 的 packing 逻辑失效或回退到 1024
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # 某些版本的 TRL/Unsloth 需要明确更新 pad_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
# [Fix] 显式告诉 Tokenizer 它的最大长度，防止 SFTTrainer 读取 config.json 中的旧值
tokenizer.model_max_length = SEQ_LENGTH
# ==========================================
# 2. 启用 Sliding Window Attention (SWA)
# ==========================================
if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = SLIDING_WINDOW_SIZE
else:
    model.config.sliding_window = SLIDING_WINDOW_SIZE
    print(f"警告：该架构默认不显示 SWA 属性，已强制注入 config 喵。")

# ==========================================
# 3. 动态配置 LoRA
# ==========================================
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

# ==========================================
# 4. TTT 手术 (Architecture Surgery)
# ==========================================
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

# ==========================================
# 5. 梯度管理
# ==========================================
if USE_TTT:
    # 开启 TTT 参数的梯度
    for mod in ttt_modules:
        mod.init_A.requires_grad = True
        mod.init_B.requires_grad = True
        if hasattr(mod, 'target_projector'):
            mod.target_projector.weight.requires_grad = True

# 验证可训练参数
print("\n=== 可训练参数确认 ===")
trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"Total trainable tensors: {len(trainable_names)}")
if USE_TTT and any("target_projector" in n for n in trainable_names):
    print("状态: TTT Meta-Network 梯度正常开启 喵！")

# ==========================================
# 6. 数据加载 (智能缓存版)
# ==========================================
import os
from datasets import load_from_disk

# 检查是否存在预处理好的数据文件夹
if os.path.exists(PROCESSED_DATA_PATH):
    print(f"🚀 发现预处理数据集: {PROCESSED_DATA_PATH}，直接加载！(跳过 Tokenizing)")
    dataset = load_from_disk(PROCESSED_DATA_PATH)
    is_pre_packed = True # 标记：数据已经是打包好的了
else:
    print(f"🐢 未发现预处理数据，加载原始 JSONL: {CPT_DATA_PATH} (需要在线处理)")
    dataset = load_dataset("json", data_files=CPT_DATA_PATH, split="train")
    is_pre_packed = False

print(f"Dataset Loaded. Samples: {len(dataset)}")


# ==========================================
# 7. Trainer 配置
# ==========================================
token_callback = TokenTrackingCallback(seq_length=SEQ_LENGTH)
num_proc = os.cpu_count()

if num_proc is None: num_proc = 4
# [Logic Switch] 
# 如果加载的是预处理数据 (is_pre_packed=True)，则 packing=False (因为已经 pack 过了)
# 如果加载的是原始数据 (is_pre_packed=False)，则 packing=True (让 Trainer 去做)
should_pack = not is_pre_packed

sft_config = SFTConfig(
    output_dir = output_dir_name,
    max_seq_length = SEQ_LENGTH,
    packing = should_pack, 
    dataset_text_field = "text" if should_pack else None,
    dataset_num_proc = num_proc,

    per_device_train_batch_size = 1,  
    gradient_accumulation_steps = 4,
    
    warmup_ratio = 0.05,
    num_train_epochs = 1,           # 跑满整个数据集 1 遍 喵
    learning_rate = CPT_LEARNING_RATE,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_torch",
    weight_decay = 0.01,
    lr_scheduler_type = "cosine",
    seed = 3407,
    save_strategy = "steps",
    save_steps = 200,
    report_to = "wandb",
    remove_unused_columns = True, 
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = sft_config, # 将 SFTConfig 传入 args
    callbacks=[TTTSaveCallback, token_callback, TTTDebugCallback] if USE_TTT else [token_callback],
)

# ==========================================
# 8. 执行训练 (含 Resume 修复逻辑)
# ==========================================
print(f"开始 CPT 训练, 模式: {method_tag} 喵！")

last_checkpoint = get_last_checkpoint(output_dir_name)

if last_checkpoint:
    print(f"发现历史 Checkpoint: {last_checkpoint}")
    # === 关键修复: 手动加载 TTT 权重 ===
    if USE_TTT:
        ttt_weights_path = os.path.join(last_checkpoint, "ttt_weights.pt")
        if os.path.exists(ttt_weights_path):
            print(f"正在恢复 TTT 状态 from: {ttt_weights_path}...")
            # map_location='cpu' 防止显存激增，load_state_dict 会自动处理 device
            ttt_state_dict = torch.load(ttt_weights_path, map_location='cpu')
            
            # strict=False 是必须的，因为 model 包含 LoRA 和 Base weights，而 ttt_state_dict 只有 TTT 部分
            keys = model.load_state_dict(ttt_state_dict, strict=False)
            print(f"TTT 权重恢复成功。Unexpected keys: {len(keys.unexpected_keys)}")
        else:
            print("【严重警告】找到 Checkpoint 但缺失 ttt_weights.pt！TTT 层将重置为随机初始化！")
    trainer.train(resume_from_checkpoint=True)
else:
    print("未找到 Checkpoint，开始全新训练。")
    print("\n=== [DEBUG] 检查 DataLoader 输出长度 ===")
    # 模拟 Trainer 的 DataLoader
    train_dataloader = trainer.get_train_dataloader()
    # 获取第一个 Batch
    for batch in train_dataloader:
        input_ids = batch['input_ids']
        print(f"实际输入 Batch Shape: {input_ids.shape}")
        print(f"实际输入 Sequence Length: {input_ids.shape[1]}")
        print(f"Trainer Config Max Length: {trainer.args.max_seq_length if hasattr(trainer.args, 'max_seq_length') else 'Unknown'}")
        # 或者检查 processing_class (Tokenizer)
        print(f"Trainer Tokenizer Max Length: {trainer.processing_class.model_max_length}")
        if input_ids.shape[1] < 2048:
            print("❌ 警告：数据依然很短！Packing 没有生效！")
            print("可能原因：Tokenizer 限制、数据源本身过短且不可拼接、或 Unsloth 优化冲突。")
        else:
            print("✅ 数据长度正常，看起来已经 Pack 好了。")
        break
    print("========================================\n")
    trainer.train()

# ==========================================
# 9. 最终保存
# ==========================================
print(f"\n=== 训练结束，正在保存 ===")
save_name = "model_cpt_ttt_final" if USE_TTT else "model_cpt_lora_final"
model.save_pretrained(save_name)
tokenizer.save_pretrained(save_name)
if USE_TTT:
    final_ttt_path = os.path.join(save_name, "ttt_weights.pt")
    ttt_state_dict = {}
    for name, param in model.named_parameters():
        if any(k in name for k in ["target_projector", "init_A", "init_B", "ttt_layer"]):
            ttt_state_dict[name] = param.cpu()
    torch.save(ttt_state_dict, final_ttt_path)
    print(f"TTT 最终权重已独立保存至: {final_ttt_path}")
print(f"所有工作完成！模型位于: {save_name} 喵！")