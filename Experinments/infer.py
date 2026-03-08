from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
import sys

# --- 1. 配置 ---
lora_model_path = "lora_luna_anatomy" 
max_seq_length = 32768
dtype = None
load_in_4bit = True

# LoRA超参数 (用于恢复权重)
LORA_R = 8
LORA_ALPHA = 16
DEFAULT_SCALING = LORA_ALPHA / LORA_R # = 2.0

# --- 2. 加载模型 ---
print(f"🔄 Loading Adapter from: {lora_model_path} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# --- 3. 核心工具：层级消融管理器 (The Scalpel) ---
class AblationManager:
    def __init__(self, model, default_scaling=2.0):
        self.model = model
        self.default_scaling = default_scaling
        
        # 定义模块分组
        self.MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
        self.ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def set_scaling(self, layer_idx, module_group, scale_value):
        """
        精细化控制 LoRA 的开关。
        :param layer_idx: int, 层号。如果为 None 或 -1，则应用到所有层。
        :param module_group: str, 'mlp', 'attn', or 'all'
        :param scale_value: float, 0.0 为禁用，default_scaling 为启用
        """
        target_suffixes = []
        if module_group == "mlp":
            target_suffixes = self.MLP_MODULES
        elif module_group == "attn":
            target_suffixes = self.ATTN_MODULES
        elif module_group == "all":
            target_suffixes = self.MLP_MODULES + self.ATTN_MODULES
        else:
            print(f"Unknown module group: {module_group}")
            return

        count = 0
        for name, module in self.model.named_modules():
            # 1. 检查层号 (如果是 None/-1 则匹配所有层)
            if layer_idx is not None and layer_idx != -1:
                if f"layers.{layer_idx}." not in name:
                    continue
            
            # 2. 检查是否属于目标模块 (如 q_proj)
            # 我们检查 name 是否以 target_suffixes 中的任意一个结尾
            # 注意：LoRA 模块名通常是 ...q_proj 或 ...q_proj.lora_B 等，
            # 为了保险，我们检查 name 中是否包含该 suffix 且 hasattr(module, "scaling")
            is_target_module = any(suffix in name for suffix in target_suffixes)
            
            if is_target_module and hasattr(module, "scaling"):
                # [兼容性处理]
                if isinstance(module.scaling, dict):
                    if "default" in module.scaling:
                        module.scaling["default"] = scale_value
                        count += 1
                elif isinstance(module.scaling, (float, int)):
                    module.scaling = float(scale_value)
                    count += 1
        
        status = "OFF" if scale_value == 0 else "ON"
        layer_str = "ALL Layers" if (layer_idx is None or layer_idx == -1) else f"Layer {layer_idx}"
        print(f"Set [{layer_str}] - [{module_group.upper()}] -> {status} (Modified {count} modules).")

    # --- 便捷函数 ---
    def disable_all(self):
        self.set_scaling(-1, "all", 0.0)

    def restore_all(self):
        self.set_scaling(-1, "all", self.default_scaling)
        
    def enable_only_mlp(self):
        self.disable_all() # 先全关
        self.set_scaling(-1, "mlp", self.default_scaling) # 再开 MLP

    def enable_only_attn(self):
        self.disable_all() # 先全关
        self.set_scaling(-1, "attn", self.default_scaling) # 再开 Attention

    def set_scaling_range(self, start_layer, end_layer, module_group, scale_value):
        """
        批量控制层级范围 [start_layer, end_layer) 的 LoRA 开关。
        例如: start=10, end=20 会影响层号 10, 11, ..., 19。
        """
        print(f"⚡ Batch Setting: Layers {start_layer} to {end_layer-1} [{module_group.upper()}] -> {scale_value}")
        for i in range(start_layer, end_layer):
            self.set_scaling(i, module_group, scale_value)

# --- 4. 定义推理函数 ---
def ask_luna(question, prefix=""):
    messages = [{"role": "user", "content": question}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    attention_mask = torch.ones_like(input_ids)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print(f"\n❓ Question: {question}")
    print(f"🤖 Luna ({prefix}): ", end="")
    
    model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=256, # 测试时不需要太长，节省时间
        use_cache=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("") # 换行

# --- 5. 执行实验主循环 ---
if __name__ == "__main__":
    
    # 初始化手术刀
    surgeon = AblationManager(model, default_scaling=DEFAULT_SCALING)
    
    # 实验问题
    mini_test_suite = ["你好呀，敢问小姐芳名？", "师傅，你是干什么工作的？", "Who are you?"]

    # 实验计划：(Name, Action_Function)
    # 我们使用 lambda 来延迟执行具体的配置动作
    experiments = [
        # ("Baseline (Full LoRA)", lambda: surgeon.restore_all()),
        
        # # 假设：如果 Luna 是事实知识，它应该住在 MLP 里
        # ("Hypothesis A: Only MLP Active (Facts)", lambda: surgeon.enable_only_mlp()),
        
        # # 假设：如果 Luna 只是改变了说话的方式/路由，Attention 会起作用
        # ("Hypothesis B: Only Attn Active (Routing)", lambda: surgeon.enable_only_attn()),
        
        # # 进阶：结合你之前的热力图发现 (假设 Layer 6 梯度最大)
        # # 验证 Layer 6 的 MLP 是不是核心存储点
        # ("Surgery: Only Layer 6 MLP Active", lambda: (
        #     surgeon.disable_all(), 
        #     surgeon.set_scaling(6, "mlp", DEFAULT_SCALING)
        # )),

        ("Block 1: Bottom Layers (0-10) MLP Only", lambda: (
            surgeon.disable_all(),
            surgeon.set_scaling_range(0, 10, "mlp", DEFAULT_SCALING)
        )),
        
        ("Block 2: Middle Layers (10-20) MLP Only", lambda: (
            surgeon.disable_all(),
            surgeon.set_scaling_range(10, 20, "mlp", DEFAULT_SCALING)
        )),
        
        ("Block 3: Top Layers (20-30) MLP Only", lambda: (
            surgeon.disable_all(),
            surgeon.set_scaling_range(20, 30, "mlp", DEFAULT_SCALING)
        )),

        ("Block 4: Top Layers (30-36) MLP Only", lambda: (
            surgeon.disable_all(),
            surgeon.set_scaling_range(30, 36, "mlp", DEFAULT_SCALING)
        )),

        ("Block 5: Top Layers (20-32) MLP Only", lambda: (
            surgeon.disable_all(),
            surgeon.set_scaling_range(20, 32, "mlp", DEFAULT_SCALING)
        )),

        ("Block 5: Top Layers (25-36) MLP Only", lambda: (
            surgeon.disable_all(),
            surgeon.set_scaling_range(25, 36, "mlp", DEFAULT_SCALING)
        )),
    ]

    print(f"🚀 Starting Module-Level Ablation Study...\n")

    for exp_name, setup_func in experiments:
        print(f"\n{'='*50}")
        print(f"🧪 Experiment: {exp_name}")
        print(f"{'='*50}")

        # 1. 执行环境配置
        setup_func()
        
        # 2. 运行测试
        for q in mini_test_suite:
            # prefix 也可以传 exp_name 简化版
            ask_luna(q, prefix=exp_name.split(':')[0])
            
    # 实验结束后恢复
    surgeon.restore_all()
    print("\n✅ All experiments completed.")