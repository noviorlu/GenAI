import torch
import os
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import load_dataset
from model import TTT_DownProj_Wrapper
from config import *

# === 评估配置 ===

USE_TTT = False                        # TTT模式开关 (Baseline设为False)
USE_LORA = False                        # LoRA模式开关 (Native设为False)
STRIDE = 512                          # 滑动窗口步长
if USE_TTT:
    ADAPTER_PATH = "model_cpt_ttt_final"
elif USE_LORA:
    ADAPTER_PATH = "model_cpt_lora_final"
else:
    ADAPTER_PATH = None                # Native model, no adapter

# === 数据集选择 ===
# "wikitext-2-raw-v1" 是最通用的 NLP 评估基准之一
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "test" 

def load_model_for_eval(adapter_path=None):
    print(f"=== 正在加载模型 (TTT={USE_TTT}, LoRA={USE_LORA}) 喵... ===")
    
    # 1. 加载 Base Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_ID,
        max_seq_length = SEQ_LENGTH,
        dtype = None,
        load_in_4bit = False,
    )

    # 2. 设置 Sliding Window (SWA)
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = SLIDING_WINDOW_SIZE
    else:
        model.config.sliding_window = SLIDING_WINDOW_SIZE
    
    # 3. TTT 架构重建 (Surgery)
    if USE_TTT:
        print("正在重建 TTT 架构...")
        embed_tokens = model.model.embed_tokens
        device = model.device
        ttt_modules = []

        for layer in model.model.layers:
            original_down = layer.mlp.down_proj
            wrapper = TTT_DownProj_Wrapper(original_down, embed_tokens)
            wrapper.to(device)
            layer.mlp.down_proj = wrapper
            ttt_modules.append(wrapper)
        
        # 注册 Hook
        def input_capture_hook(module, args, kwargs):
            input_ids = None
            if 'input_ids' in kwargs: input_ids = kwargs['input_ids']
            elif len(args) > 0: 
                if isinstance(args[0], dict) and 'input_ids' in args[0]: input_ids = args[0]['input_ids']
                elif isinstance(args[0], torch.Tensor): input_ids = args[0]
            
            if input_ids is not None:
                for mod in ttt_modules: mod.current_input_ids = input_ids
        
        model.register_forward_pre_hook(input_capture_hook, with_kwargs=True)

    # 4. 加载权重
    if adapter_path is not None:
        print("正在加载 LoRA 与 TTT 权重...")
        model.load_adapter(adapter_path)
        
        if USE_TTT:
            ttt_weights_path = os.path.join(adapter_path, "ttt_weights.pt")
            if os.path.exists(ttt_weights_path):
                ttt_state_dict = torch.load(ttt_weights_path, map_location='cpu')
                model.load_state_dict(ttt_state_dict, strict=False)
                print("TTT 专属参数加载成功 喵！")
            else:
                raise FileNotFoundError("未找到 ttt_weights.pt，评估无法继续！")
    else:
        print("使用原生模型（无 LoRA，无 TTT）")

    FastLanguageModel.for_inference(model)
    return model, tokenizer

def calculate_perplexity(model, tokenizer):
    # === 修改点：加载 HuggingFace 开源数据集 ===
    print(f"\n正在下载/加载数据集: {DATASET_NAME} - {DATASET_CONFIG} ...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    
    # === 数据预处理 ===
    # WikiText 包含很多空字符串或者只有换行符的条目，这在计算 PPL 时是噪音
    # 我们只保留长度 > 0 的文本，并用 \n\n 连接
    print("正在预处理 WikiText 数据...")
    text_list = [t for t in dataset["text"] if len(t.strip()) > 0]
    raw_text = "\n\n".join(text_list)
    
    print(f"数据集加载完毕。总字符数: {len(raw_text)}")
    
    # Tokenization
    encodings = tokenizer(raw_text, return_tensors="pt")
    
    # 显存保护：限制最大 Context
    max_length = model.config.max_position_embeddings
    if max_length > 4096: max_length = 4096 
    
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0

    print(f"开始计算 PPL (Total Tokens: {seq_len}, Context: {max_length})...")
    
    pbar = tqdm(range(0, seq_len, STRIDE))
    
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc 
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        
        # 将 Context 部分 mask 掉 (-100)，只计算新生成的 token 的 loss
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        
        if end_loc == seq_len: break

    # 计算最终 PPL
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

if __name__ == "__main__":
    model, tokenizer = load_model_for_eval(ADAPTER_PATH)
    
    try:
        ppl = calculate_perplexity(model, tokenizer)
        print(f"\n=====================================")
        print(f"Benchmark: {DATASET_NAME} ({DATASET_SPLIT})")
        if USE_TTT:
            model_type = "TTT-Hybrid"
        elif USE_LORA:
            model_type = "LoRA-Baseline"
        else:
            model_type = "Native"
        print(f"Model: {model_type}")
        print(f"Perplexity (PPL): {ppl:.4f}")
        print(f"=====================================")
    except Exception as e:
        print(f"评估失败: {e}")