import os
import torch
from datasets import load_dataset, load_from_disk
from unsloth import FastLanguageModel
from config import *
from tqdm import tqdm
from itertools import chain

def preprocess_and_save():
    print(f"⚙️ 开始本地预处理流程...")
    print(f"   -> 原始数据: {CPT_DATA_PATH}")
    print(f"   -> 目标路径: {PROCESSED_DATA_PATH}")

    # 1. 加载 Tokenizer (配置必须和训练完全一致)
    print("📥 加载 Tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_ID,
        max_seq_length = SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    
    # [Fix] 补丁逻辑保持一致
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载原始数据
    dataset = load_dataset("json", data_files=CPT_DATA_PATH, split="train")
    
    # 3. Tokenize (第一步：转 ID)
    # CPT 只需要 input_ids，labels 会在 Trainer 里自动通过 Shift 生成
    column_names = dataset.column_names
    
    def tokenize_function(examples):
        # 加上 EOS token 很重要，分隔不同的文章
        return tokenizer(
            [text + tokenizer.eos_token for text in examples["text"]],
            truncation=False, # 先不截断，后面还要拼接
            padding=False,
            return_attention_mask=False, # CPT 通常不需要 padding mask (全是1)
        )

    print("🔄 正在 Tokenize (可能需要几分钟)...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(), # 本地火力全开
        remove_columns=column_names, # 移除原始文本，只留 input_ids
        desc="Tokenizing",
    )

    # 4. Packing (第二步：拼接 + 切块)
    # 这是最耗时的部分，本地做完 A100 就爽了
    block_size = SEQ_LENGTH

    def group_texts(examples):
        # 拼接所有文本
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # 丢弃最后不够一个 block 的部分
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # 切分成块
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # CPT 任务：Labels = Input_IDs (自回归)
        result["labels"] = result["input_ids"].copy()
        return result

    print(f"📦 正在执行 Packing (Seq_Len={block_size})...")
    packed_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        desc="Packing",
    )

    # 5. 保存到磁盘
    print(f"💾 保存处理后的数据到: {PROCESSED_DATA_PATH}")
    packed_dataset.save_to_disk(PROCESSED_DATA_PATH)
    print("✅ 预处理完成！")

if __name__ == "__main__":
    preprocess_and_save()