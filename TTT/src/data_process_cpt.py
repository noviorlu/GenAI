import re
import os
import json
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer

# === 配置 ===
from config import MODEL_ID, DATASET_REPO, SEQ_LENGTH, SUBSETS

OUTPUT_FILE = "./data/cpt_dataset.jsonl"
CACHE_DIR = "./hf_cache"

# === 清洗函数 ===
def clean_text(text):
    # 删除连续出现4次以上的非数字和字母的字符
    text = re.sub(r'([^a-zA-Z0-9])\1{4,}', r'\1\1\1\1', text)
    # 删除连续出现3次以上的数字和字母组成的子串
    text = re.sub(r'([a-zA-Z0-9]{3,}?)\1+', r'\1', text)
    return text

def prepare_cpt_data():
    print(f"🚀 开始准备 CPT 数据，目标长度: {SEQ_LENGTH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    all_datasets = []
    
    # 1. 加载数据
    for subset in SUBSETS:
        print(f"📚 正在加载 {subset}...")
        # streaming=True 节省内存
        ds = load_dataset(DATASET_REPO, subset, split="train", streaming=True, trust_remote_code=True)
        # 取前 2000 条用于验证机制 (正式跑可以加大)
        ds_head = ds.take(2000) 
        all_datasets.append(Dataset.from_list(list(ds_head)))

    # 2. 合并与乱序
    if len(all_datasets) > 0:
        full_ds = concatenate_datasets(all_datasets).shuffle(seed=42)
    else:
        raise ValueError("没有加载到任何数据喵！")
    
    print(f"🔄 开始处理 {len(full_ds)} 条原始数据...")

    # 3. 处理函数：清洗 -> Tokenize -> Chunking
    def process_function(examples):
        # 字段兼容性检查 (不同数据集字段名可能不同)
        text_column = "content" if "content" in examples else "text"
        cleaned_texts = [clean_text(t) for t in examples[text_column]]
        
        # Tokenize (不截断，不填充，我们要手动切)
        tokenized = tokenizer(cleaned_texts, truncation=False, add_special_tokens=False)
        
        input_ids_list = []
        for ids in tokenized["input_ids"]:
            # 滑动窗口/切块逻辑
            # CPT 数据通常不需要 overlap，直接切分即可
            for i in range(0, len(ids), SEQ_LENGTH):
                chunk = ids[i : i + SEQ_LENGTH]
                # 丢弃最后过短的尾巴，保证 Tensor 形状一致
                if len(chunk) == SEQ_LENGTH:
                    input_ids_list.append(chunk)
                    
        return {"input_ids": input_ids_list}

    # 4. 执行映射
    lm_dataset = full_ds.map(
        process_function,
        batched=True,
        remove_columns=full_ds.column_names, # 移除原始文本，只留 input_ids
        num_proc=4
    )
    
    # 5. 展平 (Map 返回的是 List[List]，我们需要 List)
    # 这一步是为了确保每一行 jsonl 只有一条 input_ids
    print("📦 正在展平数据...")
    final_data = []
    for item in lm_dataset:
        for chunk in item["input_ids"]:
            final_data.append({"input_ids": chunk})
    
    print(f"💾 正在保存至 {OUTPUT_FILE}...")
    
    # 手动写入 JSONL，确保格式被 Unsloth/Trainer 完美兼容
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in final_data:
            json.dump(entry, f)
            f.write("\n")
            
    print(f"✅ 处理完成！获得 {len(final_data)} 条定长序列 ({SEQ_LENGTH} tokens) 喵！")

if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.makedirs("./data")
    prepare_cpt_data()