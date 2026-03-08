import os
import json
import random
import glob
from datasets import load_dataset
from tqdm import tqdm

# === 配置区域 ===
OUTPUT_DIR = "./data"
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cpt_corpus.jsonl")

# 长度阈值 (针对 TTT，我们需要长上下文)
MIN_TEXT_LEN = 4096       # 中文长文本最小字符数
MIN_CODE_LEN = 1024       # 代码最小长度 (代码信息密度高，阈值可稍低)
MAX_SAMPLES_PER_SOURCE = 20000 # 限制每个源的最大样本数 (防止下载几TB数据撑爆硬盘，可按需修改)

# 数据集 ID (经过筛选的高质量源)
DS_CONFIG = {
    "longdata": {
        "hf_id": "yuyijiong/LongData-Corpus",
        "subset": "default",
        "split": "train",
        "ratio": 0.7, # 混合比例
        "is_code": False
    },
    "cosmopedia": {
        "hf_id": "opencsg/chinese-cosmopedia", # 合成教科书，逻辑性极强
        "subset": None,
        "split": "train",
        "ratio": 0.3,
        "is_code": False
    }
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_or_download(key, config):
    """
    检查本地是否有 raw_{key}.jsonl，没有则下载并过滤
    """
    file_name = f"raw_{key}.jsonl"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    
    data_buffer = []

    # 1. 检查本地是否存在
    if os.path.exists(file_path):
        print(f"📦 发现本地缓存: {file_name}，正在加载...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data_buffer.append(json.loads(line))
        print(f"   -> 加载了 {len(data_buffer)} 条数据。")
        return data_buffer

    # 2. 本地没有，开始下载
    print(f"⬇️ 未找到 {file_name}，开始从 HuggingFace 下载 {config['hf_id']}...")
    try:
        # Streaming 模式：防止内存爆炸，只下载需要的部分
        ds = load_dataset(
            config['hf_id'], 
            config['subset'], 
            split=config['split'], 
            streaming=True
        )
        
        count = 0
        pbar = tqdm(total=MAX_SAMPLES_PER_SOURCE, desc=f"Processing {key}")
        
        for sample in ds:
            if count >= MAX_SAMPLES_PER_SOURCE:
                break
            
            # 提取文本 (不同数据集字段名可能不同)
            text = ""
            if "text" in sample: text = sample["text"]
            elif "content" in sample: text = sample["content"]
            elif "code" in sample: text = sample["code"] # 针对某些代码库
            
            # 过滤逻辑
            limit = MIN_CODE_LEN if config['is_code'] else MIN_TEXT_LEN
            
            if len(text) >= limit:
                # 统一格式 {"text": ...}
                entry = {"text": text, "source": key}
                data_buffer.append(entry)
                
                # 实时写入本地缓存，防止中断
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
                count += 1
                pbar.update(1)
        
        print(f"✅ {key} 处理完成，共收集 {len(data_buffer)} 条长样本。")
        return data_buffer

    except Exception as e:
        print(f"❌ 下载/处理 {key} 时出错: {e}")
        return []

def mix_and_export(data_map):
    print("\n🥣 正在执行混合策略 (Mixing Strategy)...")
    
    final_corpus = []
    
    # 计算总数基准
    # 简单策略：找出最大的数据集，按比例采样其他数据集 (或者截断)
    # 这里我们采用“全部利用 + 比例加权”策略
    
    all_samples = []
    for key, data in data_map.items():
        # 给每个样本打上来源标签（虽然最后不需要，但方便调试）
        all_samples.extend(data)
    
    print(f"   -> 原始池总样本数: {len(all_samples)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_samples)
    
    # 导出
    print(f"💾 正在写入最终文件: {FINAL_OUTPUT_FILE}")
    with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in tqdm(all_samples):
            # 再次确认格式，只保留 text 字段以减小体积
            clean_item = {"text": item["text"]}
            f.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
            
    print(f"\n🎉 成功！CPT 数据集已准备就绪。")
    print(f"   路径: {FINAL_OUTPUT_FILE}")
    print(f"   总行数: {len(all_samples)}")
    print("   你现在可以运行 train_cpt.py 了 喵！")

def main():
    ensure_dir(OUTPUT_DIR)
    
    loaded_data = {}
    
    # 遍历配置下载/加载所有数据
    for key, config in DS_CONFIG.items():
        loaded_data[key] = load_or_download(key, config)
        
    # 混合
    mix_and_export(loaded_data)

if __name__ == "__main__":
    main()