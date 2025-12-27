import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    subprocess.check_call(cmd, shell=True)

def setup_environment():
    print("🚀 开始配置 Colab 环境...")
    
    # 1. 安装 Unsloth (根据 GPU 自动选择版本)
    # Colab 的 T4 和 Ampere (A100/H100) 需要不同的安装指令
    try:
        import torch
        major_version, minor_version = torch.cuda.get_device_capability()
        if major_version >= 8:
            print(f"检测到高端 GPU (Compute Capability {major_version}.{minor_version}) -> 安装 Ampere 版本")
            run_cmd('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
        else:
            print(f"检测到 T4/V100 (Compute Capability {major_version}.{minor_version}) -> 安装通用版本")
            run_cmd('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    except Exception:
        # 默认安装
        run_cmd('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')

    # 2. 安装其他依赖
    print("📦 正在安装依赖库...")
    run_cmd("pip install --no-deps packaging ninja EINOPS")
    run_cmd(f"pip install -r requirements.txt")
    
    # 3. 挂载 Google Drive (可选，但强烈建议)
    if os.path.exists("/content"):
        from google.colab import drive
        if not os.path.exists("/content/drive"):
            print("🔗 正在挂载 Google Drive...")
            drive.mount('/content/drive')
            
            # 创建一个软链接，把你 Drive 里的数据映射到当前目录的 ./data
            # 这样你的代码里写的 "./data" 在云端也能找到数据
            # 假设你在 Drive 里存数据的路径是 /content/drive/MyDrive/Luna_Project/data
            drive_data_path = "/content/drive/MyDrive/Luna_Project/data" 
            local_data_path = "./data"
            
            if os.path.exists(drive_data_path) and not os.path.exists(local_data_path):
                print(f"📂 建立数据软链接: {drive_data_path} -> {local_data_path}")
                os.symlink(drive_data_path, local_data_path)

    print("✅ 环境配置完成！可以开始训练了 喵！")

if __name__ == "__main__":
    setup_environment()