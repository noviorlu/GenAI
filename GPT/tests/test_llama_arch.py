import torch
import torch.nn.functional as F
from lm.model import LlamaLM

def test_llama_architecture():
    print("=== 开始验证 Llama-Tiny 架构 ===")
    
    # 1. 定义超参数 (Tiny Config)
    conf = {
        "n_vocab": 100,      # 词表大小
        "n_embd": 32,        # 维度 (必须能被 n_head 整除)
        "n_head": 4,         # 头数
        "n_positions": 64,   # 序列长度
        "n_layer": 2,        # 层数
        "p_dropout": 0.0
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # 2. 初始化模型
    try:
        model = LlamaLM(**conf).to(device)
        print("✅ 模型初始化成功")
        print(f"   参数量: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    # 3. 构造伪造数据
    batch_size = 8
    seq_len = 32
    input_ids = torch.randint(0, conf['n_vocab'], (batch_size, seq_len)).to(device)
    
    # 4. 前向传播测试 (Forward Pass)
    try:
        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, conf['n_vocab'])
        print("✅ 前向传播成功 (Forward Pass)")
        print(f"   Output Shape: {logits.shape}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 反向传播测试 (Backward Pass)
    try:
        loss = F.cross_entropy(logits.view(-1, conf['n_vocab']), input_ids.view(-1))
        loss.backward()
        print("✅ 反向传播成功 (Backward Pass)")
    except Exception as e:
        print(f"❌ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. 过拟合测试 (Overfit on Single Batch)
    print("\n=== 正在运行过拟合测试 (Sanity Check) ===")
    print("目标: Loss 应该快速下降")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    
    # 固定一个 Batch
    fixed_input = torch.randint(0, conf['n_vocab'], (4, 16)).to(device)
    # 简单的 Next Token Prediction 任务
    targets = fixed_input.clone() 
    
    # 简单的 Training Loop
    for step in range(50):
        optimizer.zero_grad()
        
        logits = model(fixed_input)
        
        # Shift Logits & Labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, conf['n_vocab']), 
            shift_labels.view(-1)
        )
        
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")

    if loss.item() < 0.1:
        print("\n🎉 验证成功！模型能够过拟合数据，架构逻辑正确！")
    else:
        print("\n⚠️ 警告：Loss 下降缓慢，可能需要检查初始化或超参数，但代码没有崩溃。")

if __name__ == "__main__":
    test_llama_architecture()