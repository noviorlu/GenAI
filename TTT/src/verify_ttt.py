import torch
import torch.nn as nn
import copy
from model import TTT_DownProj_Wrapper, CLIP_THRESHOLD

# === 模拟配置 ===
class MockConfig:
    def __init__(self):
        self.sliding_window = 1024

# 模拟一个简单的 MLP 层
class MockLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
    def forward(self, x):
        return x @ self.weight.t()

def test_ttt_module():
    print("\n" + "="*40)
    print("🧪 正在启动 In-Place TTT 模组验证程序...")
    print("="*40)

    # 1. 初始化环境
    B, SEQ, DIM = 2, 64, 32 # Batch, Seq, Hidden
    base_layer = MockLinear(DIM, DIM)
    
    # 包装 TTT
    ttt_layer = TTT_DownProj_Wrapper(base_layer)
    # 强制开启 TTT 模式（绕过 eval 检查）
    ttt_layer.force_ttt_in_eval = True 
    
    # 模拟输入数据
    x = torch.randn(B, SEQ, DIM, requires_grad=True)

    # ==========================================
    # 测试 1: 梯度流检查 (Gradient Flow)
    # ==========================================
    print("\n[1/3] 检查梯度流 (Gradient Flow)...")
    output = ttt_layer(x)
    loss = output.sum()
    loss.backward()

    has_grad = False
    if ttt_layer.target_projector.weight.grad is not None:
        grad_norm = ttt_layer.target_projector.weight.grad.norm().item()
        print(f"✅ Target Projector 梯度正常! Norm: {grad_norm:.4f}")
        has_grad = True
    else:
        print("❌ 严重错误: Target Projector 没有梯度！TTT 无法学习！")

    if ttt_layer.init_A.grad is not None:
        print(f"✅ Init_A 梯度正常.")
    
    if not has_grad:
        return

    # ==========================================
    # 测试 2: 因果性检查 (Causality Check)
    # ==========================================
    print("\n[2/3] 检查因果性 (Causality)...")
    # 构造两个序列：
    # Seq 1: [Prefix, Token_A]
    # Seq 2: [Prefix, Token_B]
    # TTT 是因果的，所以 Prefix 部分的输出必须完全一致，不受后面 Token 的影响。
    
    prefix_len = 30
    prefix = torch.randn(1, prefix_len, DIM)
    
    suffix_A = torch.randn(1, 10, DIM) # 结尾不同
    suffix_B = torch.randn(1, 10, DIM) # 结尾不同

    input_A = torch.cat([prefix, suffix_A], dim=1)
    input_B = torch.cat([prefix, suffix_B], dim=1)

    # 运行 TTT
    ttt_layer.eval() # 切换到 eval 模式，但 force_ttt_in_eval=True
    with torch.no_grad():
        out_A = ttt_layer(input_A)
        out_B = ttt_layer(input_B)

    # 截取 Prefix 部分的输出
    out_A_prefix = out_A[:, :prefix_len, :]
    out_B_prefix = out_B[:, :prefix_len, :]

    # 计算差异
    diff = (out_A_prefix - out_B_prefix).abs().max().item()
    
    print(f"Prefix 输出差异: {diff:.9f}")
    
    if diff < 1e-5:
        print("✅ 因果性检查通过！未来的 Token 没有泄漏到过去。")
    else:
        print("❌ 严重错误: 因果性泄漏！Prefix 输出不一致！")
        print("这意味着 Token T 在更新时看到了 T+1 或之后的 Token。")
        print("请检查 forward 中的 padding 或 chunking 逻辑。")

    # ==========================================
    # 测试 3: Clipping 触发测试
    # ==========================================
    print("\n[3/3] 检查 Clipping 机制...")
    # 故意制造巨大的梯度
    huge_input = torch.randn(B, SEQ, DIM) * 100 
    
    # 临时 Hook
    last_clip_ratio = 0
    def hook_fn(module, input, output):
        nonlocal last_clip_ratio
        if "ttt/clip_ratio" in module.last_debug_metrics:
            last_clip_ratio = module.last_debug_metrics["ttt/clip_ratio"]
            
    handle = ttt_layer.register_forward_hook(hook_fn)
    _ = ttt_layer(huge_input)
    handle.remove()
    
    print(f"当前 Clip Ratio: {last_clip_ratio * 100:.1f}%")
    if last_clip_ratio > 0:
        print("✅ Clipping 机制正在工作。")
    else:
        print("⚠️ Clipping 未触发（可能是输入还不够大，或阈值过高）。")

if __name__ == "__main__":
    test_ttt_module()