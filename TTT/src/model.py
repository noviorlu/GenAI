import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

# === 辅助函数：联合裁剪 ===
def clip_tt_delta_(deltas, max_norm, norm_type=2.0):
    if isinstance(deltas, torch.Tensor):
        deltas = [deltas]
    norms_sq = [
        torch.sum(d.pow(norm_type), dim=(-2, -1), keepdim=True, dtype=torch.float32) 
        for d in deltas
    ]
    
    total_norm_sq = sum(norms_sq)
    total_norm = torch.pow(total_norm_sq, 1.0 / norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    
    if len(deltas) > 0:
        clip_coef = clip_coef.to(deltas[0].dtype)
    
    clipped_deltas = [d * clip_coef for d in deltas]
    return total_norm, clipped_deltas

class TTT_DownProj_Wrapper(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.h_dim = base_layer.out_features
        self.k_dim = base_layer.in_features
        
        # Init Params
        self.init_A = nn.Parameter(torch.randn(self.h_dim, LORA_RANK, dtype=torch.bfloat16) * 0.01)
        self.init_B = nn.Parameter(torch.randn(LORA_RANK, self.k_dim, dtype=torch.bfloat16) * 0.01)
        
        # Normalization & Projector
        self.ttt_norm = nn.LayerNorm(self.k_dim, dtype=torch.bfloat16)
        self.target_projector = nn.Linear(self.k_dim, self.h_dim, bias=False, dtype=torch.float32)
        nn.init.orthogonal_(self.target_projector.weight)
        self.target_projector.to(dtype=torch.bfloat16)
        
        self.scaling = LORA_ALPHA / LORA_RANK
        self.last_debug_metrics = {}

        self.force_ttt_in_eval = False

    def compute_fast_update(self, A, B, Z, V):
        projected_input = torch.matmul(Z, B.t()) 
        delta_A = torch.matmul(V.transpose(1, 2), projected_input)
        projected_error = torch.matmul(V, A)
        delta_B = torch.matmul(projected_error.transpose(1, 2), Z)
        return delta_A, delta_B

    def forward(self, x):
        base_out = self.base_layer(x)
        
        should_run_ttt = self.training or getattr(self, "force_ttt_in_eval", False)
        if not should_run_ttt:
            lora_out = (x @ self.init_B.t()) @ self.init_A.t()
            return base_out + lora_out * self.scaling

        # === TTT Logic ===
        with torch.no_grad():
            x_detached = x.detach() 
            x_normed = self.ttt_norm(x_detached) # LayerNorm
            shifted_x = F.pad(x_normed[:, 1:, :], (0, 0, 0, 1))

        V = self.target_projector(shifted_x)

        # Prepare Chunks
        B_sz, Seq, _ = x.shape
        remainder = Seq % CHUNK_SIZE
        if remainder != 0:
            pad_len = CHUNK_SIZE - remainder
            Z_padded = F.pad(x, (0, 0, 0, pad_len))
            V_padded = F.pad(V, (0, 0, 0, pad_len))
        else:
            Z_padded, V_padded = x, V
            pad_len = 0
            
        new_seq_len = Z_padded.shape[1]
        num_chunks = new_seq_len // CHUNK_SIZE
        Z_chunks = Z_padded.view(B_sz * num_chunks, CHUNK_SIZE, self.k_dim)
        V_chunks = V_padded.view(B_sz * num_chunks, CHUNK_SIZE, self.h_dim)

        # Compute Gradients (Batch * Chunks 并行计算)
        dA_all, dB_all = self.compute_fast_update(
            self.init_A, self.init_B, Z_chunks, V_chunks
        )
        
        # === [Optimization] Vectorized Scan ===
        dA_view = dA_all.view(B_sz, num_chunks, self.h_dim, LORA_RANK)
        dB_view = dB_all.view(B_sz, num_chunks, LORA_RANK, self.k_dim)
        
        # 1. 计算 Inclusive Cumsum (当前 chunk 包含自己的梯度)
        # dim=1 是 chunks 维度
        acc_dA_inclusive = torch.cumsum(dA_view, dim=1)
        acc_dB_inclusive = torch.cumsum(dB_view, dim=1)
        
        # 2. 构造 Shift 后的 Exclusive Scan (当前 chunk 只包含之前的梯度)
        # 构造全零的初始状态 (对应第0个 chunk 没有任何历史梯度)
        zeros_A = torch.zeros(B_sz, 1, self.h_dim, LORA_RANK, device=dA_view.device, dtype=dA_view.dtype)
        zeros_B = torch.zeros(B_sz, 1, LORA_RANK, self.k_dim, device=dB_view.device, dtype=dB_view.dtype)
        
        # 拼接: [0, cumsum_0, cumsum_1, ..., cumsum_{N-2}]
        # 我们丢弃最后一个 cumsum (cumsum_{N-1})，因为它对后续没有贡献
        acc_dA = torch.cat([zeros_A, acc_dA_inclusive[:, :-1, ...]], dim=1)
        acc_dB = torch.cat([zeros_B, acc_dB_inclusive[:, :-1, ...]], dim=1)
        
        # Apply Updates
        lr = TTT_LEARNING_RATE * self.scaling
        acc_dA_flat = acc_dA.view(-1, self.h_dim, LORA_RANK)
        acc_dB_flat = acc_dB.view(-1, LORA_RANK, self.k_dim)
        
        raw_delta_A = lr * acc_dA_flat
        raw_delta_B = lr * acc_dB_flat
        
        # 计算 total_norm (修复了 numel > 1 的问题)
        # 注意：这里 raw_delta 包含了 batch * chunks 个梯度
        total_norm, [final_delta_A, final_delta_B] = clip_tt_delta_(
            [raw_delta_A, raw_delta_B], 
            max_norm=CLIP_THRESHOLD
        )
        
        # Apply Delta
        # final_delta_A 形状: [Batch * Chunks, H, R]
        # init_A 形状: [H, R] -> 需要广播到每个 Chunk
        init_A_exp = self.init_A.unsqueeze(0).expand(acc_dA_flat.shape[0], -1, -1)
        init_B_exp = self.init_B.unsqueeze(0).expand(acc_dB_flat.shape[0], -1, -1)
        
        A_eff = init_A_exp - final_delta_A
        B_eff = init_B_exp - final_delta_B
        
        # Parallel Computation of Output (所有 chunk 同时算)
        mid = torch.bmm(Z_chunks, B_eff.transpose(1, 2))
        lora_out_flat = torch.bmm(mid, A_eff.transpose(1, 2))
        
        lora_out = lora_out_flat.view(B_sz, new_seq_len, self.h_dim)
        if pad_len > 0:
            lora_out = lora_out[:, :Seq, :]

        if should_run_ttt:
            # 修复 total_norm 转 scalar 的问题
            if isinstance(total_norm, torch.Tensor) and total_norm.numel() > 1:
                scalar_norm = total_norm.mean().item()
            elif isinstance(total_norm, torch.Tensor):
                scalar_norm = total_norm.item()
            else:
                scalar_norm = total_norm

            self.last_debug_metrics = {
                "ttt/V_norm_avg": V.norm(p=2, dim=-1).mean().item(),
                "ttt/delta_norm_total": scalar_norm,  
                "ttt/clip_ratio": (total_norm > CLIP_THRESHOLD).float().mean().item() if isinstance(total_norm, torch.Tensor) else 0.0
            }
    
        return base_out + lora_out * self.scaling