import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class TTT_DownProj_Wrapper(nn.Module):
    """
    A wrapper that replaces the down_proj layer.
    It wraps the original frozen Unsloth Linear4bit layer and adds
    In-place TTT logic (Meta-Learning LoRA).
    """
    def __init__(self, base_layer, embed_tokens):
        super().__init__()
        self.base_layer = base_layer  # Frozen Unsloth Layer
        self.embed_tokens = embed_tokens
        
        # Dimensions
        self.h_dim = base_layer.out_features
        self.k_dim = base_layer.in_features
        
        # === Meta-Parameters (Learnable) ===
        # init_A: (h, r) initialized to 0
        self.init_A = nn.Parameter(torch.zeros(self.h_dim, LORA_RANK, dtype=torch.bfloat16))
        # init_B: (r, k) initialized with small noise
        self.init_B = nn.Parameter(torch.randn(LORA_RANK, self.k_dim, dtype=torch.bfloat16) * 0.02)
        
        # Target Generation Network (W_target)
        # Assuming embedding dim matches model input dim
        embed_dim = embed_tokens.weight.shape[1]
        self.target_projector = nn.Linear(embed_dim, self.h_dim, bias=False, dtype=torch.bfloat16)
        
        # Scaling
        self.scaling = LORA_ALPHA / LORA_RANK
        
        # State
        self.current_input_ids = None

    def compute_fast_update(self, A, B, Z, V):
        """
        Calculates Delta A and Delta B using the projection rule.
        Delta A = V^T @ (Z @ B^T)
        Delta B = (V @ A)^T @ Z
        """
        # 输入 A, B, Z, V 全部已经是 bfloat16，直接计算即可
        # 1. Project Input: (Batch, k) @ (k, r) -> (Batch, r)
        projected_input = torch.matmul(Z, B.t()) 
        
        # 2. Gradient for A: (h, Batch) @ (Batch, r) -> (h, r)
        delta_A = torch.matmul(V.transpose(1, 2), projected_input)
        
        # 3. Project Error: (Batch, h) @ (h, r) -> (Batch, r)
        projected_error = torch.matmul(V, A)
        
        # 4. Gradient for B: (r, Batch) @ (Batch, k) -> (r, k)
        delta_B = torch.matmul(projected_error.transpose(1, 2), Z)
        return delta_A, delta_B

    def forward(self, x):
        # x is (Batch, Seq, k_dim) in BFloat16
        
        # 1. Base Forward
        base_out = self.base_layer(x)
        
        # If not training or no inputs captured, act as static LoRA
        if not self.training or self.current_input_ids is None:
            # Fallback: Static LoRA using init params
            lora_out = (x @ self.init_B.t()) @ self.init_A.t()
            return base_out + lora_out * self.scaling

        # === TTT Meta-Training Logic ===
        input_ids = self.current_input_ids
        
        # A. Target Generation (V)
        # 1. 获取 Embedding (这一步不需要梯度，保持 detach 状态节省显存)
        with torch.no_grad():
            embeds = self.embed_tokens(input_ids).transpose(1, 2) # (B, D, S)
            
            # Strict Lookahead Logic (Shift 1 + Causal Pad)
            shift = 1
            shifted = embeds[..., shift:] 
            shifted = F.pad(shifted, (0, shift)) 
            
        # 2. === Meta-Network ===
        # Projector (W_target)
        V = self.target_projector(shifted.transpose(1, 2))

        # B. Prepare Chunks
        B_sz, Seq, _ = x.shape
        
        # Handle Padding for non-divisible sequence lengths
        remainder = Seq % CHUNK_SIZE
        if remainder != 0:
            pad_len = CHUNK_SIZE - remainder
            # Pad Z and V (Batch, Seq+Pad, Dim)
            Z_padded = F.pad(x, (0, 0, 0, pad_len))
            V_padded = F.pad(V, (0, 0, 0, pad_len))
        else:
            Z_padded, V_padded = x, V
            pad_len = 0
            
        new_seq_len = Z_padded.shape[1]
        num_chunks = new_seq_len // CHUNK_SIZE
        
        # Reshape to (Batch * Num_Chunks, Chunk_Size, Dim)
        # This allows parallel computation of gradients for all chunks
        Z_chunks = Z_padded.view(B_sz * num_chunks, CHUNK_SIZE, self.k_dim)
        V_chunks = V_padded.view(B_sz * num_chunks, CHUNK_SIZE, self.h_dim)
        
        # C. Compute Inner Loop Gradients (Parallel)
        # We calculate delta for ALL chunks at once using current Init params
        # Note: This implies we update from Init for every chunk (Simplified TTT),
        # or we are preparing for the Scan step.
        dA_all, dB_all = self.compute_fast_update(
            self.init_A, self.init_B, Z_chunks, V_chunks
        )
        
        # Reshape back to separate Batch and Chunks for Scan
        # dA_view: (B, N, h, r)
        dA_view = dA_all.view(B_sz, num_chunks, self.h_dim, LORA_RANK)
        dB_view = dB_all.view(B_sz, num_chunks, LORA_RANK, self.k_dim)
        
        # D. Causal Cumulative Sum (Scan)
        # acc_dA[i] = Sum(dA[0]...dA[i-1]) -> The update applied TO chunk i
        # We use a list to be Autograd safe (avoid in-place ops)
        
        acc_dA_list = []
        acc_dB_list = []
        
        # Initial accumulator (Zeros) for the first chunk
        curr_sum_A = torch.zeros_like(dA_view[:, 0]) 
        curr_sum_B = torch.zeros_like(dB_view[:, 0])
        
        acc_dA_list.append(curr_sum_A)
        acc_dB_list.append(curr_sum_B)
        
        for i in range(num_chunks - 1):
            # Accumulate gradients from previous chunks
            # Note: We do NOT use in-place += to keep gradient history valid
            curr_sum_A = curr_sum_A + dA_view[:, i]
            curr_sum_B = curr_sum_B + dB_view[:, i]
            
            acc_dA_list.append(curr_sum_A)
            acc_dB_list.append(curr_sum_B)
            
        # Stack: (B, N, h, r)
        acc_dA = torch.stack(acc_dA_list, dim=1)
        acc_dB = torch.stack(acc_dB_list, dim=1)
        
        # E. Apply Updates & Compute Output
        # Effective Weight = Init - LR * Accumulated_Delta
        # We process all chunks in parallel again using (B*N) flattening
        
        lr = TTT_LEARNING_RATE * self.scaling
        
        # Flatten accumulators: (B*N, h, r)
        acc_dA_flat = acc_dA.view(-1, self.h_dim, LORA_RANK)
        acc_dB_flat = acc_dB.view(-1, LORA_RANK, self.k_dim)
        
        # Expand Init params to match batch size: (B*N, h, r)
        init_A_exp = self.init_A.unsqueeze(0).expand(acc_dA_flat.shape[0], -1, -1)
        init_B_exp = self.init_B.unsqueeze(0).expand(acc_dB_flat.shape[0], -1, -1)
        
        # Calculate Effective Weights
        A_eff = init_A_exp - lr * acc_dA_flat
        B_eff = init_B_exp - lr * acc_dB_flat
        
        # LoRA Forward: Z @ B_eff^T @ A_eff^T
        # Z_chunks: (B*N, Chunk_Size, k)
        # B_eff: (B*N, r, k) -> transpose(1,2) -> (B*N, k, r)
        mid = torch.bmm(Z_chunks, B_eff.transpose(1, 2))
        
        # mid: (B*N, Chunk_Size, r)
        # A_eff: (B*N, h, r) -> transpose(1,2) -> (B*N, r, h)
        lora_out_flat = torch.bmm(mid, A_eff.transpose(1, 2)) # (B*N, C, h)
        
        # Reshape back to sequence
        lora_out = lora_out_flat.view(B_sz, new_seq_len, self.h_dim)
        
        # Remove Padding if applied
        if pad_len > 0:
            lora_out = lora_out[:, :Seq, :]
            
        return base_out + lora_out * self.scaling