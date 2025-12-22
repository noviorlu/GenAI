import torch
import torch.nn as nn
import torch.nn.functional as F

class TTT_DownProj_Wrapper(nn.Module):
    """
    A wrapper that replaces the down_proj layer.
    It wraps the original frozen Unsloth Linear4bit layer and adds
    In-place TTT logic (Meta-Learning LoRA).
    """
    def __init__(self, base_layer, config, embed_tokens):
        super().__init__()
        self.base_layer = base_layer  # Frozen Unsloth Layer
        self.config = config
        self.embed_tokens = embed_tokens
        
        # Dimensions
        self.h_dim = base_layer.out_features
        self.k_dim = base_layer.in_features
        
        # === Meta-Parameters (Learnable) ===
        # init_A: (h, r) initialized to 0
        self.init_A = nn.Parameter(torch.zeros(self.h_dim, config.lora_rank))
        # init_B: (r, k) initialized with small noise
        self.init_B = nn.Parameter(torch.randn(config.lora_rank, self.k_dim) * 0.02)
        
        # Target Generation Network (W_target)
        # Assuming embedding dim matches model input dim
        embed_dim = embed_tokens.weight.shape[1]
        self.target_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=0, bias=False)
        self.target_projector = nn.Linear(embed_dim, self.h_dim, bias=False)
        
        # Scaling
        self.scaling = config.lora_alpha / config.lora_rank
        
        # State
        self.current_input_ids = None

    def compute_fast_update(self, A, B, Z, V):
        """
        Calculates Delta A and Delta B using the projection rule.
        Delta A = V^T @ (Z @ B^T)
        Delta B = (V @ A)^T @ Z
        """
        # Ensure float32 for stability during meta-training
        orig_dtype = Z.dtype
        A_f, B_f = A.float(), B.float()
        Z_f, V_f = Z.float(), V.float()

        # 1. Project Input: (Batch, k) @ (k, r) -> (Batch, r)
        projected_input = torch.matmul(Z_f, B_f.t()) 
        
        # 2. Gradient for A: (h, Batch) @ (Batch, r) -> (h, r)
        delta_A = torch.matmul(V_f.transpose(1, 2), projected_input)
        
        # 3. Project Error: (Batch, h) @ (h, r) -> (Batch, r)
        projected_error = torch.matmul(V_f, A_f)
        
        # 4. Gradient for B: (r, Batch) @ (Batch, k) -> (r, k)
        delta_B = torch.matmul(projected_error.transpose(1, 2), Z_f)
        
        # === 探针 1: 验证 TTT 更新量 ===
        # 如果这里打印出 0，说明 TTT 没效果
        if not getattr(self, "_logged_delta_stats", False):
            print(f"   [DEBUG] Inner Loop Stats:")
            print(f"   -> Z norm: {Z.norm().item():.4f}")
            print(f"   -> V norm (Target): {V.norm().item():.4f} (如果为0，说明W_target没工作)")
            print(f"   -> Delta A norm: {delta_A.norm().item():.4f}")
            self._logged_delta_stats = True # 只打印一次防止刷屏

        return delta_A.to(orig_dtype), delta_B.to(orig_dtype)

    def forward(self, x):
        # x is the hidden state Z (Batch, Seq, k_dim)
        
        # 1. Base Forward (Frozen Unsloth Kernel)
        base_out = self.base_layer(x)
        
        # === 修改开始 ===
        # 打印当前状态，看看 input_ids 到底是啥
        # 注意：为了防止刷屏，可以加个随机判定或者只打印一次
        if self.current_input_ids is None and self.training:
             # 打印一次警告（利用 getattr 防止重复打印）
            if not getattr(self, "_warned_missing_input", False):
                print(f"⚠️ [WARNING] Layer {self} - input_ids missing! Fallback to Static LoRA.")
                self._warned_missing_input = True
        # === 修改结束 ===

        # If not training or no inputs captured, act as static LoRA
        if not self.training or self.current_input_ids is None:
            # Fallback: Static LoRA using init params
            lora_out = (x @ self.init_B.t()) @ self.init_A.t()
            return base_out + lora_out * self.scaling

        # === TTT Meta-Training Logic ===
        # 如果能运行到这里，说明 TTT 启动了
        if not getattr(self, "_logged_ttt_start", False):
            print(f"✅ [SUCCESS] Layer {self} - TTT Logic Triggered! Chunk Size: {self.config.chunk_size}")
            self._logged_ttt_start = True

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
            
            # Conv Padding
            conv_pad = self.target_conv.kernel_size[0] - 1
            shifted = F.pad(shifted, (0, conv_pad))

        # 2. === 关键修改：Meta-Network 计算必须在 no_grad 之外！===
        # 这样梯度才能从 Loss -> V -> target_projector (W_target)
        
        # Conv1D (如果有参数的话需要梯度，虽然这里你是随机初始化的，但未来可能想训练它)
        conv_out = self.target_conv(shifted)
        
        # Truncate
        conv_out = conv_out[..., :input_ids.size(1)]
        
        # Projector (W_target) -> 必须有梯度！
        V = self.target_projector(conv_out.transpose(1, 2)) # (B, S, h)
        # === 探针 2: 验证 W_target 是否有值 ===
        # 检查 projector 的权重是否全为 0
        if not getattr(self, "_logged_w_target", False):
            w_norm = self.target_projector.weight.norm().item()
            print(f"🔍 [INSPECT] Layer {self.h_dim} W_target Weight Norm: {w_norm:.4f}")
            if w_norm == 0:
                print("   ⚠️ 警告: W_target 权重为 0！可能初始化失败！")
            else:
                print("   ✅ 确认: W_target 已初始化且有值。")
            self._logged_w_target = True

        # B. Prepare Chunks
        B_sz, Seq, _ = x.shape
        chunk_size = self.config.chunk_size
        
        # Handle Padding for non-divisible sequence lengths
        remainder = Seq % chunk_size
        if remainder != 0:
            pad_len = chunk_size - remainder
            # Pad Z and V (Batch, Seq+Pad, Dim)
            Z_padded = F.pad(x, (0, 0, 0, pad_len))
            V_padded = F.pad(V, (0, 0, 0, pad_len))
        else:
            Z_padded, V_padded = x, V
            pad_len = 0
            
        new_seq_len = Z_padded.shape[1]
        num_chunks = new_seq_len // chunk_size
        
        # Reshape to (Batch * Num_Chunks, Chunk_Size, Dim)
        # This allows parallel computation of gradients for all chunks
        Z_chunks = Z_padded.view(B_sz * num_chunks, chunk_size, self.k_dim)
        V_chunks = V_padded.view(B_sz * num_chunks, chunk_size, self.h_dim)
        
        # C. Compute Inner Loop Gradients (Parallel)
        # We calculate delta for ALL chunks at once using current Init params
        # Note: This implies we update from Init for every chunk (Simplified TTT),
        # or we are preparing for the Scan step.
        dA_all, dB_all = self.compute_fast_update(
            self.init_A, self.init_B, Z_chunks, V_chunks
        )
        
        # Reshape back to separate Batch and Chunks for Scan
        # dA_view: (B, N, h, r)
        dA_view = dA_all.view(B_sz, num_chunks, self.h_dim, self.config.lora_rank)
        dB_view = dB_all.view(B_sz, num_chunks, self.config.lora_rank, self.k_dim)
        
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
        
        lr = self.config.ttt_learning_rate * self.scaling
        
        # Flatten accumulators: (B*N, h, r)
        acc_dA_flat = acc_dA.view(-1, self.h_dim, self.config.lora_rank)
        acc_dB_flat = acc_dB.view(-1, self.config.lora_rank, self.k_dim)
        
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