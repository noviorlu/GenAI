import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class InPlaceTTT_MLP(nn.Module):
    """
    Jointly Trainable LoRA-Hijacked In-Place TTT Module.
    Fixed by Luna: 
    - Auto-detects device/dtype
    - Auto-detects intermediate expansion size (3072 vs 1024)
    - Persists state between Prefill and Decode
    """
    def __init__(self, base_mlp: nn.Module, config, ttt_lr: float = TTT_LR):
        super().__init__()
        self.config = config
        self.ttt_lr = ttt_lr
        
        # 1. Capture Base Components (Frozen Base, Trainable Adapters)
        self.gate_proj = base_mlp.gate_proj
        self.up_proj = base_mlp.up_proj
        self.act_fn = base_mlp.act_fn
        self.down_proj = base_mlp.down_proj 
        
        # 2. Extract LoRA Components
        if not hasattr(self.down_proj, "lora_A") or not hasattr(self.down_proj, "lora_B"):
            raise ValueError("Error: LoRA adapters missing! Please inject LoRA before hijacking.")
            
        self.lora_A = self.down_proj.lora_A['default']
        self.lora_B = self.down_proj.lora_B['default']
        
        # 3. Meta-Parameter: The Target Generator
        self.rank = self.lora_A.out_features
        
        # --- FIX 1: Dimensions & Device Alignment ---
        # TTT Input is 'z' (post-activation), so we must match down_proj.in_features (e.g., 3072)
        # TTT Output is added to final result, so it must match hidden_size (e.g., 1024)
        input_dim = self.down_proj.in_features 
        output_dim = config.hidden_size      

        # Detect device/dtype from existing LoRA weights to avoid CPU/GPU mismatch
        target_device = self.lora_A.weight.device
        target_dtype = self.lora_A.weight.dtype 

        self.ttt_target_proj = nn.Linear(
            input_dim,    
            output_dim,   
            bias=False, 
            dtype=target_dtype,
            device=target_device 
        )
        # --------------------------------------------
        
        # State Cache for Inference
        self.inference_state = None 

    def get_views(self, x, z):
        """
        Generate Views for TTT.
        """
        # Key View: Reusing LoRA_A (Shared Encoder)
        k_view = self.lora_A(z) # [Batch, Seq, Rank]
        
        # Value Target: Produced by Meta-Network
        v_target = self.ttt_target_proj(z) # [Batch, Seq, Hidden]
        
        return k_view, v_target

    def differentiable_ttt_scan(self, k_seq, v_seq):
        """
        The Inner Loop (Meta-Learning Core).
        Returns:
            - corrections: Sequence of outputs for loss calculation
            - final_state: The final W_t matrix to carry over (CRITICAL fix)
        """
        B, S, R = k_seq.shape
        _, _, H = v_seq.shape
        
        curr_delta_B = torch.zeros(B, H, R, device=k_seq.device, dtype=k_seq.dtype)
        corrections = []
        
        for t in range(S):
            k_t = k_seq[:, t, :].unsqueeze(2)  # [B, R, 1]
            v_t = v_seq[:, t, :].unsqueeze(2)  # [B, H, 1]
            
            # 1. Apply (Causal Prediction using OLD state)
            pred_correction = torch.matmul(curr_delta_B, k_t).squeeze(2) # [B, H]
            corrections.append(pred_correction)
            
            # 2. Update (Generate NEW state)
            # Reconstruct target (Self-Supervised Loss)
            error = pred_correction - v_t.squeeze(2) # [B, H]
            
            # Grad = Error * k_t^T
            grad = torch.matmul(error.unsqueeze(2), k_t.transpose(1, 2)) # [B, H, R]
            
            # SGD Step
            curr_delta_B = curr_delta_B - self.ttt_lr * grad
        
        # FIX 2: Return BOTH the sequence and the final state
        return torch.stack(corrections, dim=1), curr_delta_B

    def forward_inference_step(self, k_step, v_step):
        """
        Stateful update for inference (Token-by-Token).
        """
        B, R = k_step.shape
        _, H = v_step.shape
        
        # Init state if needed (Safety check)
        if self.inference_state is None:
             self.inference_state = torch.zeros(B, H, R, device=k_step.device, dtype=k_step.dtype)
             
        k_t = k_step.unsqueeze(2) 
        v_t = v_step.unsqueeze(2) 
        
        # 1. Apply
        curr_delta_B = self.inference_state
        pred_correction = torch.matmul(curr_delta_B, k_t).squeeze(2)
        
        # 2. Update
        error = pred_correction - v_t.squeeze(2)
        grad = torch.matmul(error.unsqueeze(2), k_t.transpose(1, 2))
        
        # Update Cache (In-Place)
        self.inference_state = curr_delta_B - self.ttt_lr * grad
        
        return pred_correction

    def forward(self, x):
        # 1. Standard MLP Forward (Compute z)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        z = self.act_fn(gate) * up 
        
        # 2. Get Views 
        k_view, v_target = self.get_views(x, z)
        
        # 3. TTT Operation
        if self.training:
            # Training: We need the full sequence for loss, state is irrelevant after this batch
            ttt_correction, _ = self.differentiable_ttt_scan(k_view, v_target)
        else:
            # Inference Mode
            if x.shape[1] > 1: 
                # Prefill: Run scan, BUT SAVE THE STATE!
                ttt_correction, final_state = self.differentiable_ttt_scan(k_view, v_target)
                
                # FIX 3: Context Persistence
                # Persist the learned memory into the cache for the next token generation
                self.inference_state = final_state 
            else:
                # Decode (Step-by-Step)
                k_step = k_view.squeeze(1)
                v_step = v_target.squeeze(1)
                ttt_correction = self.forward_inference_step(k_step, v_step)
                ttt_correction = ttt_correction.unsqueeze(1) # Restore seq dim
        
        # 4. Final Output = Static (Base + LoRA) + Dynamic (TTT)
        out_static = self.down_proj(z)
        return out_static + ttt_correction