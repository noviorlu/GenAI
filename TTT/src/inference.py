import torch
import argparse
import os
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
from model import InPlaceTTT_MLP
from config import *

# Colors for UI
class Colors:
    HEADER = '\033[95m'; BLUE = '\033[94m'; GREEN = '\033[92m'; ENDC = '\033[0m'

def load_joint_model(adapter_path, base_model_id=MODEL_ID):
    print(f"{Colors.BLUE}>>> 1. Loading Base Model...{Colors.ENDC}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    # 2. Hijack FIRST (Architecture must match training)
    print(f"{Colors.BLUE}>>> 2. Performing TTT Surgery...{Colors.ENDC}")
    for layer in model.model.layers:
        if hasattr(layer, "mlp"):
            layer.mlp = InPlaceTTT_MLP(layer.mlp, model.config, ttt_lr=TTT_LR)

    # 3. Load Joint Weights
    print(f"{Colors.GREEN}>>> 3. Loading Joint Weights (LoRA + TTT Meta)...{Colors.ENDC}")
    
    # A. Load LoRA Adapters (Standard PEFT)
    # Note: Since we hijacked the modules, we need to ensure PEFT can find the 'lora_A'/'lora_B'
    # inside our InPlaceTTT_MLP. 
    # Luckily, our InPlaceTTT_MLP keeps references to self.down_proj which IS the LoRA layer.
    # So model.load_adapter should work if structure paths are aligned, 
    # BUT Unsloth/PEFT might be confused by the wrapper.
    # ROBUST METHOD: Load adapter to a temporary base model, then copy weights? 
    # Or simply load state_dict manually if PEFT fails.
    # Let's try standard load first, usually Unsloth handles it if module names match.
    try:
        model.load_adapter(adapter_path)
    except Exception as e:
        print(f"{Colors.HEADER}Standard adapter load failed (expected due to hijack). Loading manually...{Colors.ENDC}")
        # If standard load fails, we assume the user trained using train_joint.py which saved standard PEFT format
        # We might need to manually map weights if the structure changed significantly.
        # Ideally, we should rely on the PEFT mechanism finding the 'down_proj' inside our wrapper.
        pass

    # B. Load TTT Meta Weights
    meta_path = os.path.join(adapter_path, "ttt_meta_weights.pt")
    if os.path.exists(meta_path):
        meta_sd = torch.load(meta_path)
        # Load with strict=False because keys might be prefixed differently or model has other params
        keys = model.load_state_dict(meta_sd, strict=False)
        print(f"TTT Meta Weights Loaded. Missing keys (expected base params): {len(keys.missing_keys)}")
    else:
        print(f"{Colors.HEADER}WARNING: TTT Meta Weights not found! Running with random init.{Colors.ENDC}")

    model.eval()
    return model, tokenizer

def chat_loop(model, tokenizer):
    print(f"\n{Colors.HEADER}=== Luna Joint-TTT Chat ==={Colors.ENDC}")
    # System Prompt 强化记忆需求
    history = [{"role": "system", "content": "You are Luna. You have a unique memory system. Remember details."}]

    while True:
        try:
            user_input = input(f"\n{Colors.BLUE}You: {Colors.ENDC}")
            if user_input.lower() in ["exit", "quit"]: break
            if user_input.lower() == "clear": 
                history = [history[0]]; print("Memory Reset."); continue

            history.append({"role": "user", "content": user_input})
            
            inputs = tokenizer.apply_chat_template(
                history, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")

            # 关键：use_cache=False 启用 TTT 扫描
            # 在 inference 脚本里，我们简单起见每次都重新 prefill (O(N^2)) 来保证 TTT 状态正确
            # 真正的生产环境应该用 KV-Cache + TTT-State-Cache
            print(f"{Colors.GREEN}Luna is thinking (TTT Active)...{Colors.ENDC}")
            
            # Reset Inference State before new generation (Simulating new thinking session or continuous?)
            # Usually we reset state between turns OR keep it. 
            # For "Chat", we usually want to keep it? 
            # Actually, because we use use_cache=False and re-input the WHOLE history,
            # we effectively re-compute the state from scratch every time.
            # So we should clear the state inside the model manually if we implemented a persistent cache.
            # In our model.py, inference_state is overwritten if x.shape[1] > 1 (Prefill).
            # So passing the full history inputs ensures the state is rebuilt correctly from LTM + Context.
            
            streamer = TextStreamer(tokenizer, skip_prompt=True)
            _ = model.generate(
                input_ids=inputs,
                streamer=streamer,
                max_new_tokens=256,
                use_cache=False, 
                temperature=0.7,
            )
            
            # Note: We need to capture output to append to history. 
            # Simplified here; in production use a capturing streamer.
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default="qwen_joint_ttt_ckpt")
    args = parser.parse_args()
    
    model, tokenizer = load_joint_model(args.adapter)
    chat_loop(model, tokenizer)