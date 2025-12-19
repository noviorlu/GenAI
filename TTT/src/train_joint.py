import torch
import os
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import wandb
from model import InPlaceTTT_MLP
from config import *
from unsloth.chat_templates import get_chat_template

def train():
    wandb.init(project=PROJ_NAME, name="run-joint-training1")

    # 1. Load Model
    print(f">>> Loading Model: {MODEL_ID}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    # 2. Inject LoRA (Standard)
    print(">>> Injecting LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
    )

    # # 3. Hijack with TTT (The Surgery)
    # print(">>> Hijacking MLP with InPlaceTTT_MLP...")
    # for layer in model.model.model.layers:
    #     if hasattr(layer, "mlp"):
    #         layer.mlp = InPlaceTTT_MLP(layer.mlp, model.config, ttt_lr=TTT_LR)

    # 3. Hijack with TTT (The Surgery)
    print(">>> Hijacking MLP with InPlaceTTT_MLP...")

    # 获取总层数
    num_layers = len(model.model.model.layers)
    print(f">>> Total Transformer Layers: {num_layers}")

    for i, layer in enumerate(model.model.model.layers):
        if hasattr(layer, "mlp"):
            # --- Analysis Block (仅在第0层打印，避免刷屏) ---
            if i == 0:
                print(f"\n{'='*20} MLP Inspection (Layer {i}) {'='*20}")
                original_mlp = layer.mlp
                print(f"Original MLP Object: {original_mlp}")
                
                # 尝试获取具体的投影层维度
                # Llama/Mistral 架构通常有 gate_proj, up_proj, down_proj
                try:
                    # 获取输入维度 (Hidden Size) 和 中间维度 (Intermediate Size)
                    # 注意：Unsloth 加载的模型可能是 Linear4bit 封装
                    gate_shape = original_mlp.gate_proj.weight.shape
                    up_shape = original_mlp.up_proj.weight.shape
                    down_shape = original_mlp.down_proj.weight.shape
                    
                    print(f"\n[Dimensions]")
                    print(f"Input Hidden Size (d_model): {down_shape[1]}") # down_proj 的输入是 intermediate, 输出是 hidden，但weight shape可能是转置的
                    print(f"Intermediate Size (d_ff):    {gate_shape[0]}") 
                    
                    print(f"\n[Projections Details]")
                    print(f"1. Gate Proj: {gate_shape} (Projects Hidden -> Intermediate)")
                    print(f"2. Up Proj:   {up_shape}   (Projects Hidden -> Intermediate)")
                    print(f"3. Down Proj: {down_shape} (Projects Intermediate -> Hidden)")

                    # 计算 MLP 部分的参数量
                    mlp_params = sum(p.numel() for p in original_mlp.parameters())
                    print(f"\n[Parameter Count]")
                    print(f"MLP Params in one layer: {mlp_params:,}")
                    print(f"Total MLP Params (all {num_layers} layers): {mlp_params * num_layers:,}")
                    
                except AttributeError as e:
                    print(f"Could not inspect internal projections (standard Llama structure not found): {e}")
                print(f"{'='*60}\n")
            # ---------------------------------------------------

            # 执行替换 (The Surgery)
            layer.mlp = InPlaceTTT_MLP(layer.mlp, model.config, ttt_lr=TTT_LR)

#     # 4. Configure Joint Training Gradients
#     print(">>> Configuring Joint Training Parameters...")
#     trainable_params = 0
#     all_params = 0
    
#     for name, param in model.named_parameters():
#         all_params += param.numel()
        
#         # Logic: 
#         # - Base Model: Frozen
#         # - LoRA (A & B): Trainable (Encoder & Static Decoder)
#         # - TTT Meta (Target Proj): Trainable (Meta Rule)
        
#         if "lora_" in name or "ttt_target_proj" in name:
#             param.requires_grad = True
#             trainable_params += param.numel()
#         else:
#             param.requires_grad = False
            
#     print(f">>> Trainable Params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")

#     # 5. Data
#     dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
#     def format_prompts(examples):
#         convos = examples["messages"]
#         texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in convos]
#         return {"text": texts}
#     dataset = dataset.map(format_prompts, batched=True)

#     # 6. Training
#     trainer = SFTTrainer( 
#         model=model,
#         tokenizer=tokenizer,
#         train_dataset=dataset,
#         dataset_text_field="text",
#         max_seq_length=MAX_SEQ_LENGTH,
#         args=TrainingArguments(
#             per_device_train_batch_size=MINI_BATCH_SIZE,
#             gradient_accumulation_steps=GRAD_ACCUM_STEPS,
#             warmup_steps=10,
#             num_train_epochs=NUM_EPOCHS,
#             learning_rate=LEARNING_RATE,
#             fp16=not torch.cuda.is_bf16_supported(),
#             bf16=torch.cuda.is_bf16_supported(),
#             logging_steps=1,
#             output_dir=OUTPUT_DIR,
#             optim="adamw_8bit",
#             report_to="wandb",
#             save_strategy="epoch",
#         ),
#     )

#     print(">>> Starting Joint Training...")
#     trainer.train()

#     # 7. Custom Save Logic (CRITICAL)
#     # SFTTrainer will save LoRA adapters, but TTT meta-weights are custom modules inside the hijack.
#     print(f">>> Saving Joint Checkpoint to {OUTPUT_DIR}...")
    
#     # Save standard LoRA adapters
#     model.save_pretrained(OUTPUT_DIR)
#     tokenizer.save_pretrained(OUTPUT_DIR)
    
#     # Save TTT Meta Weights manually
#     ttt_weights = {}
#     for name, param in model.named_parameters():
#         if "ttt_target_proj" in name:
#             ttt_weights[name] = param.cpu()
            
#     torch.save(ttt_weights, os.path.join(OUTPUT_DIR, "ttt_meta_weights.pt"))
#     print(">>> All weights saved successfully.")

if __name__ == "__main__":
    train()