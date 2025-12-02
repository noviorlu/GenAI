import argparse
import json
import os
import math
import torch.nn.functional as F

import tiktoken
import torch
from omegaconf import OmegaConf
from tqdm import trange, tqdm
from lm.model import DecoderLM, LlamaLM
from lm.utils import determine_device, enable_tf32
from lm.train import compute_language_modeling_loss


def softmax_with_temperature(
    logits: torch.FloatTensor, temperature: float
) -> torch.FloatTensor:
    """Turns logits into probabilities under softmax (with temperature)

    Args:
        logits: a 2d torch tensor of token ids (B x V)
        temperature: temperature of the softmax function

    Returns:
        a 2d torch tensor of token probabilities (B x V)
    """

    # to avoid division by 0
    temperature = max(temperature, 1e-5)
    return F.softmax(logits / temperature, dim=-1)


@torch.inference_mode()
def generate(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int = 32,
    temperature: float = 0.1,
) -> tuple[list[str], float]:
    
    disable_tqdm = (len(prefixes) == 1)
    generations = []
    losses = []

    for i in tqdm(range(0, len(prefixes), batch_size), desc="Generating", disable=disable_tqdm):
        batch_prefixes = prefixes[i : i + batch_size]
        tokenized_batch = [tokenizer.encode(p) for p in batch_prefixes]
        max_len = max(len(ids) for ids in tokenized_batch)

        input_ids_list = []
        attention_mask_list = []

        for ids in tokenized_batch:
            pad_len = max_len - len(ids)
            padded_ids = [tokenizer.eot_token] * pad_len + ids
            mask = [0.0] * pad_len + [1.0] * len(ids)
            
            input_ids_list.append(padded_ids)
            attention_mask_list.append(mask)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.float, device=device)

        logits = model(input_ids, attention_mask)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous().clone()
        shift_mask = attention_mask[..., :-1].contiguous()
        
        shift_labels[shift_mask == 0] = -100
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        losses.append(loss.item())

        curr_input_ids = input_ids.clone()
        curr_attention_mask = attention_mask.clone()

        for _ in range(max_new_tokens):
            logits = model(curr_input_ids, curr_attention_mask)
            next_token_logits = logits[:, -1, :]
            probs = softmax_with_temperature(next_token_logits, temperature)
            next_token = torch.multinomial(probs, num_samples=1)
            
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
            new_mask = torch.ones((curr_input_ids.shape[0], 1), device=device, dtype=torch.float)
            curr_attention_mask = torch.cat([curr_attention_mask, new_mask], dim=1)

        batch_generations = []
        for j, seq in enumerate(curr_input_ids):
            original_input_len = len(tokenized_batch[j])
            pad_len = max_len - original_input_len
            
            start_index = pad_len + original_input_len
            
            new_tokens_only = seq[start_index:] 
            
            text = tokenizer.decode(new_tokens_only.tolist())
            batch_generations.append(text)
        
        generations.extend(batch_generations)

    mean_loss = sum(losses) / len(losses) if losses else 0.0
    perplexity = math.exp(mean_loss)

    return generations, perplexity


def run_interactive_mode(args, model, device, tokenizer):
    print("\n" + "="*50)
    print("🤖 Interactive Mode Started")
    print(f"Config: Temp={args.temperature}, Max Tokens={args.max_new_tokens}")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue

            generations, ppl = generate(
                model=model,
                device=device,
                tokenizer=tokenizer,
                prefixes=[user_input],
                batch_size=1,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            print(f"[Input PPL: {ppl:.2f}]")
            print(f"Model: {generations[0]}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        default=None, 
        help="a json file with a list of strings as prefixes for generation",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="run in interactive mode (chat from command line)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="temperature in sampling"
    )

    args = parser.parse_args()
    config = args.config

    if not args.interactive and args.prefixes is None:
        parser.error("the following arguments are required: --prefixes (unless --interactive is set)")

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    
    model_type = config.get("model_type", "gpt2").lower()
    
    print(f"Initializing model type: {model_type}")
    
    if model_type == "llama":
        model = LlamaLM(tokenizer.n_vocab, **config.model_config).to(device)
    elif model_type == "gpt2":
        model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'gpt2' or 'llama'.")

    state_dict = torch.load(model_path, map_location=device)
    
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            new_key = k[len(unwanted_prefix):]
            state_dict[new_key] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    # generate and save outputs
    model.eval()

    if args.interactive:
        run_interactive_mode(args, model, device, tokenizer)
    else:
        with open(args.prefixes) as f:
            prefixes = [json.loads(line)["prefix"] for line in f]
        
        generations, perplexity = generate(
            model,
            device,
            tokenizer,
            prefixes,
            config.batch_size,
            args.max_new_tokens,
            args.temperature,
        )

        print(f"Average Perplexity: {perplexity}")
        generation_path = os.path.join(config.output_dir, "generation.jsonl")
        print(f"writing generations to {generation_path}")
        with open(generation_path, "w") as f:
            for prefix, generation in zip(prefixes, generations):
                json.dump({"prefix": prefix, "generation": generation}, f)
                f.write("\n")

        print("done!")


if __name__ == "__main__":
    main()
