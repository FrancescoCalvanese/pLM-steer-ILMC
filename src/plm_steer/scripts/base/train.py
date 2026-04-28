"Train custom transformer decoder on a single protein family dataset."

import argparse
import json
import os
from contextlib import nullcontext

import random
import torch
import torch.amp
import torch.optim
from torch.utils.data import DataLoader, TensorDataset, default_collate
from Bio import SeqIO
from tqdm.auto import tqdm, trange

from plm_steer.model.transformer import GPTTransformer
from plm_steer.model.tokenizer import GPTTokenizer
from plm_steer.utils.common import get_device, set_seed

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


def get_args():
    parser = argparse.ArgumentParser(description="Train transformer on protein sequences.")
    parser.add_argument(
        "--fasta-path", "-f", type=str, required=True, help="Path to input FASTA file."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=os.path.join(BASE_PATH, "checkpoints/base"),
        help="Output directory for model checkpoints.",
    )
    # Training args
    parser.add_argument(
        "--batch-size", "-b", type=int, default=512, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--max-steps", type=int, default=8000, help="Number of training steps."
    )
    parser.add_argument(
        "--log-steps", type=int, default=100, help="Number of steps between logging."
    )
    parser.add_argument(
        "--eval-steps", type=int, default=500, help="Number of steps between evaluations."
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="bfloat16", 
        help="Data type for training (float32, bfloat16, float16)."
    )
    # Model args
    parser.add_argument(
        "--num-layers", type=int, default=16, help="Number of transformer layers.",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=64, help="Hidden size of the transformer.",
    )
    parser.add_argument(
        "--num-attention-heads", type=int, default=16, help="Number of attention heads.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate."
    )
    return parser.parse_args()


def process_data(
    fasta_path: str, 
    tokenizer: GPTTokenizer, 
    val_split: int | float = 1000, 
    seed: int | None = None
):
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]
    # if seed is not None, shuffle sequences
    if seed is not None:
        random.seed(seed)
        random.shuffle(sequences)
    if isinstance(val_split, float):
        val_split = int(val_split * len(sequences))
    train_seqs = sequences[:-val_split]
    val_seqs = sequences[-val_split:]
    
    # Since datasets are quite small, encode them all at once
    def encode(sequences: list[str]):
        return tokenizer(
            sequences, 
            add_special_tokens=True, 
            padding=True, 
            return_attention_mask=False,
            return_tensors="pt"
        )["input_ids"]
        
    def collate_fn(batch):
        inputs_ids = default_collate(batch)[0]
        targets = inputs_ids[:, 1:].clone()
        inputs_ids = inputs_ids[:, :-1]
        return inputs_ids, targets
    
    train_dataset = TensorDataset(encode(train_seqs))
    val_dataset = TensorDataset(encode(val_seqs))
    
    return {"train": train_dataset, "test": val_dataset}, collate_fn    


def main():
    args = get_args()
    device = get_device()
    set_seed(1337)

    # device and dtype settings
    if not torch.cuda.is_available() or (not torch.cuda.is_bf16_supported() and args.dtype == "bfloat16"):
        print(
            "bfloat16 not supported on this device, using float32 instead. Change the 'dtype'"
            " variable to override."
        )
        args.dtype = "float32"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=ptdtype)
        if "cuda" in device.type
        else nullcontext()
    )
    # Grad scaler for float16 training
    scaler = torch.amp.GradScaler(device=device.type, enabled=(args.dtype == "float16"))
    print(f"Using device: {device} with dtype {args.dtype}")
    
    # Initialize model and tokenizer
    tokenizer = GPTTokenizer()
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "embed_dim": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_attention_heads,
        "mlp_ratio": 4,
        "dropout_p": args.dropout,
        "pad_id": tokenizer.pad_token_id,
    }
    model = GPTTransformer(**model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.3f} M")
    
    dataset_dict, collate_fn = process_data(args.fasta_path, tokenizer, val_split=1000)
    train_len = len(dataset_dict["train"])
    val_len = len(dataset_dict["test"])
    print("Number of training samples:", train_len)
    print("Number of evaluation samples:", val_len)
    train_loader = DataLoader(
        dataset_dict["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset_dict["test"], batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # create output directory and save config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_iter = iter(train_loader)
    
    for step in trange(1, args.max_steps + 1, desc="Training"):
            
        optimizer.zero_grad(set_to_none=True)
        try:
            input_ids, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, targets = next(train_iter)

        input_ids, targets = input_ids.to(device), targets.to(device)
        with ctx:
            loss = model(input_ids, targets=targets)["loss"]

        if step % args.log_steps == 0:
            tqdm.write(f"Step {step:5d} TRAIN | Loss: {loss.item():.4f}")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % args.eval_steps == 0 or step == args.max_steps:
            model.eval()
            eval_loss = 0.0
            total_samples = 0
            
            with torch.inference_mode():
                for input_ids, targets in val_loader:
                    input_ids, targets = input_ids.to(device), targets.to(device)
                    with ctx:
                        loss = model(input_ids, targets=targets)["loss"]
                    
                    batch_len = input_ids.size(0)
                    eval_loss += loss.item() * batch_len
                    total_samples += batch_len
            
            eval_loss /= max(total_samples, 1)
            tqdm.write(f"Step {step:5d} VALIDATION | Loss: {eval_loss:.4f}")

            # raw_model = getattr(model, "_orig_mod", model)
            ckpt_path = os.path.join(args.output_dir, f"iter{step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            model.train()

    print("Training complete.")
    

if __name__ == "__main__":
    main()
