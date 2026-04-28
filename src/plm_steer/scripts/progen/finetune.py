"""Script for fine-tuning ProGen3 model, possibly using distributed training."""

import argparse
import os

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from plm_steer.utils.common import set_seed
from plm_steer.utils.progen.train import get_protein_dataset, load_progen_model

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune ProGen3 model on protein family.")
    parser.add_argument(
        "--fasta-path", "-f", type=str, required=True, help="Path to input FASTA file."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=os.path.join(BASE_PATH, "checkpoints/progen3"),
        help="Output directory for model checkpoints.",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="Profluent-Bio/progen3-112m",
        help="Pretrained ProGen3 model name.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA rank for fine-tuning. Set to None (default) to enable full fine-tuning.",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", "-n", type=int, default=3, help="Number of training epochs.")
    return parser.parse_args()


def main():
    # Parse arguments and set seed for reproducibility
    args = get_args()
    set_seed(42)

    # Load model
    print("Loading model...")
    model = load_progen_model(args.model_name, lora_rank=args.lora_rank)

    # Prepare split dataset
    print("Preparing dataset...")
    dataset_dict, collate_fn = get_protein_dataset(args.fasta_path)
    print("Number of training samples:", len(dataset_dict["train"]))
    print("Number of evaluation samples:", len(dataset_dict["test"]))

    training_args = TrainingArguments(
        output_dir=args.output_dir,  # automatically created if it doesn't exist
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        max_grad_norm=1.0,
        bf16=True,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_total_limit=2,  # best and last
        load_best_model_at_end=True,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        data_collator=collate_fn,
        callbacks=[early_stopping],
    )

    trainer.train()


if __name__ == "__main__":
    main()
