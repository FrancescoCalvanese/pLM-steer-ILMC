"""Utils for ProGen3 fine-tuning on single protein family."""

import torch
from Bio import SeqIO
from datasets import Dataset
from peft import LoraConfig, get_peft_model

try:
    from progen3.modeling import ProGen3ForCausalLM
    from progen3.batch_preparer import ProGen3BatchPreparer
except ImportError:
    raise ImportError("Progen3 package not found. Please install it first.")


def get_protein_dataset(
    fasta_path: str,
    max_length: int = 1024,
    val_split: float | int = 1000,
    seed: int | None = None,
):
    """Parses FASTA and returns a split DatasetDict (train/validation)."""

    batch_preparer = ProGen3BatchPreparer()
    collate_fn = batch_preparer.pad_encodings

    def process_fn(examples: list[str]):
        processed_outputs = [
            batch_preparer.prepare_singleseq(text, reverse_sequence=False)
            for text in examples["text"]
        ]
        # Convert list of dicts to dict of lists
        return {key: [d[key] for d in processed_outputs] for key in processed_outputs[0]}

    sequences = [str(record.seq)[:max_length] for record in SeqIO.parse(fasta_path, "fasta")]
    raw_ds = Dataset.from_dict({"text": sequences})

    # Process data entries
    tokenized_ds = raw_ds.map(process_fn, batched=True, remove_columns=["text"])
    tokenized_ds.set_format(
        type="torch", columns=["input_ids", "labels", "position_ids", "sequence_ids"]
    )

    # Split dataset into train and validation
    if seed is not None:
        ds_split = tokenized_ds.train_test_split(test_size=val_split, shuffle=True, seed=seed)
    else:
        ds_split = tokenized_ds.train_test_split(test_size=val_split, shuffle=False)

    return ds_split, collate_fn


def load_progen_model(model_name, lora_rank: int | None = None):
    """Loads model with support for distributed training/FSDP."""

    # Load model with bfloat16 dtype
    model = ProGen3ForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    if lora_rank is not None:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    # Enable gradient checkpointing to save VRAM on multi-GPU setups
    # model.gradient_checkpointing_enable()  # TO CHECK
    return model
