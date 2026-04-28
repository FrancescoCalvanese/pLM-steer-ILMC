"""Compute pLDDT scores from sequences in a FASTA file using ESMFold model."""

import argparse
import os
import sys

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import EsmForProteinFolding, AutoTokenizer


def main():
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(
        description="Calculate average pLDDT and Std Dev from a FASTA file."
    )
    parser.add_argument("fasta_file", help="Path to the input FASTA file")
    parser.add_argument(
        "--max-seqs", 
        "-n", 
        type=int, 
        default=None, 
        help="Optional: Stop after processing N sequences (default: process all)"
    )
    args = parser.parse_args()

    fasta_file = args.fasta_file
    max_seqs = args.max_seqs

    # Check if file exists
    if not os.path.exists(fasta_file):
        print(f"Error: The file '{fasta_file}' was not found.")
        sys.exit(1)
        
    # Read sequences
    print(f"Reading sequences from '{fasta_file}'...")
    records = list(SeqIO.parse(fasta_file, "fasta"))

    # Apply limit if specified
    if max_seqs is not None and max_seqs < len(records):
        print(f"Limiting processing to the first {max_seqs} sequences.")
        records = records[:max_seqs]

    total_seqs = len(records)
    if total_seqs == 0:
        print("Error: No sequences found (or file is empty).")
        sys.exit(1)

    # Load ESMFold Model
    print("Loading model... (this may take a minute)")
    model_name = "facebook/esmfold_v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EsmForProteinFolding.from_pretrained(model_name, low_cpu_mem_usage=True)
    model = model.to(device)
    model.eval()

    # Fold sequences and compute pLDDT
    sequence_scores = [] 
    print(f"Processing {total_seqs} sequences on {device}...")
    for record in tqdm(records, desc="Folding", unit="seq"):
        sequence = str(record.seq)
        
        if len(sequence) > 1024:
            tqdm.write(f"Skipping {record.id}: Sequence too long ({len(sequence)})")
            continue

        # Prepare inputs
        tokenized_input = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized_input = tokenized_input.to(device)
        with torch.no_grad():
            output = model(tokenized_input)

        # Extract pLDDT & Compute Mean
        plddt_scores = output.plddt[0].cpu().numpy()
        seq_mean_plddt = np.mean(plddt_scores)
        
        sequence_scores.append(seq_mean_plddt)

    # Calculate Final Statistics
    if len(sequence_scores) > 0:
        global_average = np.mean(sequence_scores)
        global_std = np.std(sequence_scores)
        
        print("\n" + "=" * 30)
        print(f"File: {fasta_file}")
        print(f"Sequences Processed:  {len(sequence_scores)}")
        print(f"Global Average pLDDT: {global_average:.4f}")
        print(f"Standard Deviation:   {global_std:.4f}")
        print(f"Combined Output:      {global_average:.4f} ± {global_std:.4f}")
        print("=" * 30)
    else:
        print("\nNo sequences processed successfully.")


if __name__ == "__main__":
    main()
