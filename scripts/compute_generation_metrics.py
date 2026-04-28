"""
Compute metrics from fasta files
Usage: pixi run -e default fasta_analyzer.py fasta_gen fasta_train \
--d DISTANCE_THRESHOLD [--cores N_CORES]
"""

import argparse
import math
import multiprocessing
import sys

import editdistance

# Global variable for worker processes
shared_data = {}

def init_worker(data_dict):
    """Initializer: Stores datasets in worker's global scope."""
    global shared_data
    shared_data = data_dict


# Worker functions
def worker_cluster_filter(args):
    """Greedy Clustering: Returns indices to remove."""
    center_seq, candidate_indices, threshold = args
    to_remove = []
    all_seqs = shared_data['GEN']
    
    for idx in candidate_indices:
        d = editdistance.eval(center_seq, all_seqs[idx])
        if d < threshold:
            to_remove.append(idx)
    return to_remove


def worker_pairwise_stats(args):
    """
    Calculates Sum, Sum_Squared, and Count for pairwise comparisons.
    This allows us to compute Standard Deviation efficiently.
    """
    start_idx, end_idx = args
    sequences = shared_data['GEN']
    n = len(sequences)
    
    local_sum = 0
    local_sq_sum = 0
    local_count = 0
    
    for i in range(start_idx, end_idx):
        seq_i = sequences[i]
        # Compare only with j > i (Upper Triangle)
        for j in range(i + 1, n):
            d = editdistance.eval(seq_i, sequences[j])
            local_sum += d
            local_sq_sum += (d * d)
            local_count += 1
            
    return local_sum, local_sq_sum, local_count


def worker_min_intra(chunk_data):
    """Intra-set Nearest Neighbor (GEN vs GEN)."""
    ref_seqs = shared_data['GEN']
    min_dists = []
    
    for idx_i, seq_i in chunk_data:
        current_min = float('inf')
        for idx_j, seq_j in enumerate(ref_seqs):
            if idx_i == idx_j: 
                continue 
            if seq_i == seq_j:
                current_min = 0.0
                break
            d = editdistance.eval(seq_i, seq_j)
            if d < current_min:
                current_min = d
                if current_min == 0: 
                    break
        
        if current_min == float('inf'): 
            min_dists.append(0.0)
        else: 
            min_dists.append(current_min)
            
    return min_dists


def worker_min_inter(args):
    """Inter-set Nearest Neighbor (Query vs Reference)."""
    query_seqs, ref_key = args
    ref_seqs = shared_data[ref_key]
    min_dists = []
    
    for q_seq in query_seqs:
        current_min = float('inf')
        for r_seq in ref_seqs:
            d = editdistance.eval(q_seq, r_seq)
            if d < current_min:
                current_min = d
                if current_min == 0: 
                    break
        min_dists.append(current_min)
    return min_dists

# --- HELPER FUNCTIONS ---

def parse_fasta(file_path):
    sequences = []
    current_sequence = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                if line.startswith('>'):
                    if current_sequence: 
                        sequences.append("".join(current_sequence))
                    current_sequence = []
                else:
                    current_sequence.append(line.upper().replace(' ', ''))
            if current_sequence: 
                sequences.append("".join(current_sequence))
        return sequences
    except Exception as e:
        sys.exit(f"Error reading {file_path}: {e}")


def chunk_list(data, num_chunks):
    if num_chunks < 1: 
        num_chunks = 1
    k, m = divmod(len(data), num_chunks)
    return (data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_chunks))

def get_stats(values):
    """Returns (Mean, StdDev)."""
    if not values: 
        return 0.0, 0.0
    n = len(values)
    mean_val = sum(values) / n
    if n < 2: 
        return mean_val, 0.0
    
    # Variance = sum((x - mean)^2) / (n - 1)
    var = sum((x - mean_val)**2 for x in values) / (n - 1)
    return mean_val, math.sqrt(var)

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Sequence Analysis & Clustering")
    parser.add_argument("gen_fasta", help="Input dataset (GEN)")
    parser.add_argument("train_fasta", help="Training dataset (TRAIN)")
    parser.add_argument("--d", type=int, required=True, help="Distance threshold for clustering")
    parser.add_argument("--cores", type=int, default=multiprocessing.cpu_count(), help="CPU cores")
    
    args = parser.parse_args()
    
    # Load Data
    print("Loading sequences...", file=sys.stderr)
    gen_seqs = parse_fasta(args.gen_fasta)
    train_seqs = parse_fasta(args.train_fasta)
    
    N_gen = len(gen_seqs)
    N_train = len(train_seqs)
    
    # Basic Stats & Lengths
    gen_unique_count = len(set(gen_seqs))
    
    # Calculate Length Stats (Fast enough to do serially)
    len_gen_mean, len_gen_std = get_stats([len(s) for s in gen_seqs])
    len_train_mean, len_train_std = get_stats([len(s) for s in train_seqs])
    
    # Prepare Shared Data
    data_payload = {'GEN': gen_seqs, 'TRAIN': train_seqs}
    
    # Initialize Pool
    ctx = multiprocessing.get_context()
    pool = ctx.Pool(processes=args.cores, initializer=init_worker, initargs=(data_payload,))
    
    num_chunks = args.cores * 4

    # --- METRIC 3: Number of Clusters (Greedy) ---
    print(f"Calculating Clusters (threshold < {args.d})...", file=sys.stderr)
    cluster_count = 0
    if N_gen > 0:
        active_indices = list(range(N_gen))
        min_chunk_size = 100
        
        while active_indices:
            center_idx = active_indices[0]
            cluster_count += 1
            candidates = active_indices[1:]
            
            if not candidates: 
                break
            
            center_seq = gen_seqs[center_idx]
            n_cand = len(candidates)
            needed = min(args.cores * 4, max(1, n_cand // min_chunk_size))
            c_chunks = list(chunk_list(candidates, needed))
            
            task_args = [(center_seq, c, args.d) for c in c_chunks]
            results = pool.map(worker_cluster_filter, task_args)
            
            remove_set = set()
            for r in results: 
                remove_set.update(r)
            
            if remove_set:
                active_indices = [idx for idx in candidates if idx not in remove_set]
            else:
                active_indices = candidates

    # --- METRIC 4: Avg GEN-GEN (Pairwise Mean +/- Std) ---
    print("Calculating Avg GEN-GEN Distance...", file=sys.stderr)
    gen_gen_mean, gen_gen_std = 0.0, 0.0
    if N_gen >= 2:
        chunk_size = max(1, N_gen // num_chunks)
        ranges = [(i, min(i + chunk_size, N_gen)) for i in range(0, N_gen, chunk_size)]
        
        results = pool.map(worker_pairwise_stats, ranges)
        
        total_sum = sum(r[0] for r in results)
        total_sq_sum = sum(r[1] for r in results)
        total_count = sum(r[2] for r in results)
        
        if total_count > 0:
            gen_gen_mean = total_sum / total_count
            # Variance formula using sum of squares
            if total_count > 1:
                var = (total_sq_sum - (total_sum**2)/total_count) / (total_count - 1)
                gen_gen_std = math.sqrt(max(0, var))

    # --- METRIC 5: Avg Min GEN-GEN (Intra-set NN) ---
    print("Calculating Min GEN-GEN Distance...", file=sys.stderr)
    min_gen_gen_mean, min_gen_gen_std = 0.0, 0.0
    if N_gen >= 2:
        indexed_gen = list(enumerate(gen_seqs))
        gen_chunks = list(chunk_list(indexed_gen, num_chunks))
        
        results = pool.map(worker_min_intra, gen_chunks)
        all_mins = [d for sub in results for d in sub]
        min_gen_gen_mean, min_gen_gen_std = get_stats(all_mins)

    # --- METRIC 6: Avg Min GEN-TRAIN (Gen vs Train) ---
    print("Calculating Min GEN-TRAIN Distance...", file=sys.stderr)
    min_gen_train_mean, min_gen_train_std = 0.0, 0.0
    if N_gen > 0 and N_train > 0:
        gen_chunks_simple = list(chunk_list(gen_seqs, num_chunks))
        task_args = [(chunk, 'TRAIN') for chunk in gen_chunks_simple]
        
        results = pool.map(worker_min_inter, task_args)
        all_mins = [d for sub in results for d in sub]
        min_gen_train_mean, min_gen_train_std = get_stats(all_mins)

    # --- METRIC 7: Avg Min TRAIN-GEN (Train vs Gen) ---
    print("Calculating Min TRAIN-GEN Distance...", file=sys.stderr)
    min_train_gen_mean, min_train_gen_std = 0.0, 0.0
    if N_gen > 0 and N_train > 0:
        train_chunks_simple = list(chunk_list(train_seqs, num_chunks))
        task_args = [(chunk, 'GEN') for chunk in train_chunks_simple]
        
        results = pool.map(worker_min_inter, task_args)
        all_mins = [d for sub in results for d in sub]
        min_train_gen_mean, min_train_gen_std = get_stats(all_mins)

    pool.close()
    pool.join()

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print(f"RESULTS REPORT (d < {args.d})")
    print(f"GEN:   {args.gen_fasta}")
    print(f"TRAIN: {args.train_fasta}")
    print("="*50)
    
    # Formatter for alignment
    def fmt(label, val1, std=None):
        if std is not None:
            return f"{label:<30} {val1:.4f} ± {std:.4f}"
        return f"{label:<30} {val1}"

    print(f"{'Total Sequences (GEN)':<30} {N_gen}")
    print(f"{'Unique Sequences (GEN)':<30} {gen_unique_count}")
    print(f"{'Total Sequences (TRAIN)':<30} {N_train}")
    print("-" * 50)
    print(fmt("Avg Length (GEN)", len_gen_mean, len_gen_std))
    print(fmt("Avg Length (TRAIN)", len_train_mean, len_train_std))
    print("-" * 50)
    print(f"{'Clusters (GEN)':<30} {cluster_count}")
    print("-" * 50)
    print(fmt("Avg Distance (GEN-GEN)", gen_gen_mean, gen_gen_std))
    print(fmt("Avg Min Dist (GEN-GEN)", min_gen_gen_mean, min_gen_gen_std))
    print(fmt("Avg Min Dist (GEN-TRAIN)", min_gen_train_mean, min_gen_train_std))
    print(fmt("Avg Min Dist (TRAIN-GEN)", min_train_gen_mean, min_train_gen_std))
    print("="*50)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
