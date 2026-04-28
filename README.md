# pLM-steer 

<p align="center">
  <img src="imgs/graph_abst_new.png" alt="pLM-steer summary">
</p>

pLM-steer is a Python package for steering autoregressive protein Language Models (pLMs) using **Iterative Lookback Monte Carlo (ILMC)** sampling with custom-defined potentials. This allows controlled generation of protein sequences that optimize desired properties, such as thermostability, similarity to target sequences, or diversity.

The package supports both custom GPT-style transformer models and the ProGen3 model from [Profluent](https://github.com/Profluent-AI/progen3).

The `Classifier_Steering/` folder is a self-contained comparison of FUDGE and ILMC steering using a any-prefix classifier. It includes the data, checkpoints, sampling utilities, and Jupyter notebooks needed to train/evaluate the classifier and run classifier-guided sampling experiments.

## Features

- **Custom Potentials**: Define and combine multiple steering potentials (e.g., sequence similarity, amino acid enrichment ratios for thermostability)
- **MCMC Sampling**: framework to perform block-wise MCMC sampling for sequence generation
- **Model Support**: Compatible with GPT-style transformer and ProGen3 models
- **Fine-tuning**: Scripts for fine-tuning ProGen3 on specific protein families
- **Experiment Reproduction**: Pre-configured scripts for running diversity, vicinity and thermostability steering experiments
- **Classifier Steering Notebooks**: Self-contained Jupyter notebooks in `Classifier_Steering/` for comparing FUDGE and ILMC steering with a classifier

## Installation

A basic version of the package including potential definitions and ILMC sampling pipeline for custom models can simply be installed with `pip install git+https://anonymous.4open.science/r/plm-steer-BEF8.git`. 

Basic requirements only include `torch`, `tqdm`, `biopython` and `numpy`.

To reproduce experimental results, or to enable support for custom GPT-style transformer and ProGen3 models, follow instead the following steps.

### Experimental Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>  # TODO: update URL with anonymized one
   cd plm-steer
   ```

2. Install package managers and dependencies:
   ```bash
   make all
   ```
   This will:
   - Install uv and pixi if not present
   - Set up pixi environments (`base` for GPT-style transformer, `progen3` for ProGen3 model)
   - Install CUDA-related packages (flash-attn, megablocks)  
    ⚠️ Attention: CUDA 12.4 support is required for packages compatibility, otherwise build may break.
   - Clone the ProGen3 repository and install related package.

Alternatively, you can run the steps manually:
```bash
make install_managers      # Install uv and pixi to default paths, if not found
make install_deps          # Install dependencies
make setup_progen          # Clone ProGen3
make verify                # Verify torch and flash attention setup
make download_checkpoints  # Download trained models for reproduction
make download_results      # Download results for analysis
```

Checkpoints and results are downloaded from the associated [Zenodo repository](https://zenodo.org/records/18403586?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhmN2ZkYTg2LTNjNTAtNGRiMy1hZTAxLTVlZmNkNTE2NDUzOCIsImRhdGEiOnt9LCJyYW5kb20iOiIzNmQ5ZGY5MGZkZmQyM2JkNTMxYjA3NjE5NWRmYmRiOSJ9.4Lc51rO-bNVWFLJiwIbKmcejJcRdt8vM3YRm2WfQGSZLTNTWQv2GcHUqoqW8XX6PSjidZxjW4qpPIreFo0tfdQ).

To activate a particular environment:
```bash
pixi shell -e default  # or pixi shell -e progen3
```

For experimental scripts (like ProGen3 fine-tuning and ILMC steering), pixi tasks that automatically use the correct environment are already defined.

To delete all pixi environments, run
```bash
make clean
```
which will remove the `.pixi` folder.

## Usage

### Basic MCMC Sampling

Run MCMC sampling with custom potentials on a pre-trained model:

```bash
# For GPT-style model
pixi run -e default python -m plm_steer.scripts.base.run_block_mc \
    --ckpt path/to/checkpoint \
    --potentials-config config.yaml \
    --num-sequences 1024 \
    --output-dir samples/

# For ProGen3 model
pixi run -e progen3 python -m plm_steer.scripts.progen.run_block_mc \
    --ckpt Profluent-Bio/progen3-112m \
    --potentials-config config.yaml \
    --num-sequences 1024 \
    --output-dir logs/
```

### Fine-tuning ProGen3

Fine-tune ProGen3 on a protein family:

```bash
pixi run -e progen3 progen-finetune \
    --fasta-path data/family_sequences.fasta \
    --output-dir checkpoints/progen3/family \
    --batch-size 512 \
    --epochs 20 \
    --lr 5e-4
```

### Defining Potentials

Create a YAML configuration file for potentials:

```yaml
potentials:
  - type: neighbouring
    weight: 1.0
    target: "MKLVLSLSLLVLVLLVLWLQLV"
    mismatch_score: -1.0
  - type: thermostability
    weight: 0.5
    per_residue_score:
      E: 1.28
      K: 1.27
      # ... other amino acids
```

## Reproducing Experiments

The `scripts/` directory contains scripts to reproduce some experiments from the paper. These include ILMC sampling with ProGen3 fine-tuned models and scripts to compute metrics for generated sequences.

### Fine-tuning

Fine-tune ProGen3 on chorismate mutase and phage lysozyme families:

```bash
bash scripts/finetune_progen.sh
```

This will create checkpoints in `checkpoints/progen3/CM/` and `checkpoints/progen3/PL/`. They are automatically downloaded with `make all` or `make download_zenodo` commands.

### MCMC Diversity Experiments

Run diversity experiments on chorismate mutase:

```bash
bash experiments/progen/run_mc_diversity_chm_array.sh
```

This script runs MCMC sampling with different window sizes and beta values (defined in `experiments/progen/diversity_chm_array_config.csv`), generating diverse sequence variants.

### Thermostability Experiments

Run thermostability experiments:

```bash
# For chorismate mutase
bash scripts/experiments/progen/run_mc_thermostability_cm.sh

# For lysozyme
bash scripts/experiments/progen/run_mc_thermostability_pl.sh
```

### Output

All experiments save generated sequences in FASTA format to the `logs/` directory, organized by model, protein family, and experiment parameters.

## Configuration

### MCMC Parameters

- `block_size`: Size of sequence blocks to resample (default: 2)
- `num_steps`: Number of generated blocks (default: 60)
- `num_mcmc_steps`: Number of resampling steps per iteration (default: 10)
- `beta`: Sampling temperature (default: 1.0)
- `resampling_window`: Window size for ILMC resampling (default: 5)

### Pre-defined Potential Types

- `neighbouring`: Sequence similarity potential using pairwise alignment
- `thermostability`: Amino acid enrichment ratios for thermostability

### Custom potentials

We provide the `PotentialRegistry` decorator to allow users to define custom potentials and load them from YAML configuration files. In your code, define your potential as:
```python
from plm_steer.potentials import BasePotential, PotentialRegistry

@PotentialRegistry.register("my_potential_name")
class MyPotential(BasePotential)
    def __init__(self, weight: float, transform: Callable, custom_param: float)
        # Potential initialization

    def __call__(self, query: str):
        # Computation over query input
```

Then, in the YAML configuration:
```yaml
potentials:
  - type: my_potential_name
    weight: 2.0
    custom_param: 5.0
...
```

## License

This code is licensed under CC BY 4.0 license.
