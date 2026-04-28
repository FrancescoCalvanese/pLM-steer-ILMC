# Reviewer Package: Classifier-Guided RR sDO Protein LM Sampling

This folder is self-contained for notebooks:

- `Prefix_Classifier/classifier_training.ipynb`: trains or loads the PF00196 RR homodimerization any-prefix classifier and plots metrics by prefix length.
- `Sampling/Sample.ipynb`: loads the RR small decoder-only protein language model checkpoint and the classifier, then runs unsteered or classifier-guided sampling.

## Contents

- `Training_DATA/RR_aligned.fasta`: aligned RR sequences.
- `Training_DATA/RR_pf00196_labels.txt`: binary PF00196 labels, one per FASTA record.
- `checkpoints/RR_sDO/RR_sDO_iter17999.pt`: trained RR small decoder-only protein language model checkpoint.
- `checkpoints/Prefix_Classifier/pf00196_prefix_string_classifier.pt`: trained prefix classifier checkpoint.
- `Prefix_Classifier/pf00196_prefix_string_metrics.csv`: saved classifier metrics table.
- `Sampling/generated_samples/`: generated FASTA samples.
- `Sampling/sampling_utils/`: local sampling/model compatibility code imported by `Sampling/Sample.ipynb`.

## Setup

Create an environment with Python 3.10+ and install:

```bash
pip install -r requirements.txt
```

GPU is recommended for sampling. The classifier notebook can run on CPU, but full retraining/evaluation over all prefix lengths may take longer.

## Running

Start Jupyter from this folder:

```bash
jupyter notebook
```

Then run:

1. `Prefix_Classifier/classifier_training.ipynb`
2. `Sampling/Sample.ipynb`

The classifier notebook defaults to loading `checkpoints/Prefix_Classifier/pf00196_prefix_string_classifier.pt` if it already exists. Set `FORCE_RETRAIN = True` inside the notebook to retrain.
