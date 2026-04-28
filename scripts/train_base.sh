#!/bin/bash

echo "Training base model on chorismate mutase family"
pixi run base-train \
    -f data/Russ_Uniprot_Unaligned_unique_filtered.fasta \
    -o checkpoints/base/chorismate_mutase \
    -b 512 \
    --lr 1e-3 \
    --max-steps 8000 \
    --log-steps 100 \
    --eval-steps 500 \
    --dtype bfloat16


echo "Training base model on phage lysozyme family"
pixi run base-train \
    -f data/Phage_lysozyme_Unaligned_Filtered.fasta \
    -o checkpoints/base/lysozyme \
    -b 512 \
    --lr 1e-3 \
    --max-steps 5000 \
    --log-steps 100 \
    --eval-steps 500 \
    --dtype bfloat16

echo "Done"