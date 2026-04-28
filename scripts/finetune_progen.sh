#!/bin/bash

echo "Finetuning ProGen3 model on chorismate mutase dataset"
pixi run progen-finetune \
    -f data/chorismate_mutase.fasta \
    -o checkpoints/progen3/CM \
    -b 512 \
    -n 20 \
    --lr 5e-4 \

echo "Finetuning ProGen3 model on phage lysozime dataset"
pixi run progen-finetune \
    -f data/phage_lysozyme.fasta \
    -o checkpoints/progen3/PL \
    -b 512 \
    -n 20 \
    --lr 5e-4 \

echo "Done!"
