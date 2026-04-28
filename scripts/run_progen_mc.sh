#!/bin/bash

pixi run progen-run-mc \
    --output-dir logs/progen3/CM/example \
    --ckpt checkpoints/progen3/CM/checkpoint-170 \
    --num-sequences 1024 \
    --beta 1.1 \
    --block-size 2 \
    --num-mcmc-steps 10 \
    --num-steps 50 \
    --save-interval 10 \
    --resampling-window 5 \
    --seed 42
