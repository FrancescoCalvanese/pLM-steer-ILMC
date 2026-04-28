#!/bin/bash

pixi run base-run-mc \
    --output-dir logs/sDO/CM/example \
    --ckpt checkpoints/sDO/CM/step7500.pt \
    --num-sequences 1024 \
    --beta 1.1 \
    --block-size 2 \
    --num-mcmc-steps 10 \
    --num-steps 60 \
    --save-interval 10 \
    --resampling-window 5 \
    --seed 42
