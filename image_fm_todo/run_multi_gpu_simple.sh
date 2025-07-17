#!/bin/bash

# Install accelerate if not already installed
pip install accelerate

# Configure accelerate for multi-GPU (you can also run 'accelerate config' interactively)
export ACCELERATE_USE_DEEPSPEED=false
export ACCELERATE_MIXED_PRECISION=fp16

# Launch multi-GPU training with accelerate
accelerate launch --multi_gpu --num_processes=4 train.py \
    --batch_size 32 \
    --train_num_steps 100000 \
    --warmup_steps 200 \
    --log_interval 200 \
    --sample_log_interval 2000 \
    --max_num_images_per_cat 3000 \
    --sigma_min 0.001 \
    --seed 63 \
    --image_resolution 64 \
    --use_cfg \
    --cfg_dropout 0.1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision fp16
