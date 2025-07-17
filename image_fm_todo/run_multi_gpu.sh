#!/bin/bash

# Install accelerate if not already installed
pip install accelerate

# Launch multi-GPU training with accelerate
accelerate launch --config_file accelerate_config.yaml train.py \
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
