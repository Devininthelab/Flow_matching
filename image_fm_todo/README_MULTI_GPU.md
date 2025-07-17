# Multi-GPU Training with Accelerate

This guide explains how to run your Flow Matching training code on 4 GPUs using Hugging Face Accelerate.

## Setup

1. **Install Accelerate and other dependencies:**
```bash
pip install accelerate wandb
# or install all requirements
pip install -r requirements.txt
```

2. **Configure Accelerate (Option 1 - Interactive):**
```bash
accelerate config
```
When prompted, choose:
- Compute environment: LOCAL_MACHINE
- Distributed type: MULTI_GPU
- How many machines: 1
- How many processes: 4
- GPU IDs: all
- Mixed precision: fp16 (recommended for faster training)

3. **Configure Accelerate (Option 2 - Use provided config):**
```bash
# Use the provided accelerate_config.yaml
accelerate launch --config_file accelerate_config.yaml train.py [args]
```

## Running Multi-GPU Training

### Method 1: Using the launch script (Recommended)
```bash
./run_multi_gpu_simple.sh
```

### Method 2: Using accelerate launch directly
```bash
accelerate launch --multi_gpu --num_processes=4 train.py --use_cfg --batch_size 32 --mixed_precision fp16
```

### Method 3: Using the config file
```bash
accelerate launch --config_file accelerate_config.yaml train.py --use_cfg --batch_size 32
```

## Key Changes Made

1. **Added Accelerate integration:**
   - Replaced manual device management with `accelerator.device`
   - Added `accelerator.prepare()` for model, optimizer, dataloader, and scheduler
   - Used `accelerator.backward()` instead of `loss.backward()`

2. **Process synchronization:**
   - Only main process handles file I/O, logging, and model saving
   - Progress bars only shown on local main process
   - Loss gathering across all processes for accurate logging

3. **Model access:**
   - Access model through `fm.module` when using multi-GPU (wrapped in DistributedDataParallel)

4. **Mixed precision support:**
   - Added `--mixed_precision` argument (fp16 recommended for faster training)
   - Accelerate handles automatic mixed precision

## Performance Tips

1. **Batch size scaling:** With 4 GPUs, you can increase batch size proportionally (e.g., 32 -> 128)
2. **Mixed precision:** Use `--mixed_precision fp16` for faster training
3. **Gradient accumulation:** Use `--gradient_accumulation_steps` if you want effective larger batch sizes
4. **Data loading:** Ensure `num_workers` in DataLoader is set appropriately

## Expected Performance Improvement

- **~4x faster training** with 4 GPUs (ideal case)
- **Memory efficiency** with mixed precision training
- **Better GPU utilization** with proper batch size scaling

## Troubleshooting

1. **Out of memory:** Reduce batch size or enable gradient accumulation
2. **Slow data loading:** Increase `num_workers` in DataLoader
3. **Uneven GPU utilization:** Check if your model/data is properly distributed

## Example Command

```bash
accelerate launch --multi_gpu --num_processes=4 train.py \
    --batch_size 128 \
    --train_num_steps 100000 \
    --use_cfg \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 1
```

This will run your Flow Matching training on 4 GPUs with mixed precision for optimal performance.
