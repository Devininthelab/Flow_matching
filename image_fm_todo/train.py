import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dataset import AFHQDataModule, get_data_iterator, tensor_to_pil_image
from dotmap import DotMap
from fm import FlowMatching, FMScheduler
from network import UNet
from pytorch_lightning import seed_everything
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import wandb
from PIL import Image
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
matplotlib.use("Agg")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now

def concat_images_horizontally(images):
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_im

def trajectory_to_video(traj):
    """
    Saves videos for each sample in the batch from a trajectory list.
    
    Args:
        traj (List[torch.Tensor]): List of length T, each tensor shape [BS, C, H, W].
        save_dir (str or Path): Directory to save video files.
        fps (int): Frames per second for the output video.
    """
    T = len(traj)
    BS, C, H, W = traj[0].shape
    videos = []
    for sample_idx in range(BS):
        # Collect trajectory for the sample: list of [C, H, W] tensors
        sample_traj = [traj[t][sample_idx].detach().cpu() for t in range(T)]  # length T
        pil_images = tensor_to_pil_image(torch.stack(sample_traj), single_image=False)
        # Convert each PIL image to numpy
        frames = [np.array(img) for img in pil_images]  # each is [H, W, C]
        video = np.stack(frames, axis=0)  # [T, H, W, C]
        video = video.transpose(0, 3, 1, 2)  # [T, C, H, W]
        videos.append(video)
    return videos

def main(args):
    """config"""
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    
    config = DotMap()
    config.update(vars(args))
    config.device = accelerator.device

    now = get_current_time()
    assert args.use_cfg, f"In Assignment 7, we sample images with CFG setup only."

    if args.use_cfg:
        save_dir = Path(f"results/cfg_fm-{now}")
    else:
        save_dir = Path(f"results/fm-{now}")
    
    # Only create directory on main process
    if accelerator.is_main_process:
        save_dir.mkdir(exist_ok=True, parents=True)
        print(f"save_dir: {save_dir}")

    # Use accelerate's seed setting
    set_seed(config.seed)

    if accelerator.is_main_process:
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    """######"""

    # Initialize wandb only on main process
    if accelerator.is_main_process:
        run = wandb.init(
            entity="winvswon78-nanyang-technological-university-singapore",   
            project="flow_matching_ahq",
            name=f"cfg_flow_matching-{now}" if args.use_cfg else f"flow_matching-{now}",
            config=config,
            dir=save_dir,
        )


    image_resolution = 64
    ds_module = AFHQDataModule(
        "./data",
        batch_size=config.batch_size,
        num_workers=4,
        max_num_images_per_cat=config.max_num_images_per_cat,
        image_resolution=image_resolution
    )

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    # Set up the scheduler
    fm_scheduler = FMScheduler(sigma_min=args.sigma_min)

    network = UNet(
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=getattr(ds_module, "num_classes", None),
    )

    fm = FlowMatching(network, fm_scheduler)

    optimizer = torch.optim.Adam(fm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    # Prepare everything with accelerator
    fm, optimizer, train_dl, scheduler = accelerator.prepare(
        fm, optimizer, train_dl, scheduler
    )
    train_it = get_data_iterator(train_dl)

    step = 0
    losses = []
    with tqdm(initial=step, total=config.train_num_steps, disable=not accelerator.is_local_main_process) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0 and accelerator.is_main_process:
                fm.eval()
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()
                shape = (4, 3, fm.module.image_resolution, fm.module.image_resolution)
                if args.use_cfg:
                    class_label = torch.tensor([1,1,2,3]).to(accelerator.device)
                    samples = fm.module.sample(shape, class_label=class_label, guidance_scale=7.5, verbose=False)
                else:
                    samples = fm.module.sample(shape, return_traj=False, verbose=False)
                pil_images = tensor_to_pil_image(samples)
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                accelerator.save(fm.state_dict(), f"{save_dir}/last.ckpt")
                if wandb.run is not None:
                    wandb.save(str(save_dir / "last.ckpt"))
                fm.train()

            # For wandb logging
            if step % config.sample_log_interval == 0 and accelerator.is_main_process:
                if args.use_cfg:
                    print("Enabling CFG sampling.")
                    fm.eval()
                    # For class 1
                    print()
                    print(f"Step {step}, logging samples to wandb with CFG.")
                    print("####################### Sample category 1 #######################")
                    class_labels = torch.tensor([1, 2, 3], dtype=torch.long).to(accelerator.device)
                    shape = (3, 3, fm.module.image_resolution, fm.module.image_resolution)
                    samples = fm.module.sample(shape=shape, return_traj=True, class_label=class_labels, guidance_scale=7.5) # use guidance scale as in sample.py
                    videos = trajectory_to_video(samples)
                    wandb_videos = []
                    for i, video in enumerate(videos):
                        assert video.shape == (51, 3, 64, 64), f"Expected video shape (51, 3, 64, 64), got {video.shape}"
                        wandb_videos.append(wandb.Video(video, fps=30, format="mp4", caption=f"cfg_sample_step_{step}_class_{i+1}"))
                    if wandb.run is not None:
                        wandb.log(
                            {f"samples_step_{step}": wandb_videos}, step=step
                        )
                    fm.train()
                else:
                    fm.eval()
                    print()
                    print(f"Step {step}, logging samples to wandb.")
                    shape = (3, 3, fm.module.image_resolution, fm.module.image_resolution)
                    samples = fm.module.sample(shape=shape, return_traj=True)
                    videos = trajectory_to_video(samples)
                    wandb_videos = []
                    for i, video in enumerate(videos):
                        assert video.shape == (51, 3, 64, 64), f"Expected video shape (51, 3, 64, 64), got {video.shape}"
                        wandb_videos.append(wandb.Video(video, fps=30, format="mp4"))
                    if wandb.run is not None:
                        wandb.log(
                            {f"samples_step_{step}": wandb_videos}, step=step
                        )
                    fm.train()


            img, label = next(train_it)
            
            with accelerator.accumulate(fm):
                if args.use_cfg:  # Conditional, CFG training
                    loss = fm.get_loss(img, class_label=label)
                else:  # Unconditional training
                    loss = fm.get_loss(img)
                
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            # Gather loss across all processes for logging
            gathered_loss = accelerator.gather(loss).mean()
            losses.append(gathered_loss.item())
            
            if accelerator.is_local_main_process:
                pbar.set_description(f"Loss: {gathered_loss.item():.4f}")

            step += 1
            if accelerator.is_local_main_process:
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU device (deprecated when using accelerate)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--sample_log_interval", type=int, default=2000)
    parser.add_argument(
        "--max_num_images_per_cat",
        type=int,
        default=3000,
        help="max number of images per category for AFHQ dataset",
    )
    parser.add_argument("--sigma_min", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--image_resolution", type=int, default=64)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision training")
    args = parser.parse_args()
    main(args)
