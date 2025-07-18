import argparse

import numpy as np
import torch
from dataset import tensor_to_pil_image, AFHQDataModule
from fm import FlowMatching, FMScheduler
from network import UNet
from pathlib import Path


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"
    
    # Recreate the network and scheduler (same as training)
    image_resolution = 64
    
    # Try to create dataset module, but handle if data path doesn't exist
    try:
        ds_module = AFHQDataModule(
            "./data",
            batch_size=1,  # Just to get num_classes
            num_workers=1,
            max_num_images_per_cat=10,
            image_resolution=image_resolution
        )
        num_classes = getattr(ds_module, "num_classes", 3)  # Default to 3 for AFHQ
    except:
        print("Warning: Could not load dataset, using default num_classes=3")
        num_classes = 3  # AFHQ has 3 classes: cat, dog, wild
    
    # Recreate the network exactly as in training
    network = UNet(
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=True,  # CFG should be enabled
        cfg_dropout=0.1,
        num_classes=num_classes,
    )
    
    # Recreate the scheduler
    fm_scheduler = FMScheduler(sigma_min=0.001)
    
    # Create FlowMatching model
    fm = FlowMatching(
        network=network,
        fm_scheduler=fm_scheduler,
    )
    
    # Load the checkpoint using our custom method
    fm.load_accelerate(args.ckpt_path)
    fm.eval()
    fm = fm.to(device)

    print(f"Model loaded from {args.ckpt_path}")
    print(f"Network CFG support: {fm.network.use_cfg}")
    print(f"Number of classes: {getattr(fm.network, 'num_classes', 'Unknown')}")
    print(f"Image resolution: {fm.image_resolution}")


    total_num_samples = 500
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        if args.use_cfg:  # Enable CFG sampling
            assert fm.network.use_cfg, f"The model was not trained to support CFG."
            shape = (B, 3, fm.image_resolution, fm.image_resolution)
            # Generate random class labels (1, 2, 3 for cat, dog, wild)
            class_labels = torch.randint(1, 4, (B,)).to(device)
            samples = fm.sample(
                shape,
                num_inference_timesteps=50,  # Increased for better quality
                class_label=class_labels,
                guidance_scale=args.cfg_scale,
            )
            print(f"Generated batch {i+1}/{num_batches} with classes: {class_labels.cpu().tolist()}")
        else:
            raise NotImplementedError("In Assignment 7, we sample images with CFG setup only.")

        pil_images = tensor_to_pil_image(samples)

        for j, img in zip(range(sidx, eidx), pil_images):
            img.save(save_dir / f"{j}.png")
            print(f"Saved the {j}-th image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    args = parser.parse_args()
    main(args)
    # python sampling.py --use_cfg --ckpt_path results/cfg_fm-07-17-114436/last.ckpt --save_dir ../result/eval_sampling
