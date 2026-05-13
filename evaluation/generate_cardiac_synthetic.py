"""
Cardiac MRI Synthetic Data Generation Script.

Generates synthetic cardiac MRI data for 5 disease categories using trained BiFlowNet + AutoEncoder.
Supports both patch-based (128x128) and full-resolution (256x256) generation.

Usage:
    python evaluation/generate_cardiac_synthetic.py \
        --AE-ckpt /path/to/stage2_checkpoint.ckpt \
        --model-ckpt /path/to/biflownet_checkpoint.pt \
        --output-dir /path/to/output \
        --mode patch  # or 'full' for full resolution
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import numpy as np
from ddpm.BiFlowNet import GaussianDiffusion, BiFlowNet
from AutoEncoder.model.PatchVolume import patchvolumeAE
import torchio as tio


# Cardiac MRI disease mapping
DISEASE_MAPPING = {
    0: "ARVC",   # Arrhythmogenic Right Ventricular Cardiomyopathy
    1: "DCM",    # Dilated Cardiomyopathy
    2: "HCM",    # Hypertrophic Cardiomyopathy
    3: "LVNC",   # Left Ventricular Non-Compaction
    4: "RCM",    # Restrictive Cardiomyopathy
}

# Resolution mapping for cardiac MRI
# Latent space: [25, 16, 16] -> Decoded: [25, 128, 128] (patch)
# For full resolution: [25, 32, 32] -> Decoded: [25, 256, 256]
RESOLUTION_MAPPING = {
    "patch": {
        "latent_size": (25, 16, 16),
        "output_size": "128x128x25",
    },
    "full": {
        "latent_size": (25, 32, 32),
        "output_size": "256x256x25",
    },
}


def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create output directory
    output_dir = os.path.join(args.output_dir, f"seed_{args.seed}", args.mode)
    os.makedirs(output_dir, exist_ok=True)

    # Load BiFlowNet model
    print("Loading BiFlowNet model...")
    model = BiFlowNet(
        dim=args.model_dim,
        dim_mults=args.dim_mults,
        channels=args.volume_channels,
        init_kernel_size=3,
        cond_classes=args.num_classes,
        learn_sigma=False,
        use_sparse_linear_attn=args.use_attn,
        vq_size=args.vq_size,
        num_mid_DiT=args.num_dit,
        patch_size=args.patch_size
    ).cuda()

    # Load checkpoint (use EMA model)
    model_ckpt = torch.load(args.model_ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(model_ckpt['ema'], strict=True)
    model = model.cuda()
    model.eval()

    # Load AutoEncoder
    print("Loading AutoEncoder...")
    AE = patchvolumeAE.load_from_checkpoint(args.AE_ckpt).cuda()
    AE.eval()

    device = torch.device("cuda")

    # Get resolution config
    res_config = RESOLUTION_MAPPING[args.mode]
    latent_size = res_config["latent_size"]
    output_size_str = res_config["output_size"]

    # Initialize diffusion
    diffusion = GaussianDiffusion(
        channels=args.volume_channels,
        timesteps=args.timesteps,
        loss_type=args.loss_type,
    ).cuda()

    # Generate samples for each disease
    total_files = 0
    for disease_idx, disease_name in DISEASE_MAPPING.items():
        if disease_idx >= args.num_classes:
            break

        print(f"\nGenerating {disease_name} (class {disease_idx})...")

        for sample_idx in range(args.samples_per_disease):
            with torch.no_grad():
                # Create random noise
                z = torch.randn(1, args.volume_channels, *latent_size, device=device)
                y = torch.tensor([disease_idx], device=device)
                res = torch.tensor(latent_size, device=device) / 64.0

                # Generate latent using diffusion model
                print(f"  Sample {sample_idx + 1}/{args.samples_per_disease}: Generating latent...")
                samples = diffusion.sample(
                    model, z, y=y, res=res, strategy=args.sampling_strategy
                )

                # Denormalize latent
                samples = (((samples + 1.0) / 2.0) * (AE.codebook.embeddings.max() -
                                                       AE.codebook.embeddings.min())) + AE.codebook.embeddings.min()

                # Decode latent to volume
                print(f"  Sample {sample_idx + 1}/{args.samples_per_disease}: Decoding volume...")
                if args.mode == "patch":
                    volume = AE.decode(samples, quantize=True)
                else:
                    # Use sliding window for full resolution
                    volume = AE.decode_sliding(samples, quantize=True)

                # Save volume
                volume = volume.detach().squeeze(0).cpu()
                volume = volume.transpose(1, 3).transpose(1, 2)  # (C, D, H, W) -> (C, H, W, D)

                filename = f"{disease_name}_{sample_idx:03d}_{output_size_str}.nii.gz"
                volume_path = os.path.join(output_dir, filename)
                tio.ScalarImage(tensor=volume).save(volume_path)

                print(f"  Saved: {filename}")
                total_files += 1

                torch.cuda.empty_cache()

    print(f"\nGeneration complete!")
    print(f"Total files: {total_files}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cardiac MRI data")
    parser.add_argument("--AE-ckpt", type=str, required=True, help="Path to AutoEncoder checkpoint")
    parser.add_argument("--model-ckpt", type=str, required=True, help="Path to BiFlowNet checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--mode", type=str, default="patch", choices=["patch", "full"],
                        help="Generation mode: 'patch' (128x128) or 'full' (256x256)")
    parser.add_argument("--samples-per-disease", type=int, default=1,
                        help="Number of samples per disease category")
    parser.add_argument("--num-classes", type=int, default=7, help="Number of disease classes")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--model-dim", type=int, default=72, help="Model dimension")
    parser.add_argument("--dim-mults", nargs='+', type=int, default=[1, 1, 2, 4, 8])
    parser.add_argument("--use-attn", nargs='+', type=int, default=[0, 0, 0, 1, 1])
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--num-dit", type=int, default=1)
    parser.add_argument("--volume-channels", type=int, default=8)
    parser.add_argument("--vq-size", type=int, default=64)
    parser.add_argument("--loss-type", type=str, default='l1')
    parser.add_argument("--sampling-strategy", type=str, default='ddpm',
                        help="Sampling strategy: 'ddpm' or 'ddim'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
