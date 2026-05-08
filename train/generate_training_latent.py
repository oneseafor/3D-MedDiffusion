import sys
import os
import yaml
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
from torch.utils.data import DataLoader
from AutoEncoder.model.PatchVolume import patchvolumeAE
from dataset.Singleres_dataset import Singleres_dataset
from data_loaders.cardiac_dataset import CardiacMultiModalDataset
import torch
from os.path import join
import argparse
import torchio as tio
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


def generate(args):

    # Check if using cardiac dataset
    if args.modality:
        # Load cardiac config
        cardiac_config_path = os.path.join(project_root, "configs", "cardiac_pipeline.yaml")
        with open(cardiac_config_path, 'r') as f:
            cardiac_config = yaml.safe_load(f)

        # Get model config based on modality
        model_cfg = cardiac_config["model"][args.modality]

        # Create cardiac dataset for latent generation
        tr_dataset = CardiacMultiModalDataset(
            root_dir=cardiac_config["data"]["root_dir"],
            modality_mapping=cardiac_config["data"]["modality_mapping"],
            disease_categories=cardiac_config["data"]["disease_categories"],
            file_naming=cardiac_config["data"]["file_naming"],
            modality_type=args.modality,
            slice_alignment=cardiac_config["data"]["slice_alignment"]["strategy"],
            frame_selection=cardiac_config["data"]["frame_selection"]["strategy"],
            lge_frame_selection=cardiac_config["data"]["lge_frame_selection"]["strategy"],
            patch_size=model_cfg["autoencoder"]["patch_size"],
            patch_depth=model_cfg["autoencoder"]["patch_depth"],
            stage=1,
            augment=False,
        )

        print(f"Using cardiac dataset for {args.modality} modality")
        print(f"  Patients: {len(tr_dataset)}")

        patch_size = model_cfg["autoencoder"]["patch_size"]
    else:
        # Use original dataset
        tr_dataset = Singleres_dataset(root_dir=args.data_path, generate_latents=True)
        patch_size = 64

    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AE_ckpt = args.AE_ckpt
    AE = patchvolumeAE.load_from_checkpoint(AE_ckpt)
    AE = AE.to(device)
    AE.eval()

    output_dir = args.output_dir or os.path.dirname(AE_ckpt)

    for batch_idx, batch in enumerate(tr_dataloader):
        if args.modality:
            # Cardiac dataset returns dict
            # Use "data" key which has shape (B, C, D, H, W) instead of "volume" (B, H, W, D)
            sample = batch["data"].cuda()
            patient_ids = batch["patient_id"]
        else:
            # Original dataset returns tuple
            sample, paths = batch
            sample = sample.cuda()

        with torch.no_grad():
            z = AE.patch_encode(sample, patch_size=patch_size)
            output = ((z - AE.codebook.embeddings.min()) /
                      (AE.codebook.embeddings.max() -
                       AE.codebook.embeddings.min())) * 2.0 - 1.0

        output = output.cpu()

        if args.modality:
            # Save cardiac latents
            for idx, patient_id in enumerate(patient_ids):
                output_ = output[idx]
                # Create output path based on patient ID
                latent_path = os.path.join(output_dir, f"{args.modality}_latent", f"{patient_id}.nii.gz")
                os.makedirs(os.path.dirname(latent_path), exist_ok=True)
                img = tio.ScalarImage(tensor=output_)
                img.save(latent_path)

                if batch_idx == 0 and idx == 0:
                    print(f"  Latent shape: {output_.shape}")
        else:
            # Save original latents
            for idx, path in enumerate(paths):
                output_ = output[idx]
                dir_name = os.path.basename(os.path.dirname(path))
                latent_dir_name = dir_name + '_latents'
                path = path.replace(dir_name, latent_dir_name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                img = tio.ScalarImage(tensor=output_)
                img.save(path)

        if batch_idx % 10 == 0:
            print(f"  Processed batch {batch_idx}/{len(tr_dataloader)}")

    print(f"Latent generation complete. Output saved to: {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="Data path for original dataset")
    parser.add_argument("--AE-ckpt", type=str, required=True, help="AutoEncoder checkpoint path")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--modality", type=str, choices=["cine", "lge"], help="Modality type: cine or lge")
    parser.add_argument("--output-dir", type=str, help="Output directory for latents")
    args = parser.parse_args()

    if not args.modality and not args.data_path:
        parser.error("Either --modality or --data-path must be specified")

    generate(args)
