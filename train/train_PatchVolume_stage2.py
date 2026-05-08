
import sys
import os
import yaml
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from AutoEncoder.model.PatchVolume import patchvolumeAE,AE_finetuning
from train.callbacks import VolumeLogger
from dataset.vqgan_4x import VQGANDataset_4x
from dataset.vqgan import VQGANDataset
from data_loaders.cardiac_dataset import CardiacMultiModalDataset
import argparse
from omegaconf import OmegaConf
import torch
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

def main(cfg_path: str, modality: str = None):
    cfg = OmegaConf.load(cfg_path)
    pl.seed_everything(cfg.model.seed)

    # Check if using cardiac dataset (modality specified)
    if modality:
        # Load cardiac config
        cardiac_config_path = os.path.join(project_root, "configs", "cardiac_pipeline.yaml")
        with open(cardiac_config_path, 'r') as f:
            cardiac_config = yaml.safe_load(f)

        # Get model config based on modality
        model_cfg = cardiac_config["model"][modality]

        # Create cardiac dataset
        train_dataset = CardiacMultiModalDataset(
            root_dir=cardiac_config["data"]["root_dir"],
            modality_mapping=cardiac_config["data"]["modality_mapping"],
            disease_categories=cardiac_config["data"]["disease_categories"],
            file_naming=cardiac_config["data"]["file_naming"],
            modality_type=modality,
            slice_alignment=cardiac_config["data"]["slice_alignment"]["strategy"],
            frame_selection=cardiac_config["data"]["frame_selection"]["strategy"],
            lge_frame_selection=cardiac_config["data"]["lge_frame_selection"]["strategy"],
            patch_size=model_cfg["autoencoder"]["patch_size"],
            patch_depth=model_cfg["autoencoder"]["patch_depth"],
            stage=2,
            augment=True,
        )

        val_dataset = CardiacMultiModalDataset(
            root_dir=cardiac_config["data"]["root_dir"],
            modality_mapping=cardiac_config["data"]["modality_mapping"],
            disease_categories=cardiac_config["data"]["disease_categories"],
            file_naming=cardiac_config["data"]["file_naming"],
            modality_type=modality,
            slice_alignment=cardiac_config["data"]["slice_alignment"]["strategy"],
            frame_selection=cardiac_config["data"]["frame_selection"]["strategy"],
            lge_frame_selection=cardiac_config["data"]["lge_frame_selection"]["strategy"],
            patch_size=model_cfg["autoencoder"]["patch_size"],
            patch_depth=model_cfg["autoencoder"]["patch_depth"],
            stage=2,
            augment=False,
        )

        # Update cfg with cardiac model config
        cfg.model.downsample = model_cfg["autoencoder"]["downsample"]
        cfg.model.patch_size = model_cfg["autoencoder"]["patch_size"]
        cfg.model.batch_size = cardiac_config["training"][modality]["batch_size"]

        print(f"Using cardiac dataset for {modality} modality")
        print(f"  Patch size: {model_cfg['autoencoder']['patch_size']}")
        print(f"  Patch depth: {model_cfg['autoencoder']['patch_depth']}")
        print(f"  Downsample: {model_cfg['autoencoder']['downsample']}")
    else:
        # Use original dataset
        downsample_ratio = cfg.model.downsample[0]
        if downsample_ratio == 4:
            train_dataset = VQGANDataset_4x(
                root_dir=cfg.dataset.root_dir,augmentation=True,split='train',stage=cfg.model.stage)
            val_dataset = VQGANDataset_4x(
                root_dir=cfg.dataset.root_dir,augmentation=False,split='val')
        else:
            train_dataset = VQGANDataset(
                root_dir=cfg.dataset.root_dir,augmentation=True,split='train',stage=cfg.model.stage)
            val_dataset = VQGANDataset(
                root_dir=cfg.dataset.root_dir,augmentation=False,split='val')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,shuffle=True,
                                  num_workers=cfg.model.num_workers)


    val_dataloader = DataLoader(val_dataset, batch_size=1,
                                shuffle=True, num_workers=cfg.model.num_workers)

    # automatically adjust learning rate
    bs, lr, ngpu = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus


    print("Setting learning rate to {:.2e}, batch size to {}, ngpu to {}".format(lr, bs, ngpu))

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(VolumeLogger(
        batch_frequency=1500, max_volumes=4, clamp=True))
    callbacks.append(AE_finetuning())
    callbacks.append(EarlyStopping(
        monitor='val/recon_loss',
        patience=20,
        mode='min',
        verbose=True
    ))



    logger = TensorBoardLogger(cfg.model.default_root_dir, name="my_model")
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.model.gpus,
        default_root_dir=cfg.model.default_root_dir,
        strategy='ddp_find_unused_parameters_true',
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        check_val_every_n_epoch=1,
        num_sanity_val_steps = 2,
        logger=logger
    )
    ckpt = torch.load(cfg.model.resume_from_checkpoint)
    model = patchvolumeAE(cfg)
    model.load_state_dict(ckpt['state_dict'])
    model.cfg = cfg
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--modality", type=str, choices=["cine", "lge"], help="Modality type: cine or lge")
    args = parser.parse_args()
    main(args.config, args.modality)



