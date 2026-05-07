"""
End-to-end Smoke Test for 3D-MedDiffusion Cardiac Pipeline.

Tests the complete loop:
1. Generate mock input data
2. Load data with multi-modal cardiac Dataset
3. Create BiFlowNet + text conditioning
4. Run 1 batch forward/backward training
5. Generate synthetic output data
6. Verify output directory structure

MODIFICATION: This is a new file. Not part of the original 3D-MedDiffusion.
"""

import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def step_1_generate_mock_data(config: dict):
    """Step 1: Generate mock cardiac MRI data for pipeline validation."""
    logger.info("=" * 60)
    logger.info("STEP 1: Generating mock input data...")
    logger.info("=" * 60)

    from data_loaders.mock_data_generator import generate_mock_cardiac_data

    mock_root = config["data"]["root_dir"]
    if Path(mock_root).exists() and any(Path(mock_root).iterdir()):
        logger.info(f"Mock data already exists at {mock_root}, skipping generation.")
    else:
        generate_mock_cardiac_data(
            output_root=mock_root,
            center_name="mock_center_20240101_VIRTUAL",
            disease_categories=config["data"]["disease_categories"],
            patients_per_disease=2,
        )

    # Count generated files
    nii_count = len(list(Path(mock_root).rglob("*.nii.gz")))
    logger.info(f"Mock data ready: {nii_count} NIfTI files at {mock_root}")
    return nii_count > 0


def step_2_test_dataloader(config: dict):
    """Step 2: Test the multi-modal cardiac Dataset and DataLoader."""
    logger.info("=" * 60)
    logger.info("STEP 2: Testing Cardiac Dataset & DataLoader...")
    logger.info("=" * 60)

    from data_loaders.cardiac_dataset import CardiacMultiModalDataset

    dataset = CardiacMultiModalDataset(
        root_dir=config["data"]["root_dir"],
        modality_mapping=config["data"]["modality_mapping"],
        disease_categories=config["data"]["disease_categories"],
        file_naming=config["data"]["file_naming"],
        slice_alignment=config["data"]["slice_alignment"]["strategy"],
        patch_size=config["model"]["autoencoder"]["patch_size"],
        stage=1,
        augment=True,
    )

    logger.info(f"Dataset length: {len(dataset)}")
    assert len(dataset) > 0, "Dataset is empty!"

    # Load one sample
    sample = dataset[0]
    logger.info(f"Sample keys: {list(sample.keys())}")
    logger.info(f"  volume shape: {sample['volume'].shape}")
    logger.info(f"  disease: {sample['disease']} (idx={sample['disease_idx']})")
    logger.info(f"  patient_id: {sample['patient_id']}")
    logger.info(f"  modalities loaded: {sample['num_modalities']}")

    # Test DataLoader batching
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    batch = next(iter(loader))
    logger.info(f"Batch volume shape: {batch['volume'].shape}")
    logger.info(f"Batch disease indices: {batch['disease_idx']}")

    return dataset


def step_3_test_pdf_parser(config: dict):
    """Step 3: Test the knowledge base PDF parser (if enabled)."""
    logger.info("=" * 60)
    logger.info("STEP 3: Testing Knowledge Base PDF Parser...")
    logger.info("=" * 60)

    if not config.get("knowledge_base", {}).get("enabled", False):
        logger.info("  Knowledge base is SEALED (enabled=false). Skipping.")
        logger.info("  Reason: 文本-影像对齐尚未实现，PDF模块暂不启用。")
        return None

    from knowledge_base.create_placeholder_pdfs import create_placeholder_pdfs
    from knowledge_base.pdf_parser import PDFKnowledgeParser

    # Create placeholder PDFs
    kb_dir = str(PROJECT_ROOT / "knowledge_base")
    create_placeholder_pdfs(kb_dir)

    # Test parser
    pdf_paths = config["knowledge_base"]["pdf_paths"]

    parser = PDFKnowledgeParser(
        pdf_paths=pdf_paths,
        target_keywords=config["knowledge_base"]["parser"]["target_keywords"],
    )

    # Try parsing (may fall back to .txt if PyMuPDF not installed)
    for disease in ["RCM_niigz", "ARVC_niigz"]:
        text = parser.get_disease_text(disease)
        logger.info(f"  {disease}: {len(text)} chars extracted")
        if text:
            logger.info(f"    Preview: {text[:100]}...")

    return parser


def step_4_test_model_forward(config: dict, device: str):
    """Step 4: Create model and run 1-batch forward/backward pass."""
    logger.info("=" * 60)
    logger.info("STEP 4: Testing Model Forward/Backward (1 batch)...")
    logger.info("=" * 60)

    from ddpm.BiFlowNet_cardiac import BiFlowNetCardiac

    model_cfg = config["model"]
    latent_channels = model_cfg["autoencoder"]["embedding_dim"]
    num_classes = model_cfg["diffusion"]["num_classes"]
    kb_enabled = config.get("knowledge_base", {}).get("enabled", False)
    text_dim = model_cfg["diffusion"]["text_conditioning"]["text_embed_dim"] if kb_enabled else 0

    # Create BiFlowNet with text conditioning
    # Note: dim must be divisible by 3 for sincos positional embeddings
    model = BiFlowNetCardiac(
        dim=192,  # divisible by 3 for sincos pos embed
        cond_classes=num_classes,
        channels=latent_channels,
        dim_mults=(1, 1, 2),
        sub_volume_size=(8, 8, 8),
        patch_size=2,
        attn_heads=4,
        DiT_num_heads=4,
        resnet_groups=8,
        res_condition=True,
        text_condition_dim=text_dim,
        use_sparse_linear_attn=[0, 0, 0],
        num_mid_DiT=1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"BiFlowNet parameters: {total_params:,}")
    logger.info(f"Text conditioning dim: {text_dim}")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Simulate a training batch
    batch_size = config["training"]["batch_size"]
    latent_size = model_cfg["diffusion"]["image_size"]

    # Random latent (as if encoded by AutoEncoder)
    x = torch.randn(batch_size, latent_channels, *latent_size, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)  # class labels
    res = torch.randn(batch_size, 3, device=device)  # resolution embedding
    text_emb = torch.randn(batch_size, text_dim, device=device) if text_dim > 0 else None  # text embedding

    logger.info(f"Input shapes: x={x.shape}, t={t.shape}, y={y.shape}, res={res.shape}, text_emb={text_emb.shape if text_emb is not None else 'None (KB sealed)'}")

    # Forward pass
    model.train()
    optimizer.zero_grad()

    start_time = time.time()
    output = model(x, time=t, y=y, res=res, text_emb=text_emb)
    forward_time = time.time() - start_time

    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Forward pass time: {forward_time:.3f}s")

    # Compute loss and backward
    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)

    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time

    optimizer.step()
    logger.info(f"Loss: {loss.item():.4f}")
    logger.info(f"Backward pass time: {backward_time:.3f}s")
    logger.info("Forward/Backward pass PASSED!")

    return model


def step_5_generate_synthetic(config: dict):
    """Step 5: Generate synthetic dataset and export with correct directory structure."""
    logger.info("=" * 60)
    logger.info("STEP 5: Generating Synthetic Dataset...")
    logger.info("=" * 60)

    from generate_synthetic_dataset import generate_and_export, verify_output_structure

    paths = generate_and_export(config)

    logger.info("Verifying output structure...")
    success = verify_output_structure(config["generation"]["output_root"], config)

    if success:
        logger.info("Output structure verification PASSED!")
    else:
        logger.error("Output structure verification FAILED!")

    return success


def step_6_print_results(success: bool):
    """Step 6: Final results."""
    logger.info("=" * 60)
    if success:
        logger.info("数据生成流水线构建与结构化导出测试通过")
    else:
        logger.info("SMOKE TEST FAILED - see errors above")
    logger.info("=" * 60)


def main():
    logger.info("Starting 3D-MedDiffusion Cardiac Pipeline Smoke Test")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        logger.info("Running on CPU (no GPU detected)")
        device = "cpu"

    # Load config
    config_path = PROJECT_ROOT / "configs" / "cardiac_pipeline.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    try:
        # Step 1: Generate mock data
        assert step_1_generate_mock_data(config), "Mock data generation failed"

        # Step 2: Test DataLoader
        dataset = step_2_test_dataloader(config)

        # Step 3: Test PDF parser
        parser = step_3_test_pdf_parser(config)

        # Step 4: Test model forward/backward
        model = step_4_test_model_forward(config, device)

        # Step 5: Generate synthetic dataset
        gen_success = step_5_generate_synthetic(config)

        # Step 6: Print results
        step_6_print_results(gen_success)

        return gen_success

    except Exception as e:
        logger.error(f"Smoke test failed with exception: {e}", exc_info=True)
        step_6_print_results(False)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
