"""
Generate mock cardiac MRI data in the six-level directory structure.

Creates synthetic NIfTI files with realistic spatial structure
for pipeline validation without real patient data.
"""

import os
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def generate_mock_nifti(filepath: Path, shape: tuple, affine: np.ndarray = None):
    """Generate a synthetic NIfTI volume with cardiac-like structure."""
    try:
        import nibabel as nib
    except ImportError:
        logger.error("nibabel not installed. Run: pip install nibabel")
        return

    # Create synthetic volume with some spatial structure
    data = np.random.randn(*shape).astype(np.float32) * 0.1

    # Add a rough elliptical "heart" structure in the center
    center = np.array(shape[:3]) / 2
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2] if len(shape) > 2 else 1):
                dist = np.sqrt(
                    ((x - center[0]) / (shape[0] * 0.25)) ** 2 +
                    ((y - center[1]) / (shape[1] * 0.25)) ** 2
                )
                if len(shape) > 2:
                    dist += ((z - center[2]) / (shape[2] * 0.3)) ** 2
                dist = np.sqrt(dist)
                if dist < 1.0:
                    idx = (x, y) if len(shape) <= 2 else (x, y, z)
                    data[idx] += 0.8 * (1.0 - dist)

    # 2D slices: remove extra dimensions
    if len(shape) == 2:
        data = data[:, :, np.newaxis]
        shape = data.shape

    if affine is None:
        affine = np.eye(4)
        affine[0, 0] = 1.5  # voxel spacing
        affine[1, 1] = 1.5
        affine[2, 2] = 10.0  # thick slices for cine

    img = nib.Nifti1Image(data, affine)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(filepath))
    logger.debug(f"Generated mock NIfTI: {filepath} shape={shape}")


def generate_mock_cardiac_data(
    output_root: str,
    center_name: str = "mock_center_20240101_VIRTUAL",
    disease_categories: list = None,
    patients_per_disease: int = 2,
    patient_id_prefix: str = "mock_patient",
):
    """
    Generate a complete mock cardiac dataset in the six-level structure.

    Structure:
      output_root/
        center_name/
          disease_1/
            patient_001/
              0_final_custom_cine4ch_xxx/
                cine_4ch.nii.gz
              0_final_custom_cinesax_xxx/
                cine_sax_down.nii.gz
                cine_sax_mid.nii.gz
                cine_sax_up.nii.gz
              0_final_custom_lgesax_xxx/
                lge_sax_mid.nii.gz
          disease_2/
            ...
    """
    if disease_categories is None:
        disease_categories = ["ARVC_niigz", "RCM_niigz", "DCM_niigz", "HCM_niigz", "LVNC_niigz"]

    root = Path(output_root)
    center_dir = root / center_name
    generated_files = []

    for disease in disease_categories:
        for i in range(1, patients_per_disease + 1):
            patient_id = f"{patient_id_prefix}_{i:03d}"
            patient_dir = center_dir / disease / patient_id

            # cine4ch: long-axis view (2D-like, single slice)
            cine4ch_dir = patient_dir / "0_final_custom_cine4ch_virtual"
            cine4ch_path = cine4ch_dir / "cine_4ch.nii.gz"
            generate_mock_nifti(cine4ch_path, shape=(192, 192))
            generated_files.append(cine4ch_path)

            # cinesax: short-axis view (3 slices)
            cinesax_dir = patient_dir / "0_final_custom_cinesax_virtual"
            for suffix in ["down", "mid", "up"]:
                sax_path = cinesax_dir / f"cine_sax_{suffix}.nii.gz"
                generate_mock_nifti(sax_path, shape=(192, 192))
                generated_files.append(sax_path)

            # lgesax: late gadolinium enhancement (single slice)
            lgesax_dir = patient_dir / "0_final_custom_lgesax_virtual"
            lge_path = lgesax_dir / "lge_sax_mid.nii.gz"
            generate_mock_nifti(lge_path, shape=(192, 192))
            generated_files.append(lge_path)

    logger.info(f"Generated {len(generated_files)} mock NIfTI files in {output_root}")
    return generated_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_mock_cardiac_data(
        output_root="/home/regouchang/3D-MedDiffusion/synthetic_output/mock_input_data",
        patients_per_disease=2,
    )
