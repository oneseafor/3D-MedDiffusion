"""
Multi-modal Cardiac MRI Dataset for 3D-MedDiffusion.

Reads the six-level directory structure:
  root -> center/batch -> disease -> patient -> modality -> .nii.gz

Supports three modalities per patient: cine4ch, cinesax, lgesax.
Handles slice alignment when cinesax has 3 slices but lgesax has 1.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CardiacMultiModalDataset(Dataset):
    """
    PyTorch Dataset for multi-modal cardiac MRI data.

    For each patient (Level 3), extracts:
      - cine4ch: long-axis cine (single volume)
      - cinesax: short-axis cine (3 slices: down/mid/up)
      - lgesax:  late gadolinium enhancement (2 volumes: mid and other slice)

    Handles dimension alignment between modalities.
    """

    def __init__(
        self,
        root_dir: str,
        modality_mapping: Dict[str, str],
        disease_categories: List[str],
        file_naming: Dict[str, List[str]],
        slice_alignment: str = "mid_only",
        frame_selection: str = "mid_only",
        frame_index: int = 12,
        lge_slice_selection: str = "mid_only",
        lge_slice_index: int = 3,
        patch_size: int = 128,
        stage: int = 1,
        augment: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.modality_mapping = modality_mapping
        self.disease_categories = disease_categories
        self.file_naming = file_naming
        self.slice_alignment = slice_alignment
        self.frame_selection = frame_selection
        self.frame_index = frame_index
        self.lge_slice_selection = lge_slice_selection
        self.lge_slice_index = lge_slice_index
        self.patch_size = patch_size
        self.stage = stage
        self.augment = augment
        self.normalize = normalize

        # Discover all patient paths
        self.patient_paths = self._discover_patients()
        logger.info(f"Found {len(self.patient_paths)} patient samples across {len(disease_categories)} diseases")

        # Build disease-to-index mapping
        self.disease_to_idx = {d: i for i, d in enumerate(sorted(disease_categories))}

    def _discover_patients(self) -> List[Dict]:
        """Walk the six-level directory tree to find all valid patient entries."""
        patients = []
        for center_dir in sorted(self.root_dir.iterdir()):
            if not center_dir.is_dir():
                continue
            for disease_dir in sorted(center_dir.iterdir()):
                if not disease_dir.is_dir():
                    continue
                if disease_dir.name not in self.disease_categories:
                    continue
                for patient_dir in sorted(disease_dir.iterdir()):
                    if not patient_dir.is_dir():
                        continue
                    # Check that at least one modality exists
                    modalities_found = {}
                    for mod_key, mod_prefix in self.modality_mapping.items():
                        mod_path = patient_dir / mod_prefix
                        # Modality dir might have suffix after prefix
                        matching_dirs = [
                            d for d in patient_dir.iterdir()
                            if d.is_dir() and d.name.startswith(mod_prefix)
                        ]
                        if matching_dirs:
                            modalities_found[mod_key] = matching_dirs[0]
                    if modalities_found:
                        patients.append({
                            "center": center_dir.name,
                            "disease": disease_dir.name,
                            "patient_id": patient_dir.name,
                            "patient_path": patient_dir,
                            "modalities": modalities_found,
                        })
        return patients

    def _load_nifti_volume(self, filepath: Path) -> np.ndarray:
        """Load a NIfTI file and return as numpy array."""
        try:
            import nibabel as nib
            img = nib.load(str(filepath))
            data = img.get_fdata().astype(np.float32)
            return data
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return None

    def _load_modality(self, mod_path: Path, mod_key: str) -> Optional[np.ndarray]:
        """Load all files for a given modality, handling multi-slice cases."""
        file_names = self.file_naming.get(mod_key, [])
        volumes = []

        for fname in file_names:
            fpath = mod_path / fname
            if fpath.exists():
                vol = self._load_nifti_volume(fpath)
                if vol is not None:
                    volumes.append(vol)

        if not volumes:
            # Fallback: try to load any .nii.gz in the directory
            nii_files = sorted(mod_path.glob("*.nii.gz"))
            for f in nii_files[:3]:
                vol = self._load_nifti_volume(f)
                if vol is not None:
                    volumes.append(vol)

        if not volumes:
            return None

        if mod_key == "cinesax" and len(volumes) > 1:
            return self._align_cinesax(volumes)
        elif len(volumes) == 1:
            return volumes[0]
        else:
            # Stack along last axis if multiple volumes
            return np.stack(volumes, axis=-1)

    def _load_lgesax(self, patient_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load lgesax data from multiple lge folders.
        Each patient may have multiple lge folders (e.g., 2-3 folders).
        Returns 2 lge volumes: mid slice and another slice.
        """
        lge_prefix = self.modality_mapping.get("lgesax", "0_final_custom_lgesax")

        # Find all lge folders
        lge_folders = sorted([
            d for d in patient_path.iterdir()
            if d.is_dir() and d.name.startswith(lge_prefix)
        ])

        if not lge_folders:
            return None, None

        # Load volumes from each folder
        lge_volumes = []
        for folder in lge_folders:
            nii_files = sorted(folder.glob("*.nii.gz"))
            if nii_files:
                vol = self._load_nifti_volume(nii_files[0])
                if vol is not None:
                    lge_volumes.append(vol)

        if not lge_volumes:
            return None, None

        # If only one lge folder, duplicate it
        if len(lge_volumes) == 1:
            return lge_volumes[0], lge_volumes[0].copy()

        # If multiple lge folders, use first two
        # lge_volumes[0] = first lge (e.g., mid slice)
        # lge_volumes[1] = second lge (e.g., another slice)
        return lge_volumes[0], lge_volumes[1]

    def _align_cinesax(self, slices: List[np.ndarray]) -> np.ndarray:
        """Align cinesax multi-slice data based on configured strategy."""
        if self.slice_alignment == "mid_only":
            # Use only the middle slice (index 1 of [down, mid, up])
            return slices[len(slices) // 2]
        elif self.slice_alignment == "interpolate":
            # Average all slices
            return np.mean(np.stack(slices, axis=0), axis=0)
        elif self.slice_alignment == "stack":
            # Stack slices as channel dimension: (H, W, D) -> (C, H, W, D)
            stacked = np.stack(slices, axis=0)
            return stacked
        else:
            return slices[len(slices) // 2]

    def _select_frame(self, volume: np.ndarray) -> np.ndarray:
        """
        Select frame from cine data.

        Handles both:
        - 4D: (H, W, D, T) -> (H, W, D)
        - 3D: (H, W, T) -> (H, W, 1) when T is time frames
        """
        if volume.ndim == 4:
            # 4D data: (H, W, D, T)
            num_frames = volume.shape[3]
            if self.frame_selection == "mid_only":
                frame_idx = num_frames // 2
                return volume[:, :, :, frame_idx]
            elif self.frame_selection == "average":
                return np.mean(volume, axis=3)
            elif self.frame_selection == "stack":
                return volume
            elif self.frame_selection == "first":
                return volume[:, :, :, 0]
            elif self.frame_selection == "last":
                return volume[:, :, :, -1]
            else:
                return volume[:, :, :, num_frames // 2]

        elif volume.ndim == 3:
            # 3D data: check if last dim is time frames
            # If D=1 and T>1, it's likely (H, W, T)
            # We treat last dim as time if it's > 1 and matches expected frame count
            h, w, d = volume.shape

            # Heuristic: if d > 1, treat as time frames
            if d > 1:
                if self.frame_selection == "mid_only":
                    frame_idx = d // 2
                    return volume[:, :, frame_idx:frame_idx+1]
                elif self.frame_selection == "average":
                    avg = np.mean(volume, axis=2)
                    return avg[:, :, np.newaxis]
                elif self.frame_selection == "stack":
                    return volume
                elif self.frame_selection == "first":
                    return volume[:, :, 0:1]
                elif self.frame_selection == "last":
                    return volume[:, :, -1:]
                else:
                    return volume[:, :, d // 2:d // 2 + 1]
            else:
                return volume

        return volume

    def _select_lge_slice(self, volume: np.ndarray) -> np.ndarray:
        """Select slice from lge data (3D: H, W, S -> 3D: H, W, 1)."""
        if volume.ndim != 3:
            return volume

        num_slices = volume.shape[2]

        if self.lge_slice_selection == "mid_only":
            slice_idx = num_slices // 2
            return volume[:, :, slice_idx:slice_idx+1]
        elif self.lge_slice_selection == "average":
            avg = np.mean(volume, axis=2)
            return avg[:, :, np.newaxis]
        elif self.lge_slice_selection == "stack":
            return volume  # Keep all slices
        elif self.lge_slice_selection == "specific":
            idx = min(self.lge_slice_index, num_slices - 1)
            return volume[:, :, idx:idx+1]
        else:
            return volume[:, :, num_slices // 2:num_slices // 2 + 1]

    def _normalize_volume(self, vol: np.ndarray) -> np.ndarray:
        """Normalize volume to [-1, 1] range."""
        vmin, vmax = vol.min(), vol.max()
        if vmax - vmin < 1e-8:
            return np.zeros_like(vol)
        return 2.0 * (vol - vmin) / (vmax - vmin) - 1.0

    def _random_augment(self, vol: np.ndarray) -> np.ndarray:
        """Apply random flip and rotation augmentation."""
        if not self.augment:
            return vol
        # Random flip along axes 0 and 1
        if np.random.rand() > 0.5:
            vol = np.flip(vol, axis=0).copy()
        if np.random.rand() > 0.5:
            vol = np.flip(vol, axis=1).copy()
        # Random 90-degree rotation in axial plane
        k = np.random.randint(0, 4)
        if k > 0:
            vol = np.rot90(vol, k, axes=(0, 1)).copy()
        return vol

    def _extract_patch(self, vol: np.ndarray) -> np.ndarray:
        """Extract a random patch of size patch_size^3 from the volume."""
        shape = vol.shape
        ps = self.patch_size

        # Pad if volume is smaller than patch size
        for i in range(3):
            if shape[i] < ps:
                pad_width = [(0, 0)] * 3
                pad_width[i] = (0, ps - shape[i])
                vol = np.pad(vol, pad_width, mode='constant', constant_values=0)
                shape = vol.shape

        # Random crop
        starts = [np.random.randint(0, max(1, shape[i] - ps + 1)) for i in range(3)]
        patch = vol[starts[0]:starts[0]+ps, starts[1]:starts[1]+ps, starts[2]:starts[2]+ps]
        return patch

    def __len__(self) -> int:
        return len(self.patient_paths)

    def __getitem__(self, idx: int) -> Dict:
        patient = self.patient_paths[idx]
        result = {
            "patient_id": patient["patient_id"],
            "disease": patient["disease"],
            "disease_idx": self.disease_to_idx.get(patient["disease"], 0),
            "center": patient["center"],
        }

        # Load each modality
        loaded_volumes = {}
        for mod_key in ["cine4ch", "cinesax"]:
            if mod_key in patient["modalities"]:
                vol = self._load_modality(patient["modalities"][mod_key], mod_key)
                if vol is not None:
                    # Handle cine data (4D: H, W, D, T -> 3D: H, W, D)
                    if vol.ndim == 4:
                        vol = self._select_frame(vol)

                    if self.normalize:
                        vol = self._normalize_volume(vol)
                    vol = self._random_augment(vol)
                    loaded_volumes[mod_key] = vol

        # Load lgesax (special handling for multiple folders)
        lge1, lge2 = self._load_lgesax(patient["patient_path"])
        if lge1 is not None:
            # Handle lge data (3D: H, W, S -> 3D: H, W, 1)
            if lge1.ndim == 3 and lge1.shape[2] > 1:
                lge1 = self._select_lge_slice(lge1)
            if lge2.ndim == 3 and lge2.shape[2] > 1:
                lge2 = self._select_lge_slice(lge2)

            if self.normalize:
                lge1 = self._normalize_volume(lge1)
                lge2 = self._normalize_volume(lge2)

            lge1 = self._random_augment(lge1)
            lge2 = self._random_augment(lge2)

            loaded_volumes["lgesax"] = lge1
            loaded_volumes["lgesax2"] = lge2

        # For training: use the primary volume (lgesax preferred, else first available)
        primary_mod = "lgesax" if "lgesax" in loaded_volumes else next(iter(loaded_volumes), None)
        if primary_mod and loaded_volumes.get(primary_mod) is not None:
            vol = loaded_volumes[primary_mod]
            # Ensure 3D
            if vol.ndim == 2:
                vol = vol[:, :, np.newaxis]
            elif vol.ndim == 4:
                vol = vol[:, :, :, 0]  # Take first channel
            patch = self._extract_patch(vol)
            result["volume"] = torch.from_numpy(patch).float()
        else:
            # Fallback: return zeros
            ps = self.patch_size
            result["volume"] = torch.zeros(ps, ps, ps, dtype=torch.float32)

        # Store all modalities for generation/export
        for mod_key, vol in loaded_volumes.items():
            if vol.ndim == 2:
                vol = vol[:, :, np.newaxis]
            elif vol.ndim == 4:
                vol = vol[:, :, :, 0]
            result[f"vol_{mod_key}"] = torch.from_numpy(vol).float()

        result["num_modalities"] = len(loaded_volumes)
        return result


def build_cardiac_dataloader(config: dict, stage: int = 1, batch_size: int = 2, num_workers: int = 4):
    """Build a DataLoader from the YAML config dict."""
    from torch.utils.data import DataLoader

    dataset = CardiacMultiModalDataset(
        root_dir=config["data"]["root_dir"],
        modality_mapping=config["data"]["modality_mapping"],
        disease_categories=config["data"]["disease_categories"],
        file_naming=config["data"]["file_naming"],
        slice_alignment=config["data"]["slice_alignment"]["strategy"],
        frame_selection=config["data"]["frame_selection"]["strategy"],
        frame_index=config["data"]["frame_selection"]["frame_index"],
        lge_slice_selection=config["data"]["lge_slice_selection"]["strategy"],
        lge_slice_index=config["data"]["lge_slice_selection"]["slice_index"],
        patch_size=config["model"]["autoencoder"]["patch_size"],
        stage=stage,
        augment=(stage == 1),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader
