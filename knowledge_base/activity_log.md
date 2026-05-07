# Activity Log: 3D-MedDiffusion Cardiac Pipeline Modifications

## Overview
This log documents all modifications made to the original 3D-MedDiffusion codebase
for the cardiac multi-modal MRI generation pipeline.

**Date**: 2026-05-06
**Original repo**: https://github.com/ShanghaiTech-IMPACT/3D-MedDiffusion.git
**Environment**: med_diffusion_env (Python 3.11.11, PyTorch 2.2.0)

---

## New Files Created

### 1. `configs/cardiac_pipeline.yaml`
- **Purpose**: Configuration-driven pipeline settings
- **Content**: Data paths, modality mapping (cine4ch/cinesax/lgesax), disease categories (5 types), model architecture params, knowledge base config, training hyperparameters, generation/export settings
- **Key design**: All paths and parameters externalized to YAML; no hardcoding in source

### 2. `data_loaders/__init__.py`
- **Purpose**: Package init for data_loaders module

### 3. `data_loaders/cardiac_dataset.py`
- **Purpose**: Multi-modal cardiac MRI PyTorch Dataset
- **Class**: `CardiacMultiModalDataset`
- **Features**:
  - Walks six-level directory tree (root -> center -> disease -> patient -> modality -> nii.gz)
  - Loads all 5 cardiomyopathy types (ARVC, RCM, DCM, HCM, LVNC)
  - Extracts 3 modalities per patient: cine4ch, cinesax, lgesax
  - Handles cinesax multi-slice alignment (mid_only / interpolate / stack strategies)
  - Normalization to [-1, 1], random augmentation (flip + rotation)
  - Disease-to-index mapping for class conditioning

### 4. `data_loaders/mock_data_generator.py`
- **Purpose**: Generate synthetic NIfTI data for pipeline validation
- **Function**: `generate_mock_cardiac_data()`
- **Output**: Creates the six-level directory structure with cardiac-shaped phantom volumes

### 5. `knowledge_base/__init__.py`
- **Purpose**: Package init for knowledge_base module

### 6. `knowledge_base/pdf_parser.py`
- **Purpose**: PDF knowledge base parser for RAG/text conditioning
- **Classes**:
  - `PDFKnowledgeParser`: Extracts pathology text from PDFs using PyMuPDF, filters by keywords (LGE, fibrosis, RCM, ARVC, etc.)
  - `TextEncoder`: Transformer-based text encoder for converting pathology descriptions to embedding vectors
- **Fallback**: Works with .txt files if PyMuPDF unavailable

### 7. `knowledge_base/create_placeholder_pdfs.py`
- **Purpose**: Create placeholder PDF documents with RCM/ARVC pathology descriptions
- **Content**: LGE characteristics, myocardial tissue features, MRI findings for both diseases
- **Fallback**: Creates minimal valid PDFs without external dependencies

### 8. `ddpm/BiFlowNet_cardiac.py`
- **Purpose**: Text-conditioned wrapper around original BiFlowNet
- **Class**: `BiFlowNetCardiac`
- **MODIFICATION**: Wraps original BiFlowNet (preserved untouched), adds text conditioning pathway
- **Design**: Text embedding projected to time_dim, injected via resolution conditioning pathway
- **Rationale**: Avoids modifying original BiFlowNet internals; text info enriches conditioning signal

### 9. `generate_synthetic_dataset.py`
- **Purpose**: Synthetic dataset generation and export script
- **Functions**:
  - `generate_and_export()`: Generates RCM/ARVC data with all 3 modalities
  - `build_output_directory_tree()`: Creates exact six-level directory structure
  - `verify_output_structure()`: Validates output matches expected format
  - `save_nifti()`: Saves volumes as .nii.gz with proper affine matrices
- **Output format**: Mirrors input data structure 100%

### 10. `run_smoke_test.py`
- **Purpose**: End-to-end pipeline validation
- **Steps**:
  1. Generate mock input data
  2. Test Cardiac Dataset & DataLoader
  3. Test PDF Knowledge Base parser
  4. Test model forward/backward (1 batch)
  5. Generate synthetic dataset
  6. Verify output structure
- **Result**: "数据生成流水线构建与结构化导出测试通过"

### 11. `knowledge_base/activity_log.md` (this file)
- **Purpose**: Document all modifications

---

## Original Files Modified

**None.** All original 3D-MedDiffusion files are preserved untouched. The cardiac pipeline
is built as additive modules that import from the original codebase.

---

## New Directories Created

| Directory | Purpose |
|-----------|---------|
| `configs/` | YAML configuration files |
| `data_loaders/` | Custom PyTorch Dataset classes |
| `knowledge_base/` | PDF parser, pathology docs, activity log |
| `synthetic_output/` | Generated synthetic data output |
| `logs/` | Training logs |

---

## Environment Setup

- **Conda env**: `med_diffusion_env` (Python 3.11.11)
- **PyTorch**: 2.2.0+cpu (CUDA 11.8 compatible version available for GPU)
- **Key dependencies**: nibabel, torchio, monai, omegaconf, pytorch-lightning, einops, timm, PyMuPDF
- **CUDA**: Driver 517.00, CUDA 11.7 (WSL2 environment)

---

## Smoke Test Results

```
STEP 1: Mock data generation     — 50 NIfTI files
STEP 2: Dataset & DataLoader     — 10 patients, 3 modalities each
STEP 3: PDF Knowledge Base       — 512 chars extracted per disease
STEP 4: Model Forward/Backward   — 103M params, loss=1.1508
STEP 5: Synthetic generation     — 30 NIfTI files (RCM + ARVC)
STEP 6: Structure verification   — 30/30 passed, 0 failed
```

---

## Design Decisions

1. **Wrapper pattern for text conditioning**: Instead of modifying BiFlowNet internals,
   we wrap it with BiFlowNetCardiac which injects text embeddings via the resolution
   conditioning pathway. This preserves upstream compatibility.

2. **Additive text injection**: Text embedding is added to the time embedding (not
   concatenated), keeping dimensionality unchanged and DiTBlock compatible.

3. **Configuration-driven**: All paths, modalities, and parameters are in YAML.
   No hardcoding in source code.

4. **Six-level directory mirroring**: Output data structure exactly matches input format
   so downstream tasks can directly mount the synthetic dataset.

5. **Mock data for validation**: Since real patient data is private, we generate
   cardiac-shaped phantom volumes for pipeline testing.
