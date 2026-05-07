"""
验证配置是否与实际数据匹配。
支持分模态验证：Cine模型和LGE模型。

使用方法：
    python verify_config.py              # 验证所有模态
    python verify_config.py --cine       # 只验证Cine
    python verify_config.py --lge        # 只验证LGE
"""

import argparse
import yaml
import torch
from data_loaders.cardiac_dataset import CardiacMultiModalDataset


def verify_modality(config: dict, modality_type: str):
    """验证单个模态的配置和数据加载。"""
    print("\n" + "=" * 70)
    print(f"验证 {modality_type.upper()} 模态")
    print("=" * 70)

    # Get model config
    model_cfg = config["model"][modality_type]
    train_cfg = config["training"][modality_type]

    print(f"\n【模型配置】")
    print(f"  patch_size: {model_cfg['autoencoder']['patch_size']}")
    print(f"  patch_depth: {model_cfg['autoencoder']['patch_depth']}")
    print(f"  image_size: {model_cfg['diffusion']['image_size']}")
    print(f"  downsample: {model_cfg['autoencoder']['downsample']}")

    print(f"\n【训练配置】")
    print(f"  batch_size: {train_cfg['batch_size']}")
    print(f"  precision: {train_cfg['precision']}")

    # Create Dataset
    print("\n" + "-" * 70)
    print("创建 Dataset...")
    print("-" * 70)

    try:
        ds = CardiacMultiModalDataset(
            root_dir=config['data']['root_dir'],
            modality_mapping=config['data']['modality_mapping'],
            disease_categories=config['data']['disease_categories'],
            file_naming=config['data']['file_naming'],
            modality_type=modality_type,
            slice_alignment=config['data']['slice_alignment']['strategy'],
            patch_size=model_cfg['autoencoder']['patch_size'],
            patch_depth=model_cfg['autoencoder']['patch_depth'],
            stage=1,
            augment=False,
        )

        print(f"\n✓ Dataset 创建成功")
        print(f"  患者数量: {len(ds)}")

        if len(ds) == 0:
            print("\n⚠ 警告: 未找到患者数据")
            return False

        # Load first sample
        print("\n" + "-" * 70)
        print("加载第一个样本...")
        print("-" * 70)

        sample = ds[0]

        print(f"\n✓ 样本加载成功")
        print(f"  患者ID: {sample['patient_id']}")
        print(f"  疾病: {sample['disease']}")

        # Check volume shape
        if 'volume' in sample:
            vol_shape = sample['volume'].shape
            expected_shape = (
                model_cfg['autoencoder']['patch_size'],
                model_cfg['autoencoder']['patch_size'],
                model_cfg['autoencoder']['patch_depth']
            )

            print(f"\n【训练 patch 尺寸】")
            print(f"  实际: {vol_shape}")
            print(f"  预期: {expected_shape}")

            if vol_shape == torch.Size(expected_shape):
                print(f"  ✓ 尺寸匹配")
                return True
            else:
                print(f"  ✗ 尺寸不匹配")
                return False
        else:
            print(f"\n✗ 未找到 volume 数据")
            return False

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="验证配置与数据")
    parser.add_argument("--cine", action="store_true", help="只验证Cine模态")
    parser.add_argument("--lge", action="store_true", help="只验证LGE模态")
    args = parser.parse_args()

    print("=" * 70)
    print("3D-MedDiffusion 配置验证")
    print("=" * 70)

    # Load config
    config_path = "configs/cardiac_pipeline.yaml"
    print(f"\n加载配置: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\n【数据配置】")
    print(f"  数据根目录: {config['data']['root_dir']}")

    print(f"\n【输出配置】")
    print(f"  cine4ch: {config['generation']['output_file_naming']['cine4ch']}")
    print(f"  cinesax: {config['generation']['output_file_naming']['cinesax']}")
    print(f"  lgesax: {config['generation']['output_file_naming']['lgesax']}")

    # Verify modalities
    results = {}

    if args.cine or (not args.cine and not args.lge):
        results["cine"] = verify_modality(config, "cine")

    if args.lge or (not args.cine and not args.lge):
        results["lge"] = verify_modality(config, "lge")

    # Summary
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)

    for modality, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {modality.upper()}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n✓ 所有验证通过")
    else:
        print(f"\n⚠ 部分验证失败")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
