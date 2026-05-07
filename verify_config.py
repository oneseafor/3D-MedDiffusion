"""
验证配置是否与实际数据匹配。
运行此脚本确认：
1. 数据能正确加载
2. 帧/切片选择正确
3. 输出尺寸符合预期

使用方法：
    python verify_config.py
"""

import yaml
import torch
from data_loaders.cardiac_dataset import CardiacMultiModalDataset


def main():
    print("=" * 70)
    print("配置验证")
    print("=" * 70)

    # 加载配置
    config_path = "configs/cardiac_pipeline.yaml"
    print(f"\n加载配置: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 打印关键配置
    print("\n【数据配置】")
    print(f"  数据根目录: {config['data']['root_dir']}")
    print(f"  帧选择策略: {config['data']['frame_selection']['strategy']}")
    print(f"  LGE切片策略: {config['data']['lge_slice_selection']['strategy']}")

    print("\n【模型配置】")
    print(f"  patch_size: {config['model']['autoencoder']['patch_size']}")
    print(f"  image_size: {config['model']['diffusion']['image_size']}")
    print(f"  downsample: {config['model']['autoencoder']['downsample']}")

    print("\n【输出尺寸】")
    for mod, size in config['generation']['volume_sizes'].items():
        print(f"  {mod}: {size}")

    # 创建 Dataset
    print("\n" + "-" * 70)
    print("创建 Dataset...")
    print("-" * 70)

    try:
        ds = CardiacMultiModalDataset(
            root_dir=config['data']['root_dir'],
            modality_mapping=config['data']['modality_mapping'],
            disease_categories=config['data']['disease_categories'],
            file_naming=config['data']['file_naming'],
            slice_alignment=config['data']['slice_alignment']['strategy'],
            frame_selection=config['data']['frame_selection']['strategy'],
            frame_index=config['data']['frame_selection']['frame_index'],
            lge_slice_selection=config['data']['lge_slice_selection']['strategy'],
            lge_slice_index=config['data']['lge_slice_selection']['slice_index'],
            patch_size=config['model']['autoencoder']['patch_size'],
            stage=1,
            augment=False,  # 验证时不做增强
        )

        print(f"\n✓ Dataset 创建成功")
        print(f"  患者数量: {len(ds)}")

        if len(ds) == 0:
            print("\n⚠ 警告: 未找到患者数据，请检查数据路径和目录结构")
            return

        # 加载第一个样本
        print("\n" + "-" * 70)
        print("加载第一个样本...")
        print("-" * 70)

        sample = ds[0]

        print(f"\n✓ 样本加载成功")
        print(f"  患者ID: {sample['patient_id']}")
        print(f"  疾病: {sample['disease']}")
        print(f"  模态数量: {sample['num_modalities']}")

        # 检查各模态尺寸
        print("\n【加载后的模态尺寸】")
        for key in ['vol_cine4ch', 'vol_cinesax', 'vol_lgesax']:
            if key in sample:
                vol = sample[key]
                print(f"  {key}: {vol.shape}")

        # 检查 patch 尺寸
        if 'volume' in sample:
            print(f"\n【训练 patch 尺寸】")
            print(f"  volume: {sample['volume'].shape}")

        # 验证尺寸是否匹配预期
        print("\n" + "-" * 70)
        print("尺寸验证...")
        print("-" * 70)

        expected_sizes = config['generation']['volume_sizes']
        all_match = True

        for mod_key in ['cine4ch', 'cinesax', 'lgesax']:
            vol_key = f'vol_{mod_key}'
            if vol_key in sample:
                actual_shape = sample[vol_key].shape
                expected = expected_sizes[mod_key]
                # 转换为 (H, W, D) 格式比较
                actual_list = list(actual_shape)

                if actual_list == expected:
                    print(f"  ✓ {mod_key}: {actual_list} == {expected}")
                else:
                    print(f"  ✗ {mod_key}: {actual_list} != {expected}")
                    all_match = False

        if all_match:
            print("\n✓ 所有模态尺寸与配置匹配")
        else:
            print("\n⚠ 部分模态尺寸不匹配，请检查配置")

        # 检查 patch_size 是否合适
        patch_size = config['model']['autoencoder']['patch_size']
        print(f"\n【patch_size 检查】")
        for mod_key in ['cine4ch', 'cinesax', 'lgesax']:
            vol_key = f'vol_{mod_key}'
            if vol_key in sample:
                vol = sample[vol_key]
                h, w = vol.shape[0], vol.shape[1]
                if h >= patch_size and w >= patch_size:
                    print(f"  ✓ {mod_key}: {h}×{w} >= {patch_size}×{patch_size}")
                else:
                    print(f"  ⚠ {mod_key}: {h}×{w} < {patch_size}×{patch_size}，训练时会零填充")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("验证完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
