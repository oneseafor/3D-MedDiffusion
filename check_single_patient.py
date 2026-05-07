"""
检查单个病人的数据尺寸。
用于确认模型输入尺寸，确保配置与实际数据一致。

使用方法：
    python check_single_patient.py
"""

import os
import nibabel as nib
from pathlib import Path


def check_patient_data(patient_dir):
    """检查单个病人目录下所有模态的尺寸。"""
    patient_path = Path(patient_dir)

    print("=" * 70)
    print("单病人数据尺寸检查")
    print("=" * 70)
    print(f"\n目录: {patient_dir}\n")

    if not patient_path.exists():
        print(f"错误: 目录不存在")
        return

    # 遍历所有子目录
    all_modalities = {}

    for subdir in sorted(patient_path.iterdir()):
        if not subdir.is_dir():
            continue

        folder_name = subdir.name
        print(f"\n【{folder_name}】")

        # 查找所有 nii.gz 文件
        nii_files = sorted(subdir.glob("*.nii.gz"))

        if not nii_files:
            print(f"  未找到 .nii.gz 文件")
            continue

        for nii_file in nii_files:
            try:
                img = nib.load(str(nii_file))
                shape = img.shape
                dtype = img.get_data_dtype()
                affine = img.affine
                zooms = img.header.get_zooms()

                print(f"  文件: {nii_file.name}")
                print(f"    尺寸 (shape): {shape}")
                print(f"    数据类型: {dtype}")
                print(f"    体素尺寸 (zooms): {zooms}")

                all_modalities[folder_name] = {
                    "file": nii_file.name,
                    "shape": shape,
                    "dtype": str(dtype),
                    "zooms": zooms
                }

            except Exception as e:
                print(f"  加载失败 {nii_file.name}: {e}")

    return all_modalities


def print_config_suggestions(all_modalities):
    """根据检查结果给出配置建议。"""
    print("\n" + "=" * 70)
    print("配置建议")
    print("=" * 70)

    # 识别各模态
    cine4ch_shape = None
    cinesax_shape = None
    lgesax_shapes = []

    for folder_name, info in all_modalities.items():
        name_lower = folder_name.lower()
        shape = info["shape"]

        if "4ch" in name_lower:
            cine4ch_shape = shape
            print(f"\ncine4ch (4ch):")
            print(f"  shape: {shape}")
            print(f"  建议 volume_sizes.cine4ch: [{shape[0]}, {shape[1]}, 1]")

        elif "sax" in name_lower or "短轴" in name_lower:
            cinesax_shape = shape
            print(f"\ncinesax (短轴):")
            print(f"  shape: {shape}")
            print(f"  建议 volume_sizes.cinesax: [{shape[0]}, {shape[1]}, {shape[2]}]")

        elif "lge" in name_lower:
            lgesax_shapes.append(shape)
            print(f"\nlgesax ({folder_name}):")
            print(f"  shape: {shape}")

    # 确定 patch_size 建议
    print("\n" + "-" * 70)
    print("patch_size 建议:")
    print("-" * 70)

    all_shapes = []
    if cine4ch_shape:
        all_shapes.append(cine4ch_shape)
    if cinesax_shape:
        all_shapes.append(cinesax_shape)
    all_shapes.extend(lgesax_shapes)

    if all_shapes:
        # 取所有模态中最小的 H 和 W
        min_h = min(s[0] for s in all_shapes)
        min_w = min(s[1] for s in all_shapes)
        min_hw = min(min_h, min_w)

        print(f"\n  所有模态中:")
        print(f"    最小 H: {min_h}")
        print(f"    最小 W: {min_w}")

        if min_hw >= 128:
            print(f"\n  建议 patch_size: 128 (可完整使用数据)")
            suggested_patch = 128
        elif min_hw >= 64:
            print(f"\n  建议 patch_size: 64 (训练时随机裁剪)")
            suggested_patch = 64
        else:
            print(f"\n  建议 patch_size: {min_hw} (数据较小，直接使用)")
            suggested_patch = min_hw

        # 计算 latent size
        latent_size = suggested_patch // 8
        print(f"\n  对应 diffusion.image_size: [{latent_size}, {latent_size}, {latent_size}]")

    # lgesax 输出尺寸建议
    if lgesax_shapes:
        print("\n" + "-" * 70)
        print("lgesax 输出尺寸建议:")
        print("-" * 70)
        # 取第一个 lge 的尺寸
        lge_shape = lgesax_shapes[0]
        print(f"  建议 volume_sizes.lgesax: [{lge_shape[0]}, {lge_shape[1]}, 1]")


def main():
    # 使用用户提供的目录
    patient_dir = "/home/zirui/yuansq/CMR-AI/0_data_CMRAI/center/arvc/complete"

    all_modalities = check_patient_data(patient_dir)

    if all_modalities:
        print_config_suggestions(all_modalities)

    print("\n" + "=" * 70)
    print("检查完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
