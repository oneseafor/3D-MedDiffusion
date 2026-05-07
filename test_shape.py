"""
测试 encoder/decoder 的形状是否正确。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from omegaconf import OmegaConf

def test_shape(modality="cine"):
    """测试指定模态的 encoder/decoder 形状。"""

    # 加载配置
    cardiac_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "cardiac_pipeline.yaml")
    with open(cardiac_config_path, 'r') as f:
        cardiac_config = yaml.safe_load(f)

    model_cfg = cardiac_config["model"][modality]

    # 创建 PatchVolume 配置
    cfg = OmegaConf.create({
        "model": {
            "seed": 1234,
            "embedding_dim": model_cfg["autoencoder"]["embedding_dim"],
            "n_codes": model_cfg["autoencoder"]["n_codes"],
            "n_hiddens": model_cfg["autoencoder"]["n_hiddens"],
            "downsample": model_cfg["autoencoder"]["downsample"],
            "norm_type": "group",
            "num_groups": model_cfg["autoencoder"]["n_groups"],
            "discriminator_iter_start": 20000,
            "disc_loss_type": "hinge",
            "volume_gan_weight": 2,
            "perceptual_weight": 2,
            "l1_weight": 4.0,
            "gan_feat_weight": 4,
            "perceptual_3d": False,
            "stage": 1,
        },
        "dataset": {
            "patch_size": model_cfg["autoencoder"]["patch_size"],
            "image_channels": 1,
        }
    })

    # 创建模型
    from AutoEncoder.model.PatchVolume import patchvolumeAE
    model = patchvolumeAE(cfg)
    model.eval()

    # 创建测试输入
    patch_size = model_cfg["autoencoder"]["patch_size"]
    patch_depth = model_cfg["autoencoder"]["patch_depth"]
    batch_size = 1

    x = torch.randn(batch_size, 1, patch_depth, patch_size, patch_size)

    print(f"\n{'='*60}")
    print(f"测试 {modality.upper()} 模态")
    print(f"{'='*60}")

    print(f"\n【配置】")
    print(f"  downsample: {model_cfg['autoencoder']['downsample']}")
    print(f"  patch_size: {patch_size}")
    print(f"  patch_depth: {patch_depth}")

    print(f"\n【输入形状】")
    print(f"  x.shape: {x.shape}")

    # 测试 encoder
    with torch.no_grad():
        z = model.pre_vq_conv(model.encoder(x))
        print(f"\n【Encoder 输出】")
        print(f"  z.shape: {z.shape}")

        # 测试 codebook
        vq_output = model.codebook(z)
        embeddings = vq_output['embeddings']
        print(f"\n【Codebook 输出】")
        print(f"  embeddings.shape: {embeddings.shape}")

        # 测试 decoder
        x_recon = model.decoder(model.post_vq_conv(embeddings))
        print(f"\n【Decoder 输出】")
        print(f"  x_recon.shape: {x_recon.shape}")

    # 检查形状是否匹配
    print(f"\n【形状检查】")
    if x.shape == x_recon.shape:
        print(f"  ✓ 输入和输出形状匹配: {x.shape}")
        return True
    else:
        print(f"  ✗ 输入和输出形状不匹配!")
        print(f"    输入: {x.shape}")
        print(f"    输出: {x_recon.shape}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("测试 Encoder/Decoder 形状")
    print("=" * 60)

    results = {}
    for modality in ["cine", "lge"]:
        try:
            results[modality] = test_shape(modality)
        except Exception as e:
            print(f"\n✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            results[modality] = False

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for modality, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {modality.upper()}: {status}")
