#!/usr/bin/env python3
"""
创建示例模型文件
用于测试和演示
"""

import sys
from pathlib import Path
import torch

# 添加src路径
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from core.model_factory import ModelFactory


def create_demo_model():
    """创建演示模型"""
    print("🤖 创建演示模型...")

    # 创建模型保存目录
    models_dir = project_root / "models" / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 创建模型
    model = ModelFactory.create_model(
        model_name="efficientnetv2_s", num_classes=3, pretrained=True  # 使用预训练权重
    )

    # 保存模型
    model_path = models_dir / "best_model.pth"

    # 保存完整的checkpoint格式
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": "efficientnetv2_s",
        "num_classes": 3,
        "epoch": 50,
        "accuracy": 0.95,
        "loss": 0.1,
        "classes": ["主图", "细节", "吊牌"],
        "created_by": "demo_script",
        "notes": "这是一个演示模型，使用EfficientNetV2-S预训练权重",
    }

    torch.save(checkpoint, model_path)

    print(f"✅ 演示模型已保存: {model_path}")
    print(f"📊 模型信息:")
    print(f"   - 架构: EfficientNetV2-S")
    print(f"   - 类别数: 3")
    print(f"   - 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - 文件大小: {model_path.stat().st_size / (1024*1024):.1f} MB")

    return model_path


if __name__ == "__main__":
    try:
        model_path = create_demo_model()
        print(f"\n🎉 演示模型创建完成!")
        print(f"现在可以运行: python scripts/fast_classify.py --dry-run")
    except Exception as e:
        print(f"❌ 创建演示模型失败: {e}")
        import traceback

        traceback.print_exc()
