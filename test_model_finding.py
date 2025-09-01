#!/usr/bin/env python3
"""
测试默认模型查找逻辑
"""

import os
import sys
from pathlib import Path

def test_model_finding():
    """测试模型查找逻辑"""
    # 获取项目根目录
    project_root = Path(__file__).parent

    # 定义可能的模型路径和优先级
    possible_models = [
        # 优先使用最新的JiLing训练模型
        ("models/JiLing_baiditu_1755873239.pth", "最新训练的JiLing模型"),
        # 其他可能的JiLing模型（按时间戳降序）
        ("models/JiLing_baiditu_1755749592.pth", "JiLing训练模型"),
        # saved_models目录中的最佳模型
        ("models/saved_models/best_model.pth", "最佳训练模型"),
        # 默认模型
        ("models/clothing_classifier.pth", "默认分类模型"),
        # 演示模型
        ("models/demo_model.pth", "演示模型")
    ]

    print("🔍 测试模型文件查找...")
    print(f"项目根目录: {project_root}")
    print()

    found_models = []

    # 查找存在的模型文件
    for model_path, model_desc in possible_models:
        model_full_path = project_root / model_path
        exists = model_full_path.exists()

        status = "✅ 存在" if exists else "❌ 不存在"
        print("15")

        if exists:
            found_models.append((model_path, model_desc))

    print()
    print("📊 查找结果:")

    if found_models:
        print(f"✅ 找到 {len(found_models)} 个可用模型:")
        for i, (path, desc) in enumerate(found_models, 1):
            print(f"   {i}. {path} ({desc})")

        # 推荐使用的模型
        recommended = found_models[0]
        print(f"\n🎯 推荐使用: {recommended[0]} ({recommended[1]})")
        return True
    else:
        print("❌ 未找到任何模型文件")
        print("💡 建议: 运行训练脚本生成模型，或创建演示模型")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("JiLing服装分类系统 - 模型查找测试")
    print("=" * 50)

    success = test_model_finding()

    print("\n" + "=" * 50)
    if success:
        print("✅ 测试通过！模型查找逻辑工作正常")
    else:
        print("⚠️  测试完成，但未找到模型文件")
    print("=" * 50)
