"""默认模型查找逻辑测试。"""

from pathlib import Path


def _find_available_models(project_root: Path):
    """按优先级返回存在的模型列表。"""
    possible_models = [
        ("models/JiLing_baiditu_1755873239.pth", "最新训练的JiLing模型"),
        ("models/JiLing_baiditu_1755749592.pth", "JiLing训练模型"),
        ("models/saved_models/best_model.pth", "最佳训练模型"),
        ("models/clothing_classifier.pth", "默认分类模型"),
        ("models/demo_model.pth", "演示模型"),
    ]

    found_models = []
    for model_path, model_desc in possible_models:
        model_full_path = project_root / model_path
        if model_full_path.exists():
            found_models.append((model_path, model_desc))

    return found_models


def test_model_finding_returns_models_in_priority_order(tmp_path: Path):
    """存在多个模型时，应按预设优先级返回。"""
    first_model = tmp_path / "models" / "JiLing_baiditu_1755873239.pth"
    fallback_model = tmp_path / "models" / "saved_models" / "best_model.pth"

    first_model.parent.mkdir(parents=True, exist_ok=True)
    fallback_model.parent.mkdir(parents=True, exist_ok=True)
    first_model.write_bytes(b"test")
    fallback_model.write_bytes(b"test")

    found_models = _find_available_models(tmp_path)

    assert len(found_models) == 2
    assert found_models[0][0] == "models/JiLing_baiditu_1755873239.pth"
    assert found_models[1][0] == "models/saved_models/best_model.pth"


def test_model_finding_returns_empty_when_no_model_exists(tmp_path: Path):
    """没有模型文件时，应返回空列表。"""
    found_models = _find_available_models(tmp_path)
    assert found_models == []
