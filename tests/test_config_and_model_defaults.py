"""配置与模型默认值一致性回归测试。"""

from src.core.model_factory import ModelFactory
from src.core.pytorch_trainer import ClothingTrainer
from src.core.pytorch_classifier import ClothingClassifier
from src.utils.config_manager import config_manager


class _DummyModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


def test_model_alias_normalization():
    assert (
        ModelFactory.normalize_model_name("efficientnetv2_s") == "tf_efficientnetv2_s"
    )
    assert (
        ModelFactory.normalize_model_name("tf_efficientnetv2_s")
        == "tf_efficientnetv2_s"
    )


def test_config_model_settings_is_valid():
    settings = config_manager.get_model_settings()

    assert settings["name"] in ModelFactory.get_supported_models()
    assert settings["num_classes"] > 0
    assert settings["image_size"] > 0
    assert isinstance(settings["classes"], list)
    assert len(settings["classes"]) == settings["num_classes"]


def test_trainer_defaults_follow_config():
    settings = config_manager.get_model_settings()

    trainer = ClothingTrainer()

    assert trainer.model_name == ModelFactory.normalize_model_name(settings["name"])
    assert trainer.num_classes == settings["num_classes"]
    assert trainer.input_size == settings["image_size"]


def test_classifier_defaults_follow_config(monkeypatch):
    settings = config_manager.get_model_settings()

    monkeypatch.setattr(ClothingClassifier, "_load_model", lambda self: _DummyModel())

    classifier = ClothingClassifier(model_path="not-used-by-test.pth", device="cpu")

    assert classifier.model_name == ModelFactory.normalize_model_name(settings["name"])
    assert classifier.input_size == settings["image_size"]
    assert classifier.num_classes == settings["num_classes"]
