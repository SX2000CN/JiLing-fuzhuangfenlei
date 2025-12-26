"""
设置管理器 - 统一管理应用程序设置的持久化
"""

import json
from pathlib import Path
from typing import Any, Optional


class SettingsManager:
    """设置管理器 - 负责设置的加载、保存和访问"""

    # 默认设置
    DEFAULT_SETTINGS = {
        "appearance": {
            "theme": "dark",
            "scale": "100%"
        },
        "model": {
            "device": "auto",
            "precision": "fp32",
            "confidence_threshold": 0.5
        },
        "paths": {
            "model_dir": "models/saved_models",
            "dataset_dir": "data",
            "log_dir": "logs",
            "export_dir": "outputs"
        },
        "performance": {
            "workers": 4,
            "max_batch": 64,
            "amp_enabled": True,
            "pin_memory": True
        },
        "training": {
            "default_epochs": 50,
            "patience": 10,
            "checkpoint_freq": 5,
            "save_best_only": True
        },
        "logging": {
            "level": "INFO",
            "retention_days": 30,
            "verbose": False
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化设置管理器

        Args:
            config_path: 配置文件路径，默认为项目根目录的 gui_settings.json
        """
        if config_path is None:
            # 默认保存到项目根目录
            self._config_path = Path(__file__).resolve().parents[3] / "gui_settings.json"
        else:
            self._config_path = Path(config_path)

        self._settings = {}
        self.load()

    def load(self) -> dict:
        """从配置文件加载设置"""
        if self._config_path.exists():
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                # 合并默认设置和加载的设置
                self._settings = self._merge_settings(self.DEFAULT_SETTINGS, loaded)
            except (json.JSONDecodeError, IOError):
                self._settings = self.DEFAULT_SETTINGS.copy()
        else:
            self._settings = self.DEFAULT_SETTINGS.copy()
        return self._settings

    def save(self) -> bool:
        """保存设置到配置文件"""
        try:
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=2, ensure_ascii=False)
            return True
        except IOError:
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取设置值，支持点号分隔的嵌套键

        Args:
            key: 设置键，如 "appearance.theme" 或 "model.confidence_threshold"
            default: 默认值

        Returns:
            设置值或默认值
        """
        keys = key.split('.')
        value = self._settings
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """
        设置值，支持点号分隔的嵌套键

        Args:
            key: 设置键，如 "appearance.theme"
            value: 设置值
        """
        keys = key.split('.')
        target = self._settings
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    def get_all(self) -> dict:
        """获取所有设置"""
        return self._settings.copy()

    def update(self, settings: dict) -> None:
        """批量更新设置"""
        self._settings = self._merge_settings(self._settings, settings)

    def reset(self) -> None:
        """重置为默认设置"""
        self._settings = self.DEFAULT_SETTINGS.copy()

    @staticmethod
    def _merge_settings(base: dict, override: dict) -> dict:
        """递归合并设置字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = SettingsManager._merge_settings(result[key], value)
            else:
                result[key] = value
        return result

    def to_gui_settings(self) -> dict:
        """转换为 GUI 设置页面格式"""
        s = self._settings
        return {
            "theme": self._theme_to_display(s.get("appearance", {}).get("theme", "dark")),
            "scale": s.get("appearance", {}).get("scale", "100%"),
            "device": self._device_to_display(s.get("model", {}).get("device", "auto")),
            "precision": self._precision_to_display(s.get("model", {}).get("precision", "fp32")),
            "confidence_threshold": s.get("model", {}).get("confidence_threshold", 0.5),
            "model_dir": s.get("paths", {}).get("model_dir", "models/saved_models"),
            "dataset_dir": s.get("paths", {}).get("dataset_dir", "data"),
            "log_dir": s.get("paths", {}).get("log_dir", "logs"),
            "export_dir": s.get("paths", {}).get("export_dir", "outputs"),
            "workers": s.get("performance", {}).get("workers", 4),
            "max_batch": s.get("performance", {}).get("max_batch", 64),
            "amp_enabled": s.get("performance", {}).get("amp_enabled", True),
            "pin_memory": s.get("performance", {}).get("pin_memory", True),
            "default_epochs": s.get("training", {}).get("default_epochs", 50),
            "patience": s.get("training", {}).get("patience", 10),
            "checkpoint_freq": s.get("training", {}).get("checkpoint_freq", 5),
            "save_best_only": s.get("training", {}).get("save_best_only", True),
            "log_level": s.get("logging", {}).get("level", "INFO"),
            "log_days": s.get("logging", {}).get("retention_days", 30),
            "verbose": s.get("logging", {}).get("verbose", False),
            "auto_load_model": True,
            "remember_window": True,
            "notify_on_complete": True,
        }

    def from_gui_settings(self, gui_settings: dict) -> None:
        """从 GUI 设置页面格式更新设置"""
        if "theme" in gui_settings:
            self.set("appearance.theme", self._display_to_theme(gui_settings["theme"]))
        if "scale" in gui_settings:
            self.set("appearance.scale", gui_settings["scale"])
        if "device" in gui_settings:
            self.set("model.device", self._display_to_device(gui_settings["device"]))
        if "precision" in gui_settings:
            self.set("model.precision", self._display_to_precision(gui_settings["precision"]))
        if "confidence_threshold" in gui_settings:
            self.set("model.confidence_threshold", gui_settings["confidence_threshold"])
        if "model_dir" in gui_settings:
            self.set("paths.model_dir", gui_settings["model_dir"])
        if "dataset_dir" in gui_settings:
            self.set("paths.dataset_dir", gui_settings["dataset_dir"])
        if "log_dir" in gui_settings:
            self.set("paths.log_dir", gui_settings["log_dir"])
        if "export_dir" in gui_settings:
            self.set("paths.export_dir", gui_settings["export_dir"])
        if "workers" in gui_settings:
            self.set("performance.workers", gui_settings["workers"])
        if "max_batch" in gui_settings:
            self.set("performance.max_batch", gui_settings["max_batch"])
        if "amp_enabled" in gui_settings:
            self.set("performance.amp_enabled", gui_settings["amp_enabled"])
        if "pin_memory" in gui_settings:
            self.set("performance.pin_memory", gui_settings["pin_memory"])
        if "default_epochs" in gui_settings:
            self.set("training.default_epochs", gui_settings["default_epochs"])
        if "patience" in gui_settings:
            self.set("training.patience", gui_settings["patience"])
        if "checkpoint_freq" in gui_settings:
            self.set("training.checkpoint_freq", gui_settings["checkpoint_freq"])
        if "save_best_only" in gui_settings:
            self.set("training.save_best_only", gui_settings["save_best_only"])
        if "log_level" in gui_settings:
            self.set("logging.level", gui_settings["log_level"])
        if "log_days" in gui_settings:
            self.set("logging.retention_days", gui_settings["log_days"])
        if "verbose" in gui_settings:
            self.set("logging.verbose", gui_settings["verbose"])

    # 显示值转换辅助方法
    @staticmethod
    def _theme_to_display(theme: str) -> str:
        return {"dark": "深色模式", "light": "浅色模式", "system": "跟随系统"}.get(theme, "深色模式")

    @staticmethod
    def _display_to_theme(display: str) -> str:
        return {"深色模式": "dark", "浅色模式": "light", "跟随系统": "system"}.get(display, "dark")

    @staticmethod
    def _device_to_display(device: str) -> str:
        return {"auto": "自动检测", "cpu": "CPU", "cuda": "GPU (CUDA)"}.get(device, "自动检测")

    @staticmethod
    def _display_to_device(display: str) -> str:
        return {"自动检测": "auto", "CPU": "cpu", "GPU (CUDA)": "cuda"}.get(display, "auto")

    @staticmethod
    def _precision_to_display(precision: str) -> str:
        return {"fp32": "FP32 (默认)", "fp16": "FP16 (加速)"}.get(precision, "FP32 (默认)")

    @staticmethod
    def _display_to_precision(display: str) -> str:
        return {"FP32 (默认)": "fp32", "FP16 (加速)": "fp16"}.get(display, "fp32")
