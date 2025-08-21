"""
配置管理器
负责加载和管理YAML配置文件
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录路径
        """
        self.config_dir = Path(config_dir)
        self.configs = {}
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_name: str, required: bool = True) -> Optional[Dict[str, Any]]:
        """
        加载指定的配置文件
        
        Args:
            config_name: 配置文件名（不含扩展名）
            required: 是否为必需的配置文件
            
        Returns:
            配置字典，如果文件不存在且不是必需的则返回None
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            if required:
                raise FileNotFoundError(f"必需的配置文件不存在: {config_path}")
            else:
                self.logger.warning(f"可选配置文件不存在: {config_path}")
                return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.configs[config_name] = config
                self.logger.info(f"成功加载配置文件: {config_path}")
                return config
        except yaml.YAMLError as e:
            self.logger.error(f"解析配置文件失败 {config_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"读取配置文件失败 {config_path}: {e}")
            raise
    
    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        获取已加载的配置
        
        Args:
            config_name: 配置文件名
            
        Returns:
            配置字典，如果未加载则返回None
        """
        return self.configs.get(config_name)
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        加载所有配置文件
        
        Returns:
            所有配置的字典
        """
        config_files = [
            "model_config",
            "training_config", 
            "paths_config"
        ]
        
        for config_file in config_files:
            self.load_config(config_file, required=True)
        
        return self.configs
    
    def get_nested_value(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        获取嵌套配置值
        
        Args:
            config_name: 配置文件名
            key_path: 点分隔的键路径，如 "model.name"
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        config = self.get_config(config_name)
        if config is None:
            return default
        
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> None:
        """
        保存配置文件
        
        Args:
            config_name: 配置文件名
            config_data: 配置数据
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            self.configs[config_name] = config_data
            self.logger.info(f"成功保存配置文件: {config_path}")
            
        except Exception as e:
            self.logger.error(f"保存配置文件失败 {config_path}: {e}")
            raise
    
    def create_default_config(self, config_name: str) -> Dict[str, Any]:
        """
        创建默认配置
        
        Args:
            config_name: 配置文件名
            
        Returns:
            默认配置字典
        """
        if config_name == "user_config":
            return {
                "ui": {
                    "theme": "dark",
                    "language": "zh_CN",
                    "window": {
                        "width": 1200,
                        "height": 800,
                        "maximized": False
                    }
                },
                "processing": {
                    "auto_backup": True,
                    "batch_size": 32,
                    "confidence_threshold": 0.8
                },
                "paths": {
                    "last_input_folder": "",
                    "last_output_folder": "",
                    "last_model_path": ""
                }
            }
        
        return {}


# 全局配置管理器实例
config_manager = ConfigManager()
