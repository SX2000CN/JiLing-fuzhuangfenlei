"""
模型工厂模块
负责创建和管理不同类型的预训练模型
"""
import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """模型工厂类"""
    
    # 支持的模型配置
    SUPPORTED_MODELS = {
        'tf_efficientnetv2_s': {
            'timm_name': 'tf_efficientnetv2_s_in21ft1k',
            'description': 'EfficientNetV2-Small (TF版本)',
            'input_size': 384,
            'params': '22M',
            'recommended_batch_size': 32
        },
        'convnext_tiny': {
            'timm_name': 'convnext_tiny_in22ft1k',
            'description': 'ConvNeXt-Tiny，现代化CNN架构',
            'input_size': 224,
            'params': '29M',
            'recommended_batch_size': 32
        },
        'resnet50': {
            'timm_name': 'resnet50',
            'description': 'ResNet-50，经典CNN架构',
            'input_size': 224,
            'params': '26M',
            'recommended_batch_size': 32
        },
        'vit_base_patch16_224': {
            'timm_name': 'vit_base_patch16_224',
            'description': 'Vision Transformer Base',
            'input_size': 224,
            'params': '86M',
            'recommended_batch_size': 16
        },
        'swin_tiny_patch4_window7_224': {
            'timm_name': 'swin_tiny_patch4_window7_224',
            'description': 'Swin Transformer Tiny',
            'input_size': 224,
            'params': '28M',
            'recommended_batch_size': 32
        }
    }
    
    @classmethod
    def create_model(cls, model_name: str, num_classes: int = 3, 
                    pretrained: bool = True, **kwargs) -> nn.Module:
        """
        创建模型
        
        Args:
            model_name: 模型名称
            num_classes: 分类数量
            pretrained: 是否使用预训练权重
            **kwargs: 其他参数
            
        Returns:
            PyTorch模型
        """
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 获取timm模型名称
        timm_name = cls.SUPPORTED_MODELS[model_name]['timm_name']
        
        try:
            # 创建模型
            model = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=num_classes
            )
            
            logger.info(f"成功创建模型: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"创建模型失败: {model_name}, 错误: {str(e)}")
            raise
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """
        获取支持的模型名称列表
        
        Returns:
            模型名称列表
        """
        return list(cls.SUPPORTED_MODELS.keys())
    
    @classmethod
    def get_all_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        获取所有可用模型信息
        
        Returns:
            模型信息字典
        """
        return cls.SUPPORTED_MODELS.copy()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息字典，如果模型不存在返回None
        """
        return cls.SUPPORTED_MODELS.get(model_name)
