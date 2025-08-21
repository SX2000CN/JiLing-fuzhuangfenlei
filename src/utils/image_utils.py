"""
图像处理工具模块
提供图像预处理、数据增强和格式转换功能
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, List, Optional, Union
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """图像处理器类"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        初始化图像处理器
        
        Args:
            target_size: 目标图像尺寸 (width, height)
        """
        self.target_size = target_size
        
    def load_image(self, image_path: Union[str, Path]) -> Optional[Image.Image]:
        """
        加载图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            PIL Image对象，加载失败返回None
        """
        try:
            image = Image.open(image_path)
            
            # 转换为RGB模式（如果不是的话）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.debug(f"成功加载图像: {image_path}, 尺寸: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"加载图像失败 {image_path}: {e}")
            return None
    
    def save_image(self, image: Image.Image, save_path: Union[str, Path], 
                   quality: int = 95) -> bool:
        """
        保存图像文件
        
        Args:
            image: PIL Image对象
            save_path: 保存路径
            quality: JPEG质量 (1-100)
            
        Returns:
            保存是否成功
        """
        try:
            # 确保保存目录存在
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 根据文件扩展名保存
            save_path = Path(save_path)
            if save_path.suffix.lower() in ['.jpg', '.jpeg']:
                image.save(save_path, 'JPEG', quality=quality)
            else:
                image.save(save_path)
            
            logger.debug(f"成功保存图像: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存图像失败 {save_path}: {e}")
            return False
    
    def resize_image(self, image: Image.Image, size: Optional[Tuple[int, int]] = None,
                     maintain_aspect_ratio: bool = True) -> Image.Image:
        """
        调整图像尺寸
        
        Args:
            image: 输入图像
            size: 目标尺寸，如果为None则使用self.target_size
            maintain_aspect_ratio: 是否保持宽高比
            
        Returns:
            调整尺寸后的图像
        """
        if size is None:
            size = self.target_size
        
        if maintain_aspect_ratio:
            # 保持宽高比的缩放
            image.thumbnail(size, Image.Resampling.LANCZOS)
            
            # 创建新的图像并居中粘贴
            new_image = Image.new('RGB', size, color=(255, 255, 255))
            paste_x = (size[0] - image.size[0]) // 2
            paste_y = (size[1] - image.size[1]) // 2
            new_image.paste(image, (paste_x, paste_y))
            
            return new_image
        else:
            # 直接缩放到目标尺寸
            return image.resize(size, Image.Resampling.LANCZOS)
    
    def normalize_image(self, image: Image.Image) -> np.ndarray:
        """
        图像归一化
        
        Args:
            image: 输入图像
            
        Returns:
            归一化后的numpy数组 (0-1范围)
        """
        image_array = np.array(image, dtype=np.float32)
        return image_array / 255.0
    
    def denormalize_image(self, image_array: np.ndarray) -> Image.Image:
        """
        反归一化
        
        Args:
            image_array: 归一化的图像数组
            
        Returns:
            PIL Image对象
        """
        image_array = (image_array * 255).astype(np.uint8)
        return Image.fromarray(image_array)


class ImageAugmentor:
    """图像数据增强器"""
    
    def __init__(self):
        self.augmentations = []
    
    def add_horizontal_flip(self, probability: float = 0.5):
        """添加水平翻转"""
        self.augmentations.append(('horizontal_flip', probability))
        return self
    
    def add_vertical_flip(self, probability: float = 0.1):
        """添加垂直翻转"""
        self.augmentations.append(('vertical_flip', probability))
        return self
    
    def add_rotation(self, max_angle: float = 15, probability: float = 0.5):
        """添加旋转"""
        self.augmentations.append(('rotation', max_angle, probability))
        return self
    
    def add_brightness(self, factor_range: Tuple[float, float] = (0.8, 1.2), 
                      probability: float = 0.5):
        """添加亮度调整"""
        self.augmentations.append(('brightness', factor_range, probability))
        return self
    
    def add_contrast(self, factor_range: Tuple[float, float] = (0.8, 1.2),
                    probability: float = 0.5):
        """添加对比度调整"""
        self.augmentations.append(('contrast', factor_range, probability))
        return self
    
    def add_saturation(self, factor_range: Tuple[float, float] = (0.8, 1.2),
                      probability: float = 0.5):
        """添加饱和度调整"""
        self.augmentations.append(('saturation', factor_range, probability))
        return self
    
    def add_blur(self, radius_range: Tuple[float, float] = (0.1, 2.0),
                probability: float = 0.1):
        """添加模糊"""
        self.augmentations.append(('blur', radius_range, probability))
        return self
    
    def apply_augmentation(self, image: Image.Image) -> Image.Image:
        """
        应用数据增强
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        augmented_image = image.copy()
        
        for aug in self.augmentations:
            if np.random.random() < aug[-1]:  # 概率判断
                augmented_image = self._apply_single_augmentation(augmented_image, aug)
        
        return augmented_image
    
    def _apply_single_augmentation(self, image: Image.Image, aug_config: tuple) -> Image.Image:
        """应用单个增强操作"""
        aug_type = aug_config[0]
        
        if aug_type == 'horizontal_flip':
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        
        elif aug_type == 'vertical_flip':
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        
        elif aug_type == 'rotation':
            max_angle = aug_config[1]
            angle = np.random.uniform(-max_angle, max_angle)
            return image.rotate(angle, fillcolor=(255, 255, 255))
        
        elif aug_type == 'brightness':
            factor_range = aug_config[1]
            factor = np.random.uniform(factor_range[0], factor_range[1])
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
        
        elif aug_type == 'contrast':
            factor_range = aug_config[1]
            factor = np.random.uniform(factor_range[0], factor_range[1])
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        
        elif aug_type == 'saturation':
            factor_range = aug_config[1]
            factor = np.random.uniform(factor_range[0], factor_range[1])
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(factor)
        
        elif aug_type == 'blur':
            radius_range = aug_config[1]
            radius = np.random.uniform(radius_range[0], radius_range[1])
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        return image


def get_image_files(directory: Union[str, Path], 
                   extensions: List[str] = None) -> List[Path]:
    """
    获取目录中的所有图像文件
    
    Args:
        directory: 目录路径
        extensions: 支持的文件扩展名列表
        
    Returns:
        图像文件路径列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    directory = Path(directory)
    image_files = []
    
    if not directory.exists():
        logger.warning(f"目录不存在: {directory}")
        return image_files
    
    for ext in extensions:
        # 支持大小写不敏感
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    logger.info(f"在 {directory} 中找到 {len(image_files)} 个图像文件")
    return sorted(image_files)


def create_image_thumbnail(image_path: Union[str, Path], 
                          thumbnail_size: Tuple[int, int] = (128, 128)) -> Optional[Image.Image]:
    """
    创建图像缩略图
    
    Args:
        image_path: 图像文件路径
        thumbnail_size: 缩略图尺寸
        
    Returns:
        缩略图Image对象，失败返回None
    """
    try:
        with Image.open(image_path) as image:
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 创建缩略图
            image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
            
            # 创建正方形缩略图
            thumbnail = Image.new('RGB', thumbnail_size, color=(255, 255, 255))
            paste_x = (thumbnail_size[0] - image.size[0]) // 2
            paste_y = (thumbnail_size[1] - image.size[1]) // 2
            thumbnail.paste(image, (paste_x, paste_y))
            
            return thumbnail
            
    except Exception as e:
        logger.error(f"创建缩略图失败 {image_path}: {e}")
        return None


def validate_image(image_path: Union[str, Path]) -> bool:
    """
    验证图像文件是否有效
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        图像是否有效
    """
    try:
        with Image.open(image_path) as image:
            image.verify()  # 验证图像完整性
        return True
    except Exception:
        return False
