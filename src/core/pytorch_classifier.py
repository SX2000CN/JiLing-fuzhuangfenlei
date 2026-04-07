"""
PyTorch分类器模块
负责图像分类的核心功能
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import shutil
import json
from typing import List, Dict, Tuple, Union, Optional
import logging
from tqdm import tqdm
import time

from .model_factory import ModelFactory
from src.utils.config_manager import config_manager

logger = logging.getLogger(__name__)


class ClothingClassifier:
    """服装图片分类器"""
    
    def __init__(self,
                 model_path: str,
                 device: str = 'auto',
                 model_name: Optional[str] = None,
                 input_size: Optional[int] = None):
        """
        初始化分类器
        
        Args:
            model_path: 模型文件路径
            device: 计算设备 ('auto', 'cuda', 'cpu')
            model_name: 模型名称（默认读取 config/model_config.yaml）
            input_size: 输入尺寸（默认读取 config/model_config.yaml）
        """
        model_settings = config_manager.get_model_settings()

        resolved_model_name = model_name or model_settings["name"]
        resolved_input_size = int(input_size or model_settings["image_size"])

        if resolved_input_size <= 0:
            raise ValueError(f"输入尺寸必须大于0，当前值: {resolved_input_size}")

        self.device = self._setup_device(device)
        self.model_name = ModelFactory.normalize_model_name(resolved_model_name)
        self.input_size = resolved_input_size
        self.model_path = model_path
        self.classes = model_settings.get("classes", ['主图', '细节', '吊牌'])
        self.num_classes = len(self.classes)
        
        # 加载模型
        self.model = self._load_model()
        
        # 设置图像预处理
        self.transform = self._get_transform()
        
        logger.info(f"分类器初始化完成:")
        logger.info(f"  - 设备: {self.device}")
        logger.info(f"  - 模型: {self.model_name}")
        logger.info(f"  - 类别: {self.classes}")
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                device_name = torch.device('cuda')
                logger.info(f"自动选择GPU: {torch.cuda.get_device_name(0)}")
            else:
                device_name = torch.device('cpu')
                logger.info("GPU不可用，使用CPU")
        else:
            device_name = torch.device(device)
            logger.info(f"使用指定设备: {device}")
        
        return device_name
    
    def _load_model(self) -> nn.Module:
        """加载训练好的模型"""
        try:
            # 创建模型结构
            model = ModelFactory.create_model(
                self.model_name, 
                num_classes=self.num_classes, 
                pretrained=False
            )
            
            # 加载权重
            if Path(self.model_path).exists():
                logger.info(f"加载模型权重: {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # 处理不同的保存格式
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"加载训练信息: epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
            else:
                logger.warning(f"模型文件不存在: {self.model_path}")
                logger.warning("使用预训练权重，未加载自定义权重")
            
            # 移动到指定设备并设置为评估模式
            model.to(self.device)
            model.eval()
            
            logger.info("模型加载完成")
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _get_transform(self) -> transforms.Compose:
        """获取图像预处理管道"""
        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet标准化
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"图像预处理: 输入尺寸 {self.input_size}x{self.input_size}")
        return transform
    
    def predict_single(self, image_path: Union[str, Path]) -> Tuple[str, float, Dict]:
        """
        分类单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            Tuple[预测类别, 置信度, 详细结果]
        """
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = self.classes[predicted.item()]
                confidence_score = confidence.item()
                
                # 所有类别的概率
                all_probs = probabilities.cpu().numpy()[0]
                class_probs = {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.classes, all_probs)
                }
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'all_probabilities': class_probs,
                'image_path': str(image_path)
            }
            
            return predicted_class, confidence_score, result
            
        except Exception as e:
            logger.error(f"分类失败 {image_path}: {e}")
            raise
    
    def classify_folder(self, 
                       input_folder: Union[str, Path], 
                       output_folder: Union[str, Path],
                       confidence_threshold: float = 0.5,
                       move_files: bool = True,
                       save_results: bool = True) -> Dict:
        """
        批量分类文件夹中的图片
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            confidence_threshold: 置信度阈值
            move_files: 是否移动文件（True移动，False复制）
            save_results: 是否保存分类结果
            
        Returns:
            分类统计结果
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
        
        # 创建输出文件夹
        for class_name in self.classes:
            (output_path / class_name).mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in input_path.iterdir() 
            if f.suffix.lower() in image_extensions and f.is_file()
        ]
        
        if not image_files:
            logger.warning(f"在 {input_folder} 中未找到图片文件")
            return {'total': 0, 'processed': 0, 'failed': 0}
        
        logger.info(f"开始批量分类: {len(image_files)} 张图片")
        
        # 分类统计
        stats = {
            'total': len(image_files),
            'processed': 0,
            'failed': 0,
            'by_class': {class_name: 0 for class_name in self.classes},
            'low_confidence': 0,
            'results': [],
            'start_time': time.time()
        }
        
        # 批量处理
        for image_file in tqdm(image_files, desc="分类中"):
            try:
                # 分类
                predicted_class, confidence, result = self.predict_single(image_file)
                
                # 检查置信度
                if confidence >= confidence_threshold:
                    # 移动/复制文件
                    dest_folder = output_path / predicted_class
                    dest_file = dest_folder / image_file.name
                    
                    if move_files:
                        shutil.move(str(image_file), str(dest_file))
                    else:
                        shutil.copy2(str(image_file), str(dest_file))
                    
                    stats['by_class'][predicted_class] += 1
                    stats['processed'] += 1
                    
                    logger.debug(f"✅ {image_file.name} → {predicted_class} ({confidence:.2%})")
                else:
                    stats['low_confidence'] += 1
                    logger.warning(f"⚠️ {image_file.name}: 置信度过低 ({confidence:.2%})")
                
                # 保存结果
                if save_results:
                    stats['results'].append(result)
                
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"❌ 处理失败 {image_file.name}: {e}")
        
        # 计算耗时
        stats['processing_time'] = time.time() - stats['start_time']
        stats['speed'] = stats['total'] / stats['processing_time']  # 张/秒
        
        # 保存统计结果
        if save_results:
            self._save_classification_report(output_path, stats)
        
        self._print_summary(stats)
        return stats
    
    def batch_predict(self, image_paths: List[Union[str, Path]], batch_size: int = 32) -> List[Dict]:
        """
        批量预测（不移动文件）
        
        Args:
            image_paths: 图片路径列表
            batch_size: 批次大小
            
        Returns:
            预测结果列表
        """
        results = []
        
        logger.info(f"批量预测: {len(image_paths)} 张图片")
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="预测中"):
            batch_paths = image_paths[i:i + batch_size]
            
            for image_path in batch_paths:
                try:
                    predicted_class, confidence, result = self.predict_single(image_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"预测失败 {image_path}: {e}")
                    results.append({
                        'predicted_class': 'unknown',
                        'confidence': 0.0,
                        'error': str(e),
                        'image_path': str(image_path)
                    })
        
        return results
    
    def _save_classification_report(self, output_folder: Path, stats: Dict):
        """保存分类报告"""
        report_file = output_folder / f"classification_report_{int(time.time())}.json"
        
        # 简化stats用于JSON序列化
        simple_stats = {
            'summary': {
                'total_images': stats['total'],
                'processed': stats['processed'],
                'failed': stats['failed'],
                'low_confidence': stats['low_confidence'],
                'processing_time': f"{stats['processing_time']:.2f}s",
                'speed': f"{stats['speed']:.1f} images/sec"
            },
            'by_class': stats['by_class'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {
                'model_name': self.model_name,
                'model_path': self.model_path,
                'device': str(self.device)
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(simple_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分类报告已保存: {report_file}")
    
    def _print_summary(self, stats: Dict):
        """打印分类摘要"""
        print("\n" + "="*50)
        print("📊 分类结果摘要")
        print("="*50)
        print(f"总图片数量: {stats['total']}")
        print(f"成功处理: {stats['processed']}")
        print(f"处理失败: {stats['failed']}")
        print(f"置信度过低: {stats['low_confidence']}")
        print(f"处理时间: {stats['processing_time']:.2f}秒")
        print(f"处理速度: {stats['speed']:.1f}张/秒")
        print("\n📋 各类别统计:")
        for class_name, count in stats['by_class'].items():
            percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {class_name}: {count}张 ({percentage:.1f}%)")
        print("="*50)


if __name__ == "__main__":
    # 测试代码
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("🧪 测试分类器...")
    
    # 注意：这需要一个实际的模型文件才能完全测试
    try:
        # 创建一个虚拟模型用于测试（仅测试结构）
        model_path = "test_model.pth"
        
        # 保存一个简单的模型用于测试
        test_model = ModelFactory.create_model('tf_efficientnetv2_s', num_classes=3, pretrained=False)
        torch.save(test_model.state_dict(), model_path)
        
        # 测试分类器初始化
        classifier = ClothingClassifier(model_path, device='cpu')
        print("✅ 分类器初始化成功")
        
        # 清理测试文件
        Path(model_path).unlink()
        print("✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
