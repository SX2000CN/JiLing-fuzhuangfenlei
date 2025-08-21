#!/usr/bin/env python3
"""
JiLing服装分类系统 - 命令行版本
直接使用GUI保存的设置进行分类，无需GUI界面
"""

import sys
import os
import json
import time
import torch
from pathlib import Path
from typing import List
from PIL import Image
from tqdm import tqdm
import concurrent.futures

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from src.core.pytorch_classifier import ClothingClassifier
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已安装PySide6和相关依赖")
    sys.exit(1)


class CommandLineClassifier:
    """命令行分类器"""
    
    def __init__(self):
        # 创建QApplication（必需，即使不显示GUI）
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        
        # 初始化设置
        self.settings = QSettings("JiLing", "ClothingClassifier")
        self.classifier = None
        
    def load_gui_settings(self):
        """加载GUI中保存的设置"""
        print("📂 加载GUI设置...")
        
        # 获取记忆的路径
        self.classification_folder = self.settings.value("last_classification_folder", "")
        self.model_folder = self.settings.value("last_model_folder", "models")
        
        print(f"  上次分类路径: {self.classification_folder}")
        print(f"  模型文件夹: {self.model_folder}")
        
        return bool(self.classification_folder and os.path.exists(self.classification_folder))
    
    def find_latest_model(self):
        """查找最新的JiLing_baiditu模型"""
        models_dir = Path("models")
        if not models_dir.exists():
            return None
        
        # 查找JiLing_baiditu模型文件
        jiling_models = list(models_dir.glob("JiLing_baiditu_*.pth"))
        if jiling_models:
            # 按时间戳排序，获取最新的
            latest_model = max(jiling_models, key=lambda x: x.stat().st_mtime)
            return str(latest_model)
        
        # 回退到其他模型
        other_models = list(models_dir.glob("*.pth"))
        if other_models:
            return str(other_models[0])
        
        return None
    
    def initialize_classifier(self):
        """初始化分类器"""
        print("🤖 初始化分类器...")
        
        # 查找模型文件
        model_path = self.find_latest_model()
        if not model_path:
            print("❌ 未找到可用的模型文件")
            return False
        
        print(f"  使用模型: {model_path}")
        
        try:
            # 创建分类器，使用正确的配置
            self.classifier = ClothingClassifier(
                model_path=model_path,
                model_name='tf_efficientnetv2_s',  # 使用正确的模型名称
                device='auto'
            )
            print("✅ 分类器初始化成功")
            return True
            
        except Exception as e:
            print(f"❌ 分类器初始化失败: {e}")
            return False
    
    def get_image_files(self, folder_path: str) -> List[str]:
        """获取文件夹中的图像文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        folder = Path(folder_path)
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                image_files.append(str(file))
        
        return sorted(image_files)
    
    def _batch_predict_optimized(self, image_files: List[str], batch_size: int) -> List[dict]:
        """
        使用与GUI相同的高性能批量推理方法
        """
        results = []
        total_files = len(image_files)
        
        # 分批处理
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_paths = image_files[batch_start:batch_end]
            
            print(f"📦 处理批次 {batch_start//batch_size + 1}, 图片: {batch_start+1}-{batch_end}")
            
            # 批量预处理图像 - 多线程并行优化
            preprocess_start = time.time()
            batch_tensors = []
            valid_paths = []
            
            def preprocess_single_image(image_path):
                try:
                    # 使用PIL + 原生transform - 最佳性能平衡
                    image = Image.open(image_path).convert('RGB')
                    input_tensor = self.classifier.transform(image)
                    return image_path, input_tensor
                except Exception as e:
                    return image_path, None, str(e)
            
            # 20线程 - 经过系统测试验证的最优配置 (29.48张/秒)
            optimal_workers = 20
            
            print(f"🚀 启用{optimal_workers}线程最优配置")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                parallel_results = list(executor.map(preprocess_single_image, batch_paths))
            
            # 收集成功处理的结果
            for result in parallel_results:
                if len(result) == 2:  # 成功
                    image_path, tensor = result
                    batch_tensors.append(tensor)
                    valid_paths.append(image_path)
                else:  # 失败
                    image_path, _, error = result
                    print(f"预处理失败 {image_path}: {error}")
                    results.append({
                        'predicted_class': 'unknown',
                        'confidence': 0.0,
                        'error': error,
                        'image_path': str(image_path)
                    })
            
            preprocess_end = time.time()
            preprocess_time = preprocess_end - preprocess_start
            print(f"⏱️ 批次预处理完成，{len(batch_tensors)}张图片，耗时: {preprocess_time:.3f}秒")
            
            if not batch_tensors:
                continue
            
            # 高效GPU推理 - 专注于速度优化
            inference_start = time.time()
            try:
                batch_tensor = torch.stack(batch_tensors).to(self.classifier.device, non_blocking=True)
                
                # 单轮高效推理，专注于速度而非GPU占用率
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        outputs = self.classifier.model(batch_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidences, predicted = torch.max(probabilities, 1)
                
                inference_end = time.time()
                inference_time = inference_end - inference_start
                print(f"⏱️ GPU推理完成，{len(batch_tensors)}张图片，耗时: {inference_time:.3f}秒")
                
                # 处理批量结果
                for i, (image_path, confidence, predicted_idx) in enumerate(zip(valid_paths, confidences, predicted)):
                    predicted_class = self.classifier.classes[predicted_idx.item()]
                    confidence_score = confidence.item()
                    
                    results.append({
                        'predicted_class': predicted_class,
                        'confidence': confidence_score,
                        'image_path': str(image_path)
                    })
                
                # 清理GPU内存
                del batch_tensor, outputs, probabilities
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"批量推理失败: {e}")
                # 降级到单张处理
                for image_path in valid_paths:
                    try:
                        predicted_class, confidence, result = self.classifier.predict_single(image_path)
                        results.append(result)
                    except Exception as e2:
                        results.append({
                            'predicted_class': 'unknown',
                            'confidence': 0.0,
                            'error': str(e2),
                            'image_path': str(image_path)
                        })
        
        return results
    
    def classify_images(self, image_files: List[str]):
        """分类图像"""
        if not image_files:
            print("❌ 没有找到图像文件")
            return
        
        total_files = len(image_files)
        print(f"📊 开始分类 {total_files} 张图片...")
        print("=" * 60)
        
        # 创建输出目录 - 使用图片所在文件夹的父目录（与GUI版本一致）
        input_folder = Path(self.classification_folder)
        base_dir = input_folder.parent  # 使用父目录作为输出根目录
        output_dirs = {
            '主图': base_dir / '主图',
            '细节': base_dir / '细节', 
            '吊牌': base_dir / '吊牌'
            # 移除unknown文件夹，与GUI版本保持一致
        }
        
        for dir_path in output_dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # 分类统计
        stats = {'主图': 0, '细节': 0, '吊牌': 0}
        start_time = time.time()
        
        try:
            # GPU优化批量处理 - 使用与GUI相同的策略
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory_gb >= 10:  # RTX 3060 12GB
                    base_batch_size = 160  # 增大批次，提升GPU利用率
                elif gpu_memory_gb >= 6:
                    base_batch_size = 128
                else:
                    base_batch_size = 64
                
                if total_files < base_batch_size:
                    batch_size = min(base_batch_size, total_files)
                else:
                    batch_size = base_batch_size
            else:
                batch_size = 32  # CPU模式降级
            
            print(f"⭐ GPU优化模式 - 批次大小 {batch_size} (GPU: {gpu_memory_gb:.1f}GB)")
            
            # 使用优化的批量推理方法（与GUI相同）
            batch_results = self._batch_predict_optimized(image_files, batch_size)
            
            # 处理结果并移动文件
            for i, result in enumerate(batch_results):
                image_path = image_files[i]
                file_name = Path(image_path).name
                predicted_class = result.get('predicted_class', 'unknown')
                confidence = result.get('confidence', 0.0)
                
                # 移动文件
                source = Path(image_path)
                if predicted_class in output_dirs:
                    target = output_dirs[predicted_class] / file_name
                    if source != target:
                        source.rename(target)
                    # 更新统计
                    stats[predicted_class] += 1
                else:
                    # 如果预测类别不在已知类别中，跳过移动但显示警告
                    print(f"⚠️ 未知类别: {predicted_class}，跳过移动 {file_name}")
                    continue
                
                # 显示进度
                progress = (i + 1) / total_files * 100
                print(f"📦 [{progress:5.1f}%] {file_name} → {predicted_class} ({confidence:.2f})")
        
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断分类")
            return
        except Exception as e:
            print(f"\n❌ 分类过程出错: {e}")
            return
        
        # 显示最终统计
        elapsed_time = time.time() - start_time
        speed = total_files / elapsed_time
        
        print("=" * 60)
        print("🎉 分类完成!")
        print(f"📊 分类统计:")
        for category, count in stats.items():
            percentage = count / total_files * 100
            print(f"  {category}: {count} 张 ({percentage:.1f}%)")
        
        print(f"⏱️ 总耗时: {elapsed_time:.2f} 秒")
        print(f"⚡ 平均速度: {speed:.2f} 张/秒")
        print("=" * 60)
    
    def run(self):
        """运行分类"""
        print("🚀 JiLing服装分类系统 - 命令行版本")
        print("=" * 60)
        
        # 加载设置
        if not self.load_gui_settings():
            print("❌ 未找到GUI保存的路径设置")
            print("请先在GUI中选择并使用一次分类功能")
            self.wait_for_exit()
            return
        
        # 检查路径
        if not os.path.exists(self.classification_folder):
            print(f"❌ 分类路径不存在: {self.classification_folder}")
            self.wait_for_exit()
            return
        
        # 初始化分类器
        if not self.initialize_classifier():
            self.wait_for_exit()
            return
        
        # 获取图像文件
        image_files = self.get_image_files(self.classification_folder)
        if not image_files:
            print(f"❌ 在路径中未找到图像文件: {self.classification_folder}")
            self.wait_for_exit()
            return
        
        # 开始分类
        self.classify_images(image_files)
        
        # 等待用户按回车退出
        self.wait_for_exit()
    
    def wait_for_exit(self):
        """等待用户按回车退出"""
        print("\n💡 按回车键退出...")
        try:
            input()
        except KeyboardInterrupt:
            pass
        print("👋 再见!")


def main():
    """主函数"""
    try:
        classifier = CommandLineClassifier()
        classifier.run()
    except KeyboardInterrupt:
        print("\n👋 用户取消，再见!")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        print("💡 按回车键退出...")
        try:
            input()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
