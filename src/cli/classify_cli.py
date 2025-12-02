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
import argparse
from pathlib import Path
from typing import List
from PIL import Image
from tqdm import tqdm
import concurrent.futures

# 添加项目路径
project_root = Path(__file__).parent.parent.parent  # src/cli/ -> src/ -> 项目根目录
sys.path.insert(0, str(project_root))

try:
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from src.core.pytorch_classifier import ClothingClassifier
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Please ensure PySide6 and dependencies are installed")
    sys.exit(1)


class CommandLineClassifier:
    """命令行分类器"""
    
    def __init__(self, no_pause=False):
        # 创建QApplication（必需，即使不显示GUI）
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        
        # 初始化设置
        self.settings = QSettings("JiLing", "ClothingClassifier")
        self.classifier = None
        self.no_pause = no_pause  # 是否跳过等待
        
    def load_gui_settings(self):
        """加载GUI中保存的设置"""
        print("[INFO] Loading GUI settings...")
        
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
        print("[INFO] Initializing classifier...")

        # 查找模型文件
        model_path = self.find_latest_model()
        if not model_path:
            print("[ERROR] No available model file found")
            return False

        print(f"  Model: {model_path}")

        try:
            # 创建分类器，使用正确的配置
            self.classifier = ClothingClassifier(
                model_path=model_path,
                model_name='tf_efficientnetv2_s',  # 使用正确的模型名称
                device='auto'
            )
            print("[OK] Classifier initialized successfully")
            return True

        except Exception as e:
            print(f"[ERROR] Classifier initialization failed: {e}")
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
            
            print(f"[BATCH] Processing batch {batch_start//batch_size + 1}, images: {batch_start+1}-{batch_end}")

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

            print(f"[PERF] Using {optimal_workers} threads (optimal config)")
            
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
            print(f"[TIME] Batch preprocess done, {len(batch_tensors)} images, time: {preprocess_time:.3f}s")
            
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
                print(f"[TIME] GPU inference done, {len(batch_tensors)} images, time: {inference_time:.3f}s")
                
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
            print("[ERROR] No image files found")
            return

        total_files = len(image_files)
        print(f"[INFO] Starting classification of {total_files} images...")
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
            
            print(f"[GPU] Optimized mode - batch size {batch_size} (GPU: {gpu_memory_gb:.1f}GB)")
            
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
                    print(f"[WARN] Unknown category: {predicted_class}, skipping {file_name}")
                    continue

                # 显示进度
                progress = (i + 1) / total_files * 100
                print(f"[PROG] [{progress:5.1f}%] {file_name} -> {predicted_class} ({confidence:.2f})")

        except KeyboardInterrupt:
            print("\n[WARN] User interrupted")
            return
        except Exception as e:
            print(f"\n[ERROR] Classification error: {e}")
            return

        # 显示最终统计
        elapsed_time = time.time() - start_time
        speed = total_files / elapsed_time

        print("=" * 60)
        print("[DONE] Classification complete!")
        print(f"[STATS] Classification results:")
        for category, count in stats.items():
            percentage = count / total_files * 100
            print(f"  {category}: {count} images ({percentage:.1f}%)")

        print(f"[TIME] Total time: {elapsed_time:.2f}s")
        print(f"[PERF] Average speed: {speed:.2f} images/s")
        print("=" * 60)
    
    def run(self):
        """运行分类"""
        print("[START] JiLing Clothing Classification System - CLI")
        print("=" * 60)

        # 加载设置
        if not self.load_gui_settings():
            print("[ERROR] GUI settings not found")
            print("Please use the GUI to set up classification path first")
            self.wait_for_exit()
            return

        # 检查路径
        if not os.path.exists(self.classification_folder):
            print(f"[ERROR] Classification path not found: {self.classification_folder}")
            self.wait_for_exit()
            return

        # 初始化分类器
        if not self.initialize_classifier():
            self.wait_for_exit()
            return

        # 获取图像文件
        image_files = self.get_image_files(self.classification_folder)
        if not image_files:
            print(f"[ERROR] No image files found in: {self.classification_folder}")
            self.wait_for_exit()
            return

        # 开始分类
        self.classify_images(image_files)

        # 等待用户按回车退出
        self.wait_for_exit()

    def wait_for_exit(self):
        """等待用户按回车退出"""
        if self.no_pause:
            print("[EXIT] Goodbye!")
            return

        print("\n[TIP] Press Enter to exit...")
        try:
            input()
        except KeyboardInterrupt:
            pass
        print("[EXIT] Goodbye!")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='JiLing Clothing Classification System - CLI')
    parser.add_argument('--no-pause', action='store_true',
                       help='Do not wait for keypress after completion')
    args = parser.parse_args()

    try:
        classifier = CommandLineClassifier(no_pause=args.no_pause)
        classifier.run()
    except KeyboardInterrupt:
        print("\n[EXIT] User cancelled, goodbye!")
    except Exception as e:
        print(f"\n[ERROR] Program exception: {e}")
        if not args.no_pause:
            print("[TIP] Press Enter to exit...")
            try:
                input()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
