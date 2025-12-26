"""
分类工作线程 - 在后台执行图像分类任务
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Any

from PySide6.QtCore import QObject, Signal

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# 添加项目路径到 sys.path
_src_path = str(PROJECT_ROOT / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# 后端模块延迟导入
torch = None
BACKEND_AVAILABLE = False


def _init_backend():
    """初始化后端模块"""
    global torch, BACKEND_AVAILABLE
    if BACKEND_AVAILABLE:
        return True
    try:
        import torch as _torch
        torch = _torch
        BACKEND_AVAILABLE = True
        return True
    except ImportError:
        return False


class ClassificationWorker(QObject):
    """
    分类工作线程

    在后台执行图像分类任务，支持批量处理和文件移动。

    Signals:
        progress_updated: (int, str) - 进度百分比、状态消息
        classification_completed: (list) - 分类结果列表
    """
    progress_updated = Signal(int, str)
    classification_completed = Signal(list)

    def __init__(self, image_paths: List[str], classifier: Optional[Any] = None,
                 output_folder: Optional[str] = None):
        super().__init__()
        self.image_paths = [str(p) for p in image_paths]
        self.classifier = classifier
        self.output_folder = output_folder

    def start_classification(self) -> None:
        """开始分类任务"""
        if not _init_backend():
            self.progress_updated.emit(0, "后端模块不可用")
            self.classification_completed.emit([])
            return

        start_time = time.time()

        try:
            from PIL import Image
            import concurrent.futures

            classifier = self.classifier
            if classifier is None:
                raise Exception("未加载分类器")

            total_images = len(self.image_paths)
            results = []

            # 确定输出文件夹
            if self.output_folder:
                output_folder = Path(self.output_folder)
            else:
                output_folder = Path(self.image_paths[0]).parent

            # 创建类别文件夹
            for class_name in classifier.classes:
                (output_folder / class_name).mkdir(parents=True, exist_ok=True)

            # 智能批次大小
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 4

            if gpu_memory_gb >= 10:
                batch_size = 160
            elif gpu_memory_gb >= 6:
                batch_size = 128
            else:
                batch_size = 64

            batch_size = min(batch_size, total_images)

            # 预处理函数
            def preprocess_single_image(image_path: str):
                try:
                    image = Image.open(image_path).convert('RGB')
                    input_tensor = classifier.transform(image)
                    return image_path, input_tensor
                except Exception as e:
                    return image_path, None, str(e)

            # 批量处理
            for batch_start in range(0, total_images, batch_size):
                batch_end = min(batch_start + batch_size, total_images)
                batch_paths = self.image_paths[batch_start:batch_end]

                progress = int(batch_end * 100 / total_images)
                self.progress_updated.emit(progress, f"分类中... {batch_end}/{total_images}")

                # 多线程预处理
                batch_tensors = []
                valid_paths = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                    parallel_results = list(executor.map(preprocess_single_image, batch_paths))

                for result in parallel_results:
                    if len(result) == 2:
                        image_path, tensor = result
                        batch_tensors.append(tensor)
                        valid_paths.append(image_path)
                    else:
                        image_path, _, error = result
                        results.append({
                            'path': image_path,
                            'result': {'predicted_class': 'error', 'confidence': 0.0, 'error': error}
                        })

                if not batch_tensors:
                    continue

                # GPU推理
                try:
                    batch_tensor = torch.stack(batch_tensors).to(classifier.device, non_blocking=True)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            outputs = classifier.model(batch_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            confidences, predicted = torch.max(probabilities, 1)

                    # 处理结果并移动文件
                    for i, (image_path, confidence, predicted_idx) in enumerate(zip(valid_paths, confidences, predicted)):
                        predicted_class = classifier.classes[predicted_idx.item()]
                        confidence_score = confidence.item()

                        # 移动文件
                        dest_folder = output_folder / predicted_class
                        dest_file = dest_folder / Path(image_path).name

                        if not dest_file.exists():
                            os.rename(image_path, dest_file)

                        # 构造结果
                        all_probs = probabilities[i].cpu().numpy()
                        class_probs = {
                            class_name: float(prob)
                            for class_name, prob in zip(classifier.classes, all_probs)
                        }

                        results.append({
                            'path': str(dest_file),
                            'result': {
                                'predicted_class': predicted_class,
                                'confidence': confidence_score,
                                'class_probabilities': class_probs,
                                'image_path': str(dest_file)
                            }
                        })

                except Exception as e:
                    # 降级到单张处理
                    for image_path in valid_paths:
                        try:
                            predicted_class, confidence, result = classifier.predict_single(image_path)
                            dest_folder = output_folder / predicted_class
                            dest_file = dest_folder / Path(image_path).name
                            if not dest_file.exists():
                                os.rename(image_path, dest_file)
                            results.append({'path': str(dest_file), 'result': result})
                        except Exception as e2:
                            results.append({
                                'path': image_path,
                                'result': {'predicted_class': 'error', 'confidence': 0.0, 'error': str(e2)}
                            })

            # 完成
            total_time = time.time() - start_time
            self.progress_updated.emit(100, f"完成! {len(results)}张, {total_time:.1f}秒")
            self.classification_completed.emit(results)

        except Exception as e:
            self.progress_updated.emit(0, f"分类失败: {str(e)}")
            self.classification_completed.emit([])
