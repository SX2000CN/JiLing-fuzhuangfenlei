# 后台任务模块
from .training import TrainingWorker
from .classification import ClassificationWorker

__all__ = [
    'TrainingWorker',
    'ClassificationWorker',
]
