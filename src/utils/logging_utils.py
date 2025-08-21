"""
日志工具模块
提供统一的日志配置和管理功能
"""
import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 日志文件目录
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的日志文件数量
        
    Returns:
        配置好的logger实例
    """
    
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger("JiLing-fuzhuangfenlei")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有handlers
    logger.handlers.clear()
    
    # 创建formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if file_output:
        # 主日志文件
        main_log_file = log_path / "application.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 错误日志文件
        error_log_file = log_path / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取logger实例
    
    Args:
        name: logger名称，如果为None则使用根logger
        
    Returns:
        logger实例
    """
    if name is None:
        return logging.getLogger("JiLing-fuzhuangfenlei")
    else:
        return logging.getLogger(f"JiLing-fuzhuangfenlei.{name}")


def log_performance(func):
    """
    性能日志装饰器
    
    Usage:
        @log_performance
        def my_function():
            pass
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("performance")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"{func.__name__} 执行完成，耗时: {execution_time:.4f}秒")
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.error(f"{func.__name__} 执行失败，耗时: {execution_time:.4f}秒，错误: {str(e)}")
            raise
    
    return wrapper


def log_function_call(func):
    """
    函数调用日志装饰器
    
    Usage:
        @log_function_call
        def my_function(arg1, arg2):
            pass
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("function_calls")
        
        # 记录函数调用
        args_str = ', '.join([str(arg) for arg in args])
        kwargs_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        params_str = ', '.join(filter(None, [args_str, kwargs_str]))
        
        logger.debug(f"调用 {func.__name__}({params_str})")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} 返回: {type(result).__name__}")
            return result
            
        except Exception as e:
            logger.error(f"{func.__name__} 抛出异常: {str(e)}")
            raise
    
    return wrapper


class TrainingLogger:
    """训练过程专用logger"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建训练专用logger
        self.logger = logging.getLogger("training")
        self.logger.setLevel(logging.INFO)
        
        # 训练日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_log_file = self.log_dir / f"training_{timestamp}.log"
        
        handler = logging.FileHandler(training_log_file, encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_epoch(self, epoch: int, total_epochs: int, train_loss: float, 
                  train_acc: float, val_loss: float = None, val_acc: float = None):
        """记录epoch信息"""
        msg = f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
        
        if val_loss is not None and val_acc is not None:
            msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        
        self.logger.info(msg)
    
    def log_training_start(self, model_name: str, total_params: int, trainable_params: int):
        """记录训练开始信息"""
        self.logger.info("=" * 50)
        self.logger.info("开始训练")
        self.logger.info(f"模型: {model_name}")
        self.logger.info(f"总参数数量: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}")
        self.logger.info("=" * 50)
    
    def log_training_end(self, best_epoch: int, best_acc: float, total_time: float):
        """记录训练结束信息"""
        self.logger.info("=" * 50)
        self.logger.info("训练完成")
        self.logger.info(f"最佳epoch: {best_epoch}")
        self.logger.info(f"最佳准确率: {best_acc:.4f}")
        self.logger.info(f"总训练时间: {total_time:.2f}秒")
        self.logger.info("=" * 50)


# 初始化默认logger
default_logger = setup_logging()
