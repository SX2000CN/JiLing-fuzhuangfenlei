"""
FastAPI Backend for JiLing Clothing Classification System
提供RESTful API接口包装现有的Python分类和训练功能
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import torch
import tkinter as tk
from tkinter import filedialog
import threading

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入现有的核心模块
try:
    from src.core.pytorch_classifier import ClothingClassifier
    from src.core.pytorch_trainer import ClothingTrainer
    from src.utils.config_manager import ConfigManager
    from src.utils.logging_utils import setup_logging
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 创建FastAPI应用
app = FastAPI(
    title="JiLing Clothing Classification API",
    description="服装分类系统API接口",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite默认端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
classifier: Optional[ClothingClassifier] = None
trainer: Optional[ClothingTrainer] = None
config_manager = ConfigManager()
logger = setup_logging("INFO")
training_status = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": 0.0,
    "best_accuracy": 0.0,
    "start_time": None,
    "logs": []
}

# Pydantic模型定义
class ClassificationRequest(BaseModel):
    image_path: str
    model_name: Optional[str] = None

class TrainingConfig(BaseModel):
    data_path: str
    model_name: str
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.2

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None

class SystemInfo(BaseModel):
    current_model: Optional[str] = None
    gpu_available: bool = False
    models_count: int = 0
    python_version: str = ""
    pytorch_version: str = ""

# 文件对话框辅助函数
def open_file_dialog():
    """打开文件选择对话框"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    file_path = filedialog.askopenfilename(
        title="选择图片文件",
        filetypes=[
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("所有文件", "*.*")
        ]
    )
    root.destroy()
    return file_path

def open_folder_dialog():
    """打开文件夹选择对话框"""
    root = tk.Tk()
    root.withdraw()
    
    folder_path = filedialog.askdirectory(title="选择文件夹")
    root.destroy()
    return folder_path

# API路由
@app.get("/", response_class=FileResponse)
async def root():
    """服务根路径，返回前端页面"""
    return FileResponse("web-frontend/dist/index.html")

@app.get("/api/status")
async def get_api_status():
    """获取API状态"""
    return ApiResponse(
        success=True,
        message="API运行正常",
        data={"status": "running", "version": "1.0.0"}
    )

@app.get("/api/system/status")
async def get_system_status():
    """获取系统状态信息"""
    global classifier
    
    # 获取可用模型列表
    models_dir = project_root / "models"
    available_models = []
    if models_dir.exists():
        for model_file in models_dir.glob("*.pth"):
            available_models.append(model_file.stem)
    
    system_info = SystemInfo(
        current_model=classifier.model_name if classifier else None,
        gpu_available=torch.cuda.is_available(),
        models_count=len(available_models),
        python_version=sys.version.split()[0],
        pytorch_version=torch.__version__
    )
    
    return ApiResponse(
        success=True,
        message="系统状态获取成功",
        data=system_info.dict()
    )

@app.get("/api/models")
async def get_models():
    """获取可用模型列表"""
    models_dir = project_root / "models"
    models = []
    
    if models_dir.exists():
        for model_file in models_dir.glob("*.pth"):
            model_info = {
                "name": model_file.stem,
                "path": str(model_file),
                "size": model_file.stat().st_size if model_file.exists() else 0,
                "modified": model_file.stat().st_mtime if model_file.exists() else 0
            }
            models.append(model_info)
    
    return ApiResponse(
        success=True,
        message=f"找到 {len(models)} 个模型",
        data=models
    )

@app.post("/api/load_model")
async def load_model(request: dict):
    """加载指定模型"""
    global classifier
    
    model_name = request.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="未指定模型名称")
    
    try:
        # 查找模型文件
        models_dir = project_root / "models"
        model_path = models_dir / f"{model_name}.pth"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"模型文件不存在: {model_name}")
        
        # 创建分类器并加载模型
        classifier = ClothingClassifier()
        classifier.load_model(str(model_path))
        classifier.model_name = model_name
        
        logger.info(f"模型加载成功: {model_name}")
        
        return ApiResponse(
            success=True,
            message=f"模型 {model_name} 加载成功",
            data={"model_name": model_name, "model_path": str(model_path)}
        )
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

@app.post("/api/classify")
async def classify_image(request: ClassificationRequest):
    """对单张图片进行分类"""
    global classifier
    
    if not classifier:
        raise HTTPException(status_code=400, detail="请先加载模型")
    
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="图片文件不存在")
    
    try:
        result = classifier.predict(request.image_path)
        
        return ApiResponse(
            success=True,
            message="分类完成",
            data=result
        )
        
    except Exception as e:
        logger.error(f"图片分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")

@app.post("/api/classify/batch")
async def batch_classify(request: dict):
    """批量分类文件夹中的图片"""
    global classifier
    
    if not classifier:
        raise HTTPException(status_code=400, detail="请先加载模型")
    
    folder_path = request.get("folder_path")
    if not folder_path or not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="文件夹不存在")
    
    try:
        results = classifier.predict_folder(folder_path)
        
        return ApiResponse(
            success=True,
            message=f"批量分类完成，共处理 {len(results)} 张图片",
            data=results
        )
        
    except Exception as e:
        logger.error(f"批量分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量分类失败: {str(e)}")

@app.post("/api/train/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """开始训练模型"""
    global trainer, training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="已有训练任务在进行中")
    
    if not os.path.exists(config.data_path):
        raise HTTPException(status_code=404, detail="训练数据路径不存在")
    
    try:
        # 创建训练器
        trainer = ClothingTrainer()
        
        # 重置训练状态
        training_status.update({
            "is_training": True,
            "current_epoch": 0,
            "total_epochs": config.epochs,
            "current_loss": 0.0,
            "best_accuracy": 0.0,
            "start_time": None,
            "logs": []
        })
        
        # 在后台启动训练
        background_tasks.add_task(run_training, config)
        
        return ApiResponse(
            success=True,
            message="训练任务已启动",
            data={"status": "started", "config": config.dict()}
        )
        
    except Exception as e:
        training_status["is_training"] = False
        logger.error(f"启动训练失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")

async def run_training(config: TrainingConfig):
    """运行训练任务"""
    global trainer, training_status
    
    try:
        # 这里需要实现训练逻辑
        # 由于原有的训练器可能是同步的，这里需要适配
        import time
        training_status["start_time"] = time.time()
        
        # 模拟训练过程 - 实际应该调用trainer的方法
        for epoch in range(config.epochs):
            training_status["current_epoch"] = epoch + 1
            training_status["current_loss"] = 0.5 - epoch * 0.01  # 模拟损失下降
            training_status["logs"].append(f"Epoch {epoch + 1}/{config.epochs} completed")
            
            # 每个epoch暂停1秒（实际训练会更久）
            await asyncio.sleep(1)
            
            if not training_status["is_training"]:  # 检查是否被停止
                break
        
        training_status["is_training"] = False
        training_status["logs"].append("训练完成")
        
    except Exception as e:
        training_status["is_training"] = False
        training_status["logs"].append(f"训练失败: {str(e)}")
        logger.error(f"训练过程中出错: {e}")

@app.post("/api/train/stop")
async def stop_training():
    """停止训练"""
    global training_status
    
    if not training_status["is_training"]:
        raise HTTPException(status_code=400, detail="当前没有训练任务")
    
    training_status["is_training"] = False
    training_status["logs"].append("训练已停止")
    
    return ApiResponse(
        success=True,
        message="训练已停止",
        data=training_status
    )

@app.get("/api/train/status")
async def get_training_status():
    """获取训练状态"""
    return ApiResponse(
        success=True,
        message="训练状态获取成功",
        data=training_status
    )

@app.post("/api/file/select-file")
async def select_file():
    """打开文件选择对话框"""
    try:
        # 在新线程中运行对话框，避免阻塞
        def run_dialog():
            return open_file_dialog()
        
        # 使用线程池执行文件对话框
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_dialog)
            file_path = future.result(timeout=30)  # 30秒超时
        
        if file_path:
            return ApiResponse(
                success=True,
                message="文件选择成功",
                data={"path": file_path}
            )
        else:
            return ApiResponse(
                success=False,
                message="用户取消选择",
                data=None
            )
            
    except Exception as e:
        logger.error(f"文件选择失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件选择失败: {str(e)}")

@app.post("/api/file/select-folder")
async def select_folder():
    """打开文件夹选择对话框"""
    try:
        def run_dialog():
            return open_folder_dialog()
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_dialog)
            folder_path = future.result(timeout=30)
        
        if folder_path:
            return ApiResponse(
                success=True,
                message="文件夹选择成功",
                data={"path": folder_path}
            )
        else:
            return ApiResponse(
                success=False,
                message="用户取消选择",
                data=None
            )
            
    except Exception as e:
        logger.error(f"文件夹选择失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件夹选择失败: {str(e)}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件"""
    try:
        # 创建上传目录
        upload_dir = project_root / "uploads"
        upload_dir.mkdir(exist_ok=True)
        
        # 保存文件
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return ApiResponse(
            success=True,
            message="文件上传成功",
            data={"path": str(file_path), "filename": file.filename}
        )
        
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

# 静态文件服务
app.mount("/", StaticFiles(directory="web-frontend/dist", html=True), name="static")

if __name__ == "__main__":
    # 自动加载默认模型
    try:
        models_dir = project_root / "models"
        if models_dir.exists():
            pth_files = list(models_dir.glob("*.pth"))
            if pth_files:
                # 加载第一个找到的模型
                default_model = pth_files[0]
                classifier = ClothingClassifier()
                classifier.load_model(str(default_model))
                classifier.model_name = default_model.stem
                logger.info(f"自动加载模型: {default_model.stem}")
    except Exception as e:
        logger.warning(f"自动加载模型失败: {e}")
    
    # 启动服务器
    print("启动JiLing服装分类系统API服务器...")
    print("前端地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境建议设为False
        log_level="info"
    )
