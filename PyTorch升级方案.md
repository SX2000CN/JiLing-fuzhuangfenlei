# 服装挂拍分类系统 - PyTorch升级方案

## 📋 项目概述

### 当前工作流程
```
Canon相机拍摄 → CR3 RAW + XMP → 批量调色 → JPG导出 → AI自动分类
├── 主图：完整服装展示 (正面挂拍，背景干净)
├── 细节：局部特写 (面料、工艺、设计细节)
└── 吊牌：商品标签 (价格、成分、货号信息)
```

### 数据规模
- **总数据量**: 20,000+ 张照片
- **当前处理**: 146个RAW文件 → 81个JPG文件
- **目标文件夹**: `D:\桌面\筛选\{主图|细节|吊牌}\`

## 🎯 技术方案

### 架构设计：双模式运行
```
服装分类系统
├── 🖥️ 桌面应用程序 (PySide6)
│   ├── 训练模块：模型训练、参数调整、进度监控
│   ├── 分类模块：批量分类、实时预览、结果统计
│   ├── 设置模块：路径配置、模型选择、参数设置
│   └── 模型管理：加载、保存、切换模型
│
└── ⚡ 独立分类脚本 (Fast Mode)
    ├── 从桌面程序读取配置
    ├── 命令行快速执行
    ├── 批量处理优化
    └── 结果日志输出
```

### 技术栈升级
| 组件 | 当前版本 | 升级版本 | 优势 |
|------|---------|----------|------|
| **深度学习框架** | TensorFlow 2.20.0 | PyTorch + timm | GPU支持更稳定 |
| **模型架构** | EfficientNetB0 | EfficientNetV2-S/ConvNeXt-Tiny | 性能提升30%+ |
| **训练尺寸** | 512×512 | 512×512 (保持) | 适合RTX 3060 12G |
| **桌面界面** | PyQt5 | PySide6 | 现代化UI，官方支持 |

## 📁 新项目文件结构

```
JiLing-fuzhuangfenlei/
├── README.md
├── requirements.txt                # PyTorch依赖
├── config/
│   ├── model_config.yaml          # 模型配置
│   ├── training_config.yaml       # 训练配置
│   └── paths_config.yaml          # 路径配置
├── src/
│   ├── main.py                     # 主程序入口
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py          # 主窗口
│   │   ├── training_tab.py         # 训练界面
│   │   ├── classification_tab.py   # 分类界面
│   │   ├── model_manager_tab.py    # 模型管理
│   │   └── settings_tab.py         # 设置界面
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pytorch_trainer.py      # PyTorch训练器
│   │   ├── pytorch_classifier.py   # PyTorch分类器
│   │   ├── data_loader.py          # 数据加载器
│   │   ├── model_factory.py        # 模型工厂
│   │   └── config_manager.py       # 配置管理器
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py          # 图像处理工具
│       ├── logging_utils.py        # 日志工具
│       └── metrics_utils.py        # 评估指标
├── scripts/
│   ├── fast_classify.py            # 独立分类脚本
│   ├── batch_process.py            # 批量处理脚本
│   └── model_converter.py          # 模型转换工具
├── models/
│   ├── saved_models/               # 保存的模型
│   ├── checkpoints/                # 训练检查点
│   └── exports/                    # 导出的模型
├── data/
│   ├── train/                      # 训练数据
│   │   ├── 主图/
│   │   ├── 细节/
│   │   └── 吊牌/
│   ├── val/                        # 验证数据
│   └── test/                       # 测试数据
├── logs/                           # 训练日志
├── outputs/                        # 分类输出
└── docs/                           # 文档
    ├── installation.md             # 安装指南
    ├── usage.md                    # 使用说明
    └── api.md                      # API文档
```

## 🚀 实施步骤

### 第一阶段：环境准备 (1-2小时)

#### 1.1 创建新的Python环境
```bash
# 创建虚拟环境
python -m venv JiLing-fuzhuangfenlei-env

# 激活环境 (Windows)
JiLing-fuzhuangfenlei-env\Scripts\activate

# 激活环境 (Linux/Mac)
source JiLing-fuzhuangfenlei-env/bin/activate
```

#### 1.2 安装PyTorch和依赖
```bash
# 安装PyTorch (CUDA 11.8版本，适配RTX 3060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装timm (预训练模型库)
pip install timm

# 安装其他依赖
pip install PySide6
pip install pillow opencv-python
pip install numpy pandas matplotlib seaborn
pip install pyyaml tqdm
pip install scikit-learn
```

#### 1.3 验证GPU支持
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
```

### 第二阶段：核心模块开发 (1-2天)

#### 2.1 模型工厂 (`src/core/model_factory.py`)
```python
import timm
import torch.nn as nn

class ModelFactory:
    @staticmethod
    def create_model(model_name: str, num_classes: int = 3, pretrained: bool = True):
        """创建预训练模型"""
        if model_name == 'efficientnetv2_s':
            model = timm.create_model('efficientnetv2_s', pretrained=pretrained, num_classes=num_classes)
        elif model_name == 'convnext_tiny':
            model = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=num_classes)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        return model
```

#### 2.2 PyTorch训练器 (`src/core/pytorch_trainer.py`)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

class ClothingTrainer:
    def __init__(self, model_name='efficientnetv2_s', num_classes=3, device='cuda'):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        self.model = None
        
    def build_model(self):
        """构建模型"""
        self.model = timm.create_model(
            self.model_name, 
            pretrained=True, 
            num_classes=self.num_classes
        )
        self.model.to(self.device)
        
    def train(self, train_loader, val_loader, epochs=50):
        """训练模型"""
        # 实现训练逻辑
        pass
```

#### 2.3 PyTorch分类器 (`src/core/pytorch_classifier.py`)
```python
import torch
from torchvision import transforms
from PIL import Image
import timm

class ClothingClassifier:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        self.classes = ['主图', '细节', '吊牌']
        
    def classify_image(self, image_path):
        """分类单张图片"""
        # 实现分类逻辑
        pass
        
    def batch_classify(self, image_paths, batch_size=32):
        """批量分类"""
        # 实现批量分类逻辑
        pass
```

### 第三阶段：桌面应用升级 (2-3天)

#### 3.1 现代化UI设计（参考微信电脑版）

##### **界面布局设计**
```
JiLing-fuzhuangfenlei 主界面
├── 🎨 左侧导航栏 (80px宽)
│   ├── 🤖 训练模块 (图标+文字)
│   ├── 📸 分类模块 (图标+文字)  
│   ├── 🔧 模型管理 (图标+文字)
│   ├── ⚙️ 设置模块 (图标+文字)
│   └── ℹ️ 关于信息 (图标+文字)
│
└── 📱 主内容区域
    ├── 顶部标题栏 (40px高)
    ├── 功能操作区 (动态内容)
    ├── 进度显示区 (可折叠)
    └── 状态信息栏 (30px高)
```

##### **配色方案（深色主题）**
```css
/* 主要颜色 */
--bg-primary: #2F2F2F      /* 主背景色 */
--bg-secondary: #1E1E1E    /* 侧边栏背景 */
--bg-hover: #3A3A3A        /* 悬停效果 */
--bg-active: #404040       /* 激活状态 */

/* 文字颜色 */
--text-primary: #FFFFFF    /* 主文字 */
--text-secondary: #B0B0B0  /* 次要文字 */
--text-disabled: #666666   /* 禁用文字 */

/* 强调色（AI主题） */
--accent-primary: #4A90E2  /* 主强调色（蓝色） */
--accent-secondary: #7B68EE /* 次强调色（紫色） */
--accent-success: #4CAF50  /* 成功色（绿色） */
--accent-warning: #FF9800  /* 警告色（橙色） */
--accent-error: #F44336    /* 错误色（红色） */

/* 边框和分割线 */
--border-color: #404040    /* 边框颜色 */
--divider-color: #333333   /* 分割线颜色 */
```

##### **组件设计规范**
```python
# PySide6样式表示例
MAIN_STYLE = """
QMainWindow {
    background-color: #2F2F2F;
    color: #FFFFFF;
}

/* 左侧导航栏 */
QFrame#sidebar {
    background-color: #1E1E1E;
    border-right: 1px solid #404040;
    min-width: 80px;
    max-width: 80px;
}

/* 导航按钮 */
QPushButton#nav_button {
    background-color: transparent;
    border: none;
    padding: 15px 10px;
    text-align: center;
    color: #B0B0B0;
    border-radius: 8px;
    margin: 5px;
}

QPushButton#nav_button:hover {
    background-color: #3A3A3A;
    color: #FFFFFF;
}

QPushButton#nav_button:checked {
    background-color: #4A90E2;
    color: #FFFFFF;
}

/* 主内容区域 */
QFrame#content_area {
    background-color: #2F2F2F;
    border-radius: 12px;
    margin: 10px;
}

/* 进度条 */
QProgressBar {
    border: 2px solid #404040;
    border-radius: 8px;
    background-color: #1E1E1E;
    text-align: center;
    color: #FFFFFF;
}

QProgressBar::chunk {
    background-color: #4A90E2;
    border-radius: 6px;
}
"""
```

- 升级训练标签页为PyTorch后端
- 增强分类标签页的批量处理能力
- 添加模型性能监控和可视化
- 集成配置导出功能
- 现代化UI主题和高DPI支持

#### 3.2 具体界面模块设计

##### **主窗口代码示例 (`src/ui/main_window.py`)**
```python
from PySide6.QtWidgets import (QMainWindow, QHBoxLayout, QVBoxLayout, 
                             QFrame, QPushButton, QStackedWidget, QLabel)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QFont

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JiLing-fuzhuangfenlei")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(MAIN_STYLE)
        
        # 创建主布局
        self.setup_ui()
        
    def setup_ui(self):
        # 主容器
        main_widget = QFrame()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 左侧导航栏
        self.sidebar = self.create_sidebar()
        main_layout.addWidget(self.sidebar)
        
        # 右侧内容区域
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("content_area")
        main_layout.addWidget(self.content_stack)
        
        self.setCentralWidget(main_widget)
        
    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(5, 20, 5, 20)
        layout.setSpacing(10)
        
        # 导航按钮
        nav_items = [
            ("🤖", "训练", "training"),
            ("📸", "分类", "classification"),  
            ("🔧", "模型", "models"),
            ("⚙️", "设置", "settings"),
        ]
        
        for icon, text, name in nav_items:
            btn = self.create_nav_button(icon, text, name)
            layout.addWidget(btn)
            
        layout.addStretch()  # 底部弹性空间
        return sidebar
        
    def create_nav_button(self, icon, text, name):
        btn = QPushButton(f"{icon}\n{text}")
        btn.setObjectName("nav_button")
        btn.setCheckable(True)
        btn.setFixedSize(70, 70)
        btn.clicked.connect(lambda: self.switch_page(name))
        return btn
```

##### **各模块界面设计**

**🤖 训练模块界面**
```
┌─────────────────────────────────────────┐
│ 📊 模型训练中心                          │
├─────────────────────────────────────────┤
│ 模型选择: [EfficientNetV2-S ▼] [ConvNeXt-Tiny] │
│ 数据路径: [D:\data\train\] [📁浏览]       │
│ 训练参数: 批次大小[32] 学习率[0.001]      │
│                                         │
│ ┌─ 训练进度 ─────────────────────────────┐ │
│ │ Epoch: 15/50  ████████░░░ 65%        │ │
│ │ Loss: 0.245   Accuracy: 94.2%        │ │
│ │ ETA: 2h 15m   GPU: RTX3060 (85%)     │ │
│ └───────────────────────────────────────┘ │
│                                         │
│ [🚀开始训练] [⏸️暂停] [📊查看日志]        │
└─────────────────────────────────────────┘
```

**📸 分类模块界面**
```
┌─────────────────────────────────────────┐
│ 🎯 智能图片分类                          │
├─────────────────────────────────────────┤
│ 输入目录: [D:\桌面\筛选\JPG\] [📁浏览]    │
│ 输出目录: [D:\桌面\筛选\] [📁浏览]        │
│ 选择模型: [best_model.pth ▼] [🔄刷新]   │
│                                         │
│ ┌─ 预览区域 ─────────────────────────────┐ │
│ │ [图片1] [图片2] [图片3] [图片4]       │ │
│ │  主图    细节    吊牌    主图         │ │
│ │ 95.2%   87.6%   99.1%   92.8%        │ │
│ └───────────────────────────────────────┘ │
│                                         │
│ [⚡快速分类] [🎨批量预览] [📋导出配置]     │
└─────────────────────────────────────────┘
```

#### 3.3 新增功能
- **快速分类按钮**: 一键启动独立脚本
- **模型对比工具**: EfficientNetV2-S vs ConvNeXt-Tiny
- **性能监控**: GPU使用率、内存占用、处理速度
- **结果统计**: 各类别数量、置信度分布

### 第四阶段：独立分类脚本 (1天)

#### 4.1 快速分类脚本 (`scripts/fast_classify.py`)
```python
#!/usr/bin/env python3
import argparse
import json
import torch
from pathlib import Path
import shutil
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='服装图片快速分类')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--input', type=str, help='输入文件夹路径')
    parser.add_argument('--output', type=str, help='输出文件夹路径')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--batch-size', type=int, default=32, help='批处理大小')
    parser.add_argument('--confidence', type=float, default=0.5, help='置信度阈值')
    
    args = parser.parse_args()
    
    # 读取配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 执行分类
    classifier = ClothingClassifier(args.model or config['model_path'])
    classifier.batch_classify_folder(
        input_folder=args.input or config['input_folder'],
        output_folder=args.output or config['output_folder'],
        batch_size=args.batch_size,
        confidence_threshold=args.confidence
    )

if __name__ == '__main__':
    main()
```

#### 4.2 配置文件模板 (`config.json`)
```json
{
    "model_config": {
        "model_name": "efficientnetv2_s",
        "model_path": "models/saved_models/best_model.pth",
        "num_classes": 3,
        "image_size": 512
    },
    "paths": {
        "input_folder": "D:/桌面/筛选/JPG",
        "output_folder": "D:/桌面/筛选",
        "log_folder": "logs"
    },
    "classification": {
        "batch_size": 32,
        "confidence_threshold": 0.5,
        "classes": ["主图", "细节", "吊牌"]
    },
    "processing": {
        "delete_source": true,
        "create_subfolders": true,
        "save_statistics": true
    }
}
```

## 📊 性能预期

### 模型对比 (512×512, RTX 3060 12G)
| 模型 | 参数量 | 训练速度 | 推理速度 | 显存占用 | 预期准确率 |
|------|--------|----------|----------|----------|------------|
| **EfficientNetV2-S** | 22M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~3-4GB | 95%+ |
| **ConvNeXt-Tiny** | 29M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~4-5GB | 96%+ |

### 处理速度预期
- **训练**: 20,000张图片 → 2-4小时完成
- **推理**: 1000张图片 → 2-3分钟完成
- **批量分类**: 81张图片 → 5-10秒完成

## 🔧 使用方式

### 桌面应用模式
```bash
# 启动桌面应用
python src/main.py
```

### 快速脚本模式
```bash
# 使用默认配置
python scripts/fast_classify.py

# 使用自定义参数
python scripts/fast_classify.py \
  --input "D:/桌面/筛选/JPG" \
  --output "D:/桌面/筛选" \
  --model "models/best_model.pth" \
  --batch-size 16 \
  --confidence 0.6
```

## 📈 升级优势

### 技术优势
1. **GPU支持更稳定**: PyTorch的CUDA支持比TensorFlow更可靠
2. **模型性能更好**: EfficientNetV2-S比EfficientNetB0准确率提升5-10%
3. **训练速度更快**: 优化的数据加载和混合精度训练
4. **社区支持更好**: timm库提供1000+预训练模型

### 工作流程优势
1. **双模式运行**: 桌面应用 + 独立脚本，灵活高效
2. **配置化管理**: 参数调整无需修改代码
3. **批量处理优化**: 20,000+张图片快速处理
4. **结果可视化**: 详细的统计和分析报告

## 🎯 实施建议

### 优先级排序
1. **P0**: 环境搭建和GPU验证
2. **P1**: 核心分类模块开发
3. **P1**: 独立分类脚本
4. **P2**: 桌面应用升级
5. **P3**: 高级功能和优化

### 风险评估
- **低风险**: PyTorch安装和基础功能
- **中风险**: 现有代码迁移
- **高风险**: 大规模数据训练

### 成功指标
- [ ] GPU成功识别并使用
- [ ] 模型训练准确率 > 95%
- [ ] 批量分类速度 > 100张/分钟
- [ ] 桌面应用功能完整性

## 📚 参考资料

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [timm模型库](https://github.com/rwightman/pytorch-image-models)
- [EfficientNetV2论文](https://arxiv.org/abs/2104.00298)
- [ConvNeXt论文](https://arxiv.org/abs/2201.03545)

---

**准备开始升级？** 

## 🚀 **下一步行动**

您现在可以：

1. **新建项目文件夹**：`JiLing-fuzhuangfenlei`
2. **移动这个MD文档过去**
3. **按照文档创建项目结构**
4. **从第一阶段开始实施**

准备好开始实施了吗？我们可以从环境搭建开始！💪
