# 项目结构说明

## 目录结构

```
JiLing-fuzhuangfenlei/
├── src/                    # 源代码
│   ├── core/              # 核心功能
│   │   ├── model_factory.py      # 模型工厂
│   │   ├── pytorch_classifier.py # PyTorch 分类器
│   │   └── pytorch_trainer.py    # 训练器
│   ├── gui/               # GUI 组件
│   │   ├── main_window.py        # 传统界面
│   │   └── native_ui.py          # 新界面 (VS Code 风格) ⭐
│   └── utils/             # 工具函数
│       ├── config_manager.py
│       ├── image_utils.py
│       └── logging_utils.py
│
├── launchers/              # 启动器目录
│   ├── launch.py           # 统一启动器
│   ├── gui_main.py         # 传统界面启动
│   ├── gui_launcher.py     # 通用启动器
│   ├── classify_cli.py     # 命令行分类工具
│   ├── run_classify.bat    # 批处理分类
│   └── 快速启动-传统界面.bat
│
├── scripts/               # 脚本工具
│   └── fast_classify.py  # 快速分类脚本
│
├── tests/                 # 测试文件
│
├── docs/                  # 文档
│   ├── 启动说明.md
│   ├── 使用说明.txt
│   ├── 部署包说明.md
│   ├── 项目完整文档.md
│   └── PyTorch升级方案.md
│
├── .github/               # GitHub 配置
│   └── instructions/      # 开发规范
│       ├── 系统设计规范.md      # VS Code 设计系统参考
│       └── 回复永远使用中文.md
│
├── config/                # 配置文件
│   ├── model_config.yaml
│   ├── paths_config.yaml
│   └── training_config.yaml
│
├── data/                  # 数据集
│   ├── train/            # 训练集
│   ├── val/              # 验证集
│   └── test/             # 测试集
│
├── models/                # 模型文件
│   ├── saved_models/     # 保存的模型
│   ├── checkpoints/      # 检查点
│   └── exports/          # 导出的模型
│
├── MiSans/                # MiSans 字体资源
│
├── outputs/               # 输出结果
├── logs/                  # 日志文件
├── build/                 # 构建输出
│
├── config.json            # 主配置文件
├── requirements.txt       # Python 依赖
├── README.md              # 项目说明
├── LICENSE.txt            # 许可证
├── start.bat              # 主启动入口
├── 启动-新界面.bat         # 新界面启动器 ⭐
└── 启动-传统界面.bat       # 传统界面启动器
```

## 快速开始

### 方式 1: 启动新界面 (推荐)
双击 `启动-新界面.bat`

### 方式 2: 启动传统界面
双击 `启动-传统界面.bat`

### 方式 3: Python 直接运行
```bash
# 新界面 (VS Code 风格)
python src/gui/native_ui.py

# 传统界面
python launchers/gui_main.py

# 命令行分类
python launchers/classify_cli.py
```

## 界面说明

### 新界面 (native_ui.py)
- 基于 VS Code 设计系统
- 使用 MiSans 字体
- iOS 风格超椭圆圆角
- 现代化参数面板
- 终端式输出区域

### 传统界面 (main_window.py)
- 经典 Tab 标签页布局
- 完整的训练和分类功能
- 兼容性更好

## 核心功能

### 模型训练
- 支持多种预训练模型 (EfficientNet, ResNet, ViT)
- 可配置训练参数 (轮数、批次、学习率)
- 实时训练日志输出

### 图像分类
- 批量图像分类
- 支持多种图像格式
- 分类结果导出

## 环境要求

- Python 3.11+
- PySide6 6.10.0+
- PyTorch (CPU/CUDA)
- timm (模型库)

详见 `requirements.txt`
