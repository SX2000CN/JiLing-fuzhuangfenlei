# 项目结构说明

## 目录结构

```
JiLing-fuzhuangfenlei/
├── launchers/              # 启动器目录
│   ├── launch.py           # 统一启动器 (支持现代/传统界面)
│   ├── modern_gui_main.py  # 现代化界面 (Ant Design 风格)
│   ├── gui_main.py         # 传统界面
│   ├── 启动器.bat           # 批处理启动菜单
│   ├── 快速启动-现代界面.bat
│   ├── 快速启动-传统界面.bat
│   ├── 测试环境.bat
│   └── run_classify.bat
│
├── src/                    # 源代码
│   ├── core/              # 核心功能
│   │   ├── model_factory.py      # 模型工厂
│   │   ├── pytorch_classifier.py # PyTorch 分类器
│   │   └── pytorch_trainer.py    # 训练器
│   ├── gui/               # GUI 组件
│   │   └── main_window.py
│   └── utils/             # 工具函数
│       ├── config_manager.py
│       ├── image_utils.py
│       └── logging_utils.py
│
├── web-frontend/          # Web 前端 (React + TypeScript + Ant Design)
│   ├── src/
│   │   ├── components/   # React 组件
│   │   ├── pages/        # 页面
│   │   ├── services/     # API 服务
│   │   └── store/        # 状态管理
│   └── package.json
│
├── scripts/               # 脚本工具
│   └── fast_classify.py  # 快速分类脚本
│
├── tests/                 # 测试文件
│   ├── test_model_finding.py
│   ├── memory_test.py
│   ├── interactive_test.py
│   └── create_demo_model.py
│
├── docs/                  # 文档
│   ├── 启动说明.md
│   ├── 使用说明.txt
│   ├── 部署包说明.md
│   ├── 项目完整文档.md
│   └── PyTorch升级方案.md
│
├── build/                 # 构建工具
│   ├── make_installer.bat
│   ├── project_cleanup.bat
│   └── 分类测试整理.bat
│
├── temp/                  # 临时文件
│   ├── fix_text.py
│   └── fast_classify.log
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
├── outputs/               # 输出结果
├── logs/                  # 日志文件
│
├── api_server.py          # API 服务器
├── api_requirements.txt   # API 依赖
├── web_shell.py           # Web 外壳 (QtWebEngine)
├── classify_cli.py        # 命令行分类工具
├── config.json            # 主配置文件
├── requirements.txt       # Python 依赖
├── README.md              # 项目说明
├── LICENSE.txt            # 许可证
└── start.bat              # 主启动入口 ⭐
```

## 快速开始

### 方式 1: 使用主启动器 (推荐)
双击 `start.bat`，选择启动方式

### 方式 2: 直接启动
- 现代界面: `launchers\快速启动-现代界面.bat`
- 传统界面: `launchers\快速启动-传统界面.bat`
- 命令行工具: `launchers\run_classify.bat`

### 方式 3: Python 直接运行
```bash
# 现代界面
python launchers/launch.py --mode modern

# 传统界面
python launchers/launch.py --mode traditional

# 命令行分类
python classify_cli.py
```

## 主要组件

### 1. 启动系统
- **launch.py**: 统一启动器，支持自动进程清理
- **modern_gui_main.py**: 现代化界面 (Ant Design 风格)
- **gui_main.py**: 传统 PySide6 界面

### 2. Web 系统
- **web-frontend/**: React + TypeScript + Ant Design 前端
- **api_server.py**: FastAPI 后端服务
- **web_shell.py**: Qt WebEngine 桌面包装器

### 3. 核心功能
- **src/core/**: 模型训练、分类核心逻辑
- **src/utils/**: 工具函数 (图像处理、配置管理、日志)

## 环境要求

- Python 3.11+
- PySide6 6.10.0+
- PyTorch (CPU/CUDA)
- Node.js 16+ (Web 前端)

详见 `requirements.txt` 和 `api_requirements.txt`
