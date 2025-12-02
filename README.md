# 服装挂拍分类系统 - JiLing-fuzhuangfenlei

## 📋 项目简介

基于PyTorch的智能服装图片自动分类系统，支持将服装拍摄照片自动分类为：
- **主图**：完整服装展示（正面挂拍，背景干净）
- **细节**：局部特写（面料、工艺、设计细节）
- **吊牌**：商品标签（价格、成分、货号信息）

## 🚀 快速开始

**最简单的方式：双击 `start.bat` 启动主菜单！**

### 启动选项
1. **现代化界面** (推荐) - Ant Design 风格，功能完整
2. **传统界面** - 经典 PySide6 界面
3. **命令行工具** - 批量分类处理
4. **测试环境** - 检查依赖安装

### 快捷启动
- 新界面: `启动-新界面.bat`（根目录）
- 传统界面: `启动-传统界面.bat`（根目录）或 `launchers\快速启动-传统界面.bat`
- 命令行分类: `launchers\run_classify.bat`（生产环境核心工具）

## 🎯 功能特性

### 桌面应用
- **双界面**：新界面 (VS Code 风格) + 传统 UI (PySide6)
- **自动进程管理**：重启时自动关闭旧进程
- **实时预览**：分类结果即时显示

### Web 前端
- **React + TypeScript**：现代化前端技术栈
- **Ant Design**：企业级 UI 组件库
- **响应式设计**：适配各种屏幕尺寸

### 核心功能
- **高性能训练**：支持EfficientNetV2-S和ConvNeXt-Tiny模型
- **GPU加速**：优化的CUDA支持，适配RTX 3060
- **批量处理**：快速处理20,000+张图片
- **智能分类**：主图/细节/吊牌自动识别

## 🔧 系统要求

- Python 3.11+ (已包含在 `.conda` 目录)
- PySide6 6.10.0+
- PyTorch 2.0+ (支持CPU/CUDA)
- 8GB+ RAM
- 20GB+ 可用磁盘空间

## 📦 安装说明

### 已配置环境
项目已包含完整的 Conda 环境 (`.conda` 目录)，所有批处理文件会自动使用正确的 Python 解释器。

### 手动配置 (可选)
如果需要重新配置环境:
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动应用
```bash
# 桌面应用模式
python src/main.py

# 快速分类模式
python scripts/fast_classify.py
```

## 📁 项目结构

```
JiLing-fuzhuangfenlei/
├── README.md
├── requirements.txt
├── start.bat                   # 主菜单启动器
├── 启动-新界面.bat             # 快捷启动-新界面
├── 启动-传统界面.bat           # 快捷启动-传统界面
├── launchers/                  # 启动器目录
│   ├── classify_cli.py        # 命令行分类工具（生产核心）
│   ├── run_classify.bat       # 生产环境启动脚本
│   ├── launch.py              # 通用启动器
│   └── gui_main.py            # GUI启动入口
├── src/                        # 源代码
│   ├── core/                  # 核心算法
│   │   ├── model_factory.py   # 模型工厂
│   │   ├── pytorch_classifier.py  # 分类器
│   │   └── pytorch_trainer.py # 训练器
│   ├── gui/                   # 图形界面
│   │   ├── main_window.py     # 传统界面
│   │   └── native_ui.py       # 新界面
│   └── utils/                 # 工具函数
├── config/                     # 配置文件
├── models/                     # 模型文件
├── data/                       # 数据集
├── logs/                       # 日志文件
└── docs/                       # 项目文档
```

## 🎨 使用说明

详见 [docs/项目完整文档.md](docs/项目完整文档.md) 或 [快速开始.md](快速开始.md)

## 📈 性能指标

- **训练时间**：20,000张图片 → 2-4小时
- **推理速度**：1000张图片 → 2-3分钟
- **准确率**：95%+ (EfficientNetV2-S)

## 📄 许可证

MIT License
