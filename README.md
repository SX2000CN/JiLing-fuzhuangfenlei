# 服装分类系统

基于 PyTorch 的智能服装图片自动分类系统，能够将服装拍摄照片自动分类为三个类别。

## 分类类别

- **主图** - 完整服装展示（正面挂拍，背景干净）
- **细节** - 局部特写（面料、工艺、设计细节）
- **吊牌** - 商品标签（价格、成分、货号信息）

## 技术栈

- **深度学习**: PyTorch 2.0+ / timm (EfficientNetV2-S)
- **图形界面**: PySide6
- **图像处理**: Pillow / OpenCV

## 快速开始

### 环境要求

- Python 3.11+
- 8GB+ RAM
- GPU (可选，支持 CUDA 加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动方式

**方式一：命令行工具（推荐，生产环境）**
```bash
python launchers/classify_cli.py
# 或双击 launchers/run_classify.bat
```

**方式二：传统图形界面**
```bash
python launchers/launch.py
# 或双击 launchers/快速启动-传统界面.bat
```

**方式三：新版图形界面（VS Code 风格）**
```bash
python src/gui/native_ui.py
# 或双击 launchers/启动-新界面.bat
```

## 项目结构

```
服装分类系统/
├── src/                    # 源代码
│   ├── core/              # 核心算法
│   │   ├── pytorch_classifier.py   # 分类器
│   │   ├── pytorch_trainer.py      # 训练器
│   │   └── model_factory.py        # 模型工厂
│   ├── gui/               # 图形界面
│   │   ├── main_window.py          # 传统界面
│   │   ├── native_ui.py            # 新版界面
│   │   ├── components/             # UI组件库
│   │   ├── pages/                  # 页面模块
│   │   └── styles/                 # 设计系统
│   ├── cli/               # 命令行工具
│   │   └── classify_cli.py         # 批量分类
│   └── utils/             # 工具函数
├── config/                # 配置文件
│   ├── model_config.yaml          # 模型配置
│   ├── training_config.yaml       # 训练配置
│   └── paths_config.yaml          # 路径配置
├── models/                # 模型文件
├── data/                  # 数据集
│   ├── train/            # 训练数据
│   ├── val/              # 验证数据
│   └── test/             # 测试数据
├── launchers/             # 启动脚本
├── docs/                  # 详细文档
├── UI设计规范/            # UI设计规范文档
├── fonts/                 # 字体资源 (MiSans)
├── logs/                  # 运行日志
└── outputs/               # 输出结果
```

## 功能模块

### 命令行分类工具 (主要工作程序)
- 批量图片分类
- 多线程并行处理（20线程）
- 自动文件组织
- 实时进度显示

### 传统图形界面
- 完整的训练和分类功能
- 参数配置面板
- 实时日志输出

### 新版图形界面 (开发中)
- VS Code 风格设计
- 无边框窗口
- 现代化组件库

## 配置说明

主要配置文件位于 `config/` 目录：

| 文件 | 说明 |
|------|------|
| `model_config.yaml` | 模型类型、类别数、图像尺寸 |
| `training_config.yaml` | 训练参数、优化器、数据增强 |
| `paths_config.yaml` | 数据路径、输出路径 |

## 性能指标

- **推理速度**: 27+ 张/秒 (RTX 3060)
- **准确率**: 95%+ (EfficientNetV2-S)
- **批量处理**: 20,000 张图片约 2-3 分钟

## 文档

- [UI设计规范](UI设计规范/README.md) - 界面设计规范文档
- [开发者指南](docs/开发者指南.md) - 代码结构与API说明
- [部署说明](docs/部署包说明.md) - 部署与打包指南
- [项目完整文档](docs/项目完整文档.md) - 详细功能说明

## 许可证

MIT License
