# 服装挂拍分类系统 - JiLing-fuzhuangfenlei

## 📋 项目简介

基于PyTorch的智能服装图片自动分类系统，支持将服装拍摄照片自动分类为：
- **主图**：完整服装展示（正面挂拍，背景干净）
- **细节**：局部特写（面料、工艺、设计细节）
- **吊牌**：商品标签（价格、成分、货号信息）

## 🎯 功能特性

- **双模式运行**：桌面应用 + 独立脚本
- **现代化界面**：基于PySide6的深色主题UI
- **高性能训练**：支持EfficientNetV2-S和ConvNeXt-Tiny模型
- **GPU加速**：优化的CUDA支持，适配RTX 3060
- **批量处理**：快速处理20,000+张图片

## 🔧 系统要求

- Python 3.8+
- CUDA 11.8+ (推荐RTX 3060或更高)
- 8GB+ RAM
- 20GB+ 可用磁盘空间

## 📦 安装步骤

### 1. 创建Python环境
```bash
python -m venv JiLing-fuzhuangfenlei-env
JiLing-fuzhuangfenlei-env\Scripts\activate
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
├── config/                     # 配置文件
├── src/                        # 源代码
│   ├── main.py                # 主程序入口
│   ├── ui/                    # 用户界面
│   ├── core/                  # 核心功能
│   └── utils/                 # 工具函数
├── scripts/                   # 独立脚本
├── models/                    # 模型文件
├── data/                      # 数据集
├── logs/                      # 日志文件
└── outputs/                   # 输出结果
```

## 🎨 使用说明

详见 [docs/usage.md](docs/usage.md)

## 📈 性能指标

- **训练时间**：20,000张图片 → 2-4小时
- **推理速度**：1000张图片 → 2-3分钟
- **准确率**：95%+ (EfficientNetV2-S)

## 📄 许可证

MIT License
