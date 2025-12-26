# Design: connect-frontend-backend

## 架构概述

本设计文档描述如何将模块化 GUI 与后端核心模块完全对接。

## 当前架构

```
┌─────────────────────────────────────────────────────────────┐
│                    NativeUI (native_ui.py)                  │
│  - 主窗口框架                                                │
│  - 侧边栏导航                                                │
│  - 页面切换                                                  │
└─────────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
    ┌────────────┐      ┌────────────┐      ┌────────────┐
    │ TrainingPage│      │ClassifyPage│      │SettingsPage│
    │  训练页面   │      │  分类页面   │      │  设置页面   │
    │  ✅ 已连接  │      │  ⚠️ 部分   │      │  ❌ 未连接  │
    └────────────┘      └────────────┘      └────────────┘
         ↓                    ↓
    ┌────────────┐      ┌────────────┐
    │TrainingWorker│    │ClassifyWorker│
    │  训练线程   │      │  分类线程   │
    └────────────┘      └────────────┘
         ↓                    ↓
    ┌────────────┐      ┌────────────┐
    │ClothingTrainer│   │ClothingClassifier│
    │  训练器     │      │  分类器     │
    └────────────┘      └────────────┘
```

## 目标架构

```
┌─────────────────────────────────────────────────────────────┐
│                    NativeUI (native_ui.py)                  │
│  - 主窗口框架                                                │
│  - 设置管理器 (新增)                                         │
│  - 模型状态管理 (新增)                                       │
└─────────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
    ┌────────────┐      ┌────────────┐      ┌────────────┐
    │ TrainingPage│      │ClassifyPage│      │SettingsPage│
    │  训练页面   │      │  分类页面   │      │  设置页面   │
    │  ✅ 完全   │      │  ✅ 完全   │      │  ✅ 完全   │
    └────────────┘      └────────────┘      └────────────┘
         ↓                    ↓                    ↓
    ┌────────────┐      ┌────────────┐      ┌────────────┐
    │TrainingWorker│    │ClassifyWorker│    │SettingsManager│
    │  训练线程   │      │  分类线程   │      │  设置管理   │
    │  +AMP      │      │  +阈值过滤  │      │  +持久化   │
    │  +早停     │      │  +模型加载  │      │            │
    └────────────┘      └────────────┘      └────────────┘
         ↓                    ↓
    ┌────────────┐      ┌────────────┐
    │ClothingTrainer│   │ClothingClassifier│
    │  +AMP      │      │  +阈值过滤  │
    │  +早停     │      │            │
    └────────────┘      └────────────┘
```

## 关键设计决策

### 1. 设置管理策略

**方案 A：集中式设置管理器（推荐）**
- 创建 `SettingsManager` 类管理所有设置
- 设置变更时自动保存到 `config.json`
- 启动时自动加载设置

**方案 B：分散式设置传递**
- 每个页面独立管理自己的设置
- 需要手动传递设置到各个组件

**选择方案 A**，因为：
- 统一的设置入口，易于维护
- 自动持久化，用户体验更好
- 避免设置不一致问题

### 2. 模型状态管理

**当前问题**：分类页面需要知道模型是否已加载，但没有全局状态管理。

**解决方案**：
- 在 `NativeUI` 中维护 `_current_model` 状态
- 模型加载成功后更新状态
- 分类前检查模型状态

### 3. 信号连接设计

```python
# 设置页面信号
SettingsPage.theme_changed → NativeUI._on_theme_changed
SettingsPage.settings_changed → NativeUI._on_settings_changed (新增)

# 分类页面信号
ClassificationPage.load_model_requested → NativeUI._on_load_model (新增)
ClassificationPage.start_classification_requested → NativeUI._on_start_classification

# 训练页面信号
TrainingPage.start_training_requested → NativeUI._on_start_training
TrainingPage.stop_training_requested → NativeUI._on_stop_training
```

### 4. 配置文件结构

```json
{
  "appearance": {
    "theme": "dark",
    "scale": "100%"
  },
  "model": {
    "device": "auto",
    "precision": "fp32",
    "confidence_threshold": 0.5
  },
  "paths": {
    "model_dir": "models/saved_models",
    "dataset_dir": "data",
    "log_dir": "logs",
    "export_dir": "outputs"
  },
  "performance": {
    "workers": 4,
    "max_batch": 64,
    "amp_enabled": true,
    "pin_memory": true
  },
  "training": {
    "default_epochs": 50,
    "patience": 10,
    "checkpoint_freq": 5,
    "save_best_only": true
  },
  "logging": {
    "level": "INFO",
    "retention_days": 30,
    "verbose": false
  }
}
```

### 5. 早停机制设计

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 停止训练
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False
```

### 6. 混合精度训练设计

```python
# 在 ClothingTrainer 中添加
if self.amp_enabled:
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

## 依赖关系

```
阶段一（P0）
├── 1. 设置管理器 ← 无依赖
├── 2. 模型加载 ← 无依赖
└── 3. 置信度阈值 ← 依赖 1

阶段二（P1）
├── 4. 数据加载参数 ← 依赖 1
├── 5. 早停机制 ← 无依赖
└── 6. 混合精度 ← 依赖 1

阶段三（P2）
├── 7. 训练可视化 ← 无依赖
└── 8. 分类报告 ← 无依赖
```

## 测试策略

1. **单元测试**：测试 `SettingsManager` 的保存/加载功能
2. **集成测试**：测试设置变更是否正确传递到后端
3. **端到端测试**：完整的训练/分类流程测试
