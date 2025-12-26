# Tasks: connect-frontend-backend

## 阶段一：核心功能对接（P0）

### 1. 创建设置管理器
- [x] 创建 `src/gui/utils/settings_manager.py`
- [x] 实现 `SettingsManager` 类
  - [x] `load()` - 从 `config.json` 加载设置
  - [x] `save()` - 保存设置到 `config.json`
  - [x] `get(key)` - 获取单个设置
  - [x] `set(key, value)` - 设置单个值
- [x] 在 `src/gui/utils/__init__.py` 中导出
- **验证**：手动修改设置后重启应用，确认设置被保留

### 2. 连接设置页面到主窗口
- [x] 在 `MainWindow.__init__()` 中初始化 `SettingsManager`
- [x] 连接 `SettingsPage.settings_changed` 信号
- [x] 实现 `_on_settings_changed()` 方法
- [x] 启动时调用 `SettingsPage.set_settings()` 加载已保存设置
- **验证**：修改设置后检查 `gui_settings.json` 是否更新

### 3. 实现模型加载功能
- [x] 在 `MainWindow` 中使用 `current_classifier` 属性
- [x] 连接 `ClassificationPage.load_model_requested` 信号
- [x] 实现 `_load_classify_model()` 方法
  - [x] 创建 `ClothingClassifier` 实例
  - [x] 更新 `ClassificationPage` 的模型状态显示
- [x] 实现 `_use_default_model()` 方法
- **验证**：加载模型后状态显示"已加载"

### 4. 应用置信度阈值
- [x] 修改 `ClassificationWorker.__init__()` 添加 `confidence_threshold` 参数
- [x] 在 `MainWindow._start_classification()` 中从设置读取阈值
- [x] 在分类结果中标记低于阈值的项（`uncertain` 字段）
- [x] 在 `ClassificationPage.set_results()` 中用红色标记不确定项
- **验证**：设置高阈值后，低置信度结果被标记

---

## 阶段二：训练增强（P1）

### 5. 传递数据加载参数
- [x] 修改 `TrainingWorker` 传递 `num_workers` 参数
- [x] 在 `MainWindow._start_training()` 中从设置读取 `workers` 值
- [x] 传递给 `trainer.create_data_loaders()`
- **验证**：检查数据加载器使用的线程数

### 6. 实现早停机制
- [x] 在 `src/core/pytorch_trainer.py` 中添加 `EarlyStopping` 类
- [x] 在 `TrainingWorker` 中使用早停机制
- [x] 从设置读取 `patience` 参数
- [x] 早停时发送日志消息
- **验证**：设置小 patience 值，验证训练提前停止

### 7. 实现混合精度训练
- [x] 修改 `ClothingTrainer.__init__()` 添加 `amp_enabled` 参数
- [x] 在 `train_epoch()` 方法中使用 `torch.cuda.amp`
- [x] 添加 GPU 兼容性检查（仅 CUDA 可用时启用）
- [x] 从设置读取 AMP 开关状态
- **验证**：启用 AMP 后检查训练速度提升

---

## 阶段三：可视化增强（P2）

### 8. 训练历史可视化
- [x] 训练完成后调用 `trainer.plot_history()`
- [x] 保存图表到模型同目录
- [x] 在日志中输出图表保存路径
- **验证**：训练完成后检查图表文件生成

### 9. 分类报告展示
- [x] 分类完成后生成统计摘要
- [x] 在结果区域显示统计摘要
  - [x] 各类别数量
  - [x] 平均置信度
  - [x] 低置信度项数量
- [x] 添加 `ClassificationPage.set_results()` 方法
- [x] 添加 `ClassificationPage.show_summary()` 方法
- **验证**：分类完成后显示统计信息

---

## 依赖关系

```
任务 1 (设置管理器) ← 无依赖
任务 2 (连接设置) ← 依赖任务 1
任务 3 (模型加载) ← 无依赖
任务 4 (置信度阈值) ← 依赖任务 1, 2
任务 5 (数据加载参数) ← 依赖任务 1, 2
任务 6 (早停机制) ← 无依赖
任务 7 (混合精度) ← 依赖任务 1, 2
任务 8 (训练可视化) ← 无依赖
任务 9 (分类报告) ← 无依赖
```

## 可并行任务

- 任务 1, 3, 6, 8, 9 可以并行开发
- 任务 2 完成后，任务 4, 5, 7 可以并行开发

## 预估工作量

| 任务 | 预估时间 | 复杂度 | 状态 |
|------|---------|--------|------|
| 1. 设置管理器 | 1小时 | 低 | ✅ 完成 |
| 2. 连接设置 | 1小时 | 低 | ✅ 完成 |
| 3. 模型加载 | 2小时 | 中 | ✅ 完成 |
| 4. 置信度阈值 | 30分钟 | 低 | ✅ 完成 |
| 5. 数据加载参数 | 30分钟 | 低 | ✅ 完成 |
| 6. 早停机制 | 1.5小时 | 中 | ✅ 完成 |
| 7. 混合精度 | 2小时 | 中 | ✅ 完成 |
| 8. 训练可视化 | 30分钟 | 低 | ✅ 完成 |
| 9. 分类报告 | 1小时 | 低 | ✅ 完成 |
| **总计** | **~10小时** | - | **全部完成** |
