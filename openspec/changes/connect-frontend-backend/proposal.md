# Proposal: connect-frontend-backend

## Summary

将新版模块化 GUI 的所有前端功能与后端核心模块完全对接，确保用户界面的每个功能都能正确调用后端逻辑，同时补齐后端缺失的功能。

## Motivation

当前新版 GUI（`src/gui/app.py` 及 `pages/` 模块）采用了模块化架构，但存在以下对接问题：

### 🔴 严重问题（功能不可用）

| 问题 | 影响 |
|------|------|
| 设置页面未连接到主窗口 | 用户修改的所有设置都无法生效 |
| 模型加载功能未实现 | 分类页面无法加载自定义模型 |
| 置信度阈值未应用 | 分类结果不受用户设置影响 |

### 🟡 中等问题（功能受限）

| 问题 | 影响 |
|------|------|
| 数据加载线程数未传递 | 无法优化数据加载性能 |
| 早停机制未实现 | 训练无法提前停止，浪费资源 |
| 混合精度训练未实现 | 无法加速 GPU 训练 |

### 🟢 轻微问题（功能缺失）

| 问题 | 影响 |
|------|------|
| 训练历史可视化未使用 | 用户无法查看训练曲线 |
| 分类报告 UI 缺失 | 无法查看详细分类报告 |

## Solution

### 阶段一：核心功能对接（P0）

1. **连接设置页面**
   - 在 `app.py` 中读取设置页面的配置
   - 将设置传递给训练/分类工作线程
   - 实现设置的持久化（保存到 `config.json`）

2. **实现模型加载**
   - 处理 `ClassificationPage.load_model_requested` 信号
   - 调用 `ClothingClassifier.load_model()` 加载模型
   - 更新模型状态显示

3. **应用置信度阈值**
   - 从设置页面读取阈值
   - 传递给 `ClassificationWorker`
   - 在分类结果中标记低置信度项

### 阶段二：训练增强（P1）

4. **传递数据加载参数**
   - 读取 `workers` 设置
   - 传递给 `trainer.create_data_loaders()`

5. **实现早停机制**
   - 在 `pytorch_trainer.py` 中添加早停逻辑
   - 监控验证损失，连续 N 轮不下降则停止

6. **实现混合精度训练**
   - 使用 `torch.cuda.amp` 实现 AMP
   - 根据设置开关启用/禁用

### 阶段三：可视化增强（P2）

7. **训练历史可视化**
   - 训练完成后调用 `plot_history()`
   - 在终端区域显示图表路径

8. **分类报告展示**
   - 读取生成的 JSON 报告
   - 在结果区域显示统计信息

## Scope

### 涉及文件

| 文件 | 修改类型 |
|------|---------|
| `src/gui/app.py` | 修改 - 添加设置连接和信号处理 |
| `src/gui/workers/classification.py` | 修改 - 添加置信度阈值参数 |
| `src/gui/workers/training.py` | 修改 - 添加设置参数传递 |
| `src/core/pytorch_trainer.py` | 修改 - 添加早停和 AMP |
| `src/core/pytorch_classifier.py` | 修改 - 添加阈值过滤 |

### 不涉及

- 主题系统（已有独立提案 `rewrite-light-theme`）
- UI 组件样式
- 传统界面 `main_window.py`

## Risks

1. **设置持久化格式**：需要确定 `config.json` 的结构，避免与现有配置冲突
2. **AMP 兼容性**：部分旧 GPU 不支持 FP16，需要添加检测和回退逻辑
3. **早停阈值**：需要合理设置默认值，避免过早停止

## References

- [PyTorch AMP 文档](https://pytorch.org/docs/stable/amp.html)
- [早停最佳实践](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- 现有后端实现：`src/core/pytorch_trainer.py`, `src/core/pytorch_classifier.py`
