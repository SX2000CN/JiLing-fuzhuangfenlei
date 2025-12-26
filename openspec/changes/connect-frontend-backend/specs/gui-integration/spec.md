# Spec: gui-integration

## Overview

定义 GUI 与后端核心模块的集成规范，确保前端功能正确调用后端逻辑。

---

## ADDED Requirements

### Requirement: 设置管理

GUI 必须提供统一的设置管理机制，支持设置的持久化和跨组件共享。

#### Scenario: 设置保存
- **Given** 用户在设置页面修改了配置
- **When** 用户切换到其他页面或关闭应用
- **Then** 设置自动保存到 `config.json`

#### Scenario: 设置加载
- **Given** 应用启动
- **When** 主窗口初始化完成
- **Then** 从 `config.json` 加载设置并应用到各页面

#### Scenario: 设置应用
- **Given** 用户修改了训练相关设置（如 workers、AMP）
- **When** 用户开始训练
- **Then** 训练使用最新的设置值

---

### Requirement: 模型加载

分类页面必须支持加载用户指定的模型文件。

#### Scenario: 加载模型成功
- **Given** 用户点击"加载模型"按钮
- **When** 选择有效的 `.pth` 或 `.pt` 文件
- **Then** 模型加载成功，状态显示"已加载"

#### Scenario: 加载模型失败
- **Given** 用户点击"加载模型"按钮
- **When** 选择无效或损坏的模型文件
- **Then** 显示错误提示，状态保持"未加载"

#### Scenario: 使用默认模型
- **Given** 用户点击"使用默认模型"按钮
- **When** 默认模型路径有效
- **Then** 加载默认模型，状态显示"已加载（默认）"

---

### Requirement: 置信度阈值

分类结果必须支持按置信度阈值过滤和标记。

#### Scenario: 低置信度标记
- **Given** 设置中置信度阈值为 0.7
- **When** 分类结果中某项置信度为 0.5
- **Then** 该项在结果表格中被标记为"不确定"

#### Scenario: 阈值实时生效
- **Given** 用户修改置信度阈值
- **When** 开始新的分类任务
- **Then** 使用新的阈值进行过滤

---

### Requirement: 早停机制

训练过程必须支持早停，避免过拟合和资源浪费。

#### Scenario: 触发早停
- **Given** 早停耐心值设置为 5
- **When** 验证损失连续 5 轮未下降
- **Then** 训练自动停止，显示"早停触发"消息

#### Scenario: 正常完成
- **Given** 早停耐心值设置为 10
- **When** 训练在 50 轮内验证损失持续下降
- **Then** 训练正常完成所有轮次

---

### Requirement: 混合精度训练

训练过程必须支持可选的混合精度（AMP）加速。

#### Scenario: 启用 AMP
- **Given** 设置中启用了混合精度训练
- **And** GPU 支持 FP16
- **When** 开始训练
- **Then** 使用 AMP 进行训练，显示"AMP 已启用"

#### Scenario: AMP 不可用
- **Given** 设置中启用了混合精度训练
- **And** 使用 CPU 或不支持 FP16 的 GPU
- **When** 开始训练
- **Then** 回退到 FP32 训练，显示"AMP 不可用，使用 FP32"

---

## MODIFIED Requirements

### Requirement: 训练参数传递

修改训练工作线程，支持从设置页面读取参数。

#### Scenario: 使用设置中的参数
- **Given** 设置中 workers=8, batch_size=32
- **When** 开始训练
- **Then** 数据加载器使用 8 个工作线程和 32 的批次大小

---

## Cross-References

- 相关规范：`ui-architecture/spec.md` - UI 模块化架构
- 相关规范：`theme-system/spec.md` - 主题系统（设置中的主题切换）
