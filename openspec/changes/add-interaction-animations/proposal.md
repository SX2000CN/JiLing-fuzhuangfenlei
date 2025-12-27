# Proposal: add-interaction-animations

## 概述

为 JiLing GUI 添加完整的交互动画系统，采用 macOS 风格（流畅弹性，300ms 过渡，spring 缓动），提升用户体验。

## 动机

当前界面缺乏交互反馈动画，用户操作时感觉生硬：
- 页面切换是瞬间跳转，没有过渡效果
- 按钮悬停/点击没有动画反馈
- 复选框切换没有动画
- 进度条没有平滑过渡
- 主题切换是瞬间变化

## 目标

1. 建立统一的动画系统（AnimationManager）
2. 实现 macOS 风格的 spring 缓动曲线
3. 为所有核心组件添加交互动画
4. 确保动画性能不影响主线程

## 范围

### 包含

| 动画类型 | 组件 | 效果 |
|---------|------|------|
| 页面切换 | QStackedWidget | 淡入淡出 + 滑动 |
| 按钮悬停 | SidebarButton, ActionButton | 背景色渐变 + 缩放 |
| 按钮点击 | 所有按钮 | 按下缩放 + 释放弹回 |
| 复选框 | VSCheckBox | 勾选标记动画 |
| 进度条 | QProgressBar | 数值平滑过渡 |
| 主题切换 | 全局 | 颜色渐变过渡 |
| 面板展开 | 参数面板 | 宽度/高度动画 |
| 输入框聚焦 | PathInput, QLineEdit | 边框高亮动画 |

### 不包含

- 3D 变换效果
- 粒子效果
- 复杂的路径动画

## 技术方案

- 使用 PySide6 的 `QPropertyAnimation` 和 `QParallelAnimationGroup`
- 自定义 spring 缓动曲线（模拟 macOS 弹性效果）
- 创建 `AnimationManager` 单例管理所有动画
- 动画参数可通过设置调整（启用/禁用、时长）

## 风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 动画影响性能 | 中 | 使用硬件加速，限制并发动画数 |
| 动画与业务逻辑冲突 | 低 | 动画完成回调机制 |
| 低配设备卡顿 | 中 | 提供禁用动画选项 |

## 相关规格

- `gui` - GUI 基础规格
- `theme-system` - 主题系统规格
- `ui-architecture` - UI 架构规格
