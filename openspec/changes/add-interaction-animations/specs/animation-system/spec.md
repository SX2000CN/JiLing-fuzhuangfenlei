# Spec: animation-system

## 概述

GUI 交互动画系统规格，定义动画行为、时长、缓动曲线等标准。

---

## ADDED Requirements

### Requirement: ANIM-001 动画管理器

系统 SHALL 提供统一的动画管理器，集中管理所有 UI 动画。

#### Scenario: 创建动画

- Given 动画管理器已初始化
- When 调用 `AnimationManager.create_animation(widget, "opacity", 0, 1, 300)`
- Then 返回配置好的 `QPropertyAnimation` 对象
- And 动画时长为 300ms
- And 使用默认缓动曲线

#### Scenario: 全局禁用动画

- Given 动画管理器 `enabled = False`
- When 创建任何动画
- Then 动画时长自动设为 0
- And 动画立即完成

---

### Requirement: ANIM-002 macOS 风格缓动曲线

系统 SHALL 提供 macOS 风格的 spring 缓动曲线，实现弹性动画效果。

#### Scenario: Spring 缓动

- Given 使用 spring 缓动曲线
- When 动画播放
- Then 动画有轻微过冲效果
- And 最终平滑停止在目标值

#### Scenario: 缓动曲线参数

- Given spring 缓动曲线
- When 设置 `stiffness=100, damping=10`
- Then 动画表现出适度弹性
- And 过冲幅度约为目标值的 5-10%

---

### Requirement: ANIM-003 按钮交互动画

所有可点击按钮 SHALL 有悬停和点击动画反馈。

#### Scenario: 按钮悬停

- Given 按钮处于正常状态
- When 鼠标进入按钮区域
- Then 背景色在 300ms 内渐变到悬停色
- And 按钮缩放到 1.05 倍（spring 缓动）

#### Scenario: 按钮点击

- Given 按钮处于悬停状态
- When 鼠标按下
- Then 按钮在 100ms 内缩放到 0.95 倍
- When 鼠标释放
- Then 按钮在 300ms 内弹回 1.0 倍（spring 缓动）

#### Scenario: 禁用按钮

- Given 按钮处于禁用状态
- When 鼠标悬停或点击
- Then 不触发任何动画

---

### Requirement: ANIM-004 复选框动画

复选框切换 SHALL 有平滑的动画效果。

#### Scenario: 勾选动画

- Given 复选框未勾选
- When 用户点击勾选
- Then 背景色在 150ms 内渐变到选中色
- And 勾选标记在 200ms 内从起点绘制到终点

#### Scenario: 取消勾选

- Given 复选框已勾选
- When 用户点击取消
- Then 勾选标记在 100ms 内淡出
- And 背景色在 150ms 内渐变回默认色

---

### Requirement: ANIM-005 页面切换动画

页面切换 SHALL 有流畅的过渡动画。

#### Scenario: 向右切换页面

- Given 当前在页面 A
- When 切换到页面 B（index 增加）
- Then 页面 A 向左滑出并淡出（300ms）
- And 页面 B 从右侧滑入并淡入（300ms）
- And 两个动画并行执行

#### Scenario: 向左切换页面

- Given 当前在页面 B
- When 切换到页面 A（index 减少）
- Then 页面 B 向右滑出并淡出（300ms）
- And 页面 A 从左侧滑入并淡入（300ms）

---

### Requirement: ANIM-006 进度条动画

进度条数值变化 SHALL 平滑过渡。

#### Scenario: 进度更新

- Given 进度条当前值为 30%
- When 设置新值为 60%
- Then 进度条在 300ms 内平滑过渡到 60%
- And 使用 easeOut 缓动

#### Scenario: 快速连续更新

- Given 进度条正在动画中
- When 收到新的进度值
- Then 中断当前动画
- And 从当前位置开始新动画到目标值

---

## 动画参数标准

| 参数              | 值     | 说明                       |
| ----------------- | ------ | -------------------------- |
| DURATION_FAST     | 150ms  | 快速反馈（按下、淡出）     |
| DURATION_NORMAL   | 300ms  | 标准过渡（悬停、页面切换） |
| DURATION_SLOW     | 450ms  | 复杂动画（展开面板）       |
| SPRING_OVERSHOOT  | 1.1    | Spring 过冲系数            |
| SCALE_HOVER       | 1.05   | 悬停缩放比例               |
| SCALE_PRESS       | 0.95   | 按下缩放比例               |

---

## 相关规格

- `gui` - GUI 基础规格
- `theme-system` - 主题系统（颜色过渡）
