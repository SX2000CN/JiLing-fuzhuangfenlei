# Design: add-interaction-animations

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                      AnimationManager                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ EasingCurves│  │ AnimationPool│  │ AnimationPresets   │  │
│  │ - spring    │  │ - 复用动画   │  │ - fadeIn/Out      │  │
│  │ - easeOut   │  │ - 限制并发   │  │ - slideIn/Out     │  │
│  │ - bounce    │  │              │  │ - scalePress      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ AnimatedButton│    │AnimatedCheckBox│   │AnimatedPage   │
│ - hover scale │    │ - check anim  │    │ - slide/fade  │
│ - press scale │    │ - color trans │    │ - stack trans │
└───────────────┘    └───────────────┘    └───────────────┘
```

## 核心组件

### 1. AnimationManager (单例)

```python
class AnimationManager:
    """动画管理器 - 统一管理所有动画"""

    # 全局配置
    enabled: bool = True
    duration_multiplier: float = 1.0  # 动画时长倍率

    # 预设时长 (macOS 风格)
    DURATION_FAST = 150      # 快速反馈
    DURATION_NORMAL = 300    # 标准过渡
    DURATION_SLOW = 450      # 复杂动画

    # 方法
    def create_animation(widget, property, ...) -> QPropertyAnimation
    def create_spring_curve() -> QEasingCurve
    def fade_in(widget, duration) -> QPropertyAnimation
    def fade_out(widget, duration) -> QPropertyAnimation
    def scale_press(widget) -> QPropertyAnimation
    def slide_in(widget, direction) -> QPropertyAnimation
```

### 2. Spring 缓动曲线

macOS 风格的弹性缓动，使用自定义 QEasingCurve：

```python
def create_spring_curve(stiffness=100, damping=10):
    """创建 spring 缓动曲线"""
    curve = QEasingCurve(QEasingCurve.Custom)
    # 使用 bezier 近似 spring 效果
    # 或使用 QEasingCurve.OutBack 作为简化版本
    return curve
```

参数说明：
- `stiffness`: 弹簧刚度，值越大弹性越强
- `damping`: 阻尼系数，值越大衰减越快

### 3. 动画类型定义

| 动画名称 | 属性 | 时长 | 缓动 | 用途 |
|---------|------|------|------|------|
| `fadeIn` | opacity | 300ms | easeOut | 元素显示 |
| `fadeOut` | opacity | 200ms | easeIn | 元素隐藏 |
| `scalePress` | scale | 100ms | easeOut | 按钮按下 |
| `scaleRelease` | scale | 300ms | spring | 按钮释放 |
| `slideLeft` | x | 300ms | spring | 页面切换 |
| `slideRight` | x | 300ms | spring | 页面切换 |
| `expandWidth` | width | 300ms | spring | 面板展开 |
| `collapseWidth` | width | 250ms | easeOut | 面板收起 |
| `colorTransition` | color | 300ms | linear | 主题切换 |
| `checkMark` | path | 200ms | easeOut | 复选框勾选 |

## 组件改造

### SidebarButton 动画

```
状态流转:
  Normal ──hover──> Hovered ──press──> Pressed ──release──> Hovered
    │                  │                                        │
    │<────leave────────┘                                        │
    │<──────────────────────────────────────────────────────────┘

动画效果:
  - hover: 背景色渐变 (300ms), 图标缩放 1.0 -> 1.05 (300ms spring)
  - press: 整体缩放 1.0 -> 0.95 (100ms easeOut)
  - release: 整体缩放 0.95 -> 1.0 (300ms spring)
```

### 页面切换动画

```
切换流程:
  1. 当前页面 fadeOut + slideOut (并行)
  2. 新页面 fadeIn + slideIn (并行)

方向:
  - 向右切换 (index 增加): 当前页左滑出, 新页从右滑入
  - 向左切换 (index 减少): 当前页右滑出, 新页从左滑入
```

### 复选框动画

```
勾选动画:
  1. 背景色渐变 (150ms)
  2. 勾选标记路径动画 (200ms easeOut)
     - 从起点到终点逐渐绘制

取消勾选:
  1. 勾选标记淡出 (100ms)
  2. 背景色渐变 (150ms)
```

### 进度条动画

```
数值变化:
  - 使用 QPropertyAnimation 平滑过渡 value 属性
  - 时长: 300ms
  - 缓动: easeOut

完成效果:
  - 100% 时闪烁高亮 (可选)
```

## 性能考虑

1. **动画池复用**: 避免频繁创建/销毁动画对象
2. **并发限制**: 同一组件最多 3 个并行动画
3. **帧率控制**: 目标 60fps，低配设备降级到 30fps
4. **禁用选项**: 设置中提供"减少动画"开关

## 文件结构

```
src/gui/
├── animations/
│   ├── __init__.py
│   ├── manager.py          # AnimationManager
│   ├── easing.py           # 自定义缓动曲线
│   └── presets.py          # 动画预设
├── components/
│   ├── buttons.py          # 添加动画支持
│   ├── checkbox.py         # 添加动画支持
│   └── ...
└── pages/
    └── ...                 # 页面切换动画
```

## 兼容性

- PySide6 6.0+
- Windows 10/11, macOS 10.15+
- 最低配置: 4GB RAM, 集成显卡
