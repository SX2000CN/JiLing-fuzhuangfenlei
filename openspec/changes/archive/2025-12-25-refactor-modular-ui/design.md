# Design: 模块化 UI 架构

## Context

当前 `native_ui.py` 是一个约 3500 行的单文件，包含：
- 2 个 Worker 类 (后台任务)
- 3 个常量/管理器类 (LayoutConstants, FontManager, StyleSheet)
- 1 个图标类 (IconSvg)
- 14 个 UI 组件类
- 1 个主窗口类 (MainWindow)

问题：
1. 修改一处样式需要在大文件中搜索，容易改错位置
2. 颜色硬编码在 StyleSheet 类中，无法切换主题
3. 高亮边距问题难以定位（涉及多层嵌套的 margin/padding）

## Goals / Non-Goals

### Goals
- 将代码按职责拆分为独立模块
- 实现主题系统，支持深色/浅色模式切换
- 修复设置页高亮边距问题
- 保持功能完全一致，用户无感知

### Non-Goals
- 不改变现有 UI 外观设计
- 不添加新功能
- 不修改后端逻辑

## Decisions

### 1. 目录结构

```
src/gui/
├── __init__.py              # 导出 main() 和 MainWindow
├── app.py                   # QApplication 配置和启动
├── main_window.py           # MainWindow 类
│
├── theme/                   # 主题系统
│   ├── __init__.py
│   ├── manager.py           # ThemeManager 单例
│   ├── tokens.py            # 颜色 token 定义
│   ├── dark.py              # 深色主题颜色值
│   ├── light.py             # 浅色主题颜色值
│   └── styles.py            # StyleSheet 生成器
│
├── components/              # 基础 UI 组件
│   ├── __init__.py
│   ├── buttons.py           # SidebarButton
│   ├── checkbox.py          # VSCheckBox
│   ├── inputs.py            # ComboRow, FileRow, SliderRow
│   ├── cards.py             # ParamCard, ParamRow, RoundedContainer
│   ├── settings.py          # SettingsRow, SettingsCheckRow
│   ├── terminal.py          # TerminalOutput
│   └── window_controls.py   # WindowControlBar
│
├── widgets/                 # 复合组件
│   ├── __init__.py
│   ├── gpu_status.py        # GPUStatusWidget
│   └── result_table.py      # ClassificationResultTable
│
├── pages/                   # 页面视图
│   ├── __init__.py
│   ├── training.py          # 训练页面
│   ├── classification.py    # 分类页面
│   └── settings.py          # 设置页面
│
├── workers/                 # 后台任务
│   ├── __init__.py
│   ├── training.py          # TrainingWorker
│   └── classification.py    # ClassificationWorker
│
└── utils/                   # 工具类
    ├── __init__.py
    ├── constants.py         # LayoutConstants
    ├── fonts.py             # FontManager
    └── icons.py             # IconSvg
```

### 2. 主题系统设计

**Token-Based 颜色系统**：
- 定义语义化的颜色 token（如 `background.primary`, `text.foreground`）
- 深色/浅色主题分别提供 token 值
- 组件通过 token 引用颜色，不直接使用硬编码值

```python
# tokens.py - 语义化 token 定义
class ColorTokens:
    # 背景
    BG_PRIMARY = "bg.primary"
    BG_SECONDARY = "bg.secondary"
    BG_SIDEBAR = "bg.sidebar"
    BG_INPUT = "bg.input"

    # 前景/文字
    TEXT_PRIMARY = "text.primary"
    TEXT_SECONDARY = "text.secondary"
    TEXT_MUTED = "text.muted"

    # 交互
    ACCENT = "accent"
    ACCENT_HOVER = "accent.hover"
    BORDER = "border"
    BORDER_FOCUS = "border.focus"

    # 状态
    HOVER_BG = "hover.bg"
    SELECTED_BG = "selected.bg"
```

```python
# dark.py - 深色主题
DARK_THEME = {
    ColorTokens.BG_PRIMARY: "#1E1E1E",
    ColorTokens.BG_SECONDARY: "#252526",
    ColorTokens.BG_SIDEBAR: "#333333",
    ColorTokens.TEXT_PRIMARY: "#CCCCCC",
    ColorTokens.ACCENT: "#007FD4",
    ColorTokens.HOVER_BG: "rgba(90, 93, 94, 0.31)",
    # ...
}
```

```python
# light.py - 浅色主题
LIGHT_THEME = {
    ColorTokens.BG_PRIMARY: "#FFFFFF",
    ColorTokens.BG_SECONDARY: "#F3F3F3",
    ColorTokens.BG_SIDEBAR: "#E8E8E8",
    ColorTokens.TEXT_PRIMARY: "#333333",
    ColorTokens.ACCENT: "#0066B8",
    ColorTokens.HOVER_BG: "rgba(0, 0, 0, 0.04)",
    # ...
}
```

**ThemeManager 单例**：
```python
class ThemeManager:
    _instance = None

    def __init__(self):
        self._current_theme = "dark"
        self._themes = {"dark": DARK_THEME, "light": LIGHT_THEME}
        self.theme_changed = Signal(str)

    def get_color(self, token: str) -> str:
        return self._themes[self._current_theme][token]

    def set_theme(self, theme: str):
        self._current_theme = theme
        self.theme_changed.emit(theme)
```

### 3. 高亮边距问题修复

**问题根因**：
- `scroll_layout` 的 margin 和 `SettingsRow` 的 margin 混淆
- 高亮背景填充整个 widget，但文字的 padding 独立计算

**修复方案**：
- `scroll_layout`: 只负责外部间距 `(12, 0, 12, 24)`
- `SettingsRow`: 内部使用 padding 而非 margin
- 高亮效果通过 `paintEvent` 绘制，考虑 padding

```python
class SettingsRow(QWidget):
    def __init__(self, ...):
        # 使用内边距，让高亮区域与文字有间距
        self._padding = (12, 10, 12, 10)  # left, top, right, bottom

    def paintEvent(self, event):
        if self._hovered:
            painter = QPainter(self)
            # 绘制高亮背景，留出 padding
            rect = self.rect().adjusted(0, 0, 0, 0)  # 整个 widget
            painter.fillRect(rect, QColor(theme.get_color(HOVER_BG)))
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| 重构期间引入 bug | 分阶段重构，每阶段验证功能 |
| 主题切换性能 | 只刷新可见组件，使用缓存 |
| 导入路径变化 | 在 `__init__.py` 保持兼容导出 |

## Migration Plan

1. **Phase 1**: 创建目录结构，提取工具类和常量
2. **Phase 2**: 提取主题系统，保持深色模式
3. **Phase 3**: 提取基础组件
4. **Phase 4**: 提取页面和主窗口
5. **Phase 5**: 添加浅色模式
6. **Phase 6**: 删除原 `native_ui.py`

回滚：保留原文件备份，直到新代码稳定运行。

## Open Questions

- [ ] 浅色模式的具体配色方案需要确认
- [ ] 是否需要支持"跟随系统"自动切换主题？
