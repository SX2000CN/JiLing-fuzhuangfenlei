## Context

本项目使用 PySide6 (Qt for Python) 构建 GUI 界面。在 Windows 高 DPI 显示器（如 125%、150%、200% 缩放）上，当前实现导致文字和图标模糊。

### 问题代码位置

1. `native_ui.py:3856` - `os.environ["QT_SCALE_FACTOR"] = "1"` 强制缩放为 1
2. `native_ui.py:1714` - `pixmap = QPixmap(32, 32)` 固定像素尺寸
3. `native_ui.py:2279` - 另一处 `_create_svg_icon` 同样问题

## Goals / Non-Goals

### Goals
- 在任意 DPI 缩放设置下，文字清晰无锯齿
- SVG 图标锐利，无模糊边缘
- 与微信、VS Code 等原生应用渲染质量相当

### Non-Goals
- 不修改现有布局逻辑
- 不更换字体或图标设计
- 不支持运行时动态 DPI 切换（重启应用即可）

## Decisions

### 决策 1: 移除强制缩放因子

**做法**: 删除 `os.environ["QT_SCALE_FACTOR"] = "1"`

**原因**: 这行代码强制 Qt 以 100% 缩放渲染，然后由 Windows 进行位图放大，导致模糊。移除后 Qt 可以直接以正确的 DPI 渲染。

### 决策 2: SVG 图标按 devicePixelRatio 渲染

**做法**:
```python
def _create_svg_icon(self, svg_template: str, color: str, size: int = 32) -> QIcon:
    svg_content = svg_template.replace("{color}", color)
    renderer = QSvgRenderer(QByteArray(svg_content.encode()))

    # 获取设备像素比
    dpr = QApplication.primaryScreen().devicePixelRatio() if QApplication.primaryScreen() else 1.0

    # 创建高分辨率 pixmap
    pixmap = QPixmap(int(size * dpr), int(size * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    renderer.render(painter)
    painter.end()

    return QIcon(pixmap)
```

**原因**: 在 2x 显示器上，32x32 逻辑像素需要 64x64 物理像素才能锐利显示。`setDevicePixelRatio()` 告诉 Qt 这是高分辨率资源。

### 决策 3: 启用渲染提示

**做法**: 在绘制时添加渲染提示
```python
painter.setRenderHint(QPainter.Antialiasing)
painter.setRenderHint(QPainter.SmoothPixmapTransform)
painter.setRenderHint(QPainter.TextAntialiasing)
```

**原因**: 确保抗锯齿处理，平滑边缘。

### 备选方案考虑

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| 使用 QSvgWidget | 自动处理 DPI | 不适合按钮图标 | 不采用 |
| 使用 QIcon.fromTheme | 系统集成好 | 需要主题支持 | 不采用 |
| 当前方案（修正 pixmap） | 最小改动，完全可控 | 需要遍历修改 | 采用 |

## Risks / Trade-offs

- **风险**: 不同 Windows 版本的 DPI 处理可能有差异
  - **缓解**: 使用 Qt 官方推荐的 `AA_EnableHighDpiScaling` 属性，兼容性最佳

- **风险**: 首次渲染可能略慢（生成更大的 pixmap）
  - **缓解**: 图标数量有限，影响可忽略

## Migration Plan

无需迁移，直接修改代码即可。向后兼容。

## Open Questions

无
