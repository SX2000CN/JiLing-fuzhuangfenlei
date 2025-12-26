# Change: Fix High DPI Rendering for Sharp Text and Icons

## Why

当前应用在高 DPI 显示器上字体和图标显示模糊、有锯齿。与微信等原生应用相比，即使字体和图标同样小，微信的渲染清晰锐利，而本应用存在明显的模糊问题。

根本原因分析：
1. **`QT_SCALE_FACTOR = "1"` 强制禁用了缩放** - 这导致应用以 1x 分辨率渲染后被系统放大，产生模糊
2. **SVG 图标使用固定 32x32 像素渲染** - 未考虑设备像素比（devicePixelRatio），在 2x 或 3x 显示器上图标模糊
3. **字体渲染未启用抗锯齿优化** - 缺少 `AA_UseDesktopOpenGL` 或字体平滑设置

## What Changes

- **移除 `QT_SCALE_FACTOR = "1"` 限制** - 让 Qt 自动处理缩放
- **SVG 图标按设备像素比渲染** - `pixmap = QPixmap(size * devicePixelRatio)`，并设置 `pixmap.setDevicePixelRatio()`
- **启用高质量文本渲染** - 添加 `Qt.TextAntialiasing` 和渲染提示

## Impact

- Affected specs: gui (新建)
- Affected code:
  - `src/gui/native_ui.py` - main() 函数的 DPI 设置
  - `src/gui/native_ui.py` - `_create_svg_icon()` 方法（多处）
  - `src/gui/native_ui.py` - FontManager 类
