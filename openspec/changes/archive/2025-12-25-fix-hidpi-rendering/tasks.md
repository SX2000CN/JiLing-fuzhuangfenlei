## 1. Fix DPI Initialization

- [x] 1.1 在 `native_ui.py:main()` 中删除 `os.environ["QT_SCALE_FACTOR"] = "1"`
- [x] 1.2 确保 `AA_EnableHighDpiScaling` 和 `AA_UseHighDpiPixmaps` 在 QApplication 创建前设置

## 2. Fix SVG Icon Rendering

- [x] 2.1 修改 `WindowControlButtons._create_svg_icon()` (约 line 1710) 以支持 devicePixelRatio
- [x] 2.2 修改 `MainWindow._create_svg_icon()` (约 line 2284) 使用相同逻辑
- [x] 2.3 (不适用) ClassificationWindow 和 SettingsWindow 共用 MainWindow 的 _create_svg_icon
- [x] 2.4 (不适用) 同上
- [x] 2.5 修改 `SidebarButton._update_icon()` 方法以支持高 DPI 渲染

## 3. Add Render Hints

- [x] 3.1 在所有 QPainter 绑定处添加 `setRenderHint(QPainter.Antialiasing)`
- [x] 3.2 添加 `setRenderHint(QPainter.SmoothPixmapTransform)` 用于图像缩放
- [x] 3.3 MainWindow paintEvent 已有 TextAntialiasing

## 4. Testing

- [ ] 4.1 在 100% 缩放下测试应用，验证布局正常
- [ ] 4.2 在 125% 缩放下测试，验证文字和图标清晰
- [ ] 4.3 在 150% 缩放下测试，验证文字和图标清晰
- [ ] 4.4 对比微信或其他应用，确认渲染质量相当
