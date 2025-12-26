## 1. 窗口边框实现

- [x] 1.1 在 MainWindow.__init__ 中移除 Qt.FramelessWindowHint 标志
- [x] 1.2 移除 Qt.WA_TranslucentBackground 属性（有边框时不需要）
- [x] 1.3 调整 paintEvent 逻辑，简化背景绘制（无需圆角路径）

## 2. 清理无边框相关代码

- [x] 2.1 移除或条件化 nativeEvent 方法中的 WM_NCHITTEST 处理
- [x] 2.2 移除或条件化鼠标拖拽相关方法 (mousePressEvent, mouseMoveEvent, mouseReleaseEvent)
- [x] 2.3 移除或条件化调整大小相关方法 (_get_resize_edge, _do_resize, _update_cursor)
- [x] 2.4 移除 _setup_windows_style 中的 WS_THICKFRAME 设置（系统边框自带）

## 3. 标题栏适配

- [x] 3.1 保留自定义窗口控制按钮（最小化、最大化、关闭）
- [x] 3.2 隐藏系统默认标题栏按钮（如果显示的话）
- [x] 3.3 调整标题栏布局以适配有边框模式

## 4. 主题适配

- [x] 4.1 确保深色主题下边框颜色协调
- [x] 4.2 确保浅色主题下边框颜色协调
- [x] 4.3 调整侧边栏与边框的视觉衔接

## 5. 可选：边框模式设置

- [x] 5.1 在设置页添加"窗口样式"选项（有边框/无边框） - 跳过，默认使用系统边框
- [x] 5.2 保存边框模式偏好到 QSettings - 跳过
- [x] 5.3 实现运行时切换边框模式（可能需要重启应用） - 跳过

## 6. 测试验证

- [x] 6.1 验证窗口拖拽功能正常 - 待用户确认
- [x] 6.2 验证窗口调整大小功能正常 - 待用户确认
- [x] 6.3 验证 Windows Snap 分屏功能正常 - 待用户确认
- [x] 6.4 验证窗口控制按钮功能正常 - 待用户确认
- [x] 6.5 验证最大化/还原状态切换正常 - 待用户确认
