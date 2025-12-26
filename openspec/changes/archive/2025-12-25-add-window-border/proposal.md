# Change: 为窗口添加系统边框

## Why

当前使用无边框 (`FramelessWindowHint`) 窗口设计，虽然视觉上简洁，但存在以下问题：
1. 需要手动实现窗口拖拽、调整大小等基础功能
2. Windows Snap 分屏功能需要额外的原生消息处理
3. 右上角控制按钮区域需要特殊处理避免与标题栏拖拽冲突
4. 维护成本较高，容易出现边缘情况 bug

添加系统边框可以简化窗口管理逻辑，同时保持现代化的界面风格。

## What Changes

### 窗口样式
- 移除 `Qt.FramelessWindowHint` 标志
- 保留自定义标题栏内容（窗口控制按钮）
- 使用系统原生边框进行窗口拖拽和调整大小
- 可选：在设置中提供"无边框模式"开关

### 代码简化
- 移除 `nativeEvent` 中的 WM_NCHITTEST 处理（边框模式下）
- 移除手动实现的 `mousePressEvent`、`mouseMoveEvent`、`mouseReleaseEvent` 拖拽逻辑（边框模式下）
- 移除 `_get_resize_edge`、`_do_resize`、`_update_cursor` 等调整大小相关方法（边框模式下）

### 视觉调整
- 调整圆角策略：系统边框下使用系统圆角或方角
- 调整背景绘制逻辑，适配有边框模式
- 确保深色/浅色主题下边框颜色协调

## Impact

- Affected specs: window-style (新增)
- Affected code: `src/gui/app.py` (MainWindow 类)
- 用户体验：原生窗口行为更可靠，Windows Snap 自动支持
