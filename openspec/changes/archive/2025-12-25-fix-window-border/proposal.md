# Change: 修复窗口边框显示问题

## Why

当前窗口边框实现存在以下问题：
1. **训练/分类页面**：边框只在四个圆角处露出一点点，被子控件完全覆盖
2. **设置页面**：侧栏区域的边框被覆盖，只有右侧和底部可见
3. **最大化状态**：边框仍然显示，应该在最大化时隐藏

根本原因：CSS `border` 设置在 `centralWidget` 上，但布局使用 `contentsMargins(0, 0, 0, 0)`，子控件完全贴边覆盖了边框。

## What Changes

### 行业主流做法：设置 contentsMargins

修改布局边距，为边框留出空间：

```python
# 当前（错误）
main_layout.setContentsMargins(0, 0, 0, 0)

# 修复后（正确）
b = self.BORDER_WIDTH
main_layout.setContentsMargins(b, b, b, b)
```

### 最大化状态处理
- 最大化时：移除边框样式，设置 `contentsMargins(0, 0, 0, 0)`
- 还原时：恢复边框样式和边距

## Impact

- Affected code: `src/gui/app.py` (_setup_ui, _toggle_maximize)
- 改动量：约 10 行代码
- 这是 Qt 应用添加边框的标准做法
