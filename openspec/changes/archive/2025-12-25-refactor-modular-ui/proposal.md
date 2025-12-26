# Change: 重构 UI 为模块化架构并添加主题系统

## Why

当前 `native_ui.py` 文件超过 3500 行，包含 20+ 个类，维护困难且修改容易出错。高亮边距等样式问题难以定位和修复。同时只支持深色模式，缺乏浅色模式支持。

## What Changes

### 架构重构
- 将单一 `native_ui.py` 文件拆分为模块化目录结构
- 按职责分离：主题、组件、页面、工具类、后台任务
- 每个模块独立可测试、可维护

### 主题系统
- 新增 `ThemeManager` 主题管理器，支持运行时切换主题
- 实现深色模式 (Dark) 和浅色模式 (Light)
- 颜色和样式表与主题解耦，通过 token 引用

### 样式修复
- 修复设置页高亮边距问题：使用正确的 padding 和 margin 分离
- 统一组件样式规范，避免内联样式覆盖

## Impact

- Affected specs: ui-architecture (新增), theme-system (新增)
- Affected code: `src/gui/native_ui.py` -> `src/gui/` 目录
- **BREAKING**: 删除原 `native_ui.py`，替换为模块化结构
