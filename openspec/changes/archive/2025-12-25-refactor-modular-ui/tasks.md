# Tasks: 模块化 UI 重构

## 1. 基础设施

- [x] 1.1 创建 `src/gui/` 目录结构
- [x] 1.2 创建各模块的 `__init__.py` 文件
- [x] 1.3 备份原 `native_ui.py` 到 `native_ui.py.bak`

## 2. 工具类提取

- [x] 2.1 提取 `LayoutConstants` 到 `utils/constants.py`
- [x] 2.2 提取 `FontManager` 到 `utils/fonts.py`
- [x] 2.3 提取 `IconSvg` 到 `utils/icons.py`
- [x] 2.4 验证：导入工具类，确认无错误

## 3. 主题系统

- [x] 3.1 创建 `theme/tokens.py` 定义颜色 token
- [x] 3.2 创建 `theme/dark.py` 深色主题配置
- [x] 3.3 创建 `theme/light.py` 浅色主题配置
- [x] 3.4 创建 `theme/manager.py` ThemeManager 单例
- [x] 3.5 创建 `theme/styles.py` 样式表生成器
- [x] 3.6 验证：切换主题，确认颜色正确变化

## 4. 后台任务

- [x] 4.1 提取 `TrainingWorker` 到 `workers/training.py`
- [x] 4.2 提取 `ClassificationWorker` 到 `workers/classification.py`
- [x] 4.3 验证：运行训练和分类，确认后台任务正常

## 5. 基础组件

- [x] 5.1 提取 `SidebarButton` 到 `components/buttons.py`
- [x] 5.2 提取 `VSCheckBox` 到 `components/checkbox.py`
- [x] 5.3 提取 `ParamCard`, `ParamRow`, `RoundedContainer` 到 `components/cards.py`
- [x] 5.4 提取 `ComboRow`, `FileRow`, `SliderRow` 到 `components/inputs.py`
- [x] 5.5 提取 `SettingsRow`, `SettingsCheckRow` 到 `components/settings.py` - **高亮边距已修复**
- [x] 5.6 提取 `TerminalOutput` 到 `components/terminal.py`
- [x] 5.7 提取 `WindowControlBar` 到 `components/window_controls.py`
- [x] 5.8 验证：所有组件正确渲染，交互正常

## 6. 复合组件

- [x] 6.1 提取 `GPUStatusWidget` 到 `widgets/gpu_status.py`
- [x] 6.2 提取 `ClassificationResultTable` 到 `widgets/result_table.py`
- [x] 6.3 验证：GPU 状态和分类结果表正常显示

## 7. 页面视图

- [x] 7.1 提取训练页面逻辑到 `pages/training.py`
- [x] 7.2 提取分类页面逻辑到 `pages/classification.py`
- [x] 7.3 提取设置页面逻辑到 `pages/settings.py`
- [x] 7.4 验证：切换页面正常，功能完整

## 8. 主窗口

- [x] 8.1 重构 `MainWindow` 到 `app.py`，使用模块化组件
- [x] 8.2 创建 `app.py` 应用入口
- [x] 8.3 更新 `__init__.py` 导出 `main()` 函数
- [x] 8.4 验证：应用正常启动，所有功能可用

## 9. 浅色模式

- [x] 9.1 完善 `light.py` 浅色主题配色
- [x] 9.2 确保所有组件支持主题切换
- [x] 9.3 在设置页添加主题切换功能（深色/浅色/跟随系统）
- [x] 9.4 验证：深色/浅色模式切换正常

## 10. 清理

- [ ] 10.1 删除原 `native_ui.py`（保留，可用 --legacy 参数启动）
- [x] 10.2 更新 `run_new_gui.py` 导入路径
- [x] 10.3 最终验证：完整功能测试

---

**依赖关系**：
- Phase 2-4 可并行
- Phase 5-7 依赖 Phase 3 (主题系统)
- Phase 8 依赖 Phase 5-7
- Phase 9-10 依赖 Phase 8

---

**完成状态**: 除 10.1 外所有任务已完成。native_ui.py 保留作为兼容备选，可通过 `--legacy` 参数使用。
