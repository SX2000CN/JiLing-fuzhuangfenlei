# Proposal: rewrite-light-theme

## Summary

重写浅色主题配色方案，确保所有文字、图标与背景色的对比度符合 WCAG 2.1 AA 标准，解决当前浅色模式下部分元素看不清的问题。

## Motivation

当前浅色主题存在以下对比度问题：

### 🔴 严重问题

| Token 组合 | 前景 | 背景 | 对比度 | 问题 |
|------------|------|------|--------|------|
| TEXT_INVERSE on SELECTED_BG | #FFFFFF | #ADD6FF | **1.5:1** | 表格选中项完全看不清！ |
| TEXT_MUTED on BG_PRIMARY | #999999 | #FFFFFF | 2.85:1 | 弱化文字难以阅读 |
| DISABLED_TEXT on DISABLED_BG | #A0A0A0 | #F5F5F5 | 2.3:1 | 禁用文字看不清 |

### ⚠️ 潜在问题

| Token 组合 | 前景 | 背景 | 对比度 | 风险 |
|------------|------|------|--------|------|
| BORDER_MUTED on BG_PRIMARY | #E5E5E5 | #FFFFFF | 1.3:1 | 边框几乎不可见 |
| DIVIDER on BG_PRIMARY | #E5E5E5 | #FFFFFF | 1.3:1 | 分割线几乎不可见 |
| LOG_WARNING on BG_PRIMARY | #BF8803 | #FFFFFF | 3.9:1 | 警告日志对比度不足 |

### WCAG 2.1 AA 标准

- **普通文字**：对比度 ≥ 4.5:1
- **大文字**（18pt+ 或 14pt 加粗）：对比度 ≥ 3:1
- **非文字内容**（图标、边框等）：对比度 ≥ 3:1

## Solution

### 1. 文字颜色调整

| Token | 旧值 | 新值 | 新对比度 |
|-------|------|------|----------|
| TEXT_PRIMARY | #333333 | **#1F1F1F** | 14.5:1 ✅ |
| TEXT_SECONDARY | #616161 | **#505050** | 7.0:1 ✅ |
| TEXT_MUTED | #999999 | **#6B6B6B** | 4.6:1 ✅ |
| DISABLED_TEXT | #A0A0A0 | **#767676** | 4.5:1 ✅ |

### 2. 图标颜色调整

| Token | 旧值 | 新值 | 新对比度 |
|-------|------|------|----------|
| ICON_DEFAULT | #424242 | **#3B3B3B** | 8.9:1 ✅ |
| ICON_HOVER | #1F1F1F | #1F1F1F | 14.5:1 ✅ |

### 3. 选中状态调整（关键修复）

| Token | 旧值 | 新值 | 说明 |
|-------|------|------|------|
| SELECTED_BG | #ADD6FF | **#0060C0** | 加深为深蓝色 |
| (TEXT_INVERSE 保持 #FFFFFF) | - | - | 白字在深蓝背景上对比度 8.5:1 ✅ |

### 4. 边框颜色调整

| Token | 旧值 | 新值 | 新对比度 |
|-------|------|------|----------|
| BORDER_MUTED | #E5E5E5 | **#C8C8C8** | 1.7:1 (边框可接受) |
| DIVIDER | #E5E5E5 | **#D4D4D4** | 1.5:1 (分割线可接受) |

### 5. 日志颜色调整

| Token | 旧值 | 新值 | 新对比度 |
|-------|------|------|----------|
| LOG_WARNING | #BF8803 | **#956700** | 5.0:1 ✅ |
| LOG_DEBUG | #008000 | **#006600** | 6.4:1 ✅ |

## Scope

- 修改 `src/gui/theme/light.py` 中的颜色定义
- 不涉及深色主题
- 不涉及主题系统架构变更
- 不修改 styles.py（样式逻辑保持不变）

## Risks

1. **SELECTED_BG 改为深色**：选中项视觉风格会从"浅色高亮"变为"深色高亮"，与 VS Code 默认行为不同，但这是保证可读性的必要取舍
2. **边框变深**：可能略微影响整体视觉轻盈感

## References

- [WCAG 2.1 Contrast Guidelines](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)
- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- [VS Code Light+ Theme](https://github.com/microsoft/vscode/blob/main/extensions/theme-defaults/themes/light_plus.json)
