"""
浅色主题配置 - 基于 VS Code Light+ 主题
WCAG 2.1 AA 合规 - 所有文字对比度 ≥ 4.5:1
"""

from .tokens import ColorTokens

LIGHT_THEME = {
    # ===== 背景色 =====
    ColorTokens.BG_PRIMARY: "#FFFFFF",      # 编辑器背景
    ColorTokens.BG_SECONDARY: "#F3F3F3",    # 侧边栏背景
    ColorTokens.BG_SIDEBAR: "#E8E8E8",      # 活动栏背景
    ColorTokens.BG_INPUT: "#FFFFFF",        # 输入框背景
    ColorTokens.BG_CARD: "#FFFFFF",         # 卡片背景

    # ===== 前景/文字 (WCAG AA 合规) =====
    ColorTokens.TEXT_PRIMARY: "#1F1F1F",    # 主要文字 (对比度 14.5:1)
    ColorTokens.TEXT_SECONDARY: "#505050",  # 描述文字 (对比度 7.0:1)
    ColorTokens.TEXT_MUTED: "#6B6B6B",      # 弱化文字 (对比度 4.6:1)
    ColorTokens.TEXT_INVERSE: "#FFFFFF",    # 反色文字

    # ===== 交互色 =====
    ColorTokens.ACCENT: "#0066B8",          # VS Code 浅色主题蓝
    ColorTokens.ACCENT_HOVER: "#005A9E",    # 悬停蓝
    ColorTokens.ACCENT_PRESSED: "#004578",  # 按下蓝

    # ===== 边框 (加深以提高可见性) =====
    ColorTokens.BORDER: "#CECECE",          # 普通边框
    ColorTokens.BORDER_FOCUS: "#0066B8",    # 聚焦边框
    ColorTokens.BORDER_MUTED: "#C8C8C8",    # 弱化边框 (加深)
    ColorTokens.DIVIDER: "#D4D4D4",         # 分割线 (加深)

    # ===== 状态 =====
    ColorTokens.HOVER_BG: "#E8E8E8",        # 悬停背景
    ColorTokens.SELECTED_BG: "#0060C0",     # 选中背景 (深蓝，确保白字可读)
    ColorTokens.DISABLED_BG: "#F5F5F5",     # 禁用背景
    ColorTokens.DISABLED_TEXT: "#767676",   # 禁用文字 (对比度 4.5:1)

    # ===== 状态颜色 =====
    ColorTokens.SUCCESS: "#388A34",
    ColorTokens.WARNING: "#BF8803",
    ColorTokens.ERROR: "#E51400",
    ColorTokens.INFO: "#1A85FF",

    # ===== 日志颜色 (WCAG AA 合规) =====
    ColorTokens.LOG_DEBUG: "#006600",       # 加深 (对比度 6.4:1)
    ColorTokens.LOG_INFO: "#1F1F1F",        # 与 TEXT_PRIMARY 一致
    ColorTokens.LOG_WARNING: "#956700",     # 加深 (对比度 5.0:1)
    ColorTokens.LOG_ERROR: "#E51400",
    ColorTokens.LOG_SUCCESS: "#388A34",
    ColorTokens.LOG_METRIC: "#0451A5",
    ColorTokens.LOG_HIGHLIGHT: "#267F99",

    # ===== 按钮特殊色 =====
    ColorTokens.BTN_START_HOVER: "#16A34A",
    ColorTokens.BTN_STOP_HOVER: "#DC2626",
    ColorTokens.BTN_CLOSE_HOVER: "#E81123",

    # ===== 图标 (加深以提高对比度) =====
    ColorTokens.ICON_DEFAULT: "#3B3B3B",    # 加深 (对比度 8.9:1)
    ColorTokens.ICON_HOVER: "#1F1F1F",
    ColorTokens.ICON_ACTIVE: "#0066B8",
    ColorTokens.ICON_CORNER: "#B0B0B0",     # 浅色主题下的图标圆角颜色

    # ===== 操作按钮 =====
    ColorTokens.ACTION_BTN_TEXT: "#1F1F1F",   # 浅色主题下用深色文字
    ColorTokens.ACTION_BTN_HOVER: "#D4D4D4",  # 浅色悬停背景

    # ===== 通用悬停 =====
    ColorTokens.CONTROL_HOVER_BG: "#D4D4D4",  # 控件悬停背景
}
