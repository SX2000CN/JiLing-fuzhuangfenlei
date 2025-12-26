"""
深色主题配置 - 基于 VS Code Dark+ 主题
"""

from .tokens import ColorTokens

DARK_THEME = {
    # ===== 背景色 =====
    ColorTokens.BG_PRIMARY: "#1E1E1E",      # 编辑器背景
    ColorTokens.BG_SECONDARY: "#252526",    # 侧边栏背景
    ColorTokens.BG_SIDEBAR: "#333333",      # 活动栏背景
    ColorTokens.BG_INPUT: "#3C3C3C",        # 输入框背景
    ColorTokens.BG_CARD: "#1E1E1E",         # 卡片背景

    # ===== 前景/文字 =====
    ColorTokens.TEXT_PRIMARY: "#CCCCCC",    # 主要文字
    ColorTokens.TEXT_SECONDARY: "#9D9D9D",  # 描述文字
    ColorTokens.TEXT_MUTED: "#6D6D6D",      # 弱化文字
    ColorTokens.TEXT_INVERSE: "#FFFFFF",    # 反色文字

    # ===== 交互色 =====
    ColorTokens.ACCENT: "#007FD4",          # VS Code 蓝
    ColorTokens.ACCENT_HOVER: "#1177BB",    # 悬停蓝
    ColorTokens.ACCENT_PRESSED: "#0E639C",  # 按下蓝

    # ===== 边框 =====
    ColorTokens.BORDER: "#505050",          # 普通边框
    ColorTokens.BORDER_FOCUS: "#007FD4",    # 聚焦边框
    ColorTokens.BORDER_MUTED: "#454545",    # 弱化边框/分割线
    ColorTokens.DIVIDER: "#2D2D2D",         # 分割线

    # ===== 状态 =====
    ColorTokens.HOVER_BG: "#2A2D2E",              # 悬停背景（VS Code 标准）
    ColorTokens.SELECTED_BG: "#094771",           # 选中背景
    ColorTokens.DISABLED_BG: "#2D2D2D",     # 禁用背景
    ColorTokens.DISABLED_TEXT: "#5D5D5D",   # 禁用文字

    # ===== 状态颜色 =====
    ColorTokens.SUCCESS: "#89D185",
    ColorTokens.WARNING: "#CCA700",
    ColorTokens.ERROR: "#F48771",
    ColorTokens.INFO: "#75BEFF",

    # ===== 日志颜色 =====
    ColorTokens.LOG_DEBUG: "#6A9955",
    ColorTokens.LOG_INFO: "#E5E5E5",
    ColorTokens.LOG_WARNING: "#CCA700",
    ColorTokens.LOG_ERROR: "#F48771",
    ColorTokens.LOG_SUCCESS: "#89D185",
    ColorTokens.LOG_METRIC: "#9CDCFE",
    ColorTokens.LOG_HIGHLIGHT: "#4EC9B0",

    # ===== 按钮特殊色 =====
    ColorTokens.BTN_START_HOVER: "#13C468",
    ColorTokens.BTN_STOP_HOVER: "#C42B1C",
    ColorTokens.BTN_CLOSE_HOVER: "#E81123",

    # ===== 图标 =====
    ColorTokens.ICON_DEFAULT: "#C5C5C5",
    ColorTokens.ICON_HOVER: "#FFFFFF",
    ColorTokens.ICON_ACTIVE: "#FFFFFF",
    ColorTokens.ICON_CORNER: "#818181",     # 深色主题下的图标圆角颜色

    # ===== 操作按钮 =====
    ColorTokens.ACTION_BTN_TEXT: "#FFFFFF",   # 深色主题下用白色文字
    ColorTokens.ACTION_BTN_HOVER: "#5A5A5A",  # 深色悬停背景

    # ===== 通用悬停 =====
    ColorTokens.CONTROL_HOVER_BG: "#3E3E3E",  # 控件悬停背景
}
