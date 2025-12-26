"""
颜色 Token 定义 - 语义化的颜色常量

使用 Token 而非硬编码颜色值，便于主题切换
"""


class ColorTokens:
    """语义化颜色 Token"""

    # ===== 背景色 =====
    BG_PRIMARY = "bg.primary"           # 主背景（编辑器区域）
    BG_SECONDARY = "bg.secondary"       # 次要背景（侧边栏）
    BG_SIDEBAR = "bg.sidebar"           # 活动栏背景
    BG_INPUT = "bg.input"               # 输入框背景
    BG_CARD = "bg.card"                 # 卡片背景

    # ===== 前景/文字 =====
    TEXT_PRIMARY = "text.primary"       # 主要文字
    TEXT_SECONDARY = "text.secondary"   # 次要文字（描述）
    TEXT_MUTED = "text.muted"           # 弱化文字
    TEXT_INVERSE = "text.inverse"       # 反色文字（按钮上的白字）

    # ===== 交互色 =====
    ACCENT = "accent"                   # 强调色/主色
    ACCENT_HOVER = "accent.hover"       # 强调色悬停
    ACCENT_PRESSED = "accent.pressed"   # 强调色按下

    # ===== 边框 =====
    BORDER = "border"                   # 普通边框
    BORDER_FOCUS = "border.focus"       # 聚焦边框
    BORDER_MUTED = "border.muted"       # 弱化边框
    DIVIDER = "divider"                 # 分割线

    # ===== 状态 =====
    HOVER_BG = "hover.bg"               # 悬停背景
    SELECTED_BG = "selected.bg"         # 选中背景
    DISABLED_BG = "disabled.bg"         # 禁用背景
    DISABLED_TEXT = "disabled.text"     # 禁用文字

    # ===== 状态颜色 =====
    SUCCESS = "status.success"          # 成功
    WARNING = "status.warning"          # 警告
    ERROR = "status.error"              # 错误
    INFO = "status.info"                # 信息

    # ===== 日志颜色 =====
    LOG_DEBUG = "log.debug"
    LOG_INFO = "log.info"
    LOG_WARNING = "log.warning"
    LOG_ERROR = "log.error"
    LOG_SUCCESS = "log.success"
    LOG_METRIC = "log.metric"
    LOG_HIGHLIGHT = "log.highlight"

    # ===== 按钮特殊色 =====
    BTN_START_HOVER = "btn.start.hover"     # 开始按钮悬停
    BTN_STOP_HOVER = "btn.stop.hover"       # 停止按钮悬停
    BTN_CLOSE_HOVER = "btn.close.hover"     # 关闭按钮悬停

    # ===== 图标 =====
    ICON_DEFAULT = "icon.default"
    ICON_HOVER = "icon.hover"
    ICON_ACTIVE = "icon.active"
    ICON_CORNER = "icon.corner"  # 图标圆角像素颜色

    # ===== 操作按钮 =====
    ACTION_BTN_TEXT = "action.btn.text"  # 底部操作按钮文字
    ACTION_BTN_HOVER = "action.btn.hover"  # 操作按钮悬停背景

    # ===== 通用悬停 =====
    CONTROL_HOVER_BG = "control.hover.bg"  # 控件悬停背景
