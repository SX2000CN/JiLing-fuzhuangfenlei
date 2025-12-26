"""
布局常量类 - 统一管理 UI 布局相关的常量

参考 VS Code Design System:
- 间距使用 4px 的倍数
- 圆角统一使用 2px 或 4px
"""


class LayoutConstants:
    """布局常量"""

    # 窗口尺寸
    WINDOW_WIDTH = 990
    WINDOW_HEIGHT = 660

    # 侧边栏
    SIDEBAR_WIDTH = 60
    SIDEBAR_BUTTON_SIZE = 50
    SIDEBAR_ICON_SIZE = 24

    # 参数区
    PARAM_AREA_WIDTH = 380
    CARD_PADDING = 16
    CARD_SPACING = 12

    # 终端区
    TERMINAL_TITLE_HEIGHT = 60
    BOTTOM_BAR_HEIGHT = 80
    PROGRESS_BAR_HEIGHT = 20

    # 控制按钮
    CONTROL_BTN_WIDTH = 46
    CONTROL_BTN_HEIGHT = 32
    CONTROL_ICON_SIZE = 12

    # 圆角
    CORNER_RADIUS = 10  # 窗口圆角
    CARD_RADIUS = 4     # 卡片圆角
    INPUT_RADIUS = 2    # 输入框圆角

    # 间距
    SPACING_XS = 4
    SPACING_SM = 8
    SPACING_MD = 12
    SPACING_LG = 16
    SPACING_XL = 24
