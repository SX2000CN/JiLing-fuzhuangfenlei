"""
样式表生成器 - 基于主题动态生成 Qt 样式表
"""

from .tokens import ColorTokens
from .manager import theme_manager


class StyleSheet:
    """
    样式表生成器

    所有样式表都通过方法动态生成，以支持主题切换
    """

    @staticmethod
    def _c(token: str) -> str:
        """获取颜色值的快捷方法"""
        return theme_manager.get_color(token)

    # ===== 颜色属性（兼容旧代码） =====
    @classmethod
    @property
    def VS_EDITOR_BG(cls) -> str:
        return cls._c(ColorTokens.BG_PRIMARY)

    @classmethod
    @property
    def VS_SIDEBAR_BG(cls) -> str:
        return cls._c(ColorTokens.BG_SECONDARY)

    @classmethod
    @property
    def VS_ACTIVITY_BAR_BG(cls) -> str:
        return cls._c(ColorTokens.BG_SIDEBAR)

    @classmethod
    @property
    def VS_FOREGROUND(cls) -> str:
        return cls._c(ColorTokens.TEXT_PRIMARY)

    @classmethod
    @property
    def VS_DESCRIPTION(cls) -> str:
        return cls._c(ColorTokens.TEXT_SECONDARY)

    @classmethod
    @property
    def VS_FOCUS_BORDER(cls) -> str:
        return cls._c(ColorTokens.ACCENT)

    @classmethod
    @property
    def VS_INPUT_BG(cls) -> str:
        return cls._c(ColorTokens.BG_INPUT)

    @classmethod
    @property
    def VS_INPUT_BORDER(cls) -> str:
        return cls._c(ColorTokens.BORDER)

    @classmethod
    @property
    def VS_BUTTON_BG(cls) -> str:
        return cls._c(ColorTokens.ACCENT)

    @classmethod
    @property
    def VS_BUTTON_FG(cls) -> str:
        return cls._c(ColorTokens.TEXT_INVERSE)

    @classmethod
    @property
    def VS_BUTTON_HOVER(cls) -> str:
        return cls._c(ColorTokens.ACCENT_HOVER)

    @classmethod
    @property
    def VS_DIVIDER(cls) -> str:
        return cls._c(ColorTokens.BORDER_MUTED)

    # 兼容旧名称
    @classmethod
    @property
    def SIDEBAR_BG(cls) -> str:
        return cls._c(ColorTokens.BG_SIDEBAR)

    @classmethod
    @property
    def PARAM_BG(cls) -> str:
        return cls._c(ColorTokens.BG_SECONDARY)

    @classmethod
    @property
    def TERMINAL_BG(cls) -> str:
        return cls._c(ColorTokens.BG_PRIMARY)

    @classmethod
    @property
    def CARD_BG(cls) -> str:
        return cls._c(ColorTokens.BG_CARD)

    @classmethod
    @property
    def DIVIDER(cls) -> str:
        return cls._c(ColorTokens.BORDER_MUTED)

    @classmethod
    @property
    def TEXT_WHITE(cls) -> str:
        return cls._c(ColorTokens.TEXT_INVERSE)

    @classmethod
    @property
    def TEXT_GRAY(cls) -> str:
        return cls._c(ColorTokens.TEXT_PRIMARY)

    @classmethod
    @property
    def ICON_BG(cls) -> str:
        return cls._c(ColorTokens.ICON_DEFAULT)

    # 日志颜色
    @classmethod
    @property
    def COLORS(cls) -> dict:
        return {
            'DEBUG': cls._c(ColorTokens.LOG_DEBUG),
            'INFO': cls._c(ColorTokens.LOG_INFO),
            'WARNING': cls._c(ColorTokens.LOG_WARNING),
            'ERROR': cls._c(ColorTokens.LOG_ERROR),
            'SUCCESS': cls._c(ColorTokens.LOG_SUCCESS),
            'METRIC': cls._c(ColorTokens.LOG_METRIC),
            'HIGHLIGHT': cls._c(ColorTokens.LOG_HIGHLIGHT),
        }

    # ===== 样式表生成方法 =====

    @classmethod
    def main_window(cls) -> str:
        return f"""
            QMainWindow {{
                background-color: {cls._c(ColorTokens.BG_PRIMARY)};
            }}
        """

    MAIN_WINDOW = property(lambda self: StyleSheet.main_window())

    @classmethod
    def sidebar(cls) -> str:
        return f"""
            QWidget#sidebar {{
                background-color: {cls._c(ColorTokens.BG_SIDEBAR)};
                border-top-left-radius: 10px;
                border-bottom-left-radius: 10px;
                border-right: 1px solid {cls._c(ColorTokens.BORDER_MUTED)};
            }}
        """

    SIDEBAR = property(lambda self: StyleSheet.sidebar())

    @classmethod
    def sidebar_btn(cls) -> str:
        return f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-left: 2px solid transparent;
                color: {cls._c(ColorTokens.ICON_DEFAULT)};
                font-size: 24px;
            }}
            QPushButton:hover {{
                background-color: transparent;
                color: {cls._c(ColorTokens.ICON_HOVER)};
            }}
            QPushButton:checked {{
                border-left: 2px solid {cls._c(ColorTokens.ACCENT)};
                color: {cls._c(ColorTokens.ICON_ACTIVE)};
            }}
        """

    SIDEBAR_BTN = property(lambda self: StyleSheet.sidebar_btn())

    @classmethod
    def param_area(cls) -> str:
        return f"""
            QWidget#paramArea {{
                background-color: {cls._c(ColorTokens.BG_SECONDARY)};
                border-right: 1px solid {cls._c(ColorTokens.BORDER_MUTED)};
            }}
        """

    PARAM_AREA = property(lambda self: StyleSheet.param_area())

    @classmethod
    def page_title(cls) -> str:
        return f"""
            QLabel {{
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                font-size: 11px;
                font-weight: bold;
                text-transform: uppercase;
            }}
        """

    PAGE_TITLE = property(lambda self: StyleSheet.page_title())

    @classmethod
    def param_card(cls) -> str:
        return f"""
            QFrame {{
                background-color: transparent;
                border: none;
            }}
        """

    PARAM_CARD = property(lambda self: StyleSheet.param_card())

    @classmethod
    def param_label(cls) -> str:
        return f"""
            QLabel {{
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                font-size: 13px;
                font-weight: normal;
            }}
        """

    PARAM_LABEL = property(lambda self: StyleSheet.param_label())

    @classmethod
    def input_box(cls) -> str:
        # 根据主题选择箭头图标
        arrow_svg = "src/gui/chevron_down_dark.svg" if theme_manager.current_theme == "light" else "src/gui/chevron_down.svg"
        return f"""
            QLineEdit, QComboBox {{
                background-color: {cls._c(ColorTokens.BG_INPUT)};
                border: 1px solid {cls._c(ColorTokens.BORDER)};
                border-radius: 2px;
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                font-size: 12px;
                padding: 2px 6px;
                min-height: 26px;
                max-height: 26px;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border: 1px solid {cls._c(ColorTokens.BORDER_FOCUS)};
            }}
            QComboBox::drop-down {{
                width: 20px;
                border: none;
                background-color: transparent;
            }}
            QComboBox::down-arrow {{
                image: url({arrow_svg});
                width: 12px;
                height: 12px;
                margin-right: 6px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {cls._c(ColorTokens.BG_INPUT)};
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                border: 1px solid {cls._c(ColorTokens.BORDER)};
                border-radius: 4px;
                selection-background-color: {cls._c(ColorTokens.SELECTED_BG)};
                padding: 4px;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 4px 8px;
                border-radius: 2px;
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {cls._c(ColorTokens.SELECTED_BG)};
            }}
        """

    INPUT_BOX = property(lambda self: StyleSheet.input_box())

    @classmethod
    def input_box_narrow(cls) -> str:
        """窄版输入框 - 190px 宽"""
        return f"""
            QLineEdit {{
                background-color: {cls._c(ColorTokens.BG_INPUT)};
                border: 1px solid {cls._c(ColorTokens.BORDER)};
                border-radius: 2px;
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                font-size: 12px;
                padding: 2px 6px;
                min-height: 26px;
                max-height: 26px;
                min-width: 190px;
                max-width: 190px;
            }}
            QLineEdit:focus {{
                border: 1px solid {cls._c(ColorTokens.BORDER_FOCUS)};
            }}
        """

    INPUT_BOX_NARROW = property(lambda self: StyleSheet.input_box_narrow())

    @classmethod
    def browse_button(cls) -> str:
        """浏览按钮 - 26x26"""
        return f"""
            QPushButton {{
                background-color: {cls._c(ColorTokens.ACCENT)};
                border: none;
                border-radius: 2px;
                color: {cls._c(ColorTokens.TEXT_INVERSE)};
                font-size: 12px;
                padding-bottom: 2px;
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.ACCENT_HOVER)};
            }}
        """

    BROWSE_BUTTON = property(lambda self: StyleSheet.browse_button())

    @classmethod
    def slider(cls) -> str:
        return f"""
            QSlider::groove:horizontal {{
                height: 4px;
                background: {cls._c(ColorTokens.BG_INPUT)};
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {cls._c(ColorTokens.ACCENT)};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 12px;
                height: 12px;
                margin: -4px 0;
                background: {cls._c(ColorTokens.TEXT_PRIMARY)};
                border-radius: 6px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {cls._c(ColorTokens.TEXT_INVERSE)};
            }}
        """

    SLIDER = property(lambda self: StyleSheet.slider())

    @classmethod
    def slider_label(cls) -> str:
        return f"""
            QLabel {{
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                font-size: 11px;
            }}
        """

    SLIDER_LABEL = property(lambda self: StyleSheet.slider_label())

    @classmethod
    def value_box(cls, width: int = 35) -> str:
        return f"""
            QLabel {{
                background-color: {cls._c(ColorTokens.BG_INPUT)};
                border: 1px solid {cls._c(ColorTokens.BORDER)};
                border-radius: 2px;
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                font-size: 12px;
                min-width: {width}px;
                max-width: {width}px;
                min-height: 26px;
                max-height: 26px;
                padding: 2px;
            }}
        """

    VALUE_BOX = property(lambda self: StyleSheet.value_box())

    @classmethod
    def terminal_area(cls) -> str:
        return f"""
            QWidget#terminalArea {{
                background-color: {cls._c(ColorTokens.BG_PRIMARY)};
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }}
        """

    TERMINAL_AREA = property(lambda self: StyleSheet.terminal_area())

    @classmethod
    def terminal_title(cls) -> str:
        return f"""
            QLabel {{
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                font-size: 24px;
                font-weight: normal;
            }}
        """

    TERMINAL_TITLE = property(lambda self: StyleSheet.terminal_title())

    @classmethod
    def terminal_output(cls) -> str:
        return f"""
            QTextEdit {{
                background-color: {cls._c(ColorTokens.BG_PRIMARY)};
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                border: none;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.4;
            }}
        """

    TERMINAL_OUTPUT = property(lambda self: StyleSheet.terminal_output())

    @classmethod
    def control_btn(cls) -> str:
        return f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.CONTROL_HOVER_BG)};
            }}
        """

    CONTROL_BTN = property(lambda self: StyleSheet.control_btn())

    @classmethod
    def close_btn(cls, corner_radius: int = 9) -> str:
        return f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-top-right-radius: {corner_radius}px;
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.BTN_CLOSE_HOVER)};
                color: #FFFFFF;
            }}
        """

    CLOSE_BTN = property(lambda self: StyleSheet.close_btn())

    @classmethod
    def btn_start(cls) -> str:
        return f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 0px;
                color: {cls._c(ColorTokens.ACTION_BTN_TEXT)};
                font-size: 24px;
                font-weight: 300;
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.BTN_START_HOVER)};
                color: #FFFFFF;
            }}
        """

    BTN_START = property(lambda self: StyleSheet.btn_start())

    @classmethod
    def btn_stop(cls, corner_radius: int = 9) -> str:
        return f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-bottom-right-radius: {corner_radius}px;
                color: {cls._c(ColorTokens.ACTION_BTN_TEXT)};
                font-size: 24px;
                font-weight: 300;
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.BTN_STOP_HOVER)};
                color: #FFFFFF;
            }}
        """

    BTN_STOP = property(lambda self: StyleSheet.btn_stop())

    @classmethod
    def divider_h(cls) -> str:
        return f"""
            QFrame {{
                background-color: {cls._c(ColorTokens.BORDER_MUTED)};
                max-height: 1px;
                min-height: 1px;
            }}
        """

    DIVIDER_H = property(lambda self: StyleSheet.divider_h())

    @classmethod
    def result_table(cls) -> str:
        return f"""
            QTableWidget {{
                background-color: {cls._c(ColorTokens.BG_PRIMARY)};
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                border: none;
                gridline-color: {cls._c(ColorTokens.BORDER_MUTED)};
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }}
            QTableWidget::item {{
                padding: 4px 8px;
                border-bottom: 1px solid {cls._c(ColorTokens.BORDER_MUTED)};
            }}
            QTableWidget::item:selected {{
                background-color: {cls._c(ColorTokens.SELECTED_BG)};
                color: {cls._c(ColorTokens.TEXT_INVERSE)};
            }}
            QTableWidget::item:hover {{
                background-color: {cls._c(ColorTokens.HOVER_BG)};
            }}
            QHeaderView::section {{
                background-color: {cls._c(ColorTokens.BG_INPUT)};
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                border: none;
                border-bottom: 1px solid {cls._c(ColorTokens.BORDER_MUTED)};
                padding: 6px 8px;
                font-weight: bold;
                font-size: 11px;
                text-transform: uppercase;
            }}
            QTableWidget QScrollBar:vertical {{
                background: {cls._c(ColorTokens.BG_PRIMARY)};
                width: 10px;
                border: none;
            }}
            QTableWidget QScrollBar::handle:vertical {{
                background: rgba(121, 121, 121, 0.4);
                border-radius: 5px;
                min-height: 30px;
            }}
            QTableWidget QScrollBar::handle:vertical:hover {{
                background: rgba(100, 100, 100, 0.7);
            }}
            QTableWidget QScrollBar::add-line:vertical,
            QTableWidget QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """

    RESULT_TABLE = property(lambda self: StyleSheet.result_table())

    @classmethod
    def progress_bar(cls) -> str:
        return f"""
            QProgressBar {{
                background-color: {cls._c(ColorTokens.BG_INPUT)};
                border: none;
                border-radius: 2px;
                height: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {cls._c(ColorTokens.ACCENT)};
                border-radius: 2px;
            }}
        """

    PROGRESS_BAR = property(lambda self: StyleSheet.progress_bar())

    @classmethod
    def btn_classify(cls) -> str:
        return f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 0px;
                color: {cls._c(ColorTokens.ACTION_BTN_TEXT)};
                font-size: 24px;
                font-weight: 300;
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.ACCENT)};
                color: #FFFFFF;
            }}
        """

    BTN_CLASSIFY = property(lambda self: StyleSheet.btn_classify())

    @classmethod
    def btn_clear(cls) -> str:
        return f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-bottom-right-radius: 10px;
                color: {cls._c(ColorTokens.ACTION_BTN_TEXT)};
                font-size: 24px;
                font-weight: 300;
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.ACTION_BTN_HOVER)};
            }}
        """

    BTN_CLEAR = property(lambda self: StyleSheet.btn_clear())

    @classmethod
    def settings_area(cls) -> str:
        return f"""
            QWidget#settingsArea {{
                background-color: {cls._c(ColorTokens.BG_PRIMARY)};
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }}
        """

    SETTINGS_AREA = property(lambda self: StyleSheet.settings_area())

    @classmethod
    def settings_section_title(cls) -> str:
        return f"""
            QLabel {{
                color: {cls._c(ColorTokens.ACCENT)};
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                padding: 16px 0px 8px 0px;
                margin: 0;
            }}
        """

    SETTINGS_SECTION_TITLE = property(lambda self: StyleSheet.settings_section_title())

    @classmethod
    def settings_row(cls) -> str:
        return f"""
            QWidget {{
                background-color: transparent;
                border-radius: 3px;
                padding: 0;
            }}
            QWidget:hover {{
                background-color: {cls._c(ColorTokens.HOVER_BG)};
            }}
        """

    SETTINGS_ROW = property(lambda self: StyleSheet.settings_row())

    @classmethod
    def settings_row_label(cls) -> str:
        return f"""
            QLabel {{
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                font-size: 13px;
                padding: 0;
                background-color: transparent;
            }}
        """

    SETTINGS_ROW_LABEL = property(lambda self: StyleSheet.settings_row_label())

    @classmethod
    def settings_row_desc(cls) -> str:
        return f"""
            QLabel {{
                color: {cls._c(ColorTokens.TEXT_SECONDARY)};
                font-size: 12px;
                padding: 0;
                background-color: transparent;
            }}
        """

    SETTINGS_ROW_DESC = property(lambda self: StyleSheet.settings_row_desc())

    @classmethod
    def settings_checkbox(cls) -> str:
        return f"""
            QCheckBox {{
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                font-size: 13px;
                spacing: 8px;
                background-color: transparent;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {cls._c(ColorTokens.BORDER)};
                border-radius: 3px;
                background-color: {cls._c(ColorTokens.BG_INPUT)};
            }}
            QCheckBox::indicator:checked {{
                background-color: {cls._c(ColorTokens.ACCENT)};
                border-color: {cls._c(ColorTokens.ACCENT)};
            }}
            QCheckBox::indicator:hover {{
                border-color: {cls._c(ColorTokens.ACCENT)};
            }}
            QCheckBox::indicator:checked:hover {{
                background-color: {cls._c(ColorTokens.ACCENT_HOVER)};
            }}
        """

    SETTINGS_CHECKBOX = property(lambda self: StyleSheet.settings_checkbox())

    @classmethod
    def settings_spinbox(cls) -> str:
        return f"""
            QSpinBox, QDoubleSpinBox {{
                background-color: {cls._c(ColorTokens.BG_INPUT)};
                border: 1px solid {cls._c(ColorTokens.BORDER)};
                border-radius: 2px;
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                font-size: 12px;
                padding: 2px 6px;
                min-height: 24px;
                max-height: 24px;
                min-width: 80px;
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {cls._c(ColorTokens.BORDER_FOCUS)};
            }}
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 16px;
                border: none;
                background-color: transparent;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: {cls._c(ColorTokens.CONTROL_HOVER_BG)};
            }}
        """

    SETTINGS_SPINBOX = property(lambda self: StyleSheet.settings_spinbox())

    @classmethod
    def btn_settings_action(cls) -> str:
        return f"""
            QPushButton {{
                background-color: {cls._c(ColorTokens.ACCENT)};
                border: none;
                border-radius: 2px;
                color: {cls._c(ColorTokens.TEXT_INVERSE)};
                font-size: 12px;
                padding: 6px 16px;
                min-height: 26px;
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.ACCENT_HOVER)};
            }}
            QPushButton:pressed {{
                background-color: {cls._c(ColorTokens.ACCENT_PRESSED)};
            }}
        """

    BTN_SETTINGS_ACTION = property(lambda self: StyleSheet.btn_settings_action())

    @classmethod
    def btn_settings_secondary(cls) -> str:
        return f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid {cls._c(ColorTokens.BORDER)};
                border-radius: 2px;
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                font-size: 12px;
                padding: 6px 16px;
                min-height: 26px;
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.CONTROL_HOVER_BG)};
                border-color: {cls._c(ColorTokens.ACCENT)};
            }}
        """

    BTN_SETTINGS_SECONDARY = property(lambda self: StyleSheet.btn_settings_secondary())

    @classmethod
    def settings_scroll(cls) -> str:
        return f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
                border-bottom-right-radius: 10px;
            }}
            QScrollArea > QWidget > QWidget {{
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 10px;
                border: none;
                margin-bottom: 10px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(121, 121, 121, 0.4);
                border-radius: 2px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba(100, 100, 100, 0.7);
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
                background: transparent;
            }}
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
        """

    SETTINGS_SCROLL = property(lambda self: StyleSheet.settings_scroll())

    @classmethod
    def primary_button(cls) -> str:
        """主按钮样式"""
        return f"""
            QPushButton {{
                background-color: {cls._c(ColorTokens.ACCENT)};
                color: {cls._c(ColorTokens.TEXT_INVERSE)};
                border: none;
                border-radius: 2px;
                padding: 0 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.ACCENT_HOVER)};
            }}
            QPushButton:pressed {{
                background-color: {cls._c(ColorTokens.ACCENT_PRESSED)};
            }}
        """

    PRIMARY_BUTTON = property(lambda self: StyleSheet.primary_button())

    @classmethod
    def secondary_button(cls) -> str:
        """次要按钮样式"""
        return f"""
            QPushButton {{
                background-color: {cls._c(ColorTokens.BG_INPUT)};
                color: {cls._c(ColorTokens.TEXT_PRIMARY)};
                border: none;
                border-radius: 2px;
                padding: 0 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.CONTROL_HOVER_BG)};
            }}
        """

    SECONDARY_BUTTON = property(lambda self: StyleSheet.secondary_button())

    @classmethod
    def action_button_classify(cls) -> str:
        """分类操作按钮"""
        return f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 0px;
                color: {cls._c(ColorTokens.ACTION_BTN_TEXT)};
                font-size: 24px;
                font-weight: 300;
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.ACCENT)};
                color: #FFFFFF;
            }}
        """

    ACTION_BUTTON_CLASSIFY = property(lambda self: StyleSheet.action_button_classify())

    @classmethod
    def action_button_clear(cls, corner_radius: int = 9) -> str:
        """清空操作按钮"""
        return f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-bottom-right-radius: {corner_radius}px;
                color: {cls._c(ColorTokens.ACTION_BTN_TEXT)};
                font-size: 24px;
                font-weight: 300;
            }}
            QPushButton:hover {{
                background-color: {cls._c(ColorTokens.ACTION_BTN_HOVER)};
            }}
        """

    ACTION_BUTTON_CLEAR = property(lambda self: StyleSheet.action_button_clear())
