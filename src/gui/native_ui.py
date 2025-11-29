"""
Native Qt UI - 使用原生 PySide6 实现的界面
完全按照 Figma 设计稿还原
"""
import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QLineEdit, QTextEdit,
    QFrame, QFileDialog, QSizePolicy, QSpacerItem, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QStackedWidget,
    QButtonGroup, QScrollArea, QCheckBox, QSpinBox, QDoubleSpinBox,
    QGridLayout
)
from PySide6.QtCore import Qt, QSize, Signal, Slot, QPoint, QByteArray
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QPainter, QPen, QBrush, QFontDatabase, QPixmap, QPainterPath, QRegion
from PySide6.QtSvg import QSvgRenderer

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# MiSans 字体路径
FONT_DIR = PROJECT_ROOT / "MiSans" / "MiSans 开发下载字重"


class FontManager:
    """字体管理器 - 加载并管理 MiSans 字体"""
    
    _fonts_loaded = False
    _font_ids = []
    
    # 字体名称常量
    FAMILY = "MiSans"
    
    @classmethod
    def load_fonts(cls):
        """加载 MiSans 字体族"""
        if cls._fonts_loaded:
            return True
        
        font_files = [
            "MiSans-Regular.ttf",    # 400
            "MiSans-Medium.ttf",     # 500
            "MiSans-Semibold.ttf",   # 600
        ]
        
        for font_file in font_files:
            font_path = FONT_DIR / font_file
            if font_path.exists():
                font_id = QFontDatabase.addApplicationFont(str(font_path))
                if font_id != -1:
                    cls._font_ids.append(font_id)
                    families = QFontDatabase.applicationFontFamilies(font_id)
                    print(f"已加载字体: {font_file} -> {families}")
                else:
                    print(f"字体加载失败: {font_file}")
            else:
                print(f"字体文件不存在: {font_path}")
        
        cls._fonts_loaded = True
        return len(cls._font_ids) > 0
    
    @classmethod
    def get_font(cls, size: int, weight: int = QFont.Normal) -> QFont:
        """
        获取指定大小和字重的 MiSans 字体
        
        Args:
            size: 字体大小 (px)
            weight: 字重 - QFont.Normal(400), QFont.Medium(500), QFont.DemiBold(600)
        
        Returns:
            QFont 对象
        """
        font = QFont(cls.FAMILY)
        font.setPixelSize(size)
        font.setWeight(weight)
        
        # 优化字体渲染
        font.setStyleStrategy(QFont.PreferAntialias)
        font.setHintingPreference(QFont.PreferFullHinting)
        
        return font
    
    @classmethod
    def title_font(cls) -> QFont:
        """页面标题字体 - VS Code Section Header Style (11px Bold Uppercase)"""
        # 使用 DemiBold (600) 对应 MiSans-Semibold，避免使用未加载的 Bold (700)
        font = cls.get_font(11, QFont.DemiBold)
        font.setCapitalization(QFont.AllUppercase)
        return font
    
    @classmethod
    def header_font(cls) -> QFont:
        """大标题字体 - 24px"""
        return cls.get_font(24, QFont.Normal)
    
    @classmethod
    def label_font(cls) -> QFont:
        """参数标签字体 - VS Code Standard (13px)"""
        # 恢复使用 Medium (500) 字重，保持 MiSans 的质感
        return cls.get_font(13, QFont.Medium)
    
    @classmethod
    def input_font(cls) -> QFont:
        """输入框文字字体 - VS Code Standard (12px)"""
        # 调整为 12px 以避免比标签显得更大
        return cls.get_font(12, QFont.Normal)
    
    @classmethod
    def small_font(cls) -> QFont:
        """小号字体 - 用于描述文字 (11px)"""
        return cls.get_font(11, QFont.Normal)
    
    @classmethod
    def slider_tick_font(cls) -> QFont:
        """滑块刻度字体 - VS Code Small (11px)"""
        return cls.get_font(11, QFont.Normal)
    
    @classmethod
    def button_font(cls) -> QFont:
        """底部按钮字体 - VS Code Standard (13px)"""
        # 按钮文字使用 Medium (500) 更清晰
        return cls.get_font(13, QFont.Medium)

    @classmethod
    def action_button_font(cls) -> QFont:
        """底部操作按钮字体 - 24px Light"""
        return cls.get_font(24, QFont.Light)


class StyleSheet:
    """样式表定义 - 基于 VS Code Design System"""
    
    # VS Code 颜色常量
    VS_EDITOR_BG = "#1E1E1E"       # 编辑器背景
    VS_SIDEBAR_BG = "#252526"      # 侧边栏背景
    VS_ACTIVITY_BAR_BG = "#333333" # 活动栏背景
    VS_FOREGROUND = "#CCCCCC"      # 默认前景色
    VS_FOCUS_BORDER = "#007FD4"    # 聚焦边框色
    VS_INPUT_BG = "#3C3C3C"        # 输入框背景
    VS_INPUT_BORDER = "#505050"    # 输入框边框
    VS_BUTTON_BG = "#0E639C"       # 按钮背景
    VS_BUTTON_FG = "#FFFFFF"       # 按钮文字
    VS_BUTTON_HOVER = "#1177BB"    # 按钮悬停
    VS_DIVIDER = "#454545"         # 分割线
    
    # 映射到现有布局变量
    SIDEBAR_BG = VS_ACTIVITY_BAR_BG  # 左侧图标栏对应 Activity Bar
    PARAM_BG = VS_SIDEBAR_BG         # 参数设置区对应 Side Bar
    TERMINAL_BG = VS_EDITOR_BG       # 终端输出区对应 Editor
    CARD_BG = VS_EDITOR_BG           # 卡片背景使用编辑器背景色，形成层级
    DIVIDER = VS_DIVIDER
    TEXT_WHITE = "#FFFFFF"           # 标题等高亮文字
    TEXT_GRAY = VS_FOREGROUND        # 普通文字
    ICON_BG = "#C5C5C5"              # 图标颜色
    
    MAIN_WINDOW = f"""
        QMainWindow {{
            background-color: {VS_EDITOR_BG};
        }}
    """
    
    SIDEBAR = f"""
        QWidget#sidebar {{
            background-color: {SIDEBAR_BG};
            border-top-left-radius: 10px;
            border-bottom-left-radius: 10px;
            border-right: 1px solid {DIVIDER};
        }}
    """
    
    SIDEBAR_BTN = f"""
        QPushButton {{
            background-color: transparent;
            border: none;
            border-left: 2px solid transparent;
            color: {ICON_BG};
            font-size: 24px;
        }}
        QPushButton:hover {{
            background-color: transparent;
            color: {TEXT_WHITE};
        }}
        QPushButton:checked {{
            border-left: 2px solid {TEXT_WHITE};
            color: {TEXT_WHITE};
        }}
    """
    
    PARAM_AREA = f"""
        QWidget#paramArea {{
            background-color: {PARAM_BG};
            border-right: 1px solid {DIVIDER};
        }}
    """
    
    PAGE_TITLE = f"""
        QLabel {{
            color: {TEXT_GRAY};
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
        }}
    """
    
    PARAM_CARD = f"""
        QFrame {{
            background-color: transparent;
            border: none;
        }}
    """
    
    PARAM_LABEL = f"""
        QLabel {{
            color: {TEXT_GRAY};
            font-size: 13px;
            font-weight: normal;
        }}
    """
    
    INPUT_BOX = f"""
        QLineEdit, QComboBox {{
            background-color: {VS_INPUT_BG};
            border: 1px solid {VS_INPUT_BORDER};
            border-radius: 2px;
            color: {VS_FOREGROUND};
            font-size: 12px;
            padding: 2px 6px;
            min-height: 26px;
            max-height: 26px;
        }}
        QLineEdit:focus, QComboBox:focus {{
            border: 1px solid {VS_FOCUS_BORDER};
        }}
        QComboBox::drop-down {{
            width: 20px;
            border: none;
            background-color: transparent;
        }}
        QComboBox::down-arrow {{
            image: url(src/gui/chevron_down.svg);
            width: 12px;
            height: 12px;
            margin-right: 6px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {VS_INPUT_BG};
            color: {VS_FOREGROUND};
            border: 1px solid {VS_DIVIDER};
            selection-background-color: #04395E;
        }}
    """
    
    SLIDER = f"""
        QSlider::groove:horizontal {{
            height: 4px;
            background: {VS_INPUT_BG};
            border-radius: 2px;
        }}
        QSlider::sub-page:horizontal {{
            background: {VS_FOCUS_BORDER};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            width: 12px;
            height: 12px;
            margin: -4px 0;
            background: {VS_FOREGROUND};
            border-radius: 6px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {TEXT_WHITE};
        }}
    """
    
    SLIDER_LABEL = f"""
        QLabel {{
            color: {TEXT_GRAY};
            font-size: 11px;
        }}
    """
    
    VALUE_BOX = f"""
        QLabel {{
            background-color: {VS_INPUT_BG};
            border: 1px solid {VS_INPUT_BORDER};
            border-radius: 2px;
            color: {VS_FOREGROUND};
            font-size: 12px;
            min-width: 35px;
            max-width: 35px;
            min-height: 26px;
            max-height: 26px;
            padding: 2px;
        }}
    """
    
    TERMINAL_AREA = f"""
        QWidget#terminalArea {{
            background-color: {TERMINAL_BG};
            border-top-right-radius: 10px;
            border-bottom-right-radius: 10px;
        }}
    """
    
    TERMINAL_TITLE = f"""
        QLabel {{
            color: {TEXT_WHITE};
            font-size: 24px;
            font-weight: normal;
        }}
    """
    
    TERMINAL_OUTPUT = f"""
        QTextEdit {{
            background-color: {TERMINAL_BG};
            color: {VS_FOREGROUND};
            border: none;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.4;
        }}
    """
    
    CONTROL_BTN = f"""
        QPushButton {{
            background-color: transparent;
            border: none;
            color: {VS_FOREGROUND};
        }}
        QPushButton:hover {{
            background-color: #3E3E3E;
        }}
    """
    
    CLOSE_BTN = f"""
        QPushButton {{
            background-color: transparent;
            border: none;
            border-top-right-radius: 10px;
            color: {VS_FOREGROUND};
        }}
        QPushButton:hover {{
            background-color: #E81123;
            color: #FFFFFF;
        }}
    """
    
    BTN_START = f"""
        QPushButton {{
            background-color: transparent;
            border: none;
            border-radius: 0px;
            color: {TEXT_WHITE};
            font-size: 24px;
            font-weight: 300;
        }}
        QPushButton:hover {{
            background-color: #13C468;
        }}
    """

    BTN_STOP = f"""
        QPushButton {{
            background-color: transparent;
            border: none;
            border-bottom-right-radius: 10px;
            color: {TEXT_WHITE};
            font-size: 24px;
            font-weight: 300;
        }}
        QPushButton:hover {{
            background-color: #c42b1c;
        }}
    """
    
    DIVIDER_H = f"""
        QFrame {{
            background-color: {DIVIDER};
            max-height: 1px;
            min-height: 1px;
        }}
    """
    
    # ===== 分类页面样式 =====
    RESULT_TABLE = f"""
        QTableWidget {{
            background-color: {TERMINAL_BG};
            color: {VS_FOREGROUND};
            border: none;
            gridline-color: {VS_DIVIDER};
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
        }}
        QTableWidget::item {{
            padding: 4px 8px;
            border-bottom: 1px solid {VS_DIVIDER};
        }}
        QTableWidget::item:selected {{
            background-color: #094771;
            color: {TEXT_WHITE};
        }}
        QTableWidget::item:hover {{
            background-color: #2A2D2E;
        }}
        QHeaderView::section {{
            background-color: {VS_INPUT_BG};
            color: {VS_FOREGROUND};
            border: none;
            border-bottom: 1px solid {VS_DIVIDER};
            padding: 6px 8px;
            font-weight: bold;
            font-size: 11px;
            text-transform: uppercase;
        }}
        QTableWidget QScrollBar:vertical {{
            background: {TERMINAL_BG};
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
    
    PROGRESS_BAR = f"""
        QProgressBar {{
            background-color: {VS_INPUT_BG};
            border: none;
            border-radius: 2px;
            height: 4px;
            text-align: center;
        }}
        QProgressBar::chunk {{
            background-color: {VS_FOCUS_BORDER};
            border-radius: 2px;
        }}
    """
    
    # 分类页面的操作按钮样式
    BTN_CLASSIFY = f"""
        QPushButton {{
            background-color: transparent;
            border: none;
            border-radius: 0px;
            color: {TEXT_WHITE};
            font-size: 24px;
            font-weight: 300;
        }}
        QPushButton:hover {{
            background-color: #0E639C;
        }}
    """
    
    BTN_CLEAR = f"""
        QPushButton {{
            background-color: transparent;
            border: none;
            border-bottom-right-radius: 10px;
            color: {TEXT_WHITE};
            font-size: 24px;
            font-weight: 300;
        }}
        QPushButton:hover {{
            background-color: #5A5A5A;
        }}
    """
    
    # ===== 设置页面样式 =====
    SETTINGS_AREA = f"""
        QWidget#settingsArea {{
            background-color: {TERMINAL_BG};
            border-top-right-radius: 10px;
            border-bottom-right-radius: 10px;
        }}
    """
    
    SETTINGS_SECTION_TITLE = f"""
        QLabel {{
            color: {VS_FOCUS_BORDER};
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            padding: 16px 0px 8px 0px;
            margin: 0;
        }}
    """
    
    # 设置行 - 带悬停高亮
    SETTINGS_ROW = f"""
        QWidget {{
            background-color: transparent;
            border-radius: 3px;
            padding: 0;
        }}
        QWidget:hover {{
            background-color: rgba(90, 93, 94, 0.31);
        }}
    """
    
    SETTINGS_ROW_LABEL = f"""
        QLabel {{
            color: {VS_FOREGROUND};
            font-size: 13px;
            padding: 0;
            background-color: transparent;
        }}
    """
    
    SETTINGS_ROW_DESC = f"""
        QLabel {{
            color: {TEXT_GRAY};
            font-size: 12px;
            padding: 0;
            background-color: transparent;
        }}
    """
    
    SETTINGS_CHECKBOX = f"""
        QCheckBox {{
            color: {VS_FOREGROUND};
            font-size: 13px;
            spacing: 8px;
            background-color: transparent;
        }}
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 1px solid {VS_INPUT_BORDER};
            border-radius: 3px;
            background-color: {VS_INPUT_BG};
        }}
        QCheckBox::indicator:checked {{
            background-color: {VS_FOCUS_BORDER};
            border-color: {VS_FOCUS_BORDER};
            image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNCIgaGVpZ2h0PSIxNCIgdmlld0JveD0iMCAwIDE0IDE0IiBmaWxsPSJub25lIj48cGF0aCBkPSJNMTEuNjY2NyAzLjVMNS4yNSA5LjkxNjY3TDIuMzMzMzQgNyIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=);
        }}
        QCheckBox::indicator:hover {{
            border-color: {VS_FOCUS_BORDER};
        }}
        QCheckBox::indicator:checked:hover {{
            background-color: #1177BB;
        }}
    """
    
    SETTINGS_SPINBOX = f"""
        QSpinBox, QDoubleSpinBox {{
            background-color: {VS_INPUT_BG};
            border: 1px solid {VS_INPUT_BORDER};
            border-radius: 2px;
            color: {VS_FOREGROUND};
            font-size: 12px;
            padding: 2px 6px;
            min-height: 24px;
            max-height: 24px;
            min-width: 80px;
        }}
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {VS_FOCUS_BORDER};
        }}
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            width: 16px;
            border: none;
            background-color: transparent;
        }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: #3E3E3E;
        }}
    """
    
    BTN_SETTINGS_ACTION = f"""
        QPushButton {{
            background-color: {VS_FOCUS_BORDER};
            border: none;
            border-radius: 2px;
            color: {TEXT_WHITE};
            font-size: 12px;
            padding: 6px 16px;
            min-height: 26px;
        }}
        QPushButton:hover {{
            background-color: #1177BB;
        }}
        QPushButton:pressed {{
            background-color: #0E639C;
        }}
    """
    
    BTN_SETTINGS_SECONDARY = f"""
        QPushButton {{
            background-color: transparent;
            border: 1px solid {VS_INPUT_BORDER};
            border-radius: 2px;
            color: {VS_FOREGROUND};
            font-size: 12px;
            padding: 6px 16px;
            min-height: 26px;
        }}
        QPushButton:hover {{
            background-color: #3E3E3E;
            border-color: {VS_FOCUS_BORDER};
        }}
    """
    
    SETTINGS_SCROLL = f"""
        QScrollArea {{
            background-color: transparent;
            border: none;
        }}
        QScrollArea > QWidget > QWidget {{
            background-color: transparent;
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 10px;
            border: none;
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


class IconSvg:
    """SVG 图标定义 - 来自 Figma 设计稿"""
    
    # 关闭按钮 14x14
    CLOSE = '''<svg xmlns="http://www.w3.org/2000/svg" width="15" height="14" viewBox="0 0 15 14" fill="none">
      <path d="M12.2931 0.293125C12.6829 -0.0977872 13.3172 -0.0976305 13.7072 0.293125C14.0983 0.683138 14.0982 1.31718 13.7072 1.70719L1.70715 13.7072C1.3172 14.0982 0.683039 14.0982 0.293091 13.7072C-0.0976938 13.3172 -0.0978275 12.6831 0.293091 12.2931L12.2931 0.293125Z" fill="{color}"/>
      <path d="M0.477295 0.293125C0.867168 -0.0977849 1.50139 -0.0976235 1.89136 0.293125L13.8914 12.2931C14.2825 12.6832 14.2825 13.3172 13.8914 13.7072C13.5015 14.0982 12.8672 14.0982 12.4773 13.7072L0.477295 1.70719C0.0865257 1.31717 0.0863646 0.683052 0.477295 0.293125Z" fill="{color}"/>
    </svg>'''
    
    # 训练图标 29x32 - 侧边栏
    TRAIN = '''<svg xmlns="http://www.w3.org/2000/svg" width="29" height="32" viewBox="0 0 29 32" fill="none">
      <path d="M16.1807 0C20.0502 0 23.6034 1.34351 26.3877 3.58398L28.9893 6.15625L26.6602 8.45898L26.6592 8.45801L19.0312 16L25.5723 22.4678L25.5703 22.4707L28.9951 25.8564L26.4551 28.3691L26.4512 28.3652C23.6575 30.6366 20.0798 32 16.1807 32C7.2443 31.9998 0 24.8364 0 16C0 7.16356 7.2443 0.00019469 16.1807 0ZM16.1807 3.55566C9.23019 3.55586 3.5957 9.12724 3.5957 16C3.5957 22.8728 9.23019 28.4441 16.1807 28.4443C19.0864 28.4443 21.7624 27.4704 23.8926 25.835L13.9463 16L23.8926 6.16504C21.7624 4.52957 19.0864 3.55566 16.1807 3.55566ZM18.0938 5.33301C19.0865 5.33326 19.8916 6.12964 19.8916 7.11133C19.8915 8.09291 19.0864 8.88842 18.0938 8.88867C17.1009 8.88867 16.295 8.09307 16.2949 7.11133C16.2949 6.12949 17.1008 5.33301 18.0938 5.33301Z" fill="{color}"/>
    </svg>'''
    
    # 分类图标 30x27 - 侧边栏
    CLASSIFY = '''<svg xmlns="http://www.w3.org/2000/svg" width="30" height="27" viewBox="0 0 30 27" fill="none">
      <path d="M13.333 26.667H0V20H13.333V26.667ZM30 20V26.667H16.667V20H30ZM30 16.667H0V10H30V16.667ZM20 6.66699H0V0H20V6.66699ZM30 6.66699H23.333V0H30V6.66699Z" fill="{color}"/>
    </svg>'''
    
    # 设置图标 40x40 - 侧边栏 (带镂空的齿轮)
    SETTINGS = '''<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40" fill="none">
      <path d="M22.5003 3.33337C22.8466 3.33601 23.1835 3.44677 23.4642 3.64978C23.745 3.85291 23.9559 4.13882 24.0667 4.46716L25.1165 7.61658L28.0833 6.13318C28.3957 5.97906 28.7486 5.92706 29.0921 5.98376C29.4359 6.04056 29.754 6.20311 30.0003 6.44958L33.5501 10.0004C33.793 10.2453 33.9542 10.5596 34.011 10.8998C34.0677 11.24 34.0168 11.5894 33.8665 11.8998L32.3831 14.8666L35.5335 15.9164C35.8645 16.0282 36.1523 16.2417 36.3558 16.5258C36.559 16.8098 36.6672 17.1511 36.6663 17.5004V22.5004C36.6638 22.8466 36.5536 23.1836 36.3509 23.4642C36.1478 23.745 35.8618 23.9559 35.5335 24.0668L32.3831 25.1166L33.8665 28.0834C34.0206 28.3957 34.0736 28.7486 34.0169 29.0922C33.9601 29.436 33.7966 29.754 33.5501 30.0004L30.0003 33.5502C29.7566 33.7897 29.4446 33.9486 29.1077 34.0052C28.771 34.0618 28.4249 34.013 28.1165 33.8666L25.1497 32.3832L24.0999 35.5336C23.9881 35.8645 23.7747 36.1524 23.4905 36.3558C23.2067 36.5589 22.8659 36.6672 22.5169 36.6664H17.5169C17.1676 36.6672 16.8264 36.5591 16.5423 36.3558C16.2582 36.1523 16.0447 35.8645 15.9329 35.5336L14.8831 32.3832L11.9163 33.8666C11.6039 34.0206 11.2512 34.0737 10.9075 34.017C10.5638 33.9601 10.2466 33.7965 10.0003 33.5502L6.44952 30.0004C6.20661 29.7554 6.04627 29.4403 5.98956 29.1C5.93294 28.7598 5.98281 28.4104 6.13312 28.1L7.61652 25.1332L4.4671 24.0834C4.13607 23.9715 3.84821 23.7582 3.64484 23.474C3.44162 23.1901 3.33248 22.8495 3.33331 22.5004V17.5004C3.33589 17.1538 3.44659 16.8163 3.64972 16.5355C3.85284 16.2548 4.13879 16.0439 4.4671 15.933L7.61652 14.8832L6.13312 11.9164C5.97906 11.604 5.92695 11.2512 5.9837 10.9076C6.04047 10.564 6.20323 10.2466 6.44952 10.0004L10.0003 6.44958C10.2452 6.20685 10.5596 6.04631 10.8997 5.98962C11.2399 5.93294 11.5893 5.98293 11.8997 6.13318L14.8665 7.61658L15.9163 4.46716C16.0281 4.13613 16.2416 3.84827 16.5257 3.6449C16.8098 3.44153 17.1509 3.33247 17.5003 3.33337H22.5003ZM18.0335 8.66638C17.8843 9.1263 17.6365 9.54897 17.3079 9.90369C16.9795 10.2582 16.578 10.5372 16.1312 10.7211C15.6841 10.905 15.202 10.989 14.7191 10.9681C14.236 10.9473 13.7629 10.8218 13.3333 10.6L11.4837 9.64978L9.64972 11.4838L10.5999 13.3334C10.8217 13.763 10.9472 14.2361 10.9681 14.7191C10.9889 15.202 10.9049 15.6842 10.721 16.1312C10.5371 16.5781 10.2581 16.9796 9.90363 17.308C9.54891 17.6365 9.12624 17.8844 8.66632 18.0336L6.66632 18.6996V21.3002L8.66632 21.9672C9.12624 22.1163 9.54891 22.3633 9.90363 22.6918C10.2583 23.0202 10.5371 23.4224 10.721 23.8695C10.9049 24.3166 10.989 24.7987 10.9681 25.2816C10.9472 25.7644 10.8216 26.2369 10.5999 26.6664L9.59991 28.517L11.4329 30.35L13.3333 29.3998C13.7629 29.178 14.236 29.0524 14.7191 29.0316C15.202 29.0107 15.6841 29.0957 16.1312 29.2797C16.5781 29.4635 16.9795 29.7424 17.3079 30.097C17.6365 30.4517 17.8843 30.8735 18.0335 31.3334L18.6995 33.3334H21.3499L22.0169 31.3334C22.1674 30.8802 22.4136 30.4644 22.7386 30.1146C23.0635 29.7649 23.46 29.4893 23.9007 29.306C24.3417 29.1227 24.8171 29.0358 25.2943 29.0521C25.7713 29.0685 26.2392 29.187 26.6663 29.3998L28.5169 30.3998L30.3499 28.5668L29.3997 26.6664C29.178 26.2369 29.0524 25.7644 29.0316 25.2816C29.0106 24.7986 29.0956 24.3167 29.2796 23.8695C29.4634 23.4224 29.7422 23.0203 30.097 22.6918C30.4516 22.3634 30.8736 22.1163 31.3333 21.9672L33.3333 21.3002V18.6498L31.3333 17.9838C30.8802 17.8333 30.4644 17.5871 30.1146 17.2621C29.7649 16.9372 29.4893 16.5407 29.306 16.1C29.1227 15.6591 29.0358 15.1836 29.0521 14.7064C29.0684 14.2292 29.1868 13.7608 29.3997 13.3334L30.3997 11.4838L28.5667 9.64978L26.6663 10.6C26.2368 10.8217 25.7644 10.9472 25.2816 10.9681C24.7986 10.989 24.3166 10.905 23.8694 10.7211C23.4223 10.5372 23.0202 10.2583 22.6917 9.90369C22.3632 9.54897 22.1163 9.1263 21.9671 8.66638L21.3001 6.66638H18.6995L18.0335 8.66638ZM20.0003 13.3334C21.7683 13.3335 23.4641 14.0354 24.7142 15.2855C25.9645 16.5358 26.6663 18.2323 26.6663 20.0004C26.6663 21.3186 26.2756 22.6073 25.5433 23.7035C24.8108 24.7998 23.7693 25.655 22.5511 26.1595C21.333 26.6641 19.9926 26.7957 18.6995 26.5385C17.4063 26.2813 16.2178 25.6466 15.2855 24.7142C14.3533 23.782 13.7185 22.5941 13.4612 21.3011C13.204 20.0078 13.3365 18.6668 13.8411 17.4486C14.3457 16.2306 15.2001 15.1899 16.2962 14.4574C17.3925 13.7248 18.6818 13.3334 20.0003 13.3334ZM21.2757 16.9203C20.6666 16.668 19.9964 16.6023 19.3499 16.7308C18.7033 16.8595 18.1091 17.1768 17.6429 17.6429C17.1767 18.1091 16.8594 18.7034 16.7308 19.35C16.6022 19.9965 16.668 20.6667 16.9202 21.2758C17.1725 21.8847 17.5996 22.4057 18.1478 22.7719C18.6959 23.1382 19.341 23.3334 20.0003 23.3334C20.8842 23.3333 21.7317 22.9819 22.3568 22.3568C22.9818 21.7317 23.3332 20.8842 23.3333 20.0004C23.3333 19.3411 23.1381 18.696 22.7718 18.1478C22.4056 17.5997 21.8847 17.1726 21.2757 16.9203Z" fill="{color}"/>
    </svg>'''
    
    # 最大化按钮 16x16
    MAXIMIZE = '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path d="M12 0C13.0608 0 14.0797 0.425977 14.8301 1.16992C15.5797 1.92187 16 2.93594 16 4V12C16 13.0641 15.5797 14.0782 14.8301 14.8301C14.0797 15.5741 13.0608 16 12 16H4C2.93921 16 1.92031 15.5741 1.16992 14.8301C0.420312 14.0782 0 13.0641 0 12V4C0 2.93594 0.420312 1.92187 1.16992 1.16992C1.92031 0.425977 2.93921 0 4 0H12ZM4 1.59961C3.36318 1.59961 2.74936 1.85196 2.2998 2.2998C1.84936 2.74785 1.59961 3.35996 1.59961 4V12C1.59961 12.64 1.84937 13.2522 2.2998 13.7002C2.74936 14.148 3.36318 14.4004 4 14.4004H12C12.6368 14.4004 13.2506 14.148 13.7002 13.7002C14.1506 13.2522 14.4004 12.64 14.4004 12V4C14.4004 3.35996 14.1506 2.74785 13.7002 2.2998C13.2506 1.85196 12.6368 1.59961 12 1.59961H4Z" fill="{color}"/>
    </svg>'''
    
    # 最小化按钮 16x2
    MINIMIZE = '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="2" viewBox="0 0 16 2" fill="none">
      <path d="M15 0C15.552 0 16 0.447998 16 1C16 1.552 15.552 2 15 2H1C0.447998 2 0 1.552 0 1C0 0.447998 0.447998 0 1 0H15Z" fill="{color}"/>
    </svg>'''

    # 下拉箭头 16x16
    CHEVRON_DOWN = '''<svg width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="none">
      <path d="M3.15 5.65C3.35 5.45 3.66 5.45 3.85 5.65L8 9.79L12.15 5.65C12.35 5.45 12.66 5.45 12.85 5.65C13.05 5.84 13.05 6.16 12.85 6.35L8.35 10.85C8.16 11.05 7.84 11.05 7.65 10.85L3.15 6.35C2.95 6.16 2.95 5.84 3.15 5.65Z" fill="{color}"/>
    </svg>'''
    
    # 勾选图标 14x14
    CHECK = '''<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M11.6667 3.5L5.25 9.91667L2.33334 7" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>'''


class SidebarButton(QPushButton):
    """
    侧边栏按钮 - VS Code Activity Bar Style
    
    设计规格:
    - 尺寸: 90x50px (适配宽侧边栏)
    - 默认: 透明背景, #CCCCCC 图标
    - Hover: 透明背景, #FFFFFF 图标
    - Active: 左侧 2px 白色边框
    """
    
    # 图标颜色常量
    ICON_COLOR_DEFAULT = "#CCCCCC"  # VS_FOREGROUND
    ICON_COLOR_HOVER = "#FFFFFF"
    
    def __init__(self, icon_text: str = None, svg_template: str = None, icon_size: tuple = (24, 24), parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 50)
        self.setCheckable(True)
        self.setStyleSheet(StyleSheet.SIDEBAR_BTN)
        
        self._svg_template = svg_template
        self._icon_text = icon_text
        self._icon_size = icon_size  # (width, height)
        
        if svg_template:
            # 使用 SVG 图标
            self._update_icon(self.ICON_COLOR_DEFAULT)
        elif icon_text:
            # 使用 emoji 文本
            self.setText(icon_text)
    
    def _update_icon(self, color: str):
        """更新 SVG 图标颜色"""
        if not self._svg_template:
            return
        
        svg_data = self._svg_template.replace("{color}", color)
        renderer = QSvgRenderer(QByteArray(svg_data.encode()))
        w, h = self._icon_size
        pixmap = QPixmap(w, h)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        renderer.render(painter)
        painter.end()
        
        self.setIcon(QIcon(pixmap))
        self.setIconSize(QSize(w, h))
    
    def enterEvent(self, event):
        """鼠标进入时高亮图标"""
        if self._svg_template:
            self._update_icon(self.ICON_COLOR_HOVER)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """鼠标离开时恢复图标颜色"""
        if self._svg_template:
            self._update_icon(self.ICON_COLOR_DEFAULT)
        super().leaveEvent(event)


class VSCheckBox(QWidget):
    """
    自定义复选框 - VS Code 风格
    使用 QPainter 绘制勾选标记，解决 Qt 样式表不支持 image url 的问题
    """
    
    toggled = Signal(bool)  # 状态改变信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False
        self._hovered = False
        self.setFixedSize(18, 18)
        self.setCursor(Qt.PointingHandCursor)
    
    def isChecked(self) -> bool:
        return self._checked
    
    def setChecked(self, checked: bool):
        if self._checked != checked:
            self._checked = checked
            self.update()
            self.toggled.emit(checked)
    
    def toggle(self):
        self.setChecked(not self._checked)
    
    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle()
        super().mousePressEvent(event)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # 背景和边框颜色
        if self._checked:
            bg_color = QColor("#1177BB") if self._hovered else QColor(StyleSheet.VS_FOCUS_BORDER)
            border_color = bg_color
        else:
            bg_color = QColor(StyleSheet.VS_INPUT_BG)
            border_color = QColor(StyleSheet.VS_FOCUS_BORDER) if self._hovered else QColor(StyleSheet.VS_INPUT_BORDER)
        
        # 绘制背景
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setBrush(bg_color)
        painter.setPen(QPen(border_color, 1))
        painter.drawRoundedRect(rect, 3, 3)
        
        # 如果选中，绘制勾选标记
        if self._checked:
            painter.setPen(QPen(QColor("#FFFFFF"), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            # 绘制勾选路径
            check_path = [
                (4, 9),   # 起点
                (7, 12),  # 中点
                (14, 5)   # 终点
            ]
            painter.drawLine(check_path[0][0], check_path[0][1], check_path[1][0], check_path[1][1])
            painter.drawLine(check_path[1][0], check_path[1][1], check_path[2][0], check_path[2][1])
        
        painter.end()


class ParamCard(QFrame):
    """参数卡片"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(StyleSheet.PARAM_CARD)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
    
    def add_row(self, widget: QWidget, show_divider: bool = True):
        """添加一行"""
        self.layout.addWidget(widget)
        if show_divider:
            divider = QFrame()
            divider.setStyleSheet(f"background-color: {StyleSheet.DIVIDER}; margin: 0 10px;")
            divider.setFixedHeight(1)
            self.layout.addWidget(divider)


class ParamRow(QWidget):
    """参数行 - 标签 + 输入框"""
    def __init__(self, label_text: str, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)  # VS Code 风格更紧凑
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setAlignment(Qt.AlignVCenter)  # 确保垂直居中
        
        # 标签
        label = QLabel(label_text)
        label.setStyleSheet(StyleSheet.PARAM_LABEL)
        label.setFont(FontManager.label_font())
        layout.addWidget(label)
        
        layout.addStretch()
        
        # 输入区域（子类实现）
        self.input_layout = QHBoxLayout()
        self.input_layout.setSpacing(0)
        self.input_layout.setAlignment(Qt.AlignVCenter) # 输入区域也垂直居中
        layout.addLayout(self.input_layout)


class ComboRow(ParamRow):
    """下拉框行"""
    def __init__(self, label_text: str, items: list, parent=None):
        super().__init__(label_text, parent)
        
        self.combo = QComboBox()
        self.combo.addItems(items)
        self.combo.setFixedWidth(220)
        self.combo.setStyleSheet(StyleSheet.INPUT_BOX)
        self.combo.setFont(FontManager.input_font())
        self.input_layout.addWidget(self.combo)


class FileRow(ParamRow):
    """文件选择行"""
    pathSelected = Signal(str)
    
    def __init__(self, label_text: str, default_path: str = "", is_folder: bool = False, parent=None):
        super().__init__(label_text, parent)
        self.is_folder = is_folder
        
        self.path_edit = QLineEdit(default_path)
        self.path_edit.setFixedWidth(190)  # 220 - 26(icon) - 4(spacing)
        self.path_edit.setReadOnly(True)
        self.path_edit.setStyleSheet(StyleSheet.INPUT_BOX.replace("min-width: 220px;", "min-width: 190px;").replace("max-width: 220px;", "max-width: 190px;"))
        self.path_edit.setFont(FontManager.input_font())
        self.input_layout.addWidget(self.path_edit)
        
        # 添加间距
        self.input_layout.addSpacing(4)
        
        # 文件夹按钮 - 26x26px (Match input height)
        self.browse_btn = QPushButton("...")
        self.browse_btn.setFixedSize(26, 26)
        self.browse_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {StyleSheet.VS_BUTTON_BG};
                border: none;
                border-radius: 2px;
                color: {StyleSheet.VS_BUTTON_FG};
                font-size: 12px;
                padding-bottom: 2px;
            }}
            QPushButton:hover {{
                background-color: {StyleSheet.VS_BUTTON_HOVER};
            }}
        """)
        self.browse_btn.clicked.connect(self._browse)
        self.input_layout.addWidget(self.browse_btn)
    
    def _browse(self):
        if self.is_folder:
            path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "模型文件 (*.pth *.pt);;所有文件 (*)")
        
        if path:
            self.path_edit.setText(path)
            self.pathSelected.emit(path)


class SliderRow(QWidget):
    """滑块行 - 支持多种显示格式"""
    valueChanged = Signal(int)
    
    def __init__(self, label_text: str, min_val: int, max_val: int, default_val: int, 
                 suffix: str = "", display_format: str = "int", scale: float = 1.0,
                 min_label_text: str = None, max_label_text: str = None,
                 value_width: int = 50, parent=None):
        """
        Args:
            label_text: 标签文本
            min_val: 滑块最小值
            max_val: 滑块最大值
            default_val: 默认值
            suffix: 后缀
            display_format: 显示格式 - "int", "float", "percent", "lr" (学习率)
            scale: 缩放系数 (实际值 = 滑块值 * scale)
            min_label_text: 左侧刻度标签 (None则自动生成)
            max_label_text: 右侧刻度标签 (None则自动生成)
            value_width: 数值框宽度
        """
        super().__init__(parent)
        self.setFixedHeight(40)
        self.display_format = display_format
        self.suffix = suffix
        self.scale = scale
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignVCenter)
        
        # 标签
        label = QLabel(label_text)
        label.setStyleSheet(StyleSheet.PARAM_LABEL)
        label.setFont(FontManager.label_font())
        layout.addWidget(label)
        
        layout.addStretch()
        
        # 滑块容器
        slider_container = QWidget()
        slider_container.setFixedWidth(177)
        slider_layout = QVBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(2)
        
        # 滑块
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default_val)
        self.slider.setStyleSheet(StyleSheet.SLIDER)
        self.slider.valueChanged.connect(self._on_value_changed)
        slider_layout.addWidget(self.slider)
        
        # 刻度标签
        labels_layout = QHBoxLayout()
        labels_layout.setContentsMargins(0, 0, 0, 0)
        
        min_text = min_label_text if min_label_text else self._format_value(min_val)
        max_text = max_label_text if max_label_text else self._format_value(max_val)
        
        min_label = QLabel(min_text)
        min_label.setStyleSheet(StyleSheet.SLIDER_LABEL)
        min_label.setFont(FontManager.slider_tick_font())
        max_label = QLabel(max_text)
        max_label.setStyleSheet(StyleSheet.SLIDER_LABEL)
        max_label.setFont(FontManager.slider_tick_font())
        labels_layout.addWidget(min_label)
        labels_layout.addStretch()
        labels_layout.addWidget(max_label)
        slider_layout.addLayout(labels_layout)
        
        layout.addWidget(slider_container)
        
        # 数值框 - 可变宽度
        self.value_label = QLabel()
        value_box_style = StyleSheet.VALUE_BOX.replace("min-width: 35px;", f"min-width: {value_width}px;")
        value_box_style = value_box_style.replace("max-width: 35px;", f"max-width: {value_width}px;")
        self.value_label.setStyleSheet(value_box_style)
        self.value_label.setFont(FontManager.input_font())
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setFixedSize(value_width, 26)
        self._update_value_display(default_val)
        layout.addWidget(self.value_label)
    
    def _on_value_changed(self, value: int):
        self._update_value_display(value)
        self.valueChanged.emit(value)
    
    def _format_value(self, value: int) -> str:
        """根据格式类型格式化数值"""
        real_val = value * self.scale
        
        if self.display_format == "int":
            return f"{int(real_val)}{self.suffix}"
        elif self.display_format == "float":
            return f"{real_val:.2f}{self.suffix}"
        elif self.display_format == "percent":
            return f"{real_val:.1f}%"
        elif self.display_format == "lr":
            # 学习率使用科学计数法或小数
            if real_val >= 0.01:
                return f"{real_val:.3f}"
            else:
                return f"{real_val:.4f}"
        else:
            return f"{real_val}{self.suffix}"
    
    def _update_value_display(self, value: int):
        self.value_label.setText(self._format_value(value))
    
    def value(self) -> int:
        """返回滑块原始值"""
        return self.slider.value()
    
    def real_value(self) -> float:
        """返回实际值 (滑块值 * scale)"""
        return self.slider.value() * self.scale


class TerminalOutput(QTextEdit):
    """终端输出区域"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet(StyleSheet.TERMINAL_OUTPUT)
        self.append_log("系统就绪，等待开始训练...", "#13C468")
    
    def append_log(self, message: str, color: str = "#E5E5E5"):
        self.append(f'<span style="color: {color};">{message}</span>')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class SettingsRow(QWidget):
    """
    设置行组件 - VS Code 风格
    每行包含：标题、描述、输入控件
    鼠标悬停时整行高亮
    """
    
    def __init__(self, title: str, description: str = "", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(50)
        self.setCursor(Qt.PointingHandCursor)
        
        # 主布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)
        
        # 左侧文字区
        text_area = QWidget()
        text_layout = QVBoxLayout(text_area)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        
        # 标题
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(StyleSheet.SETTINGS_ROW_LABEL)
        self.title_label.setFont(FontManager.label_font())
        text_layout.addWidget(self.title_label)
        
        # 描述（可选）
        if description:
            self.desc_label = QLabel(description)
            self.desc_label.setStyleSheet(StyleSheet.SETTINGS_ROW_DESC)
            self.desc_label.setFont(FontManager.small_font())
            self.desc_label.setWordWrap(True)
            text_layout.addWidget(self.desc_label)
        
        layout.addWidget(text_area, 1)
        
        # 右侧控件区
        self.control_layout = QHBoxLayout()
        self.control_layout.setContentsMargins(0, 0, 0, 0)
        self.control_layout.setSpacing(8)
        layout.addLayout(self.control_layout)
        
        # 保存控件引用
        self._control = None
    
    def set_control(self, control: QWidget):
        """设置右侧的输入控件"""
        self._control = control
        self.control_layout.addWidget(control)
    
    def get_control(self):
        """获取控件"""
        return self._control
    
    def enterEvent(self, event):
        """鼠标进入时高亮"""
        self.setStyleSheet("background-color: rgba(90, 93, 94, 0.31); border-radius: 3px;")
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """鼠标离开时恢复"""
        self.setStyleSheet("background-color: transparent;")
        super().leaveEvent(event)


class SettingsCheckRow(QWidget):
    """
    设置复选框行 - VS Code 风格
    复选框在最左侧，使用自定义 VSCheckBox
    """
    
    def __init__(self, title: str, description: str = "", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setCursor(Qt.PointingHandCursor)
        
        # 主布局
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)
        
        # 复选框 - 使用自定义 VSCheckBox
        self.checkbox = VSCheckBox()
        layout.addWidget(self.checkbox)
        
        # 文字区
        text_area = QWidget()
        text_layout = QVBoxLayout(text_area)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        
        # 标题
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(StyleSheet.SETTINGS_ROW_LABEL)
        self.title_label.setFont(FontManager.label_font())
        text_layout.addWidget(self.title_label)
        
        # 描述（可选）
        if description:
            self.desc_label = QLabel(description)
            self.desc_label.setStyleSheet(StyleSheet.SETTINGS_ROW_DESC)
            self.desc_label.setFont(FontManager.small_font())
            self.desc_label.setWordWrap(True)
            text_layout.addWidget(self.desc_label)
        
        layout.addWidget(text_area, 1)
    
    def isChecked(self) -> bool:
        return self.checkbox.isChecked()
    
    def setChecked(self, checked: bool):
        self.checkbox.setChecked(checked)
    
    def enterEvent(self, event):
        """鼠标进入时高亮"""
        self.setStyleSheet("background-color: rgba(90, 93, 94, 0.31); border-radius: 3px;")
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """鼠标离开时恢复"""
        self.setStyleSheet("background-color: transparent;")
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """点击整行切换复选框（排除复选框自身区域）"""
        if event.button() == Qt.LeftButton:
            # 检查是否点击在复选框区域内
            checkbox_rect = self.checkbox.geometry()
            if not checkbox_rect.contains(event.pos()):
                self.checkbox.toggle()
        super().mousePressEvent(event)


class ClassificationResultTable(QTableWidget):
    """分类结果表格 - VS Code 风格"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["文件名", "分类结果", "置信度", "路径"])
        self.setStyleSheet(StyleSheet.RESULT_TABLE)
        
        # 设置表头
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        
        # 设置行为
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.SingleSelection)
        self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(False)
    
    def add_result(self, filename: str, category: str, confidence: float, path: str):
        """添加一行分类结果"""
        row = self.rowCount()
        self.insertRow(row)
        
        # 文件名
        name_item = QTableWidgetItem(filename)
        name_item.setFont(FontManager.input_font())
        self.setItem(row, 0, name_item)
        
        # 分类结果
        category_item = QTableWidgetItem(category)
        category_item.setFont(FontManager.input_font())
        # 根据类别设置颜色
        if category == "主图":
            category_item.setForeground(QColor("#4FC3F7"))
        elif category == "细节":
            category_item.setForeground(QColor("#81C784"))
        elif category == "吊牌":
            category_item.setForeground(QColor("#FFB74D"))
        self.setItem(row, 1, category_item)
        
        # 置信度
        conf_item = QTableWidgetItem(f"{confidence:.2%}")
        conf_item.setFont(FontManager.input_font())
        if confidence >= 0.9:
            conf_item.setForeground(QColor("#89D185"))
        elif confidence >= 0.7:
            conf_item.setForeground(QColor("#CCA700"))
        else:
            conf_item.setForeground(QColor("#F48771"))
        self.setItem(row, 2, conf_item)
        
        # 路径
        path_item = QTableWidgetItem(path)
        path_item.setFont(FontManager.input_font())
        path_item.setForeground(QColor("#808080"))
        self.setItem(row, 3, path_item)
        
        # 滚动到最新行
        self.scrollToBottom()
    
    def clear_results(self):
        """清空所有结果"""
        self.setRowCount(0)


class RoundedContainer(QWidget):
    """
    带抗锯齿圆角的容器控件
    用于实现完全无锯齿的 iOS 风格圆角
    """
    
    CORNER_RADIUS = 10
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._squircle_path = None
    
    def _create_squircle_path(self, rect, radius: float) -> QPainterPath:
        """创建 iOS 风格超椭圆路径"""
        path = QPainterPath()
        
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        r = min(radius, w / 2, h / 2)
        
        # Figma 60% smoothing 参数
        p = 1.528665
        arc_length = r * p
        handle_offset = r * 0.275
        arc_start = min(arc_length, w / 2, h / 2)
        
        path.moveTo(x + arc_start, y)
        path.lineTo(x + w - arc_start, y)
        path.cubicTo(x + w - handle_offset, y, x + w, y + handle_offset, x + w, y + arc_start)
        path.lineTo(x + w, y + h - arc_start)
        path.cubicTo(x + w, y + h - handle_offset, x + w - handle_offset, y + h, x + w - arc_start, y + h)
        path.lineTo(x + arc_start, y + h)
        path.cubicTo(x + handle_offset, y + h, x, y + h - handle_offset, x, y + h - arc_start)
        path.lineTo(x, y + arc_start)
        path.cubicTo(x, y + handle_offset, x + handle_offset, y, x + arc_start, y)
        path.closeSubpath()
        
        return path
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        path = self._create_squircle_path(self.rect(), self.CORNER_RADIUS)
        
        # 裁剪所有子控件到圆角区域内
        painter.setClipPath(path)
        
        # 绘制背景
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#191919"))
        painter.drawPath(path)
        
        super().paintEvent(event)


class MainWindow(QMainWindow):
    """主窗口"""
    
    # iOS 风格圆角参数
    CORNER_RADIUS = 10  # 圆角半径
    # iOS 的 continuity corner 使用 n≈5 的超椭圆，对应约 60% 平滑度
    # 标准圆形 n=2，正方形 n=∞
    SUPERELLIPSE_N = 5  # iOS 超椭圆指数
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JiLing 服装分类系统")
        self.setFixedSize(990, 660)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 拖拽相关
        self._drag_pos = None
        
        self._setup_ui()
    
    def _create_squircle_path(self, rect, radius: float) -> QPainterPath:
        """
        创建 iOS 风格的超椭圆(Squircle)圆角路径
        使用 Figma/iOS 的连续性圆角算法
        
        iOS 使用的是 "continuous corner" 算法，贝塞尔曲线控制点
        需要延伸得更远，使曲线在进入直线时更平滑
        """
        import math
        path = QPainterPath()
        
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        r = min(radius, w / 2, h / 2)
        
        # iOS/Figma 连续性圆角的关键参数
        # 普通圆角: 控制点距离 = r * 0.552284749831 (magic number for circle approximation)
        # iOS squircle: 控制点需要延伸得更远
        # 60% smoothing 对应的系数约为 1.528665 * r
        # 这使得曲线从边缘开始就有曲率，而不是突然拐弯
        
        # 魔法数字来自逆向工程 Figma 的连续性圆角
        # p = 圆角起点到控制点的距离比例
        p = 1.528665  # Figma 60% smoothing 的控制点系数
        arc_length = r * p  # 曲线段的长度
        handle_offset = r * 0.275  # 贝塞尔控制柄偏移
        
        # 圆角从边缘开始的位置
        arc_start = arc_length
        
        # 确保圆弧不超过边的一半
        arc_start = min(arc_start, w / 2, h / 2)
        
        # 从顶边中点左侧开始
        path.moveTo(x + arc_start, y)
        
        # 上边 (到右上角前)
        path.lineTo(x + w - arc_start, y)
        
        # 右上角 - iOS 风格连续曲线
        # 使用两段三次贝塞尔曲线来模拟超椭圆
        path.cubicTo(
            x + w - handle_offset, y,  # CP1: 靠近边缘
            x + w, y + handle_offset,  # CP2: 靠近边缘
            x + w, y + arc_start       # 终点
        )
        
        # 右边
        path.lineTo(x + w, y + h - arc_start)
        
        # 右下角
        path.cubicTo(
            x + w, y + h - handle_offset,
            x + w - handle_offset, y + h,
            x + w - arc_start, y + h
        )
        
        # 下边
        path.lineTo(x + arc_start, y + h)
        
        # 左下角
        path.cubicTo(
            x + handle_offset, y + h,
            x, y + h - handle_offset,
            x, y + h - arc_start
        )
        
        # 左边
        path.lineTo(x, y + arc_start)
        
        # 左上角
        path.cubicTo(
            x, y + handle_offset,
            x + handle_offset, y,
            x + arc_start, y
        )
        
        path.closeSubpath()
        return path
    
    def paintEvent(self, event):
        """
        绘制圆角窗口 - 使用最高质量渲染
        """
        painter = QPainter(self)
        
        # 启用所有抗锯齿选项
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setRenderHint(QPainter.LosslessImageRendering, True)
        
        # 使用 iOS 风格的 squircle 路径
        path = self._create_squircle_path(self.rect(), self.CORNER_RADIUS)
        
        # 绘制圆角背景
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#191919"))
        painter.drawPath(path)
        
        super().paintEvent(event)
    
    # 不使用 setMask，避免锯齿
    
    def _create_svg_icon(self, svg_template: str, color: str) -> QIcon:
        """从 SVG 模板创建图标"""
        svg_data = svg_template.replace("{color}", color)
        renderer = QSvgRenderer(QByteArray(svg_data.encode()))
        pixmap = QPixmap(renderer.defaultSize())
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return QIcon(pixmap)
    
    def _setup_ui(self):
        """设置 UI"""
        # 中央部件
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 左侧边栏
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)
        
        # 中间参数区 - 使用 QStackedWidget 实现页面切换
        self.param_stack = QStackedWidget()
        self.param_stack.setFixedWidth(380)
        
        # 训练页面参数区
        self.training_param_area = self._create_training_param_area()
        self.param_stack.addWidget(self.training_param_area)
        
        # 分类页面参数区
        self.classify_param_area = self._create_classify_param_area()
        self.param_stack.addWidget(self.classify_param_area)
        
        main_layout.addWidget(self.param_stack)
        
        # 右侧终端/结果区 - 使用 QStackedWidget 实现页面切换
        self.content_stack = QStackedWidget()
        
        # 训练页面终端区
        self.training_terminal_area = self._create_terminal_area()
        self.content_stack.addWidget(self.training_terminal_area)
        
        # 分类页面结果区
        self.classify_result_area = self._create_classify_result_area()
        self.content_stack.addWidget(self.classify_result_area)
        
        # 设置页面 (全宽) - 添加一个空的占位widget到param_stack
        self.settings_placeholder = QWidget()
        self.param_stack.addWidget(self.settings_placeholder)
        
        # 设置页面内容区
        self.settings_area = self._create_settings_area()
        self.content_stack.addWidget(self.settings_area)
        
        main_layout.addWidget(self.content_stack)
        
        # 连接侧边栏按钮的页面切换
        self._connect_sidebar_buttons()
    
    def _create_sidebar(self) -> QWidget:
        """创建侧边栏"""
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(50)  # 恢复之前的宽度
        sidebar.setStyleSheet(StyleSheet.SIDEBAR)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(10)
        
        # 训练按钮
        self.btn_train = SidebarButton(svg_template=IconSvg.TRAIN, icon_size=(24, 24))
        self.btn_train.setFixedSize(50, 50)
        self.btn_train.setChecked(True)
        layout.addWidget(self.btn_train)
        
        # 分类按钮
        self.btn_classify = SidebarButton(svg_template=IconSvg.CLASSIFY, icon_size=(24, 24))
        self.btn_classify.setFixedSize(50, 50)
        layout.addWidget(self.btn_classify)
        
        layout.addStretch()
        
        # 设置按钮
        self.btn_settings = SidebarButton(svg_template=IconSvg.SETTINGS, icon_size=(28, 28))
        self.btn_settings.setFixedSize(50, 50)
        layout.addWidget(self.btn_settings)
        
        return sidebar
    
    def _connect_sidebar_buttons(self):
        """连接侧边栏按钮实现页面切换"""
        # 创建按钮组实现互斥选择
        self.sidebar_group = QButtonGroup(self)
        self.sidebar_group.addButton(self.btn_train, 0)
        self.sidebar_group.addButton(self.btn_classify, 1)
        self.sidebar_group.addButton(self.btn_settings, 2)
        self.sidebar_group.setExclusive(True)
        
        # 连接信号
        self.btn_train.clicked.connect(lambda: self._switch_page(0))
        self.btn_classify.clicked.connect(lambda: self._switch_page(1))
        self.btn_settings.clicked.connect(lambda: self._switch_page(2))
    
    def _switch_page(self, page_index: int):
        """切换页面"""
        if page_index == 2:  # 设置页面
            # 隐藏参数区，设置页面占满整个内容区
            self.param_stack.setFixedWidth(0)
            self.param_stack.setVisible(False)
        else:
            # 显示参数区
            self.param_stack.setFixedWidth(380)
            self.param_stack.setVisible(True)
        
        self.param_stack.setCurrentIndex(page_index)
        self.content_stack.setCurrentIndex(page_index)
    
    def _create_training_param_area(self) -> QWidget:
        """创建参数区"""
        param_area = QWidget()
        param_area.setObjectName("paramArea")
        param_area.setFixedWidth(380)  # 调整宽度
        param_area.setStyleSheet(StyleSheet.PARAM_AREA)
        
        layout = QVBoxLayout(param_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 页面标题区域
        title_bar = QWidget()
        title_bar.setFixedHeight(35)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 0, 20, 0)
        
        title = QLabel("MODEL TRAINING")
        title.setStyleSheet(StyleSheet.PAGE_TITLE)
        title.setFont(FontManager.title_font())
        title_layout.addWidget(title)
        layout.addWidget(title_bar)
        
        # 内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 10, 0, 10)
        content_layout.setSpacing(0)
        
        # 参数卡片1: 训练模式、模型类型、模型位置
        card1 = ParamCard()
        
        self.combo_mode = ComboRow("训练模式", [
            "从预训练权重开始", 
            "从已有模型继续训练",
            "Fine-tuning已有模型"
        ])
        card1.add_row(self.combo_mode, show_divider=False)
        
        self.combo_model = ComboRow("模型类型", [
            "tf_efficientnetv2_s", 
            "convnext_tiny", 
            "resnet50",
            "vit_base_patch16_224", 
            "swin_tiny_patch4_window7_224"
        ])
        card1.add_row(self.combo_model, show_divider=False)
        
        self.file_model = FileRow("基础模型", "", is_folder=False)
        card1.add_row(self.file_model, show_divider=False)
        
        content_layout.addWidget(card1)
        
        # 分割线
        divider1 = QFrame()
        divider1.setStyleSheet(f"background-color: {StyleSheet.DIVIDER}; margin: 10px 20px;")
        divider1.setFixedHeight(1)
        content_layout.addWidget(divider1)
        
        # 参数卡片2: 滑块参数 (根据传统UI的参数范围)
        card2 = ParamCard()
        
        # 训练轮数: 范围1-100, 默认10
        self.slider_epochs = SliderRow(
            "训练轮数", 1, 100, 10,
            display_format="int",
            min_label_text="1", max_label_text="100"
        )
        card2.add_row(self.slider_epochs, show_divider=False)
        
        # 批次大小: 范围1-128, 默认16 (步进为2的幂次更合理)
        self.slider_batch = SliderRow(
            "批次大小", 1, 128, 16,
            display_format="int",
            min_label_text="1", max_label_text="128"
        )
        card2.add_row(self.slider_batch, show_divider=False)
        
        # 学习率: 范围0.0001-0.1, 默认0.001 (使用scale缩放)
        # 滑块值1-100对应0.0001-0.01, 默认值10对应0.001
        self.slider_lr = SliderRow(
            "学习率", 1, 100, 10,
            display_format="lr",
            scale=0.0001,
            min_label_text="0.0001", max_label_text="0.01",
            value_width=55
        )
        card2.add_row(self.slider_lr, show_divider=False)
        
        # 验证比例: 范围10%-50%, 默认20%
        self.slider_val = SliderRow(
            "验证比例", 10, 50, 20,
            display_format="percent",
            scale=1.0,
            min_label_text="10%", max_label_text="50%"
        )
        card2.add_row(self.slider_val, show_divider=False)
        
        content_layout.addWidget(card2)
        
        # 分割线
        divider2 = QFrame()
        divider2.setStyleSheet(f"background-color: {StyleSheet.DIVIDER}; margin: 10px 20px;")
        divider2.setFixedHeight(1)
        content_layout.addWidget(divider2)
        
        # 参数卡片3: 数据路径
        card3 = ParamCard()
        self.file_data = FileRow("数据路径", "", is_folder=True)
        card3.add_row(self.file_data, show_divider=False)
        content_layout.addWidget(card3)
        
        content_layout.addStretch()
        layout.addWidget(content_widget)
        
        return param_area
    
    def _create_terminal_area(self) -> QWidget:
        """创建终端区"""
        terminal_area = QWidget()
        terminal_area.setObjectName("terminalArea")
        terminal_area.setStyleSheet(StyleSheet.TERMINAL_AREA)
        
        layout = QVBoxLayout(terminal_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 顶部: 标题 + 窗口控制按钮
        top_bar = QWidget()
        top_bar.setFixedHeight(72)
        
        # 使用嵌套布局：外层垂直布局，内层水平布局
        top_outer_layout = QVBoxLayout(top_bar)
        top_outer_layout.setContentsMargins(0, 0, 0, 0)
        top_outer_layout.setSpacing(0)
        
        # 按钮行容器 - 贴顶
        btn_row = QWidget()
        btn_row.setFixedHeight(32)
        btn_row_layout = QHBoxLayout(btn_row)
        btn_row_layout.setContentsMargins(0, 0, 0, 0)
        btn_row_layout.setSpacing(0)
        
        btn_row_layout.addStretch()
        
        # 窗口控制按钮 - 贴顶对齐
        btn_min = QPushButton()
        btn_min.setFixedSize(46, 32)
        btn_min.setIcon(self._create_svg_icon(IconSvg.MINIMIZE, "#D1D1D1"))
        btn_min.setIconSize(QSize(12, 2))
        btn_min.setStyleSheet(StyleSheet.CONTROL_BTN)
        btn_min.clicked.connect(self.showMinimized)
        btn_row_layout.addWidget(btn_min)
        
        btn_max = QPushButton()
        btn_max.setFixedSize(46, 32)
        btn_max.setIcon(self._create_svg_icon(IconSvg.MAXIMIZE, "#D1D1D1"))
        btn_max.setIconSize(QSize(12, 12))
        btn_max.setStyleSheet(StyleSheet.CONTROL_BTN)
        btn_max.clicked.connect(self._toggle_maximize)
        btn_row_layout.addWidget(btn_max)
        
        btn_close = QPushButton()
        btn_close.setFixedSize(46, 32)
        btn_close.setIcon(self._create_svg_icon(IconSvg.CLOSE, "#D1D1D1"))
        btn_close.setIconSize(QSize(12, 12))
        btn_close.setStyleSheet(StyleSheet.CLOSE_BTN)
        btn_close.clicked.connect(self.close)
        btn_row_layout.addWidget(btn_close)
        
        top_outer_layout.addWidget(btn_row)
        
        # 标题行容器
        title_row = QWidget()
        title_row_layout = QHBoxLayout(title_row)
        title_row_layout.setContentsMargins(20, 0, 0, 0)
        
        # 终端标题 - MiSans Semibold 24px
        title = QLabel("终端")
        title.setStyleSheet(StyleSheet.TERMINAL_TITLE)
        title.setFont(FontManager.header_font())
        title_row_layout.addWidget(title)
        title_row_layout.addStretch()
        
        top_outer_layout.addWidget(title_row)
        
        layout.addWidget(top_bar)
        
        # 分割线
        divider1 = QFrame()
        divider1.setStyleSheet(StyleSheet.DIVIDER_H)
        layout.addWidget(divider1)
        
        # 终端输出
        self.terminal = TerminalOutput()
        layout.addWidget(self.terminal)
        
        # 分割线
        divider2 = QFrame()
        divider2.setStyleSheet(StyleSheet.DIVIDER_H)
        layout.addWidget(divider2)
        
        # 底部按钮
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(80)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        
        btn_start = QPushButton("开始训练")
        btn_start.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn_start.setStyleSheet(StyleSheet.BTN_START)
        btn_start.setFont(FontManager.action_button_font())
        btn_start.clicked.connect(self._start_training)
        bottom_layout.addWidget(btn_start)
        
        # 按钮分割线
        btn_divider = QFrame()
        btn_divider.setFixedWidth(1)
        btn_divider.setStyleSheet(f"background-color: {StyleSheet.DIVIDER};")
        bottom_layout.addWidget(btn_divider)
        
        btn_stop = QPushButton("停止训练")
        btn_stop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn_stop.setStyleSheet(StyleSheet.BTN_STOP)
        btn_stop.setFont(FontManager.action_button_font())
        btn_stop.clicked.connect(self._stop_training)
        bottom_layout.addWidget(btn_stop)
        
        layout.addWidget(bottom_bar)
        
        return terminal_area
    
    def _create_classify_param_area(self) -> QWidget:
        """创建分类页面的参数区"""
        param_area = QWidget()
        param_area.setObjectName("paramArea")
        param_area.setStyleSheet(StyleSheet.PARAM_AREA)
        
        layout = QVBoxLayout(param_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 页面标题区域
        title_bar = QWidget()
        title_bar.setFixedHeight(35)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 0, 20, 0)
        
        title = QLabel("IMAGE CLASSIFICATION")
        title.setStyleSheet(StyleSheet.PAGE_TITLE)
        title.setFont(FontManager.title_font())
        title_layout.addWidget(title)
        layout.addWidget(title_bar)
        
        # 内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 10, 0, 10)
        content_layout.setSpacing(0)
        
        # 参数卡片1: 图像选择
        card1 = ParamCard()
        
        self.classify_single_file = FileRow("单个文件", "", is_folder=False)
        card1.add_row(self.classify_single_file, show_divider=False)
        
        self.classify_folder = FileRow("文件夹", "", is_folder=True)
        card1.add_row(self.classify_folder, show_divider=False)
        
        content_layout.addWidget(card1)
        
        # 分割线
        divider1 = QFrame()
        divider1.setStyleSheet(f"background-color: {StyleSheet.DIVIDER}; margin: 10px 20px;")
        divider1.setFixedHeight(1)
        content_layout.addWidget(divider1)
        
        # 参数卡片2: 模型选择
        card2 = ParamCard()
        
        self.classify_model_file = FileRow("模型文件", "", is_folder=False)
        card2.add_row(self.classify_model_file, show_divider=False)
        
        # 模型状态显示行
        status_row = QWidget()
        status_row.setFixedHeight(40)
        status_layout = QHBoxLayout(status_row)
        status_layout.setContentsMargins(20, 0, 20, 0)
        
        status_label = QLabel("模型状态")
        status_label.setStyleSheet(StyleSheet.PARAM_LABEL)
        status_label.setFont(FontManager.label_font())
        status_layout.addWidget(status_label)
        
        status_layout.addStretch()
        
        self.model_status_indicator = QLabel("未加载")
        self.model_status_indicator.setStyleSheet(f"""
            QLabel {{
                color: #F48771;
                font-size: 12px;
                padding: 2px 8px;
                background-color: rgba(244, 135, 113, 0.1);
                border-radius: 2px;
            }}
        """)
        self.model_status_indicator.setFont(FontManager.input_font())
        status_layout.addWidget(self.model_status_indicator)
        
        card2.add_row(status_row, show_divider=False)
        
        content_layout.addWidget(card2)
        
        # 分割线
        divider2 = QFrame()
        divider2.setStyleSheet(f"background-color: {StyleSheet.DIVIDER}; margin: 10px 20px;")
        divider2.setFixedHeight(1)
        content_layout.addWidget(divider2)
        
        # 参数卡片3: 快速操作按钮
        card3 = ParamCard()
        
        # 加载模型按钮行
        btn_row1 = QWidget()
        btn_row1.setFixedHeight(40)
        btn_layout1 = QHBoxLayout(btn_row1)
        btn_layout1.setContentsMargins(20, 0, 20, 0)
        
        self.btn_load_model = QPushButton("加载模型")
        self.btn_load_model.setFixedHeight(26)
        self.btn_load_model.setStyleSheet(f"""
            QPushButton {{
                background-color: {StyleSheet.VS_BUTTON_BG};
                color: {StyleSheet.VS_BUTTON_FG};
                border: none;
                border-radius: 2px;
                padding: 0 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {StyleSheet.VS_BUTTON_HOVER};
            }}
        """)
        self.btn_load_model.setFont(FontManager.button_font())
        self.btn_load_model.clicked.connect(self._load_classify_model)
        btn_layout1.addWidget(self.btn_load_model)
        
        self.btn_use_default = QPushButton("使用默认模型")
        self.btn_use_default.setFixedHeight(26)
        self.btn_use_default.setStyleSheet(f"""
            QPushButton {{
                background-color: #3A3D41;
                color: {StyleSheet.VS_FOREGROUND};
                border: none;
                border-radius: 2px;
                padding: 0 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: #4A4D51;
            }}
        """)
        self.btn_use_default.setFont(FontManager.button_font())
        self.btn_use_default.clicked.connect(self._use_default_model)
        btn_layout1.addWidget(self.btn_use_default)
        
        btn_layout1.addStretch()
        
        card3.add_row(btn_row1, show_divider=False)
        content_layout.addWidget(card3)
        
        content_layout.addStretch()
        layout.addWidget(content_widget)
        
        return param_area
    
    def _create_classify_result_area(self) -> QWidget:
        """创建分类页面的结果区"""
        result_area = QWidget()
        result_area.setObjectName("terminalArea")
        result_area.setStyleSheet(StyleSheet.TERMINAL_AREA)
        
        layout = QVBoxLayout(result_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 顶部: 标题 + 窗口控制按钮
        top_bar = QWidget()
        top_bar.setFixedHeight(72)
        
        top_outer_layout = QVBoxLayout(top_bar)
        top_outer_layout.setContentsMargins(0, 0, 0, 0)
        top_outer_layout.setSpacing(0)
        
        # 按钮行容器 - 贴顶
        btn_row = QWidget()
        btn_row.setFixedHeight(32)
        btn_row_layout = QHBoxLayout(btn_row)
        btn_row_layout.setContentsMargins(0, 0, 0, 0)
        btn_row_layout.setSpacing(0)
        
        btn_row_layout.addStretch()
        
        # 窗口控制按钮
        btn_min = QPushButton()
        btn_min.setFixedSize(46, 32)
        btn_min.setIcon(self._create_svg_icon(IconSvg.MINIMIZE, "#D1D1D1"))
        btn_min.setIconSize(QSize(12, 2))
        btn_min.setStyleSheet(StyleSheet.CONTROL_BTN)
        btn_min.clicked.connect(self.showMinimized)
        btn_row_layout.addWidget(btn_min)
        
        btn_max = QPushButton()
        btn_max.setFixedSize(46, 32)
        btn_max.setIcon(self._create_svg_icon(IconSvg.MAXIMIZE, "#D1D1D1"))
        btn_max.setIconSize(QSize(12, 12))
        btn_max.setStyleSheet(StyleSheet.CONTROL_BTN)
        btn_max.clicked.connect(self._toggle_maximize)
        btn_row_layout.addWidget(btn_max)
        
        btn_close = QPushButton()
        btn_close.setFixedSize(46, 32)
        btn_close.setIcon(self._create_svg_icon(IconSvg.CLOSE, "#D1D1D1"))
        btn_close.setIconSize(QSize(12, 12))
        btn_close.setStyleSheet(StyleSheet.CLOSE_BTN)
        btn_close.clicked.connect(self.close)
        btn_row_layout.addWidget(btn_close)
        
        top_outer_layout.addWidget(btn_row)
        
        # 标题行容器
        title_row = QWidget()
        title_row_layout = QHBoxLayout(title_row)
        title_row_layout.setContentsMargins(20, 0, 0, 0)
        
        title = QLabel("分类结果")
        title.setStyleSheet(StyleSheet.TERMINAL_TITLE)
        title.setFont(FontManager.header_font())
        title_row_layout.addWidget(title)
        title_row_layout.addStretch()
        
        # 统计信息
        self.classify_stats_label = QLabel("共 0 张图片")
        self.classify_stats_label.setStyleSheet(f"color: {StyleSheet.VS_FOREGROUND}; font-size: 12px;")
        self.classify_stats_label.setFont(FontManager.input_font())
        title_row_layout.addWidget(self.classify_stats_label)
        title_row_layout.addSpacing(20)
        
        top_outer_layout.addWidget(title_row)
        
        layout.addWidget(top_bar)
        
        # 分割线
        divider1 = QFrame()
        divider1.setStyleSheet(StyleSheet.DIVIDER_H)
        layout.addWidget(divider1)
        
        # 进度条
        self.classify_progress = QProgressBar()
        self.classify_progress.setStyleSheet(StyleSheet.PROGRESS_BAR)
        self.classify_progress.setTextVisible(False)
        self.classify_progress.setVisible(False)
        layout.addWidget(self.classify_progress)
        
        # 分类结果表格
        self.result_table = ClassificationResultTable()
        layout.addWidget(self.result_table)
        
        # 分割线
        divider2 = QFrame()
        divider2.setStyleSheet(StyleSheet.DIVIDER_H)
        layout.addWidget(divider2)
        
        # 底部按钮
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(80)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        
        btn_classify = QPushButton("开始分类")
        btn_classify.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn_classify.setStyleSheet(StyleSheet.BTN_CLASSIFY)
        btn_classify.setFont(FontManager.action_button_font())
        btn_classify.clicked.connect(self._start_classification)
        bottom_layout.addWidget(btn_classify)
        
        # 按钮分割线
        btn_divider = QFrame()
        btn_divider.setFixedWidth(1)
        btn_divider.setStyleSheet(f"background-color: {StyleSheet.DIVIDER};")
        bottom_layout.addWidget(btn_divider)
        
        btn_clear = QPushButton("清空结果")
        btn_clear.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn_clear.setStyleSheet(StyleSheet.BTN_CLEAR)
        btn_clear.setFont(FontManager.action_button_font())
        btn_clear.clicked.connect(self._clear_classification_results)
        bottom_layout.addWidget(btn_clear)
        
        layout.addWidget(bottom_bar)
        
        return result_area
    
    def _create_settings_area(self) -> QWidget:
        """创建设置页面 - VS Code 风格"""
        settings_area = QWidget()
        settings_area.setObjectName("settingsArea")
        settings_area.setStyleSheet(StyleSheet.SETTINGS_AREA)
        
        layout = QVBoxLayout(settings_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 顶部标题栏
        top_bar = QWidget()
        top_bar.setFixedHeight(80)
        top_outer_layout = QVBoxLayout(top_bar)
        top_outer_layout.setContentsMargins(0, 0, 0, 0)
        top_outer_layout.setSpacing(0)
        
        # 窗口控制按钮
        control_bar = QWidget()
        control_bar.setFixedHeight(35)
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addStretch()
        
        btn_min = QPushButton()
        btn_min.setFixedSize(46, 35)
        btn_min.setIcon(self._create_svg_icon(IconSvg.MINIMIZE, StyleSheet.VS_FOREGROUND))
        btn_min.setIconSize(QSize(16, 16))
        btn_min.setStyleSheet(StyleSheet.CONTROL_BTN)
        btn_min.clicked.connect(self.showMinimized)
        control_layout.addWidget(btn_min)
        
        btn_max = QPushButton()
        btn_max.setFixedSize(46, 35)
        btn_max.setIcon(self._create_svg_icon(IconSvg.MAXIMIZE, StyleSheet.VS_FOREGROUND))
        btn_max.setIconSize(QSize(16, 16))
        btn_max.setStyleSheet(StyleSheet.CONTROL_BTN)
        btn_max.clicked.connect(self._toggle_maximize)
        control_layout.addWidget(btn_max)
        
        btn_close = QPushButton()
        btn_close.setFixedSize(46, 35)
        btn_close.setIcon(self._create_svg_icon(IconSvg.CLOSE, StyleSheet.VS_FOREGROUND))
        btn_close.setIconSize(QSize(14, 14))
        btn_close.setStyleSheet(StyleSheet.CLOSE_BTN)
        btn_close.clicked.connect(self.close)
        control_layout.addWidget(btn_close)
        
        top_outer_layout.addWidget(control_bar)
        
        # 标题行
        title_row = QWidget()
        title_row_layout = QHBoxLayout(title_row)
        title_row_layout.setContentsMargins(24, 0, 24, 0)
        
        title = QLabel("设置")
        title.setStyleSheet(StyleSheet.TERMINAL_TITLE)
        title.setFont(FontManager.header_font())
        title_row_layout.addWidget(title)
        title_row_layout.addStretch()
        
        top_outer_layout.addWidget(title_row)
        layout.addWidget(top_bar)
        
        # 分割线
        divider = QFrame()
        divider.setStyleSheet(StyleSheet.DIVIDER_H)
        layout.addWidget(divider)
        
        # 可滚动的设置内容区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(StyleSheet.SETTINGS_SCROLL)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(24, 0, 24, 24)
        scroll_layout.setSpacing(0)
        
        # ===== 外观设置 =====
        scroll_layout.addWidget(self._create_section_title("外观设置"))
        
        # 主题
        row_theme = SettingsRow("主题", "选择应用程序的外观主题")
        self.settings_theme = QComboBox()
        self.settings_theme.addItems(["深色模式", "浅色模式", "跟随系统"])
        self.settings_theme.setStyleSheet(StyleSheet.INPUT_BOX)
        self.settings_theme.setFont(FontManager.input_font())
        self.settings_theme.setFixedWidth(180)
        row_theme.set_control(self.settings_theme)
        scroll_layout.addWidget(row_theme)
        
        # 界面缩放
        row_scale = SettingsRow("界面缩放", "调整界面元素的大小")
        self.settings_scale = QComboBox()
        self.settings_scale.addItems(["100%", "125%", "150%", "175%", "200%"])
        self.settings_scale.setStyleSheet(StyleSheet.INPUT_BOX)
        self.settings_scale.setFont(FontManager.input_font())
        self.settings_scale.setFixedWidth(180)
        row_scale.set_control(self.settings_scale)
        scroll_layout.addWidget(row_scale)
        
        # ===== 模型设置 =====
        scroll_layout.addWidget(self._create_section_title("模型设置"))
        
        # 推理设备
        row_device = SettingsRow("推理设备", "选择用于模型推理的计算设备")
        self.settings_device = QComboBox()
        self.settings_device.addItems(["自动检测", "CPU", "GPU (CUDA)"])
        self.settings_device.setStyleSheet(StyleSheet.INPUT_BOX)
        self.settings_device.setFont(FontManager.input_font())
        self.settings_device.setFixedWidth(180)
        row_device.set_control(self.settings_device)
        scroll_layout.addWidget(row_device)
        
        # 推理精度
        row_precision = SettingsRow("推理精度", "FP16 可加速推理但可能降低精度")
        self.settings_precision = QComboBox()
        self.settings_precision.addItems(["FP32 (默认)", "FP16 (加速)"])
        self.settings_precision.setStyleSheet(StyleSheet.INPUT_BOX)
        self.settings_precision.setFont(FontManager.input_font())
        self.settings_precision.setFixedWidth(180)
        row_precision.set_control(self.settings_precision)
        scroll_layout.addWidget(row_precision)
        
        # 置信度阈值
        row_conf = SettingsRow("置信度阈值", "低于此阈值的分类结果将被标记为不确定")
        self.settings_confidence = QDoubleSpinBox()
        self.settings_confidence.setRange(0.1, 0.99)
        self.settings_confidence.setValue(0.5)
        self.settings_confidence.setSingleStep(0.05)
        self.settings_confidence.setDecimals(2)
        self.settings_confidence.setStyleSheet(StyleSheet.SETTINGS_SPINBOX)
        self.settings_confidence.setFont(FontManager.input_font())
        row_conf.set_control(self.settings_confidence)
        scroll_layout.addWidget(row_conf)
        
        # ===== 路径配置 =====
        scroll_layout.addWidget(self._create_section_title("路径配置"))
        
        # 默认模型目录
        row_model_dir = SettingsRow("默认模型目录", "模型文件的存储位置")
        self.settings_default_model = QLineEdit("models/saved_models")
        self.settings_default_model.setStyleSheet(StyleSheet.INPUT_BOX)
        self.settings_default_model.setFont(FontManager.input_font())
        self.settings_default_model.setFixedWidth(280)
        row_model_dir.set_control(self.settings_default_model)
        scroll_layout.addWidget(row_model_dir)
        
        # 数据集目录
        row_data_dir = SettingsRow("数据集目录", "训练和验证数据的存储位置")
        self.settings_dataset_dir = QLineEdit("data")
        self.settings_dataset_dir.setStyleSheet(StyleSheet.INPUT_BOX)
        self.settings_dataset_dir.setFont(FontManager.input_font())
        self.settings_dataset_dir.setFixedWidth(280)
        row_data_dir.set_control(self.settings_dataset_dir)
        scroll_layout.addWidget(row_data_dir)
        
        # 日志目录
        row_log_dir = SettingsRow("日志目录", "运行日志的存储位置")
        self.settings_log_dir = QLineEdit("logs")
        self.settings_log_dir.setStyleSheet(StyleSheet.INPUT_BOX)
        self.settings_log_dir.setFont(FontManager.input_font())
        self.settings_log_dir.setFixedWidth(280)
        row_log_dir.set_control(self.settings_log_dir)
        scroll_layout.addWidget(row_log_dir)
        
        # 导出目录
        row_export_dir = SettingsRow("导出目录", "分类结果的导出位置")
        self.settings_export_dir = QLineEdit("outputs")
        self.settings_export_dir.setStyleSheet(StyleSheet.INPUT_BOX)
        self.settings_export_dir.setFont(FontManager.input_font())
        self.settings_export_dir.setFixedWidth(280)
        row_export_dir.set_control(self.settings_export_dir)
        scroll_layout.addWidget(row_export_dir)
        
        # ===== 性能设置 =====
        scroll_layout.addWidget(self._create_section_title("性能设置"))
        
        # 数据加载线程
        row_workers = SettingsRow("数据加载线程", "用于加载数据的并行线程数")
        self.settings_workers = QSpinBox()
        self.settings_workers.setRange(1, 16)
        self.settings_workers.setValue(4)
        self.settings_workers.setStyleSheet(StyleSheet.SETTINGS_SPINBOX)
        self.settings_workers.setFont(FontManager.input_font())
        row_workers.set_control(self.settings_workers)
        scroll_layout.addWidget(row_workers)
        
        # 批处理上限
        row_batch = SettingsRow("批处理上限", "单次处理的最大图片数量")
        self.settings_max_batch = QSpinBox()
        self.settings_max_batch.setRange(1, 256)
        self.settings_max_batch.setValue(64)
        self.settings_max_batch.setStyleSheet(StyleSheet.SETTINGS_SPINBOX)
        self.settings_max_batch.setFont(FontManager.input_font())
        row_batch.set_control(self.settings_max_batch)
        scroll_layout.addWidget(row_batch)
        
        # 混合精度训练
        self.row_amp = SettingsCheckRow("启用混合精度训练 (AMP)", "使用 FP16 加速训练，同时保持模型精度")
        self.row_amp.setChecked(True)
        scroll_layout.addWidget(self.row_amp)
        
        # 内存锁定
        self.row_pin_memory = SettingsCheckRow("启用内存锁定 (Pin Memory)", "锁定内存可加速 GPU 数据传输")
        self.row_pin_memory.setChecked(True)
        scroll_layout.addWidget(self.row_pin_memory)
        
        # ===== 训练默认值 =====
        scroll_layout.addWidget(self._create_section_title("训练默认值"))
        
        # 默认训练轮数
        row_epochs = SettingsRow("默认训练轮数", "新建训练任务的默认轮数")
        self.settings_default_epochs = QSpinBox()
        self.settings_default_epochs.setRange(1, 1000)
        self.settings_default_epochs.setValue(50)
        self.settings_default_epochs.setStyleSheet(StyleSheet.SETTINGS_SPINBOX)
        self.settings_default_epochs.setFont(FontManager.input_font())
        row_epochs.set_control(self.settings_default_epochs)
        scroll_layout.addWidget(row_epochs)
        
        # 早停耐心值
        row_patience = SettingsRow("早停耐心值", "验证损失不再下降后等待的轮数")
        self.settings_patience = QSpinBox()
        self.settings_patience.setRange(1, 100)
        self.settings_patience.setValue(10)
        self.settings_patience.setStyleSheet(StyleSheet.SETTINGS_SPINBOX)
        self.settings_patience.setFont(FontManager.input_font())
        row_patience.set_control(self.settings_patience)
        scroll_layout.addWidget(row_patience)
        
        # 检查点保存间隔
        row_ckpt = SettingsRow("检查点保存间隔", "每隔多少轮保存一次模型检查点")
        self.settings_checkpoint_freq = QSpinBox()
        self.settings_checkpoint_freq.setRange(1, 100)
        self.settings_checkpoint_freq.setValue(5)
        self.settings_checkpoint_freq.setStyleSheet(StyleSheet.SETTINGS_SPINBOX)
        self.settings_checkpoint_freq.setFont(FontManager.input_font())
        row_ckpt.set_control(self.settings_checkpoint_freq)
        scroll_layout.addWidget(row_ckpt)
        
        # 仅保存最佳模型
        self.row_save_best = SettingsCheckRow("仅保存最佳模型", "只保留验证精度最高的模型")
        self.row_save_best.setChecked(True)
        scroll_layout.addWidget(self.row_save_best)
        
        # ===== 日志与调试 =====
        scroll_layout.addWidget(self._create_section_title("日志与调试"))
        
        # 日志级别
        row_log_level = SettingsRow("日志级别", "控制日志输出的详细程度")
        self.settings_log_level = QComboBox()
        self.settings_log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.settings_log_level.setCurrentIndex(1)
        self.settings_log_level.setStyleSheet(StyleSheet.INPUT_BOX)
        self.settings_log_level.setFont(FontManager.input_font())
        self.settings_log_level.setFixedWidth(180)
        row_log_level.set_control(self.settings_log_level)
        scroll_layout.addWidget(row_log_level)
        
        # 保留日志天数
        row_log_days = SettingsRow("保留日志天数", "自动清理超过指定天数的日志文件")
        self.settings_log_days = QSpinBox()
        self.settings_log_days.setRange(1, 365)
        self.settings_log_days.setValue(30)
        self.settings_log_days.setStyleSheet(StyleSheet.SETTINGS_SPINBOX)
        self.settings_log_days.setFont(FontManager.input_font())
        row_log_days.set_control(self.settings_log_days)
        scroll_layout.addWidget(row_log_days)
        
        # 详细日志
        self.row_verbose = SettingsCheckRow("启用详细日志", "记录更多调试信息，用于问题排查")
        scroll_layout.addWidget(self.row_verbose)
        
        # ===== 其他设置 =====
        scroll_layout.addWidget(self._create_section_title("其他设置"))
        
        # 自动加载模型
        self.row_auto_load = SettingsCheckRow("启动时自动加载默认模型", "程序启动时自动加载上次使用的模型")
        self.row_auto_load.setChecked(True)
        scroll_layout.addWidget(self.row_auto_load)
        
        # 记住窗口位置
        self.row_remember_window = SettingsCheckRow("记住窗口位置和大小", "下次启动时恢复窗口状态")
        self.row_remember_window.setChecked(True)
        scroll_layout.addWidget(self.row_remember_window)
        
        # 任务完成通知
        self.row_notify = SettingsCheckRow("任务完成后发送系统通知", "训练或分类完成时显示桌面通知")
        self.row_notify.setChecked(True)
        scroll_layout.addWidget(self.row_notify)
        
        # 弹性空间
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # 底部按钮区
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(60)
        bottom_bar.setStyleSheet(f"background-color: {StyleSheet.TERMINAL_BG};")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(24, 0, 24, 0)
        bottom_layout.setSpacing(12)
        
        bottom_layout.addStretch()
        
        btn_reset = QPushButton("恢复默认")
        btn_reset.setStyleSheet(StyleSheet.BTN_SETTINGS_SECONDARY)
        btn_reset.setFont(FontManager.button_font())
        btn_reset.clicked.connect(self._reset_settings)
        bottom_layout.addWidget(btn_reset)
        
        btn_save = QPushButton("保存设置")
        btn_save.setStyleSheet(StyleSheet.BTN_SETTINGS_ACTION)
        btn_save.setFont(FontManager.button_font())
        btn_save.clicked.connect(self._save_settings)
        bottom_layout.addWidget(btn_save)
        
        layout.addWidget(bottom_bar)
        
        return settings_area
    
    def _create_section_title(self, title: str) -> QLabel:
        """创建设置分区标题"""
        label = QLabel(title)
        label.setStyleSheet(StyleSheet.SETTINGS_SECTION_TITLE)
        label.setFont(FontManager.title_font())
        return label
    
    def _reset_settings(self):
        """恢复默认设置"""
        # 外观
        self.settings_theme.setCurrentIndex(0)
        self.settings_scale.setCurrentIndex(0)
        
        # 模型
        self.settings_device.setCurrentIndex(0)
        self.settings_precision.setCurrentIndex(0)
        self.settings_confidence.setValue(0.5)
        
        # 路径
        self.settings_default_model.setText("models/saved_models")
        self.settings_dataset_dir.setText("data")
        self.settings_log_dir.setText("logs")
        self.settings_export_dir.setText("outputs")
        
        # 性能
        self.settings_workers.setValue(4)
        self.settings_max_batch.setValue(64)
        self.row_amp.setChecked(True)
        self.row_pin_memory.setChecked(True)
        
        # 训练
        self.settings_default_epochs.setValue(50)
        self.settings_patience.setValue(10)
        self.settings_checkpoint_freq.setValue(5)
        self.row_save_best.setChecked(True)
        
        # 日志
        self.settings_log_level.setCurrentIndex(1)
        self.settings_log_days.setValue(30)
        self.row_verbose.setChecked(False)
        
        # 其他
        self.row_auto_load.setChecked(True)
        self.row_remember_window.setChecked(True)
        self.row_notify.setChecked(True)
        
        self.terminal.append_log("已恢复默认设置", "#FFCC00")
    
    def _save_settings(self):
        """保存设置"""
        # TODO: 实际保存设置到配置文件
        self.terminal.append_log("设置已保存", "#13C468")

    def _toggle_maximize(self):
        """切换最大化/还原"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
    
    def _start_training(self):
        """开始训练"""
        self.terminal.append_log("开始训练...", "#13C468")
        self.terminal.append_log(f"训练轮数: {self.slider_epochs.value()}")
        self.terminal.append_log(f"批次大小: {self.slider_batch.value()}")
        self.terminal.append_log(f"学习率: {self.slider_lr.real_value():.4f}")
        self.terminal.append_log(f"验证比例: {self.slider_val.value()}%")
        # TODO: 调用实际训练逻辑
    
    def _stop_training(self):
        """停止训练"""
        self.terminal.append_log("停止训练", "#FF6B6B")
        # TODO: 停止训练逻辑
    
    # ===== 分类功能 =====
    def _load_classify_model(self):
        """加载分类模型"""
        model_path = self.classify_model_file.path_edit.text()
        if not model_path:
            self._update_model_status("未选择模型", False)
            return
        
        # TODO: 实际加载模型逻辑
        self._update_model_status("已加载", True)
    
    def _use_default_model(self):
        """使用默认模型"""
        # TODO: 加载默认模型
        default_path = "models/best_model.pth"
        self.classify_model_file.path_edit.setText(default_path)
        self._update_model_status("已加载默认模型", True)
    
    def _update_model_status(self, status: str, loaded: bool):
        """更新模型状态显示"""
        if loaded:
            self.model_status_indicator.setText(status)
            self.model_status_indicator.setStyleSheet(f"""
                QLabel {{
                    color: #89D185;
                    font-size: 12px;
                    padding: 2px 8px;
                    background-color: rgba(137, 209, 133, 0.1);
                    border-radius: 2px;
                }}
            """)
        else:
            self.model_status_indicator.setText(status)
            self.model_status_indicator.setStyleSheet(f"""
                QLabel {{
                    color: #F48771;
                    font-size: 12px;
                    padding: 2px 8px;
                    background-color: rgba(244, 135, 113, 0.1);
                    border-radius: 2px;
                }}
            """)
    
    def _start_classification(self):
        """开始分类"""
        # 显示进度条
        self.classify_progress.setVisible(True)
        self.classify_progress.setValue(0)
        
        # 获取输入路径
        single_file = self.classify_single_file.path_edit.text()
        folder = self.classify_folder.path_edit.text()
        
        if not single_file and not folder:
            self._update_model_status("请选择图像", False)
            return
        
        # TODO: 实际分类逻辑
        # 这里添加演示数据
        import random
        categories = ["主图", "细节", "吊牌"]
        
        # 模拟分类结果
        demo_results = [
            ("IMG_001.jpg", random.choice(categories), random.uniform(0.7, 0.99), "/path/to/主图/IMG_001.jpg"),
            ("IMG_002.jpg", random.choice(categories), random.uniform(0.7, 0.99), "/path/to/细节/IMG_002.jpg"),
            ("IMG_003.jpg", random.choice(categories), random.uniform(0.7, 0.99), "/path/to/吊牌/IMG_003.jpg"),
            ("IMG_004.jpg", random.choice(categories), random.uniform(0.7, 0.99), "/path/to/主图/IMG_004.jpg"),
            ("IMG_005.jpg", random.choice(categories), random.uniform(0.7, 0.99), "/path/to/细节/IMG_005.jpg"),
        ]
        
        for i, (name, cat, conf, path) in enumerate(demo_results):
            self.result_table.add_result(name, cat, conf, path)
            self.classify_progress.setValue(int((i + 1) / len(demo_results) * 100))
        
        # 更新统计
        self.classify_stats_label.setText(f"共 {len(demo_results)} 张图片")
        self.classify_progress.setVisible(False)
    
    def _clear_classification_results(self):
        """清空分类结果"""
        self.result_table.clear_results()
        self.classify_stats_label.setText("共 0 张图片")
        self.classify_progress.setValue(0)
        self.classify_progress.setVisible(False)
    
    # ===== 窗口拖拽 =====
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.position().y() < 80:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None


def _ensure_icons():
    """确保图标文件存在"""
    icon_dir = PROJECT_ROOT / "src" / "gui"
    icon_dir.mkdir(parents=True, exist_ok=True)
    
    chevron_path = icon_dir / "chevron_down.svg"
    # 始终重新生成以确保颜色正确
    with open(chevron_path, "w", encoding="utf-8") as f:
        f.write(IconSvg.CHEVRON_DOWN.replace("{color}", StyleSheet.VS_FOREGROUND))


def main():
    """程序入口"""
    # 适配高 DPI
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("JiLing 服装分类系统")
    
    # 确保图标存在
    _ensure_icons()
    
    # 加载 MiSans 字体
    if FontManager.load_fonts():
        print("MiSans 字体加载成功")
        # 设置应用默认字体为 MiSans
        default_font = FontManager.get_font(12, QFont.Normal)
        app.setFont(default_font)
    else:
        print("警告: MiSans 字体加载失败，使用系统默认字体")
        font = QFont("Microsoft YaHei", 10)
        app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
