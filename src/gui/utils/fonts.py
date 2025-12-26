"""
字体管理器 - 加载并管理 MiSans 字体
"""

from pathlib import Path
from PySide6.QtGui import QFont, QFontDatabase

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
FONT_DIR = PROJECT_ROOT / "fonts"


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

        # 优化高DPI字体渲染
        # PreferAntialias: 启用抗锯齿
        # PreferNoHinting: 在高DPI显示器上禁用字体提示以避免锯齿
        font.setStyleStrategy(QFont.PreferAntialias | QFont.PreferQuality)
        font.setHintingPreference(QFont.PreferNoHinting)

        return font

    @classmethod
    def title_font(cls) -> QFont:
        """页面标题字体 - VS Code Section Header Style (12px Bold Uppercase)"""
        font = cls.get_font(12, QFont.DemiBold)
        font.setCapitalization(QFont.AllUppercase)
        font.setLetterSpacing(QFont.PercentageSpacing, 105)
        return font

    @classmethod
    def header_font(cls) -> QFont:
        """大标题字体 - 24px"""
        return cls.get_font(24, QFont.Normal)

    @classmethod
    def label_font(cls) -> QFont:
        """参数标签字体 - VS Code Standard (13px)"""
        return cls.get_font(13, QFont.Medium)

    @classmethod
    def input_font(cls) -> QFont:
        """输入框文字字体 - VS Code Standard (12px)"""
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
        return cls.get_font(13, QFont.Medium)

    @classmethod
    def action_button_font(cls) -> QFont:
        """底部操作按钮字体 - 24px Light"""
        return cls.get_font(24, QFont.Light)
