"""
GUI 模块入口

这是模块化 UI 的主入口点。
使用方法:
    from src.gui import main
    main()
"""

from .app import main, MainWindow

__all__ = ['main', 'MainWindow']
