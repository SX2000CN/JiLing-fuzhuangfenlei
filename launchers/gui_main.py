#!/usr/bin/env python3
"""
JiLing服装分类系统 GUI版本启动脚本
专门用于PyInstaller打包
"""

import sys
import os
from pathlib import Path

def main():
    """GUI版本主程序入口"""
    try:
        # 设置路径
        if getattr(sys, 'frozen', False):
            # 如果是打包后的可执行文件
            application_path = Path(sys.executable).parent
        else:
            # 如果是源码运行，返回到项目根目录
            application_path = Path(__file__).parent.parent
        
        # 添加项目根路径到系统路径
        sys.path.insert(0, str(application_path))
        
        # 添加src路径到系统路径
        src_path = application_path / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        # 导入并启动GUI
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        
        # 设置高DPI缩放（PySide6中这些属性已废弃，Qt6默认启用高DPI）
        # 在Qt6中，高DPI缩放是默认启用的，不需要手动设置
        
        app = QApplication(sys.argv)
        app.setApplicationName("JiLing服装分类系统")
        app.setApplicationVersion("v1.0.0")
        
        # 设置应用样式
        app.setStyle('Fusion')
        
        # 导入并创建主窗口
        try:
            # 优先使用src.gui路径
            from src.gui.main_window import MainWindow
        except ImportError:
            try:
                # 尝试launchers目录下的main_window
                launchers_path = Path(__file__).parent
                sys.path.insert(0, str(launchers_path))
                from main_window import MainWindow
            except ImportError:
                # 最后尝试gui.main_window
                from gui.main_window import MainWindow
        
        window = MainWindow()
        window.show()
        
        # 运行应用
        sys.exit(app.exec())
        
    except ImportError as e:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        messagebox.showerror("导入错误", f"缺少必要的依赖包: {e}\n\n请确保已安装:\n- PySide6\n- PyTorch\n- timm\n\n可以尝试运行:\npip install PySide6 torch timm")
        sys.exit(1)
    except Exception as e:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        messagebox.showerror("启动错误", f"程序启动失败: {e}\n\n请检查:\n1. 是否在正确的目录下运行\n2. 配置文件是否存在\n3. 模型文件是否可用")
        sys.exit(1)

if __name__ == "__main__":
    main()
