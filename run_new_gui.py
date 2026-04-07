#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动 JiLing 服装分类系统 GUI

统一 GUI 启动入口：
- 默认：稳定新界面（src.gui.native_ui）
- --modular：模块化新架构（src.gui.app）
- --traditional：传统界面（src.gui.main_window）
"""

import sys
import os

# 确保项目根目录在 Python 路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def resolve_gui_mode(argv):
    """解析启动模式，稳定版优先。"""
    if "--modular" in argv:
        return "modular"
    if "--traditional" in argv:
        return "traditional"
    if "--legacy" in argv:
        return "stable"
    return "stable"

if __name__ == "__main__":
    mode = resolve_gui_mode(sys.argv[1:])

    if mode == "modular":
        from src.gui.app import main
    elif mode == "traditional":
        from src.gui.main_window import main
    else:
        # 默认使用稳定版新界面
        from src.gui.native_ui import main

    main()
