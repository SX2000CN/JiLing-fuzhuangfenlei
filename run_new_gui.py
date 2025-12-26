#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动 JiLing 服装分类系统 GUI

使用模块化的新架构：
- 主题支持（深色/浅色/跟随系统）
- 模块化页面（训练/分类/设置）
- VS Code 风格界面
"""

import sys
import os

# 确保项目根目录在 Python 路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 默认使用新的模块化架构
USE_LEGACY = False  # 设为 True 使用旧版 native_ui.py

if __name__ == "__main__":
    if USE_LEGACY or "--legacy" in sys.argv:
        # 使用旧版界面
        from src.gui.native_ui import main
        main()
    else:
        # 使用新的模块化界面
        from src.gui.app import main
        main()
