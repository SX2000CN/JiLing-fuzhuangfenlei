#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JiLing 服装分类系统 - GUI启动脚本
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

if __name__ == "__main__":
    try:
        from gui.main_window import main
        print("🚀 启动JiLing服装分类系统GUI...")
        main()
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有依赖：pip install PySide6 matplotlib tqdm")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动错误: {e}")
        sys.exit(1)
