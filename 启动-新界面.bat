@echo off
chcp 65001 >nul
title JiLing 服装分类系统 - 新界面
cd /d "%~dp0"
python src/gui/native_ui.py
pause
