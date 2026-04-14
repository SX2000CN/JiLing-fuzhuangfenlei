@echo off
chcp 65001 >nul
title JiLing - 新界面(模块化预览)
cd /d "%~dp0.."
".conda\Scripts\python.exe" "run_new_gui.py" --modular
pause

