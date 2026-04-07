@echo off
chcp 65001 >nul
title JiLing - 新界面(稳定)
cd /d "%~dp0.."
".conda\python.exe" "run_new_gui.py"
pause

