@echo off
:: 快速启动 - 现代界面
cd /d "%~dp0.."
call .conda\python.exe "launchers\launch.py" --mode modern
