@echo off
:: 快速启动 - 传统界面
cd /d "%~dp0.."
call .conda\python.exe "launchers\launch.py" --mode traditional
