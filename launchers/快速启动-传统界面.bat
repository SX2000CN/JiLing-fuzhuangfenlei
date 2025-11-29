@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo [*] 启动传统桌面界面...
"%~dp0..\.conda\python.exe" "%~dp0gui_main.py"
