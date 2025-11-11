@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [*] 启动现代化桌面界面...
"%~dp0.conda\python.exe" launch.py modern
