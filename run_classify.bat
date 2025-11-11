@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 使用项目 Conda 环境运行...
"%~dp0.conda\python.exe" classify_cli.py
pause
