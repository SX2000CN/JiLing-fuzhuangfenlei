@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo 测试 Python 环境
echo ========================================
echo.
echo 当前目录: %cd%
echo Python 路径: %~dp0.conda\python.exe
echo.
echo 测试 Python 版本:
"%~dp0.conda\python.exe" --version
echo.
echo 测试 PySide6:
"%~dp0.conda\python.exe" -c "from PySide6.QtWidgets import QApplication; print('PySide6 可用')"
echo.
echo ========================================
echo 测试完成
echo ========================================
pause
