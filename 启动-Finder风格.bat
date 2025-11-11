@echo off
chcp 65001 >nul
title JiLing 服装分类系统 - Finder风格界面

echo.
echo ========================================
echo   JiLing 服装分类系统
echo   Finder风格界面
echo ========================================
echo.
echo 正在启动...
echo.

cd /d "%~dp0"
call .conda\python.exe "launchers\finder_style_gui.py"

if errorlevel 1 (
    echo.
    echo ========================================
    echo   启动失败！
    echo ========================================
    echo.
    pause
)

exit
