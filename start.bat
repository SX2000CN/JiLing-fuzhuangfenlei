@echo off
chcp 65001 >nul
title JiLing 服装分类系统

echo.
echo ========================================
echo   JiLing 服装分类系统
echo ========================================
echo.
echo 请选择启动方式:
echo.
echo [1] 启动新界面 (VS Code风格)
echo [2] 启动传统界面 (PySide6)
echo [3] 命令行分类工具 (生产核心)
echo [4] 退出
echo.
echo ========================================
echo.

set /p choice=请输入选项 (1-4):

if "%choice%"=="1" goto modern
if "%choice%"=="2" goto traditional
if "%choice%"=="3" goto cli
if "%choice%"=="4" goto end

echo 无效选项，请重新选择
pause
goto start

:modern
echo.
echo 正在启动新界面...
cd /d "%~dp0"
call "%~dp0.conda\python.exe" "src\gui\native_ui.py"
goto end

:traditional
echo.
echo 正在启动传统界面...
cd /d "%~dp0"
call "%~dp0.conda\python.exe" "src\gui\main_window.py"
goto end

:cli
echo.
echo 正在启动命令行分类工具...
cd /d "%~dp0"
call "%~dp0.conda\python.exe" "src\cli\classify_cli.py"
goto end

:end
exit
