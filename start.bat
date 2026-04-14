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
echo [1] 启动新界面-稳定版 (已验证)
echo [2] 启动新界面-模块化预览 (待全量回归)
echo [3] 启动传统界面 (PySide6)
echo [4] 命令行分类工具 (生产核心)
echo [5] 退出
echo.
echo ========================================
echo.

set /p choice=请输入选项 (1-5):

if "%choice%"=="1" goto modern
if "%choice%"=="2" goto modular
if "%choice%"=="3" goto traditional
if "%choice%"=="4" goto cli
if "%choice%"=="5" goto end

echo 无效选项，请重新选择
pause
goto start

:modern
echo.
echo 正在启动新界面(稳定版)...
cd /d "%~dp0"
call "%~dp0.conda\Scripts\python.exe" "run_new_gui.py"
goto end

:modular
echo.
echo 正在启动新界面(模块化预览)...
cd /d "%~dp0"
call "%~dp0.conda\Scripts\python.exe" "run_new_gui.py" --modular
goto end

:traditional
echo.
echo 正在启动传统界面...
cd /d "%~dp0"
call "%~dp0.conda\Scripts\python.exe" "run_new_gui.py" --traditional
goto end

:cli
echo.
echo 正在启动命令行分类工具...
cd /d "%~dp0"
call "%~dp0.conda\Scripts\python.exe" "src\cli\classify_cli.py"
goto end

:end
exit

