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
echo [1] 启动现代化界面 (推荐)
echo [2] 启动传统界面
echo [3] 命令行分类工具
echo [4] 测试环境
echo [5] 退出
echo.
echo ========================================
echo.

set /p choice=请输入选项 (1-5): 

if "%choice%"=="1" goto modern
if "%choice%"=="2" goto traditional
if "%choice%"=="3" goto cli
if "%choice%"=="4" goto test
if "%choice%"=="5" goto end

echo 无效选项，请重新选择
pause
goto start

:modern
echo.
echo 正在启动现代化界面...
cd /d "%~dp0"
call "%~dp0.conda\python.exe" "launchers\launch.py" --mode modern
goto end

:traditional
echo.
echo 正在启动传统界面...
cd /d "%~dp0"
call "%~dp0.conda\python.exe" "launchers\launch.py" --mode traditional
goto end

:cli
echo.
echo 正在启动命令行工具...
cd /d "%~dp0"
call "%~dp0.conda\python.exe" classify_cli.py
goto end

:test
echo.
echo 正在测试环境...
cd /d "%~dp0"
call launchers\测试环境.bat
goto end

:end
exit
