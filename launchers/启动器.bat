@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHON_EXE=%~dp0.conda\python.exe

:menu
cls
echo ========================================
echo JiLing 服装分类系统启动器
echo ========================================
echo.
echo 请选择启动模式:
echo.
echo     1 = 现代化桌面界面 (推荐)
echo     2 = 传统桌面界面
echo     3 = 命令行分类工具
echo     4 = Web 界面
echo     5 = 退出
echo.
set /p choice=请输入选项 (1-5): 

if "%choice%"=="1" goto modern
if "%choice%"=="2" goto traditional
if "%choice%"=="3" goto cli
if "%choice%"=="4" goto web
if "%choice%"=="5" goto quit
echo.
echo [!] 无效选项，请重新选择
timeout /t 2 >nul
goto menu

:modern
echo.
echo [*] 启动现代化桌面界面...
"%PYTHON_EXE%" launch.py modern
pause
exit /b 0

:traditional
echo.
echo [*] 启动传统桌面界面...
"%PYTHON_EXE%" launch.py traditional
pause
exit /b 0

:cli
echo.
echo [*] 启动命令行分类工具...
"%PYTHON_EXE%" classify_cli.py
pause
exit /b 0

:web
echo.
echo [*] 启动 API 服务器...
echo 访问地址: http://localhost:8000
"%PYTHON_EXE%" api_server.py
pause
exit /b 0

:quit
echo.
echo 再见！
exit /b 0
