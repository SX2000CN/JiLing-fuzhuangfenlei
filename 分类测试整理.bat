@echo off
chcp 65001 >nul
echo ========================================
echo    JiLing服装分类系统 - 照片整理工具
echo ========================================
echo.

REM 设置路径
set "SOURCE_BASE=D:\桌面\筛选"
set "TARGET_DIR=D:\桌面\筛选\JPG"

REM 定义源文件夹
set "SOURCE1=%SOURCE_BASE%\吊牌"
set "SOURCE2=%SOURCE_BASE%\细节"
set "SOURCE3=%SOURCE_BASE%\主图"

echo [1/6] 检查目录结构...
echo 目标目录: %TARGET_DIR%
echo 源目录1: %SOURCE1%
echo 源目录2: %SOURCE2%
echo 源目录3: %SOURCE3%
echo.

REM 检查目标目录
if not exist "%TARGET_DIR%" (
    echo 创建目标目录: %TARGET_DIR%
    mkdir "%TARGET_DIR%"
    if errorlevel 1 (
        echo ❌ 创建目标目录失败！
        pause
        exit /b 1
    )
    echo ✅ 目标目录创建成功
) else (
    echo ✅ 目标目录已存在
)

echo.
echo [2/6] 检查源目录...

REM 检查源目录是否存在
set "SOURCE_COUNT=0"
if exist "%SOURCE1%" (
    echo ✅ 源目录1存在: %SOURCE1%
    set /a SOURCE_COUNT+=1
) else (
    echo ⚠️  源目录1不存在: %SOURCE1%
)

if exist "%SOURCE2%" (
    echo ✅ 源目录2存在: %SOURCE2%
    set /a SOURCE_COUNT+=1
) else (
    echo ⚠️  源目录2不存在: %SOURCE2%
)

if exist "%SOURCE3%" (
    echo ✅ 源目录3存在: %SOURCE3%
    set /a SOURCE_COUNT+=1
) else (
    echo ⚠️  源目录3不存在: %SOURCE3%
)

if %SOURCE_COUNT%==0 (
    echo ❌ 没有找到任何源目录！
    pause
    exit /b 1
)

echo.
echo [3/6] 统计文件数量...

REM 统计各目录的文件数量
set "TOTAL_FILES=0"

if exist "%SOURCE1%" (
    for %%f in ("%SOURCE1%\*.jpg" "%SOURCE1%\*.jpeg" "%SOURCE1%\*.png" "%SOURCE1%\*.bmp" "%SOURCE1%\*.tiff" "%SOURCE1%\*.tif") do (
        set /a TOTAL_FILES+=1
    )
)

if exist "%SOURCE2%" (
    for %%f in ("%SOURCE2%\*.jpg" "%SOURCE2%\*.jpeg" "%SOURCE2%\*.png" "%SOURCE2%\*.bmp" "%SOURCE2%\*.tiff" "%SOURCE2%\*.tif") do (
        set /a TOTAL_FILES+=1
    )
)

if exist "%SOURCE3%" (
    for %%f in ("%SOURCE3%\*.jpg" "%SOURCE3%\*.jpeg" "%SOURCE3%\*.png" "%SOURCE3%\*.bmp" "%SOURCE3%\*.tiff" "%SOURCE3%\*.tif") do (
        set /a TOTAL_FILES+=1
    )
)

echo 发现 %TOTAL_FILES% 个图片文件待整理

if %TOTAL_FILES%==0 (
    echo ⚠️  没有找到图片文件，无需整理
    pause
    exit /b 0
)

echo.
echo [4/6] 开始整理文件...

REM 移动文件
set "MOVED_COUNT=0"
set "ERROR_COUNT=0"

REM 处理第一个源目录
if exist "%SOURCE1%" (
    echo 正在处理: 吊牌目录
    for %%f in ("%SOURCE1%\*.jpg" "%SOURCE1%\*.jpeg" "%SOURCE1%\*.png" "%SOURCE1%\*.bmp" "%SOURCE1%\*.tiff" "%SOURCE1%\*.tif") do (
        echo 移动: %%~nxf
        move "%%f" "%TARGET_DIR%\" >nul 2>&1
        if errorlevel 1 (
            echo ❌ 移动失败: %%~nxf
            set /a ERROR_COUNT+=1
        ) else (
            set /a MOVED_COUNT+=1
        )
    )
)

REM 处理第二个源目录
if exist "%SOURCE2%" (
    echo 正在处理: 细节目录
    for %%f in ("%SOURCE2%\*.jpg" "%SOURCE2%\*.jpeg" "%SOURCE2%\*.png" "%SOURCE2%\*.bmp" "%SOURCE2%\*.tiff" "%SOURCE2%\*.tif") do (
        echo 移动: %%~nxf
        move "%%f" "%TARGET_DIR%\" >nul 2>&1
        if errorlevel 1 (
            echo ❌ 移动失败: %%~nxf
            set /a ERROR_COUNT+=1
        ) else (
            set /a MOVED_COUNT+=1
        )
    )
)

REM 处理第三个源目录
if exist "%SOURCE3%" (
    echo 正在处理: 主图目录
    for %%f in ("%SOURCE3%\*.jpg" "%SOURCE3%\*.jpeg" "%SOURCE3%\*.png" "%SOURCE3%\*.bmp" "%SOURCE3%\*.tiff" "%SOURCE3%\*.tif") do (
        echo 移动: %%~nxf
        move "%%f" "%TARGET_DIR%\" >nul 2>&1
        if errorlevel 1 (
            echo ❌ 移动失败: %%~nxf
            set /a ERROR_COUNT+=1
        ) else (
            set /a MOVED_COUNT+=1
        )
    )
)

echo.
echo [5/6] 整理完成！

echo ========================================
echo                整理结果
echo ========================================
echo 成功移动文件: %MOVED_COUNT% 个
if %ERROR_COUNT% gtr 0 (
    echo 移动失败文件: %ERROR_COUNT% 个
)
echo 目标目录: %TARGET_DIR%
echo ========================================

REM 显示目标目录中的文件数量
set "FINAL_COUNT=0"
for %%f in ("%TARGET_DIR%\*.jpg" "%TARGET_DIR%\*.jpeg" "%TARGET_DIR%\*.png" "%TARGET_DIR%\*.bmp" "%TARGET_DIR%\*.tiff" "%TARGET_DIR%\*.tif") do (
    set /a FINAL_COUNT+=1
)
echo 目标目录文件总数: %FINAL_COUNT% 个

if %ERROR_COUNT% gtr 0 (
    echo.
    echo ⚠️  有 %ERROR_COUNT% 个文件移动失败，请检查文件权限或是否被占用
)

echo.
echo 按任意键退出...
pause >nul
