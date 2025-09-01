@echo off
echo ========================================
echo    JiLing服装分类系统 - 项目整理工具
echo ========================================
echo.

echo [1/6] 检查项目结构...
if exist "src" (
    echo ✓ src目录存在
) else (
    echo ✗ src目录不存在
)

if exist "config" (
    echo ✓ config目录存在
) else (
    echo ✗ config目录不存在
)

if exist "models" (
    echo ✓ models目录存在
) else (
    echo ✗ models目录不存在
)

if exist "scripts" (
    echo ✓ scripts目录存在
) else (
    echo ✗ scripts目录不存在
)

echo.
echo [2/6] 检查核心文件...
if exist "src\main.py" (
    echo ✓ 主程序文件存在
) else (
    echo ✗ 主程序文件不存在
)

if exist "requirements.txt" (
    echo ✓ 依赖文件存在
) else (
    echo ✗ 依赖文件不存在
)

if exist "README.md" (
    echo ✓ 说明文档存在
) else (
    echo ✗ 说明文档不存在
)

echo.
echo [3/6] 检查配置文件...
if exist "config\model_config.yaml" (
    echo ✓ 模型配置文件存在
) else (
    echo ✗ 模型配置文件不存在
)

if exist "config\training_config.yaml" (
    echo ✓ 训练配置文件存在
) else (
    echo ✗ 训练配置文件不存在
)

if exist "config\paths_config.yaml" (
    echo ✓ 路径配置文件存在
) else (
    echo ✗ 路径配置文件不存在
)

echo.
echo [4/6] 检查模型文件...
if exist "models\*.pth" (
    echo ✓ 模型权重文件存在
) else (
    echo ⚠ 模型权重文件不存在，请先训练模型
)

echo.
echo [5/6] 清理临时文件...
if exist "*.pyc" del /q "*.pyc" 2>nul
if exist "*.pyo" del /q "*.pyo" 2>nul
if exist "__pycache__" rmdir /s /q "__pycache__" 2>nul

echo.
echo [6/6] 项目整理完成！
echo.
echo 项目结构：
echo ├── src/                 # 源代码
echo │   ├── core/           # 核心功能
echo │   ├── gui/            # 图形界面
echo │   ├── utils/          # 工具函数
echo │   └── main.py         # 主程序
echo ├── config/             # 配置文件
echo ├── models/             # 模型文件
echo ├── scripts/            # 脚本工具
echo ├── data/               # 数据目录
echo ├── logs/               # 日志文件
echo ├── outputs/            # 输出结果
echo ├── requirements.txt    # 依赖包
echo └── README.md           # 项目说明
echo.
echo 运行方式：
echo 1. 安装依赖：pip install -r requirements.txt
echo 2. 启动GUI：python src/main.py gui
echo 3. 快速分类：python src/main.py classify
echo.
pause
