# JiLing 服装分类系统

更新时间：2026-04-07

基于 PyTorch 的服装图片三分类项目（主图/细节/吊牌），提供 CLI 与多套 GUI 实现。

## 1. 入口统一状态

1. 默认新界面主入口已经统一为稳定版：`run_new_gui.py` -> `src/gui/native_ui.py`。
2. 模块化预览入口：`run_new_gui.py --modular` -> `src/gui/app.py`。
3. 传统界面入口：`run_new_gui.py --traditional` -> `src/gui/main_window.py`。
4. `launchers/` 中 GUI 脚本已统一通过 `run_new_gui.py` 路由，避免同名入口指向不同实现。

## 2. 快速启动

```bash
# 稳定新界面（默认）
python run_new_gui.py

# 模块化预览（待完整回归）
python run_new_gui.py --modular

# 传统界面
python run_new_gui.py --traditional

# CLI
python src/cli/classify_cli.py --help
```

Windows 可直接使用：

1. `launchers/启动-新界面.bat`
2. `launchers/启动-新界面-模块化预览.bat`
3. `launchers/快速启动-传统界面.bat`
4. `launchers/run_classify.bat`

## 3. 项目结构（实际）

```text
JiLing-fuzhuangfenlei/
├─ src/
│  ├─ core/
│  ├─ cli/
│  ├─ gui/
│  └─ utils/
├─ launchers/
├─ config/
├─ models/
├─ data/
├─ fonts/
├─ logs/
├─ outputs/
├─ tests/
├─ docs/
├─ build/
└─ openspec/
```

## 4. 开发与测试

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

## 5. 文档入口

1. `docs/结构规范化报告.md`
2. `docs/开发者指南.md`
3. `docs/新版UI使用说明.md`
4. `docs/项目完整文档.md`
5. `docs/项目文件结构规划.md`
6. `docs/保留冻结清理清单.md`
7. `docs/文档状态总览.md`

## 6. 当前阶段结论

项目已完成入口统一落地，当前进入“模块化新界面全流程回归 + legacy 收敛”的阶段。
