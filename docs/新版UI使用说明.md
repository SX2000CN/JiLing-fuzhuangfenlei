# 新版 UI 使用说明（入口统一版）

更新时间：2026-04-07

## 1. 先看结论

1. 现在“新界面”默认入口已经统一到 `run_new_gui.py`。
2. 默认启动稳定新界面：`src/gui/native_ui.py`。
3. 模块化版本改为预览入口：`python run_new_gui.py --modular`。

## 2. 启动方式

| 启动命令 | 实际入口 | 状态 |
|---|---|---|
| `python run_new_gui.py` | `src/gui/native_ui.py` | 稳定主线 |
| `python run_new_gui.py --modular` | `src/gui/app.py` | 模块化预览 |
| `python run_new_gui.py --traditional` | `src/gui/main_window.py` | 传统兼容 |
| `launchers/启动-新界面.bat` | `run_new_gui.py` | 稳定主线 |
| `launchers/启动-新界面-模块化预览.bat` | `run_new_gui.py --modular` | 模块化预览 |

## 3. 推荐使用

1. 日常稳定使用：`launchers/启动-新界面.bat`。
2. 新架构联调：`python run_new_gui.py --modular`。
3. 历史问题复现：`python run_new_gui.py --traditional`。

## 4. 说明

本次已消除“同名新界面指向不同实现”的入口歧义；如果界面行为不同，请先确认是否使用了 `--modular` 或 `--traditional` 参数。
