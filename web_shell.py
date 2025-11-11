#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JiLing 服装分类系统 - Web 外壳（QtWebEngine）

作用：
- 自动检查并启动后端 API（api_server.py）
- 在内嵌浏览器中加载 Web 前端（http://localhost:8000）
- 关闭窗口时，若本进程启动了后端，则一并关闭

优势：完全复用现有 Web UI，像素级一致的视觉与交互体验。
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
import webbrowser
from typing import Optional

from PySide6.QtCore import QUrl, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QAction
from PySide6.QtWebEngineWidgets import QWebEngineView

# 避免新增依赖，使用标准库检测后端可用性
try:
    # Python 3
    from urllib.request import urlopen
except Exception:  # pragma: no cover
    urlopen = None  # type: ignore


BACKEND_URL = os.environ.get("JILING_BACKEND_URL", "http://127.0.0.1:8000")
CHECK_PATHS = ["/api/status", "/", "/docs"]


def is_backend_alive(timeout: float = 0.8) -> bool:
    if urlopen is None:
        return False
    for p in CHECK_PATHS:
        try:
            with urlopen(BACKEND_URL + p, timeout=timeout) as resp:  # type: ignore
                code = getattr(resp, "status", None) or getattr(resp, "code", None)
                if code == 200:
                    return True
        except Exception:
            continue
    return False


def start_backend_if_needed() -> Optional[subprocess.Popen]:
    """如果后端未运行，则启动之；返回子进程句柄，否则返回 None。"""
    if is_backend_alive():
        print("✅ 检测到后端已运行:", BACKEND_URL)
        return None

    print("🚀 启动后端 API 服务器...")
    # 在同目录下运行 api_server.py
    python_exe = sys.executable
    proc = subprocess.Popen(
        [python_exe, "api_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )

    # 等待就绪（最多 ~20s）
    for _ in range(40):
        if is_backend_alive():
            print("✅ 后端已就绪:", BACKEND_URL)
            return proc
        time.sleep(0.5)

    print("⚠️ 后端未在预期时间内就绪，仍尝试打开前端...")
    return proc


class WebShellWindow(QMainWindow):
    def __init__(self, backend_proc: Optional[subprocess.Popen]):
        super().__init__()
        self.backend_proc = backend_proc

        self.setWindowTitle("JiLing 服装分类系统 - 桌面版（Web外壳）")
        self.resize(1280, 800)

        # Web 视图
        self.view = QWebEngineView(self)
        self.setCentralWidget(self.view)
        self.view.setUrl(QUrl(BACKEND_URL))

        # 工具栏
        toolbar = QToolBar("导航", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        act_back = QAction("← 返回", self)
        act_back.triggered.connect(self.view.back)
        toolbar.addAction(act_back)

        act_forward = QAction("前进 →", self)
        act_forward.triggered.connect(self.view.forward)
        toolbar.addAction(act_forward)

        act_reload = QAction("刷新", self)
        act_reload.triggered.connect(self.view.reload)
        toolbar.addAction(act_reload)

        act_external = QAction("外部浏览器打开", self)
        act_external.triggered.connect(lambda: webbrowser.open(BACKEND_URL))
        toolbar.addAction(act_external)

    def closeEvent(self, event):  # noqa: N802
        # 仅在由本进程拉起后端时才回收
        if self.backend_proc is not None:
            try:
                if self.backend_proc.poll() is None:
                    if os.name == "nt":
                        self.backend_proc.terminate()
                    else:
                        self.backend_proc.terminate()
                        time.sleep(0.5)
                        self.backend_proc.kill()
            except Exception:
                pass
        return super().closeEvent(event)


def main():
    # 确保工作目录正确
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    backend_proc = start_backend_if_needed()

    app = QApplication(sys.argv)
    win = WebShellWindow(backend_proc)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
