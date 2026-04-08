"""
GPU 状态监控组件
"""

import ctypes
import ctypes.wintypes
import platform
from typing import Optional

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt, QTimer

from ..theme import StyleSheet, theme_manager, ColorTokens
from ..utils import FontManager

# 可选依赖
try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


IS_WINDOWS = platform.system() == "Windows"


class _FILETIME(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", ctypes.wintypes.DWORD),
        ("dwHighDateTime", ctypes.wintypes.DWORD),
    ]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.wintypes.DWORD),
        ("dwMemoryLoad", ctypes.wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _filetime_to_int(file_time: _FILETIME) -> int:
    return (file_time.dwHighDateTime << 32) + file_time.dwLowDateTime


class GPUStatusWidget(QWidget):
    """GPU 状态监控组件"""

    def __init__(self, parent=None, compact: bool = False):
        super().__init__(parent)
        self.compact = compact
        self._last_cpu_times: Optional[tuple[int, int, int]] = None
        self._compact_mode = "cpu"

        self._setup_ui()

        # 定时更新
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_status)
        self.timer.start(2000)  # 每2秒更新

        self._update_status()

        # 监听主题变化
        theme_manager.theme_changed.connect(self._on_theme_changed)

    def _setup_ui(self):
        """构建 UI，支持紧凑模式与标准模式"""
        if self.compact:
            self.setObjectName("gpuStatusCompact")
            self.setFixedHeight(64)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(2)
            layout.setAlignment(Qt.AlignCenter)

            self.title_label = QLabel("CPU模式")
            self.title_label.setAlignment(Qt.AlignCenter)
            self.title_label.setFixedHeight(12)
            layout.addWidget(self.title_label)

            self.usage_label = QLabel("CPU --%")
            self.usage_label.setAlignment(Qt.AlignCenter)
            self.usage_label.setFixedHeight(14)
            layout.addWidget(self.usage_label)

            self.memory_label = QLabel("RAM--%")
            self.memory_label.setAlignment(Qt.AlignCenter)
            self.memory_label.setFixedHeight(14)
            layout.addWidget(self.memory_label)

            self.status_label = QLabel("")
            self.status_label.hide()

            self._apply_compact_style()
            return

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel("GPU 状态")
        title.setStyleSheet(StyleSheet.page_title())
        title.setFont(FontManager.title_font())
        layout.addWidget(title)

        usage_row = QHBoxLayout()
        usage_label = QLabel("使用率:")
        usage_label.setStyleSheet(StyleSheet.param_label())
        usage_label.setFont(FontManager.label_font())
        usage_row.addWidget(usage_label)

        self.usage_bar = QProgressBar()
        self.usage_bar.setStyleSheet(StyleSheet.progress_bar())
        self.usage_bar.setTextVisible(True)
        self.usage_bar.setFormat("%p%")
        usage_row.addWidget(self.usage_bar, 1)
        layout.addLayout(usage_row)

        memory_row = QHBoxLayout()
        memory_label = QLabel("显存:")
        memory_label.setStyleSheet(StyleSheet.param_label())
        memory_label.setFont(FontManager.label_font())
        memory_row.addWidget(memory_label)

        self.memory_bar = QProgressBar()
        self.memory_bar.setStyleSheet(StyleSheet.progress_bar())
        self.memory_bar.setTextVisible(True)
        self.memory_bar.setFormat("%p%")
        memory_row.addWidget(self.memory_bar, 1)
        layout.addLayout(memory_row)

        self.status_label = QLabel("检测中...")
        self.status_label.setStyleSheet(StyleSheet.settings_row_desc())
        self.status_label.setFont(FontManager.small_font())
        layout.addWidget(self.status_label)

    def _on_theme_changed(self, theme: str):
        if self.compact:
            self._apply_compact_style()
            return

        self.usage_bar.setStyleSheet(StyleSheet.progress_bar())
        self.memory_bar.setStyleSheet(StyleSheet.progress_bar())
        self.status_label.setStyleSheet(StyleSheet.settings_row_desc())

    def _apply_compact_style(self):
        """应用紧凑模式样式"""
        border_color = theme_manager.get_color(ColorTokens.BORDER_MUTED)
        bg_color = theme_manager.get_color(ColorTokens.BG_PRIMARY)
        usage_color = theme_manager.get_color(ColorTokens.TEXT_PRIMARY)
        memory_color = theme_manager.get_color(ColorTokens.TEXT_SECONDARY)

        self.setStyleSheet(
            f"""
            QWidget#gpuStatusCompact {{
                border: 1px solid {border_color};
                border-radius: 6px;
                background-color: {bg_color};
            }}
            """
        )

        self.title_label.setStyleSheet(
            f"color: {theme_manager.get_color(ColorTokens.TEXT_MUTED)}; font-size: 8px; font-weight: 700;"
        )

        self.usage_label.setStyleSheet(
            f"color: {usage_color}; font-size: 8px; font-weight: 700;"
        )
        self.memory_label.setStyleSheet(
            f"color: {memory_color}; font-size: 8px; font-weight: 600;"
        )

        self._set_compact_mode(self._compact_mode)

    def _set_compact_mode(self, mode: str):
        """设置紧凑模式徽标"""
        self._compact_mode = mode

        if mode == "gpu":
            title = "GPU模式"
            memory_color = theme_manager.get_color(ColorTokens.SUCCESS)
        elif mode == "cuda":
            title = "CUDA模式"
            memory_color = theme_manager.get_color(ColorTokens.ACCENT)
        elif mode == "err":
            title = "状态异常"
            memory_color = theme_manager.get_color(ColorTokens.ERROR)
        else:
            title = "CPU模式"
            memory_color = theme_manager.get_color(ColorTokens.TEXT_SECONDARY)

        self.title_label.setText(title)
        self.memory_label.setStyleSheet(
            f"color: {memory_color}; font-size: 8px; font-weight: 700;"
        )

    def _read_cpu_percent(self) -> float:
        """读取 CPU 使用率（百分比）"""
        if not IS_WINDOWS:
            return 0.0

        idle_time = _FILETIME()
        kernel_time = _FILETIME()
        user_time = _FILETIME()

        ok = ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        )
        if not ok:
            return 0.0

        sample = (
            _filetime_to_int(idle_time),
            _filetime_to_int(kernel_time),
            _filetime_to_int(user_time),
        )

        if self._last_cpu_times is None:
            self._last_cpu_times = sample
            return 0.0

        last_idle, last_kernel, last_user = self._last_cpu_times
        idle_delta = sample[0] - last_idle
        kernel_delta = sample[1] - last_kernel
        user_delta = sample[2] - last_user
        total_delta = kernel_delta + user_delta
        self._last_cpu_times = sample

        if total_delta <= 0:
            return 0.0

        usage = (total_delta - idle_delta) * 100.0 / total_delta
        return max(0.0, min(100.0, usage))

    def _read_ram_percent(self) -> float:
        """读取系统内存占用百分比"""
        if not IS_WINDOWS:
            return 0.0

        mem_status = _MEMORYSTATUSEX()
        mem_status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))
        if not ok:
            return 0.0
        return float(mem_status.dwMemoryLoad)

    def _read_gpu_status(self) -> dict:
        """读取 GPU/CUDA 状态"""
        if GPU_UTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    return {
                        "mode": "gpu",
                        "name": gpu.name,
                        "gpu_load_percent": gpu.load * 100,
                        "mem_used_gb": gpu.memoryUsed / 1024,
                        "mem_total_gb": gpu.memoryTotal / 1024,
                        "mem_percent": gpu.memoryUtil * 100,
                    }
            except Exception:
                pass

        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                mem_percent = 0.0 if mem_total <= 0 else (mem_allocated / mem_total) * 100
                return {
                    "mode": "cuda",
                    "name": device_name,
                    "gpu_load_percent": None,
                    "mem_used_gb": mem_allocated,
                    "mem_total_gb": mem_total,
                    "mem_percent": mem_percent,
                }
            except Exception:
                pass

        return {
            "mode": "none",
            "name": "未检测到 GPU",
            "gpu_load_percent": None,
            "mem_used_gb": None,
            "mem_total_gb": None,
            "mem_percent": None,
        }

    def _update_status(self):
        """更新 GPU 状态"""
        try:
            cpu_percent = self._read_cpu_percent()
            ram_percent = self._read_ram_percent()
            gpu_status = self._read_gpu_status()

            if self.compact:
                self.usage_label.setText(f"CPU {cpu_percent:.0f}%")

                if gpu_status["mode"] == "gpu":
                    gpu_load = float(gpu_status["gpu_load_percent"])
                    self._set_compact_mode("gpu")
                    self.memory_label.setText(f"GPU {gpu_load:.0f}%")
                    self.setToolTip(
                        f"CPU: {cpu_percent:.0f}%\n"
                        f"RAM: {ram_percent:.0f}%\n"
                        f"GPU: {gpu_load:.0f}%\n"
                        f"显存: {gpu_status['mem_used_gb']:.1f}/{gpu_status['mem_total_gb']:.1f}GB\n"
                        f"设备: {gpu_status['name']}"
                    )
                    return

                if gpu_status["mode"] == "cuda":
                    self._set_compact_mode("cuda")
                    self.memory_label.setText(f"CU{gpu_status['mem_used_gb']:.1f}G")
                    self.setToolTip(
                        f"CPU: {cpu_percent:.0f}%\n"
                        f"RAM: {ram_percent:.0f}%\n"
                        f"CUDA 显存: {gpu_status['mem_used_gb']:.1f}/{gpu_status['mem_total_gb']:.1f}GB\n"
                        f"设备: {gpu_status['name']}"
                    )
                    return

                self._set_compact_mode("cpu")
                self.memory_label.setText(f"RAM {ram_percent:.0f}%")
                self.setToolTip(
                    f"CPU: {cpu_percent:.0f}%\n"
                    f"RAM: {ram_percent:.0f}%\n"
                    "未检测到 GPU/CUDA"
                )
                return

            if gpu_status["mode"] == "gpu":
                self.usage_bar.setValue(int(gpu_status["gpu_load_percent"]))
                self.memory_bar.setValue(int(gpu_status["mem_percent"]))
                self.status_label.setText(
                    f"{gpu_status['name']} | CPU {cpu_percent:.0f}% RAM {ram_percent:.0f}%"
                )
                return

            if gpu_status["mode"] == "cuda":
                self.usage_bar.setValue(int(cpu_percent))
                self.memory_bar.setValue(int(gpu_status["mem_percent"]))
                self.status_label.setText(
                    f"CUDA: {gpu_status['name']} | 显存 {gpu_status['mem_used_gb']:.1f}/{gpu_status['mem_total_gb']:.1f}GB"
                )
                return

            self.usage_bar.setValue(int(cpu_percent))
            self.memory_bar.setValue(int(ram_percent))
            self.status_label.setText("CPU 模式（未检测到 GPU/CUDA）")

        except Exception:
            if self.compact:
                self._set_compact_mode("err")
                self.usage_label.setText("CPU --%")
                self.memory_label.setText("RAM --%")
                self.setToolTip("性能状态读取失败")
            else:
                self.status_label.setText("性能状态读取失败")
                self.usage_bar.setValue(0)
                self.memory_bar.setValue(0)
