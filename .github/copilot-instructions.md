# JiLing-fuzhuangfenlei Copilot Instructions

## 🏗 Project Overview
This is a hybrid desktop application for clothing classification using PyTorch. It features two interfaces:
1.  **Traditional UI:** Native Python GUI using PySide6.
2.  **Modern UI:** Web-based GUI using React + TypeScript (Frontend) and FastAPI (Backend), wrapped in QtWebEngine.

## 🏛 Architecture & Key Components

### 1. Core Logic (`src/core/`)
*   **Model Factory** (`src/core/model_factory.py`): Central place for defining and creating models. Uses `timm` library.
*   **Classifier** (`src/core/pytorch_classifier.py`): Handles inference logic.
*   **Trainer** (`src/core/pytorch_trainer.py`): Handles training loops, metrics, and saving checkpoints.

### 2. Traditional GUI (`src/gui/`)
*   **Main Window** (`src/gui/main_window.py`): The monolithic entry point for the PySide6 interface.
*   **Threading**: Heavy operations (training, inference) MUST run in background threads using `QThread` / `QObject` worker pattern (e.g., `TrainingWorker` class).
    *   *Pattern:* Create `QObject` worker -> `worker.moveToThread(thread)` -> Connect signals -> `thread.start()`.

### 3. Modern UI (`web-frontend/` & `api_server.py`)
*   **Backend**: `api_server.py` (FastAPI) exposes core functionality via REST API.
*   **Frontend**: React app in `web-frontend/`.
*   **Wrapper**: `web_shell.py` uses `QWebEngineView` to display the React app as a desktop window.

### 4. Configuration (`config/`)
*   Uses YAML files (`model_config.yaml`, `training_config.yaml`) managed by `src/utils/config_manager.py`.
*   Paths are often relative to `project_root`.

## 🛠 Critical Workflows

### Running the Application
*   **Unified Launcher**: `python launchers/launch.py` (or `start.bat`).
*   **Traditional UI Only**: `python launchers/gui_main.py`.
*   **CLI Tool**: `python classify_cli.py`.

### Building & Packaging
*   **Build Script**: `build/make_installer.bat`.
*   **Tool**: Uses PyInstaller with `.spec` files (`JiLing_GUI.spec`, etc.).
*   **Note**: Ensure all assets (config, models) are correctly included in the `.spec` file `datas` list.

## 📏 Conventions & Patterns

*   **Language**: All comments, logs, and UI text MUST be in **Chinese (Simplified)**.
*   **Path Handling**: Always use `pathlib.Path`.
    ```python
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "model_config.yaml"
    ```
*   **Error Handling**:
    *   GUI: Use `QMessageBox` for user-facing errors.
    *   Core: Log errors using `logging` module before raising exceptions.
*   **Model Loading**: Always use `ModelFactory.create_model()` instead of instantiating model classes directly.

## 🔗 Integration Points
*   **FastAPI <-> Core**: `api_server.py` imports `src.core` modules directly. Ensure thread safety when calling core functions from async API endpoints.
*   **PySide6 <-> Core**: Use Signals/Slots for all communication between UI and Core workers. NEVER call UI methods directly from a worker thread.
