@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo [*] Starting Traditional UI...
".conda\python.exe" "run_new_gui.py" --traditional
