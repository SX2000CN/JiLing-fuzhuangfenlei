@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo Starting CLI Classifier...
".conda\python.exe" "src\cli\classify_cli.py" --no-pause
pause
