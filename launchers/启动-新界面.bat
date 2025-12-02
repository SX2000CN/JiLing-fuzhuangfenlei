@echo off
chcp 65001 >nul
title JiLing - New UI
cd /d "%~dp0.."
".conda\python.exe" "src\gui\native_ui.py"
pause

