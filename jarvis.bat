@echo off
REM Launch Jarvis from any cmd window without activating the venv.
REM Put this folder on your PATH (or copy jarvis.bat somewhere on PATH) and just type: jarvis
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
"%~dp0.venv\Scripts\python.exe" -m jarvis %*
