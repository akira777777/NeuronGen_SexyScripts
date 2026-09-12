@echo off
cd /d "%~dp0"
chcp 65001 > nul
title NeuronGen Studio v2.2
cls
echo =====================================================================
echo   NeuronGen Studio v2.2
echo   Gradio UI / RTX 4070 Ti
echo =====================================================================
echo.
echo [1/2] Checking Python...
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.12+ and add to PATH.
    pause
    exit /b 1
)

echo [2/2] Starting NeuronGen Studio...
echo.
echo Open: http://127.0.0.1:7861
echo Stop: close this window or Ctrl+C
echo.
start "" "http://127.0.0.1:7861"
python main.py
pause
