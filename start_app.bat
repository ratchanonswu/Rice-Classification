@echo off
title RiceVision AI - Final Version Launcher
echo =========================================================================
echo   RiceVision AI: Rice Leaf Disease Classification ^& Explainable AI
echo   Final Version: EfficientNetV2-S vs ConvNeXtBase (Grad-CAM)
echo =========================================================================
echo.

set SCRIPT_DIR=%~dp0
set VENV_PYTHON=%SCRIPT_DIR%..\Rice Diseases Classification\ResNet50\.venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    set VENV_PYTHON=python
)

echo [1/2] Launching web browser at http://localhost:8000 ...
start "" http://localhost:8000

echo [2/2] Starting unified web server on port 8000 ...
echo       Press Ctrl+C to stop the server.
echo.

"%VENV_PYTHON%" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --app-dir "%SCRIPT_DIR%RiceVisionAI"
