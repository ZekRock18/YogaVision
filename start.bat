@echo off
setlocal EnableExtensions
title YogaVision - Launch

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"

if not exist "%VENV%\Scripts\python.exe" (
  echo [ERROR] The virtual environment (.venv) was not found.
  echo Run setup.bat first - once - to install everything.
  pause
  exit /b 1
)

rem ---------- Model check ----------
if not exist "%ROOT%models\yoga_cnn.keras" (
  echo ============================================================
  echo  WARNING: models\yoga_cnn.keras is missing.
  echo
  echo  The application will start, but every prediction page will
  echo  show "Model not available" until the model file is created.
  echo
  echo  To create it:
  echo    1. Copy models\yoga_cnn.keras and models\label_map.json
  echo       from a machine where setup + training has been run,
  echo       OR
  echo    2. Train it here:
  echo         .venv\Scripts\python scripts\train.py --config configs\train.yaml
  echo       (this needs the dataset in data\raw - see
  echo        docs\dataset_setup.md)
  echo ============================================================
  pause
)

rem ---------- Launch the application ----------
echo Starting YogaVision at http://localhost:8501
echo Press Ctrl+C in this window to stop it.
start "" http://localhost:8501

"%VENV%\Scripts\streamlit.exe" run "%ROOT%app.py" --server.headless true --server.port 8501

pause
