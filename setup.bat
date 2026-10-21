@echo off
setlocal EnableExtensions
title YogaVision - One-Time Setup

echo ============================================================
echo  YogaVision - One-Time Setup
echo  This script installs Python if missing, creates a virtual
echo  environment, installs all dependencies and runs checks.
echo  Run it ONCE from a folder containing this batch file.
echo ============================================================
echo.

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PYTHON="

rem ---------- 1. Locate a Python 3 interpreter ----------
where python >nul 2>nul && set "PYTHON=python"
if not defined PYTHON (
  where py >nul 2>nul && set "PYTHON=py -3"
)
if not defined PYTHON (
  if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)
if not defined PYTHON (
  if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
)
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

if not defined PYTHON (
  echo [ERROR] Python was not found on this computer.
  echo.
  echo Install Python 3.10 or newer from https://www.python.org/downloads/
  echo IMPORTANT: tick "Add python.exe to PATH" during installation.
  echo Then run setup.bat again.
  pause
  exit /b 1
)

echo [OK] Found Python: %PYTHON%
for /f "delims=" %%v in ('%PYTHON% --version') do echo [OK] %%v
echo.

rem ---------- 2. Create the virtual environment ----------
if not exist "%VENV%\Scripts\python.exe" (
  echo [..] Creating virtual environment in .venv ...
  %PYTHON% -m venv "%VENV%" || (echo [ERROR] Could not create the virtual environment. & pause & exit /b 1)
) else (
  echo [OK] Virtual environment already exists.
)
set "VPY=%VENV%\Scripts\python.exe"

rem ---------- 3. Install all dependencies ----------
echo [..] Upgrading pip ...
"%VPY%" -m pip install --upgrade pip

echo [..] Installing dependencies (TensorFlow, OpenCV, MediaPipe, Streamlit, ...)
echo     This downloads large packages; it can take several minutes.
"%VPY%" -r "%ROOT%requirements.txt" || (
  echo.
  echo [ERROR] Dependency installation failed. Check your internet
  echo connection and make sure your antivirus is not blocking pip.
  pause
  exit /b 1
)

echo [..] Installing the yoga vision package ...
"%VPY%" -m pip install -e . || (echo [ERROR] Package installation failed. & pause & exit /b 1)

rem ---------- 4. Verify the key packages import ----------
echo [..] Verifying installed packages ...
"%VPY%" - <<EOF || (
  echo.
  echo [ERROR] Verification failed: the installed packages could not be
  echo imported. Re-run setup.bat.
  pause
  exit /b 1
)
import numpy, pandas, cv2, sklearn, matplotlib, streamlit, mediapipe, yaml
print(f"OK numpy={numpy.__version__} opencv={cv2.__version__}")
EOF

rem ---------- 5. Sanity run of the test suite ----------
echo [..] Running the test suite ...
"%VPY%" -m pytest tests/ -q
if errorlevel 1 (
  echo.
  echo [WARN] Some tests failed. This is usually not a problem for the
  echo app itself - try start.bat and tell the teacher if anything looks off.
) else (
  echo [OK] Tests passed.
)

echo.
echo ============================================================
echo  Setup complete.
echo  Now double-click start.bat to launch the application.
echo ============================================================
pause
