@echo off
setlocal EnableExtensions
title YogaVision - Fix Model File

cd /d "%~dp0"

echo ============================================================
echo  YogaVision - Fix models folder
echo  Turns a zipped / compressed / multi-part model download
echo  into the layout the app expects.
echo ============================================================
echo.

rem ---------- Step 1: extract any zip ----------
for %%f in (*.zip *.7z) do (
  echo [..] Extracting %%f ...
  powershell -NoProfile -Command "Expand-Archive -LiteralPath '%%f' -DestinationPath '%CD%' -Force" || goto :fail
)

rem ---------- Step 2: detect which layout we have ----------
rem
rem A Keras model is either:
rem   (a) a single file  :  yoga_cnn.keras
rem   (b) a folder       :  config.json + metadata.json
rem                          + model.weights.h5
rem                          + model.weights.data   (may be absent
rem                             for very small models)
rem
rem The app (model.py) calls tf.keras.models.load_model(path)
rem where path points to models\yoga_cnn.keras.  Keras's load_model
rem accepts BOTH a single file and a directory, so a folder named
rem yoga_cnn.keras works fine as long as its 4 part files are inside.

rem Case A: we already have the right single-file .keras
if exist "yoga_cnn.keras" (
  echo [OK] yoga_cnn.keras is already here.
  goto :check
)

rem Case B: we have a Keras model directory (config.json present)
if exist "config.json" (
  if exist "metadata.json" (
    rem Rename the 4-part directory to yoga_cnn.keras so load_model works
    ren "config.json" "%TEMP%\yv_tmp_cfg.json" 2>nul
    echo [..] Detected Keras model folder. Renaming it to yoga_cnn.keras\ ...
    rem We cannot ren a folder; we build the target folder instead.
    if exist "yoga_cnn.keras" rmdir "yoga_cnn.keras"
    md "yoga_cnn.keras" || goto :fail
    copy /Y "config.json" "yoga_cnn.keras\" >nul || goto :fail
    copy /Y "metadata.json" "yoga_cnn.keras\" >nul || goto :fail
    if exist "model.weights.h5" copy /Y "model.weights.h5" "yoga_cnn.keras\" >nul
    if exist "model.weights.data" copy /Y "model.weights.data" "yoga_cnn.keras\" >nul
    echo [OK] Built models\yoga_cnn.keras\ (folder with 4 Keras parts).
    rem Clean the zip if it exists; keep the loose parts for now
    del /Q *.zip 2>nul
    goto :check
  )
  echo [ERROR] config.json found but metadata.json missing - incomplete download.
  goto :fail
)

rem Case C: a .compressed file (GitHub's delivery format)
if exist "yoga_cnn.compressed" (
  echo [..] Found yoga_cnn.compressed - renaming...
  ren "yoga_cnn.compressed" "yoga_cnn.keras" || goto :fail
  goto :check
)
if exist "yoga_cnn.keras.compressed" (
  echo [..] Found yoga_cnn.keras.compressed - renaming...
  ren "yoga_cnn.keras.compressed" "yoga_cnn.keras" || goto :fail
  goto :check
)

rem Case D: single nested folder holding the .keras
for /d %%d in (*) do (
  if not "%%d"=="yoga_cnn.keras" if exist "%%d\yoga_cnn.keras" (
    echo [..] Found yoga_cnn.keras inside %%d\ - moving up...
    move "%%d\yoga_cnn.keras" "yoga_cnn.keras" || goto :fail
    if not exist "%%d\*.*" rmdir "%%d"
    goto :check
  )
)

rem Nothing matched
echo [ERROR] Could not recognise the model layout in this folder.
echo.
echo Contents:
dir /b
echo.
echo Expected one of:
echo   yoga_cnn.keras              (single file)
echo   config.json + metadata.json (Keras folder, 4 parts)
echo   yoga_cnn.keras.zip          (zip of the above)
echo   yoga_cnn.compressed         (GitHub's delivery format)
goto :fail

:check
if not exist "yoga_cnn.keras" (
  echo [ERROR] yoga_cnn.keras still missing after the fix.
  goto :fail
)

if exist "yoga_cnn.keras\" (
  echo [OK] yoga_cnn.keras\ is a folder (Keras multi-part model).
) else (
  for %%f in ("yoga_cnn.keras") do echo [OK] yoga_cnn.keras is a file (%%~zf bytes).
)
if exist "label_map.json" (
  echo [OK] label_map.json is present.
) else (
  echo [WARN] label_map.json is missing - the app cannot start without it.
  echo        It should ship with the repo: git clone + check models\label_map.json
)
echo.
echo Now run start.bat from the project root to launch the app.
goto :done

:fail
echo.
echo ============================================================
echo  Something went wrong. See messages above.
echo ============================================================
pause
exit /b 1

:done
echo.
pause
