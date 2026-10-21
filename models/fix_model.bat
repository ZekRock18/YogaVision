@echo off
setlocal EnableExtensions
title YogaVision - Fix Model File

cd /d "%~dp0"

echo ============================================================
echo  YogaVision - Fix models folder
echo  Looks for a zipped/compressed copy of the model and turns
echo  it into models\yoga_cnn.keras
echo ============================================================
echo.

set "TARGET=yoga_cnn.keras"

rem ---------- Case 1: already correct ----------
if exist "%TARGET%" (
  echo [OK] %TARGET% is already here. Nothing to do.
  goto :done
)

rem ---------- Case 2: .compressed file ----------
if exist "yoga_cnn.compressed" (
  echo [..] Found yoga_cnn.compressed - replacing extension...
  ren "yoga_cnn.compressed" "%TARGET%" || goto :fail
  goto :verify
)

if exist "yoga_cnn.keras.compressed" (
  echo [..] Found yoga_cnn.keras.compressed - replacing extension...
  ren "yoga_cnn.keras.compressed" "%TARGET%" || goto :fail
  goto :verify
)

rem ---------- Case 3: zip archive ----------
set "ZIPFILE="
for %%f in (*.zip *.7z) do set "ZIPFILE=%%f"

if defined ZIPFILE (
  echo [..] Found %ZIPFILE% - extracting...
  powershell -NoProfile -Command "Expand-Archive -LiteralPath '%ZIPFILE%' -DestinationPath '%CD%' -Force" || goto :fail

  if exist "%TARGET%" goto :verify

  rem Model may be nested one folder inside the zip
  for /d %%d in (*) do (
    if exist "%%d\%TARGET%" (
      move "%%d\%TARGET%" "%TARGET%" || goto :fail
      if not exist "%%d\*.*" rmdir "%%d"
      goto :verify
    )
  )
  echo [ERROR] Extracted %ZIPFILE% but %TARGET% was not found inside it.
  dir /b
  goto :fail
)

rem ---------- Case 4: .keras.zst or .gz variant ----------
for %%f in (*%TARGET%.zst *%TARGET%.gz) do (
  set "COMPRESSED=%%f"
)
if defined COMPRESSED (
  echo [ERROR] Found %COMPRESSED% which is a compressed-but-still-zipped file.
  echo Download the original unzipped version instead, or share the
  echo raw .keras file with your friend again.
  goto :fail
)

echo [ERROR] Could not find the model in the models folder.
echo.
echo Look in these places on your friend's computer:
echo   - Downloads folder (the .zip / .compressed / .zst file)
echo   - Browser download history
echo Then copy that file into this models folder and re-run this bat.
echo.
dir /b
goto :fail

:verify
if not exist "%TARGET%" (
  echo [ERROR] %TARGET% still missing after the fix.
  goto :fail
)

echo [OK] %TARGET% is ready.
for %%f in ("%TARGET%") do echo [OK] File size: %%~zf bytes
if exist "label_map.json" (
  echo [OK] label_map.json is present.
) else (
  echo [WARN] label_map.json is missing - the app cannot start without it.
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
