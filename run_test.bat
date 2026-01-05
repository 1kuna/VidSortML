@echo off
setlocal enabledelayedexpansion

:: Change to script directory
cd /d "%~dp0"

:: ============================================
:: VidSortML Kill Counter - Installer & Wizard
:: ============================================

echo.
echo ========================================
echo   VidSortML Kill Counter
echo ========================================
echo.

:: ============================================
:: PHASE 1: CHECK PYTHON
:: ============================================

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VER=%%v
echo [OK] Python found ^(%PYTHON_VER%^)

:: ============================================
:: PHASE 2: VIRTUAL ENVIRONMENT
:: ============================================

if not exist ".venv" (
    echo [..] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

:: Activate venv
call .venv\Scripts\activate.bat

:: ============================================
:: PHASE 3: CHECK/INSTALL FFMPEG
:: ============================================

ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [..] FFmpeg not found. Installing via winget...
    winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install FFmpeg via winget.
        echo.
        echo Please install FFmpeg manually:
        echo   1. Download from: https://www.gyan.dev/ffmpeg/builds/
        echo   2. Extract and add the bin folder to your PATH
        echo.
        pause
        exit /b 1
    )
    echo [OK] FFmpeg installed
    echo [!!] You may need to restart this script for FFmpeg to be recognized.
    pause
)

:: ============================================
:: PHASE 4: INSTALL PYTHON PACKAGES
:: ============================================

if not exist ".venv\.installed" (
    echo [..] Installing Python packages...
    echo     This may take several minutes ^(downloading PyTorch, etc.^)
    echo.

    pip install --upgrade pip >nul 2>&1

    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements.txt
        pause
        exit /b 1
    )

    pip install "openai>=1.0.0"
    if errorlevel 1 (
        echo [ERROR] Failed to install openai package
        pause
        exit /b 1
    )

    :: Create marker file
    echo installed > .venv\.installed
    echo [OK] Python packages installed
    echo.
)

:: ============================================
:: PHASE 5: CHECK SERVER
:: ============================================

:check_server
echo [..] Checking vLLM server connection...

powershell -Command "try { $null = Invoke-WebRequest -Uri 'http://localhost:8901/health' -TimeoutSec 5 -ErrorAction Stop; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] vLLM server is not running!
    echo.
    echo Start the server in WSL2 with:
    echo   cd /mnt/c/Users/zcane/Desktop/VidSortML
    echo   ./start_server.sh
    echo.
    echo Press any key to retry, or Ctrl+C to exit...
    pause >nul
    goto check_server
)
echo [OK] Server is running
echo.

:: ============================================
:: PHASE 6: WIZARD LOOP
:: ============================================

:wizard
echo ========================================
echo   Kill Counter - Test Wizard
echo ========================================
echo.
echo Enter video path ^(or drag file here^)
echo Type 'exit' to quit
echo.

set /p "VIDEO_PATH=Path: "

:: Check for exit
if /i "!VIDEO_PATH!"=="exit" goto end

:: Strip surrounding quotes (drag-drop adds them)
set "VIDEO_PATH=!VIDEO_PATH:"=!"

:: Check if file exists
if not exist "!VIDEO_PATH!" (
    echo.
    echo [ERROR] File not found: !VIDEO_PATH!
    echo.
    goto wizard
)

:: Show processing message
echo.
echo Processing... ^(this may take 1-2 minutes^)
echo.

:: Create temp file for output
set "TEMP_OUTPUT=%TEMP%\vidsort_output_%RANDOM%.txt"

:: Run inference
python infer_omni.py "!VIDEO_PATH!" --crop > "!TEMP_OUTPUT!" 2>&1

:: Check if inference succeeded
if errorlevel 1 (
    echo [ERROR] Inference failed!
    echo.
    type "!TEMP_OUTPUT!"
    echo.
    del "!TEMP_OUTPUT!" 2>nul
    goto ask_continue
)

:: Parse TOTAL from output
set "KILLS=?"
for /f "tokens=*" %%a in ('findstr /i "^TOTAL:" "!TEMP_OUTPUT!"') do (
    set "LINE=%%a"
    for /f "tokens=2 delims=:" %%b in ("!LINE!") do (
        set "KILLS=%%b"
        :: Trim leading/trailing spaces
        for /f "tokens=* delims= " %%c in ("!KILLS!") do set "KILLS=%%c"
    )
)

:: Display result
echo.
echo ========================================
echo   KILLS DETECTED: !KILLS!
echo ========================================
echo.

:: Show full output for reference
echo Full model response:
echo ----------------------------------------
type "!TEMP_OUTPUT!"
echo ----------------------------------------
echo.

:: Cleanup temp file
del "!TEMP_OUTPUT!" 2>nul

:ask_continue
echo.
set /p "CONTINUE=Test another video? (Y/N): "
if /i "!CONTINUE!"=="Y" (
    echo.
    goto wizard
)
if /i "!CONTINUE!"=="YES" (
    echo.
    goto wizard
)

:end
echo.
echo Goodbye!
endlocal
exit /b 0
