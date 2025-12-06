@echo off
REM YOLO Project - Dependency Installation Script
REM Run this script first to install all required dependencies

echo ========================================
echo YOLO Project - Installation Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [INFO] Python found:
python --version
echo.

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [INFO] Installing required packages...
echo This may take several minutes...
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed!
    echo.
    echo Common solutions:
    echo 1. Make sure you have internet connection
    echo 2. Try running as Administrator
    echo 3. Check if antivirus is blocking the installation
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Installation completed!
echo ========================================
echo.
echo Installed packages:
pip list | findstr /i "ultralytics flask opencv torch"
echo.
echo You can now run the application using:
echo   run.bat
echo.
echo Or manually with:
echo   python app.py
echo.
pause
