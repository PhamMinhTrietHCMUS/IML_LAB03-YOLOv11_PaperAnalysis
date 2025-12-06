@echo off
REM YOLO Real-time Detection - Quick Start Script
REM This script helps you easily run the Flask application

echo ========================================
echo YOLO Real-time Detection Application
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [INFO] Python found: 
python --version
echo.

REM Check GPU availability
echo ========================================
echo Checking GPU Availability...
echo ========================================
python -c "import torch; print('PyTorch:', torch.__version__); cuda=torch.cuda.is_available(); print('CUDA Available:', cuda); print('Device:', torch.cuda.get_device_name(0) if cuda else 'CPU Only'); print('Expected FPS:', '25-35 (GPU)' if cuda else '8-15 (CPU)')"
echo ========================================
echo.

REM Check if requirements are installed
echo [INFO] Checking dependencies...
python -c "import flask, ultralytics, cv2" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Some dependencies are missing
    echo [INFO] Installing requirements...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo.
    echo [SUCCESS] Dependencies installed successfully
    echo.
) else (
    echo [SUCCESS] All dependencies are installed
    echo.
)

REM Check if model weights exist
echo [INFO] Checking model weights...
if not exist "runs\detect\yolov11\weights\best.pt" (
    echo [WARNING] YOLOv11 weights not found at runs\detect\yolov11\weights\best.pt
)
if not exist "runs\detect\yolov8\weights\best.pt" (
    echo [WARNING] YOLOv8 weights not found at runs\detect\yolov8\weights\best.pt
)
if not exist "runs\detect\yolov9\weights\best.pt" (
    echo [WARNING] YOLOv9 weights not found at runs\detect\yolov9\weights\best.pt
)
echo.

REM Start the Flask application
echo ========================================
echo Starting Flask Application...
echo ========================================
echo.
echo [INFO] Server will start at: http://localhost:5000
echo [INFO] Press Ctrl+C to stop the server
echo.
echo Opening browser in 3 seconds...
timeout /t 3 /nobreak >nul
start http://localhost:5000
echo.

python app.py

REM If we reach here, the app has stopped
echo.
echo ========================================
echo Application stopped
echo ========================================
pause
