@echo off
REM YOLO Model Evaluation Script
REM Run model evaluation and generate comparison results

echo ========================================
echo YOLO Model Evaluation
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [INFO] Starting model evaluation...
echo [INFO] This will evaluate YOLOv8, YOLOv9, and YOLOv11
echo [INFO] This may take several minutes...
echo.

python evaluate_models.py

if errorlevel 1 (
    echo.
    echo [ERROR] Evaluation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Evaluation completed!
echo ========================================
echo.
echo Results saved to: evaluation_results\
echo.

REM Open results folder
if exist "evaluation_results\comparison_results.csv" (
    echo Opening results folder...
    start evaluation_results
)

pause
