@echo off
chcp 65001 >nul
echo ============================================
echo  Reset Training Data
echo ============================================
echo.

echo Will delete:
echo   - checkpoints/ all models and history
echo   - Python cache
echo   - visualization images
echo.

set /p confirm="Confirm delete? (Y/N): "
if /i not "%confirm%"=="Y" (
    echo Operation cancelled
    pause
    exit /b
)

echo.
echo Cleaning...

REM Delete checkpoints
if exist checkpoints (
    echo [1/3] Deleting checkpoints/...
    rmdir /s /q checkpoints 2>nul
    mkdir checkpoints
)

REM Delete Python cache
echo [2/3] Deleting Python cache...
if exist __pycache__ rmdir /s /q __pycache__ 2>nul
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul

REM Delete visualization images
echo [3/3] Deleting visualization images...
if exist training_visualization.png del /q training_visualization.png 2>nul

echo.
echo ============================================
echo  Done!
echo ============================================
echo.
echo Now you can start fresh training:
echo   python train_pipeline.py --fast-test
echo.
pause
