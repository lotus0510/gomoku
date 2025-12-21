@echo off
chcp 65001 >nul
echo ============================================
echo  PyTorch 五子棋 AI 訓練（快速測試）
echo ============================================
echo.

python train_pipeline_pytorch.py --fast-test

pause
