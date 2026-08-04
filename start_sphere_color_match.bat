@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo SphereColor Match を起動します
echo URL: http://127.0.0.1:5037
echo ========================================

if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe sphere_color_match_lab.py
) else (
  python sphere_color_match_lab.py
)

pause
