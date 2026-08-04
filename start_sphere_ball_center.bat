@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "" http://127.0.0.1:5038
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe sphere_ball_center_lab.py
) else (
  python sphere_ball_center_lab.py
)
pause
