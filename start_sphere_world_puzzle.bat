@echo off
chcp 65001 >nul
cd /d %~dp0

echo ========================================
echo SphereWorld Puzzle を起動します
echo URL: http://127.0.0.1:5031
echo ========================================

if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe sphere_world_puzzle_lab.py
) else (
  python sphere_world_puzzle_lab.py
)

pause
