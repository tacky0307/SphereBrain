@echo off
setlocal
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
  echo .venv が見つかりません。先に環境を準備してください。
  pause
  exit /b 1
)

.venv\Scripts\python.exe experiments\run_structural_grid_puzzle_v1.py
if errorlevel 1 (
  echo.
  echo 実行中にエラーが発生しました。
  pause
)
endlocal
