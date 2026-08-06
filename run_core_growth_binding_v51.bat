@echo off
setlocal
cd /d "%~dp0"

echo Starting Core Growth Binding v51...
py -3.12 experiments\run_core_growth_binding_v51.py
if errorlevel 1 (
  echo.
  echo The program stopped with an error.
  pause
)
