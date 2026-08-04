@echo off
cd /d %~dp0
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe semantic_context_preservation_lab.py
) else (
  python semantic_context_preservation_lab.py
)
pause
