@echo off
cd /d %~dp0
start "" http://127.0.0.1:5023
python semantic_shared_context_formation_lab.py
pause
