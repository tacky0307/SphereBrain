@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:5027
python sphere_world_multi_context_lab.py
pause
