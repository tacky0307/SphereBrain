@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:5024
python semantic_experience_path_viewer_lab.py
pause
