@echo off
cd /d %~dp0
start "" http://127.0.0.1:5025
python sphere_world_lab.py
pause
