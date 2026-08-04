@echo off
cd /d %~dp0
start "" http://127.0.0.1:5028
python sphere_world_showcase_lab.py
pause
