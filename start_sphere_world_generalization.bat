@echo off
cd /d %~dp0
start "" http://127.0.0.1:5026
python sphere_world_generalization_lab.py
pause
