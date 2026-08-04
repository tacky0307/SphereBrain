@echo off
cd /d %~dp0
start "" http://127.0.0.1:5013
python semantic_stage_observer_lab.py
pause
