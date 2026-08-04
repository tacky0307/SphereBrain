@echo off
cd /d %~dp0
start "" http://127.0.0.1:5017
python semantic_context_improvement_v2_lab.py
pause
