@echo off
cd /d %~dp0
start "" http://127.0.0.1:5020
python semantic_novel_integration_lab.py
pause
