@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:5022
python semantic_emergent_concept_v2_lab.py
pause
