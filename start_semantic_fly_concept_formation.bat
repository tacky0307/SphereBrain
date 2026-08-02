@echo off
cd /d %~dp0
start "" http://127.0.0.1:5021
python semantic_fly_concept_formation_lab.py
pause
