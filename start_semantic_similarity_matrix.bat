@echo off
cd /d %~dp0
start "" http://127.0.0.1:5019
python semantic_similarity_matrix_lab.py
pause
