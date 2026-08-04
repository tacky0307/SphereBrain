@echo off
chcp 65001 >nul
cd /d %~dp0

echo ==========================================
echo  SphereBrain Route Match Viewer
echo  http://127.0.0.1:5029
echo ==========================================
echo.

if exist .venv\Scripts\python.exe (
    start "" http://127.0.0.1:5029
    .venv\Scripts\python.exe sphere_world_route_match_lab.py
) else (
    start "" http://127.0.0.1:5029
    python sphere_world_route_match_lab.py
)

pause
