@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo SphereTalk Room experimental app starting...
python sphere_talk_room_lab.py
pause
