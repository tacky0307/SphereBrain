@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo SphereBrain 球体経路ビジュアルを起動します...
echo URL: http://127.0.0.1:5030

echo.
python sphere_brain_sphere_visual_lab.py

if errorlevel 1 (
  echo.
  echo 起動に失敗しました。Python環境と必要ライブラリを確認してください。
  pause
)
