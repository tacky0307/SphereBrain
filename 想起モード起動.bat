@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo Sphere Brain 集中想起モード v2 を起動します。
echo このモードでは脳・記憶・研究データを書き換えません。
echo.

python recall_app_v2.py

if errorlevel 1 (
  echo.
  echo 起動に失敗しました。上のエラー内容を確認してください。
  pause
)
