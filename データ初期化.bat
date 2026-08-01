@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo SphereBrain のデータ初期化を開始します。
echo 起動中の SphereBrain や関連画面は先に閉じてください。
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    py -3 reset_project_data.py
) else (
    python reset_project_data.py
)

set EXIT_CODE=%errorlevel%
echo.
if not "%EXIT_CODE%"=="0" (
    echo 初期化は正常に完了しませんでした。
) else (
    echo 処理を終了しました。
)
pause
exit /b %EXIT_CODE%
