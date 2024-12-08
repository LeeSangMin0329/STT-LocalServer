chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running audio_device_checker.py...
venv\Scripts\python audio_device_checker.py

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause