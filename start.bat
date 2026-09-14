@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto checkdeps
where py >nul 2>nul
if errorlevel 1 (
    python -m venv .venv
) else (
    py -3 -m venv .venv
)
if errorlevel 1 goto failed
:checkdeps
".venv\Scripts\python.exe" -c "import flytrade, numpy, pandas, scipy" >nul 2>nul
if not errorlevel 1 goto run
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto failed
:run
".venv\Scripts\python.exe" run.py %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo Fly Trade could not start. Read the error above and the README troubleshooting section.
pause
exit /b 1
