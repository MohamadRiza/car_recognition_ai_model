@echo off
echo ========================================================
echo  Starting Vehicle AI Car Recognition Server (FastAPI)
echo  Port: 8000
echo ========================================================
cd /d "%~dp0"
call .\venv\Scripts\activate.bat
cd src
python api.py
pause
