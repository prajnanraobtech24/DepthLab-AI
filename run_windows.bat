@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv
  if errorlevel 1 py -3 -m venv .venv
)
call .venv\Scripts\activate.bat
if not exist .venv\dependencies_installed.flag (
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 exit /b 1
  echo installed> .venv\dependencies_installed.flag
)
python -m uvicorn server:app --host 127.0.0.1 --port 8000
