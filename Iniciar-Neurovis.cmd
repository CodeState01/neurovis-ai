@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Execute Instalar-Neurovis.cmd primeiro.
  pause
  exit /b 1
)
start "Neurovis AI" ".venv\Scripts\pythonw.exe" "desktop_app.py"
