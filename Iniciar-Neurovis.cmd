@echo off
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo Instale o Node.js em https://nodejs.org/ e tente novamente.
  pause
  exit /b 1
)
if not exist "dist\index.html" (
  echo Execute Instalar-Neurovis.cmd primeiro.
  pause
  exit /b 1
)
echo Abra http://127.0.0.1:3210 no navegador.
echo Mantenha esta janela aberta enquanto usa o Neurovis.
node server.mjs
pause
