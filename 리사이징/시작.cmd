@echo off
setlocal
cd /d "%~dp0"
set "COEX_NODE="
where node.exe >nul 2>nul
if not errorlevel 1 set "COEX_NODE=node.exe"
if not defined COEX_NODE if exist "%ProgramFiles%\nodejs\node.exe" set "COEX_NODE=%ProgramFiles%\nodejs\node.exe"
if not defined COEX_NODE (
  echo Node.js was not found. The existing editor could not be checked.
  pause
  exit /b 1
)
"%COEX_NODE%" tools\check-editor.cjs --open-existing
if errorlevel 1 (
  echo The existing editor connection was not ready. See the details above.
  pause
  exit /b 1
)
endlocal
