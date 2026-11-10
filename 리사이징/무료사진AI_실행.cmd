@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
".local-ai\env\Scripts\python.exe" tools\local-photo-server.py
pause
