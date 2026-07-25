@echo off
title Agentic Quant - Kontrol Paneli
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" agent.py
) else (
  echo [uyari] .venv bulunamadi - sistem python denenecek ^(bagimliliklar eksik olabilir^).
  python agent.py
)
pause
