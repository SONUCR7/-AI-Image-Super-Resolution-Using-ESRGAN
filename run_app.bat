@echo off
title SR Vision Lab - Comparative Image Super-Resolution
cd /d "%~dp0"
echo ===================================================
echo   Launching SR Vision Lab Dashboard (Streamlit)
echo ===================================================
streamlit run app.py
pause
