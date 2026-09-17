@echo off
REM Start ORBIT on Windows.
cd /d "%~dp0"
python -m app.seed
streamlit run frontend/streamlit_app.py
