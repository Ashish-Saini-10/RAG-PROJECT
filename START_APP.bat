@echo off
title ShowYourWork - RAG Assistant
color 0A

echo.
echo  ============================================================
echo   ShowYourWork - Academic RAG Assistant
echo   Starting up...
echo  ============================================================
echo.

:: Check if .env file exists
if not exist ".env" (
    echo  [!] No .env file found.
    echo  [!] Create a file called .env in this folder with:
    echo      GROQ_API_KEY=your_key_here
    echo.
    echo  Get your free key at: https://console.groq.com/keys
    echo.
    pause
)

:: Create data folder if missing
if not exist "data" mkdir data

:: Launch the app
echo  [+] Launching app at http://localhost:8501
echo  [+] Press CTRL+C in this window to stop the server
echo.
streamlit run app.py

pause
