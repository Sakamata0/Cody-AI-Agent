@echo off
title Cody AI Agent - Project Setup

echo ====================================================
echo                  CODY AI AGENT
echo           Autonomous AI Agent Setup
echo ====================================================
echo.
echo Powered by AWS Bedrock + LangChain
echo © 2026 Skander Boughnimi - Smartovate
echo.

echo [1/3] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/3] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [3/3] Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

if not exist .env (
    echo [INFO] Creating .env from .env.example...
    copy .env.example .env
    echo [INFO] Please edit .env with your AWS credentials.
)

echo.
echo ====================================================
echo Setup completed successfully!
echo Cody AI Agent is ready to use.
echo ====================================================
pause