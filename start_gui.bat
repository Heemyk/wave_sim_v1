@echo off
echo 🚀 Starting Acoustic Simulation GUI...
echo.
echo This will start the Electron desktop application.
echo Make sure the backend is running on localhost:8000
echo.

cd frontend

REM Check if dependencies are installed
if not exist "node_modules" (
    echo 📦 Installing dependencies...
    npm install
)

REM Start the Electron app
echo 🖥️  Starting Electron GUI...
npm run electron-dev
