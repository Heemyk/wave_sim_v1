@echo off
REM Local development setup script for Windows (no Docker)
echo Setting up Acoustic Room Simulator for local development...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed. Please install Python 3.9+ first.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed. Please install Node.js first.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install Python dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -e .

REM Install FEM dependencies (optional - comment out if you don't need FEM)
echo Installing FEM dependencies (this may take a while)...
pip install fenics dolfinx petsc4py slepc4py mpi4py gmsh

REM Install frontend dependencies
echo Installing frontend dependencies...
cd frontend
npm install
cd ..

REM Create data directories
echo Creating data directories...
if not exist "data" mkdir data
if not exist "data\meshes" mkdir data\meshes
if not exist "data\results" mkdir data\results
if not exist "data\cache" mkdir data\cache
if not exist "logs" mkdir logs

echo.
echo Setup complete! To run the application:
echo.
echo 1. Start the backend (in one terminal):
echo    venv\Scripts\activate.bat
echo    python -m backend.app.main
echo.
echo 2. Start the frontend (in another terminal):
echo    cd frontend
echo    npm start
echo.
echo The application will be available at:
echo   Frontend: http://localhost:3000
echo   Backend API: http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.
pause
