#!/bin/bash

# Local development setup script (no Docker)
echo "Setting up Acoustic Room Simulator for local development..."

# # Check if Python is installed
# if ! command -v python &> /dev/null; then
#     echo "Error: Python 3 is not installed. Please install Python 3.9+ first."
#     exit 1
# fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -e .

# Install FEM dependencies (optional - comment out if you don't need FEM)
echo "Installing FEM dependencies (this may take a while)..."
pip install fenics dolfinx petsc4py slepc4py mpi4py gmsh

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create data directories
echo "Creating data directories..."
mkdir -p data/{meshes,results,cache}
mkdir -p logs

echo ""
echo "Setup complete! To run the application:"
echo ""
echo "1. Start the backend (in one terminal):"
echo "   source venv/bin/activate"
echo "   python -m backend.app.main"
echo ""
echo "2. Start the frontend (in another terminal):"
echo "   cd frontend"
echo "   npm start"
echo ""
echo "The application will be available at:"
echo "  Frontend: http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
