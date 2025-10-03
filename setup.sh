#!/bin/bash

# Development setup script for Acoustic Room Simulator

set -e

echo "Setting up Acoustic Room Simulator development environment..."

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{meshes,results,cache}
mkdir -p logs

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Docker found. Building containers..."
    
    # Build backend container
    echo "Building backend container..."
    docker build -f docker/Dockerfile.backend -t acoustic-sim-backend .
    
    # Build frontend container
    echo "Building frontend container..."
    docker build -f docker/Dockerfile.frontend -t acoustic-sim-frontend .
    
    echo "Containers built successfully!"
    echo ""
    echo "To start the application:"
    echo "  docker-compose -f docker/docker-compose.yml up"
    echo ""
    echo "To start with Jupyter for development:"
    echo "  docker-compose -f docker/docker-compose.yml --profile dev up"
    
else
    echo "Docker not found. Please install Docker to use the containerized setup."
    echo ""
    echo "For local development without Docker:"
    echo "1. Install Python 3.9+ and Node.js 18+"
    echo "2. Install FEniCS: conda install -c conda-forge fenics dolfinx"
    echo "3. Install Python dependencies: pip install -e ."
    echo "4. Install frontend dependencies: cd frontend && npm install"
    echo "5. Start backend: python -m backend.app.main"
    echo "6. Start frontend: cd frontend && npm start"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run the test script: python fem/test_helmholtz.py"
echo "2. Start the application with Docker Compose"
echo "3. Open http://localhost:3000 in your browser"
