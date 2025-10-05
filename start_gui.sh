#!/bin/bash
echo "🚀 Starting Acoustic Simulation GUI..."
echo ""
echo "This will start the Electron desktop application."
echo "Make sure the backend is running on localhost:8000"
echo ""

# Clear problematic environment variables
unset HOST
unset HOSTNAME

# Check if backend is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  WARNING: Backend doesn't seem to be running on localhost:8000"
    echo "   Please start the backend first with: python run_backend.py"
    echo ""
fi

cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Set proper environment variables for WSL
export HOST=0.0.0.0
export BROWSER=none

# Start the Electron app
echo "🖥️  Starting Electron GUI..."
npm run electron-dev
