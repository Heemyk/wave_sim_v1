#!/bin/bash
echo "🚀 Starting Acoustic Simulation GUI (Simple Mode)..."
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

# Start Electron with the built app (this will build first if needed)
echo "🖥️  Starting Electron GUI..."
npm run electron-simple
