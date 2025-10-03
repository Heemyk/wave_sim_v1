#!/usr/bin/env python3
"""Standalone runner for the acoustic simulator backend."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set working directory to project root
os.chdir(project_root)

if __name__ == "__main__":
    import uvicorn
    from backend.app.main import app
    
    print("Starting Acoustic Room Simulator Backend...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[0]}")
    
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
