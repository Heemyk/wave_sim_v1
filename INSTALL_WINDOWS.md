# Windows Installation Guide for SfePy Migration

## The Problem
SfePy installation failed because it requires compilation of C extensions, which needs:
- Python development headers (`Python.h`)
- A proper C compiler (Visual Studio Build Tools)
- Correct virtual environment setup

## Solutions (Choose One)

### Option 1: Use Conda (Recommended)
Conda provides pre-compiled binaries that avoid compilation issues:

```bash
# Remove the problematic virtual environment
rmdir /s venv

# Install Miniconda or Anaconda, then:
conda create -n acoustic-sim python=3.11
conda activate acoustic-sim

# Install SfePy via conda-forge (pre-compiled)
conda install -c conda-forge sfepy meshio gmsh

# Install other dependencies
pip install fastapi uvicorn websockets pydantic numpy scipy h5py matplotlib plotly python-multipart aiofiles redis rq
```

### Option 2: Install Visual Studio Build Tools
If you want to stick with pip:

```bash
# 1. Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
# Install "C++ build tools" workload

# 2. Install Python development headers
# Download Python from python.org (not Microsoft Store version)
# Make sure to check "Add Python to PATH" during installation

# 3. Recreate virtual environment
rmdir /s venv
python -m venv venv
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Option 3: Use Pre-compiled Wheels (Alternative)
Try installing from a different source:

```bash
# Remove virtual environment
rmdir /s venv
python -m venv venv
venv\Scripts\activate

# Try installing SfePy from conda-forge via pip
pip install --find-links https://conda.anaconda.org/conda-forge/ sfepy

# Or try a specific version
pip install sfepy==2023.3
```

### Option 4: Docker Approach (Easiest)
Use Docker to avoid Windows compilation issues entirely:

```bash
# Build and run with Docker
docker-compose -f docker/docker-compose.yml up --build
```

## Recommended Approach
I recommend **Option 1 (Conda)** because:
- ✅ No compilation required
- ✅ Pre-tested binaries
- ✅ Better dependency management
- ✅ Works reliably on Windows

## After Successful Installation
Once SfePy is installed, test it with:
```bash
python test_sfepy_migration.py
```

## Troubleshooting
If you still have issues:
1. Make sure you're using Python 3.9-3.11 (SfePy may not support Python 3.13 yet)
2. Try using Anaconda instead of Miniconda
3. Consider using WSL2 with Ubuntu for a Linux-like environment

