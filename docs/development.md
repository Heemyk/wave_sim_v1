# Development Guide

## Quick Start

1. **Clone and setup:**
   ```bash
   git clone <repo-url>
   cd acoustic_sim
   ./setup.sh
   ```

2. **Start the application:**
   ```bash
   docker-compose -f docker/docker-compose.yml up --build
   ```

3. **Open your browser:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Architecture Overview

### Backend (Python + FEniCS)
- **FastAPI** server for REST API and WebSocket communication
- **FEniCS/dolfinx** for FEM acoustic simulations
- **Job management** system for handling multiple simulations
- **Results I/O** with HDF5/JSON storage

### Frontend (React + TypeScript)
- **React** with TypeScript for type safety
- **Three.js** via @react-three/fiber for 3D visualization
- **Plotly.js** for frequency response plots
- **WebSocket** connection for real-time updates

### Docker Setup
- **Backend container**: FEniCS + FastAPI + all Python dependencies
- **Frontend container**: React build served with nginx
- **Redis**: Job queue and caching
- **Jupyter**: Optional development environment

## Development Workflow

### Running Tests
```bash
# Test the FEM solver
python fem/test_helmholtz.py

# Run backend tests (when available)
python -m pytest backend/tests/

# Run frontend tests
cd frontend && npm test
```

### Local Development (without Docker)

1. **Backend setup:**
   ```bash
   # Install FEniCS
   conda install -c conda-forge fenics dolfinx petsc4py slepc4py
   
   # Install Python dependencies
   pip install -e .
   
   # Start backend
   python -m backend.app.main
   ```

2. **Frontend setup:**
   ```bash
   cd frontend
   npm install
   npm start
   ```

### Adding New Features

#### FEM Module (`fem/`)
- Add new solver types in `helmholtz_solver.py`
- Implement boundary conditions
- Add preconditioners in `preconditioners.py`

#### Backend API (`backend/app/`)
- Add new endpoints in `api/routes.py`
- Update schemas in `schemas.py`
- Add workers in `workers/`

#### Frontend (`frontend/src/`)
- Add components in `components/`
- Update 3D visualization in `App.tsx`
- Add new plot types

## Configuration

### Simulation Parameters
See `examples/sample_room/config.yaml` for a complete configuration example.

Key parameters:
- **Room geometry**: Box, cylinder, or custom mesh
- **Boundary conditions**: Impedance/absorption coefficients
- **Sources**: Point, line, or surface sources with signal types
- **Mesh**: Element order, target size, refinement
- **Simulation**: Frequency range, solver type, tolerance

### Performance Tuning

#### For Low-End Machines
```yaml
mesh:
  target_h: 0.2  # Larger elements
  element_order: 1  # Linear elements

simulation:
  fmax: 1000  # Lower frequency limit
  df: 100  # Larger frequency steps
  solver_type: "direct"  # More memory but faster for small problems
```

#### For High-End Machines
```yaml
mesh:
  target_h: 0.05  # Smaller elements
  element_order: 2  # Quadratic elements

simulation:
  fmax: 8000  # Higher frequency limit
  df: 20  # Smaller frequency steps
  solver_type: "iterative"  # Less memory, scales better
```

## GPU Migration Strategy

The current implementation is CPU-based but designed for easy GPU migration:

### Current CPU Implementation
- FEniCS uses PETSc for linear algebra
- Direct solvers (LU) for small problems
- Iterative solvers (GMRES) for large problems

### Future GPU Implementation
1. **Replace PETSc with CuPy/CuBLAS** for GPU linear algebra
2. **Use CUDA kernels** for mesh operations
3. **Implement GPU-accelerated preconditioners**
4. **Add GPU memory management** for large meshes

### Migration Path
```python
# Current (CPU)
from petsc4py import PETSc
solver = PETSc.KSP().create()

# Future (GPU)
import cupy as cp
solver = cp.linalg.solve(A_gpu, b_gpu)
```

## Troubleshooting

### Common Issues

1. **FEniCS Import Error**
   ```bash
   # Solution: Install via conda-forge
   conda install -c conda-forge fenics dolfinx
   ```

2. **Memory Issues**
   - Reduce mesh resolution (`target_h`)
   - Use iterative solver
   - Limit frequency range

3. **Slow Performance**
   - Use direct solver for small problems
   - Increase frequency step (`df`)
   - Use lower element order

4. **Docker Build Fails**
   ```bash
   # Clean build
   docker system prune -a
   docker-compose build --no-cache
   ```

### Debug Mode

Enable debug logging:
```bash
# Backend
export LOG_LEVEL=DEBUG
python -m backend.app.main

# Frontend
REACT_APP_DEBUG=true npm start
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Code Style
- Python: Black + isort + mypy
- TypeScript: ESLint + Prettier
- Pre-commit hooks configured

## Performance Benchmarks

### Typical Performance (8-core CPU, 32GB RAM)

| Mesh Size | Frequency Range | Element Order | Time per Freq |
|-----------|----------------|---------------|---------------|
| 10K DOFs  | 20-1000 Hz     | 1             | ~2 seconds    |
| 50K DOFs  | 20-2000 Hz     | 1             | ~10 seconds   |
| 100K DOFs | 20-1000 Hz     | 2             | ~30 seconds   |

### Memory Usage
- Direct solver: ~8x matrix size
- Iterative solver: ~2x matrix size
- Mesh storage: ~100MB per 50K DOFs

## Future Enhancements

1. **Ray Tracing Module**: High-frequency geometric acoustics
2. **Hybrid Coupling**: Seamless FEM ↔ ray tracing
3. **GPU Acceleration**: CUDA-based linear algebra
4. **Real-time Simulation**: WebGL-based visualization
5. **Machine Learning**: Neural network-based acceleration
6. **Cloud Deployment**: Kubernetes-based scaling
