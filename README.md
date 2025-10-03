# Acoustic Room Simulator

A browser-deployable acoustic room simulator using FEM (FEniCS) for low-mid frequencies and geometric acoustics for high frequencies.

## Features

- **Physics-accurate FEM**: FEniCS-based Helmholtz equation solver for low-mid frequencies
- **Hybrid approach**: Geometric acoustics/ray tracing for high frequencies
- **Interactive browser interface**: WebGL visualization with three.js
- **Local deployment**: Docker-based setup for reproducible environments
- **GPU-ready**: Designed for future GPU acceleration migration

## Quick Start

```bash
# Clone and start the application
git clone <repo-url>
cd acoustic_sim
docker-compose up --build
```

Then open http://localhost:3000 in your browser.

## Architecture

- **Backend**: FastAPI + FEniCS (Python) for FEM computations
- **Frontend**: React + TypeScript + three.js for 3D visualization
- **Communication**: WebSocket for real-time results, REST for job control
- **Data**: HDF5/XDMF for field data, JSON for configuration

## Project Structure

```
acoustic_sim/
├── backend/          # FastAPI server + FEM workers
├── frontend/         # React app with 3D viewer
├── fem/             # FEniCS physics core
├── raytracer/       # Geometric acoustics
├── docker/          # Docker configurations
├── examples/        # Sample rooms and configs
└── notebooks/       # Jupyter demos
```

## Development

See [docs/development.md](docs/development.md) for detailed setup instructions.

## License

MIT License - see [LICENSE](LICENSE) file.
