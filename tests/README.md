# Test Suite for Acoustic Room Simulation System

This directory contains comprehensive tests for the acoustic room simulation system, organized into three main categories:

## 📁 Directory Structure

```
tests/
├── unit/                           # Unit tests for individual components
│   ├── test_backend_components.py  # Backend component unit tests
│   └── __init__.py
├── integration/                    # Integration tests for end-to-end workflows
│   ├── test_end_to_end_workflows.py # Comprehensive E2E tests
│   └── __init__.py
├── fem/                           # FEM solver tests
│   ├── test_helmholtz.py          # Helmholtz solver tests
│   └── __init__.py
├── run_tests.py                   # Unified test runner
├── conftest.py                    # Pytest configuration and fixtures
├── README.md                      # This file
└── __init__.py
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)
Test individual backend components in isolation:
- **Schema validation** - Data model validation
- **Job management** - Job lifecycle and status tracking
- **Results I/O** - File operations and data persistence
- **FEM worker** - Integration with Helmholtz solver
- **API endpoints** - REST API functionality

### Integration Tests (`tests/integration/`)
Test complete workflows and system integration:
- **Complete workflow** - Job submission → processing → results
- **WebSocket communication** - Real-time status updates
- **3D visualization data** - Mesh and field data extraction
- **Error handling** - Edge cases and error scenarios
- **Backend persistence** - Data survival across restarts

### FEM Tests (`tests/fem/`)
Test the finite element method solver:
- **Helmholtz solver** - Acoustic wave equation solving
- **Mesh generation** - 3D geometry creation
- **Boundary conditions** - Acoustic boundary handling
- **Frequency analysis** - Multi-frequency computations

## 🚀 Running Tests

### Quick Start
```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --unit           # Unit tests only
python tests/run_tests.py --integration    # Integration tests only
python tests/run_tests.py --fem            # FEM tests only
python tests/run_tests.py --quick          # Unit + FEM (fast)
python tests/run_tests.py --backend        # Unit + Integration
```

### Using pytest directly
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/fem/ -v

# Run with markers
python -m pytest -m "not slow" -v         # Skip slow tests
python -m pytest -m "requires_backend" -v # Only backend tests
```

### Individual test files
```bash
# Run specific test files
python tests/unit/test_backend_components.py
python tests/integration/test_end_to_end_workflows.py
python tests/fem/test_helmholtz.py
```

## 📋 Test Requirements

### Prerequisites
- Python 3.8+
- All project dependencies installed
- Backend running on `localhost:8000` (for integration tests)

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start backend (for integration tests)
python run_backend.py
```

## 🎯 Test Coverage

The test suite covers:

### ✅ Core Functionality
- [x] Job submission and management
- [x] FEM simulation execution
- [x] Results generation and storage
- [x] WebSocket real-time updates
- [x] 3D visualization data extraction

### ✅ Data Validation
- [x] Complex number serialization/deserialization
- [x] Mesh geometry data (vertices, cells)
- [x] Pressure field data (magnitude, phase, real/imag)
- [x] Sensor readings validation
- [x] Acoustic metrics verification

### ✅ Error Handling
- [x] Invalid job IDs
- [x] Malformed requests
- [x] Backend connectivity issues
- [x] Simulation failures
- [x] Data corruption scenarios

### ✅ Performance
- [x] Job processing times
- [x] Memory usage validation
- [x] Large dataset handling
- [x] Concurrent job processing

## 🔧 Test Configuration

### Pytest Markers
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.fem` - FEM solver tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_backend` - Tests needing backend

### Fixtures Available
- `temp_dir` - Temporary directory for test files
- `sample_simulation_request` - Standard test request
- `backend_url` - Backend URL configuration
- `is_backend_running` - Backend availability check

## 📊 Test Results

Test results are automatically saved to `tests/test_results.json` when using the test runner with `--save-results` flag.

### Result Format
```json
{
  "unit": {
    "success": true,
    "duration": "2.34s",
    "stdout": "...",
    "stderr": ""
  },
  "integration": {
    "success": true,
    "workflow_test": {...},
    "visualization_test": {...}
  },
  "fem": {
    "success": true,
    "duration": "1.23s"
  }
}
```

## 🐛 Troubleshooting

### Common Issues

**Backend not running for integration tests:**
```bash
# Start the backend first
python run_backend.py

# Then run integration tests
python tests/run_tests.py --integration
```

**FEM solver dependencies missing:**
```bash
# Install SfePy (Windows-compatible)
conda install -c conda-forge sfepy meshio gmsh

# Or install via pip
pip install sfepy meshio gmsh
```

**Tests timing out:**
- Check backend is responsive
- Increase timeout in test configuration
- Run tests individually to isolate issues

### Debug Mode
```bash
# Run with verbose output
python -m pytest tests/ -v -s

# Run specific test with debug
python tests/integration/test_end_to_end_workflows.py
```

## 📈 Continuous Integration

The test suite is designed to work in CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    python tests/run_tests.py --quick --save-results
    
- name: Run Integration Tests
  run: |
    python run_backend.py &
    sleep 10
    python tests/run_tests.py --integration
```

## 🤝 Contributing

When adding new tests:

1. **Choose the right category** - Unit, Integration, or FEM
2. **Follow naming conventions** - `test_*.py` files
3. **Add appropriate markers** - Use pytest markers
4. **Include docstrings** - Document test purpose
5. **Update this README** - Document new test coverage

### Test Writing Guidelines

- **Unit tests** should be fast and isolated
- **Integration tests** should test complete workflows
- **FEM tests** should validate acoustic physics
- Use fixtures for common test data
- Include both positive and negative test cases
- Add performance benchmarks where appropriate
