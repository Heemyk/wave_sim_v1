"""
Pytest configuration for the Acoustic Room Simulation System.

This file provides shared fixtures and configuration for all tests.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_simulation_request():
    """Create a sample simulation request for testing."""
    return {
        "name": "Test Simulation",
        "room": {
            "type": "box",
            "dimensions": [4.0, 3.0, 2.5],
            "center": [0.0, 0.0, 0.0]
        },
        "boundaries": {
            "walls": {"alpha": 0.1},
            "floor": {"alpha": 0.2},
            "ceiling": {"alpha": 0.15}
        },
        "sources": [{
            "position": [2.0, 1.5, 1.0],
            "signal": {"type": "sine", "frequency": 100.0, "amplitude": 1.0}
        }],
        "listeners": [
            {"id": "listener_1", "position": [1.0, 1.0, 1.0]},
            {"id": "listener_2", "position": [3.0, 2.0, 1.5]}
        ],
        "simulation": {
            "fmin": 100.0,
            "fmax": 200.0,
            "fstep": 100.0,
            "element_order": 1
        }
    }


@pytest.fixture
def backend_url():
    """Backend URL for integration tests."""
    return "http://localhost:8000"


@pytest.fixture
def websocket_url(backend_url):
    """WebSocket URL for integration tests."""
    return backend_url.replace("http", "ws")


@pytest.fixture
def is_backend_running(backend_url):
    """Check if backend is running."""
    import requests
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "fem: marks tests as FEM solver tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_backend: marks tests that require backend to be running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add markers based on file location
        test_path = Path(item.fspath)
        if "integration" in str(test_path):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(test_path):
            item.add_marker(pytest.mark.unit)
        elif "fem" in str(test_path):
            item.add_marker(pytest.mark.fem)
        
        # Mark integration tests as slow
        if "integration" in str(test_path):
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.requires_backend)
