#!/usr/bin/env python3
"""
Comprehensive unit tests for backend components.

This test suite validates individual backend components in isolation:
1. Schema validation and data models
2. Job manager functionality (both in-memory and persistent)
3. Results I/O operations
4. FEM worker integration
5. API endpoint functionality
6. Error handling and edge cases

Usage:
    python -m pytest tests/unit/test_backend_components.py -v
    python tests/unit/test_backend_components.py
"""

import asyncio
import json
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
import unittest
from unittest.mock import patch, MagicMock
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Backend imports
from backend.app.schemas import (
    SimulationRequest, SimulationConfig, RoomGeometry, BoundaryConditions,
    SourceConfig, SensorConfig, JobStatus, SimulationResult, FrequencyResult,
    ImpedanceConfig, MeshConfig, OutputConfig
)
from backend.app.jobs.job_manager import JobManager
from backend.app.jobs.persistent_job_manager import PersistentJobManager
from backend.app.io.results_io import ResultsIO
from backend.app.workers.fem_worker import FEMWorker


def create_test_request(name: str = "Test Job") -> SimulationRequest:
    """Create a standardized test request."""
    return SimulationRequest(
        name=name,
        room=RoomGeometry(type="box", dimensions=[2.0, 2.0, 2.0], center=[0.0, 0.0, 0.0]),
        boundaries=BoundaryConditions(
            walls=ImpedanceConfig(alpha=0.1),
            floor=ImpedanceConfig(alpha=0.2),
            ceiling=ImpedanceConfig(alpha=0.15)
        ),
        sources=[SourceConfig(
            id="source_1",
            position=[1.0, 1.0, 1.0],
            signal={"type": "sine", "f0": 100.0, "f1": 200.0, "amplitude": 1.0}
        )],
        mesh=MeshConfig(element_order=1),
        simulation=SimulationConfig(fmin=100.0, fmax=200.0, df=50.0),
        output=OutputConfig(sensors=[SensorConfig(id="listener_1", position=[0.5, 0.5, 0.5])])
    )


class TestSchemas(unittest.TestCase):
    """Test schema validation and data models."""
    
    def test_simulation_request_validation(self):
        """Test SimulationRequest schema validation."""
        request = create_test_request("Schema Test")
        
        self.assertEqual(request.name, "Schema Test")
        self.assertEqual(request.room.dimensions, [2.0, 2.0, 2.0])
        self.assertEqual(len(request.sources), 1)
        self.assertEqual(len(request.output.sensors), 1)
        self.assertEqual(request.simulation.fmin, 100.0)
        self.assertEqual(request.simulation.fmax, 200.0)
        
    def test_invalid_simulation_request(self):
        """Test SimulationRequest validation with invalid data."""
        # Test with invalid frequency range (fmax <= fmin)
        with self.assertRaises(Exception):
            SimulationRequest(
                name="Test",
                room=RoomGeometry(type="box", dimensions=[2.0, 2.0, 2.0], center=[0.0, 0.0, 0.0]),
                boundaries=BoundaryConditions(walls=ImpedanceConfig(alpha=0.1)),
                sources=[SourceConfig(
                    id="source_1",
                    position=[1.0, 1.0, 1.0],
                    signal={"type": "sine", "f0": 100.0, "f1": 200.0, "amplitude": 1.0}
                )],
                mesh=MeshConfig(element_order=1),
                simulation=SimulationConfig(fmin=200.0, fmax=100.0, df=50.0),  # Invalid: fmax < fmin
                output=OutputConfig(sensors=[SensorConfig(id="listener_1", position=[0.5, 0.5, 0.5])])
            )


class TestJobManager(unittest.TestCase):
    """Test job management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.job_manager = JobManager()
        
    @pytest.mark.asyncio
    async def test_job_submission(self):
        """Test job submission and status tracking."""
        job_id = str(uuid.uuid4())
        request = create_test_request("Job Submission Test")
        
        status = JobStatus(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Test job",
            created_at="2024-01-01T00:00:00"
        )
        
        await self.job_manager.submit_job(job_id, request, status)
        
        # Verify job was submitted
        retrieved_status = await self.job_manager.get_job_status(job_id)
        self.assertIsNotNone(retrieved_status)
        self.assertEqual(retrieved_status.job_id, job_id)
        self.assertEqual(retrieved_status.status, "pending")
        
    @pytest.mark.asyncio
    async def test_job_status_updates(self):
        """Test job status update functionality."""
        job_id = str(uuid.uuid4())
        request = create_test_request("Status Update Test")
        
        status = JobStatus(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Test job",
            created_at="2024-01-01T00:00:00"
        )
        
        await self.job_manager.submit_job(job_id, request, status)
        
        # Update status
        await self.job_manager.update_job_status(job_id, "running", 0.5, "Processing...")
        
        updated_status = await self.job_manager.get_job_status(job_id)
        self.assertEqual(updated_status.status, "running")
        self.assertEqual(updated_status.progress, 0.5)
        self.assertEqual(updated_status.message, "Processing...")
        
    @pytest.mark.asyncio
    async def test_job_cancellation(self):
        """Test job cancellation."""
        job_id = str(uuid.uuid4())
        request = create_test_request("Cancellation Test")
        
        status = JobStatus(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Test job",
            created_at="2024-01-01T00:00:00"
        )
        
        await self.job_manager.submit_job(job_id, request, status)
        
        # Cancel job
        success = await self.job_manager.cancel_job(job_id)
        self.assertTrue(success)
        
        # Verify job is cancelled
        cancelled_status = await self.job_manager.get_job_status(job_id)
        self.assertEqual(cancelled_status.status, "cancelled")
        
    @pytest.mark.asyncio
    async def tearDown(self):
        """Clean up after tests."""
        await self.job_manager.cleanup()


class TestPersistentJobManager(unittest.TestCase):
    """Test persistent job management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.job_manager = PersistentJobManager(data_dir=self.temp_dir)
        
    @pytest.mark.asyncio
    async def test_persistence_across_restarts(self):
        """Test that jobs persist across manager restarts."""
        job_id = str(uuid.uuid4())
        request = create_test_request("Persistent Test Job")
        
        status = JobStatus(
            job_id=job_id,
            status="completed",
            progress=1.0,
            message="Test completed",
            created_at="2024-01-01T00:00:00"
        )
        
        # Submit job
        await self.job_manager.submit_job(job_id, request, status)
        
        # Create new manager instance (simulating restart)
        new_manager = PersistentJobManager(data_dir=self.temp_dir)
        
        # Verify job is still there
        retrieved_status = await new_manager.get_job_status(job_id)
        self.assertIsNotNone(retrieved_status)
        self.assertEqual(retrieved_status.job_id, job_id)
        self.assertEqual(retrieved_status.status, "completed")
        
        # Verify request is still there
        retrieved_request = await new_manager.get_job_request(job_id)
        self.assertIsNotNone(retrieved_request)
        self.assertEqual(retrieved_request.name, "Persistent Test Job")


class TestResultsIO(unittest.TestCase):
    """Test results I/O operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_io = ResultsIO()
        
    @pytest.mark.asyncio
    async def test_results_save_and_load(self):
        """Test saving and loading simulation results."""
        job_id = str(uuid.uuid4())
        
        # Create test results
        frequency_result = FrequencyResult(
            frequency=100.0,
            sensor_data={
                "listener_1": 0.001 + 0.002j,
                "listener_2": 0.0005 + 0.001j
            },
            metadata={"test": "data"}
        )
        
        results = SimulationResult(
            job_id=job_id,
            config=create_test_request("Results Test"),
            frequencies=[frequency_result],
            metadata={"performance": {"solve_time": 1.5}}
        )
        
        # Save results
        await self.results_io.save_results(job_id, results)
        
        # Verify file was created
        results_file = Path(self.temp_dir) / f"{job_id}" / "frequencies.json"
        self.assertTrue(results_file.exists())
        
        # Load results
        loaded_results = await self.results_io.load_results(job_id)
        self.assertIsNotNone(loaded_results)
        self.assertEqual(loaded_results.job_id, job_id)
        self.assertEqual(len(loaded_results.frequencies), 1)
        
        # Verify complex number handling
        sensor_data = loaded_results.frequencies[0].sensor_data
        self.assertEqual(sensor_data["listener_1"], 0.001 + 0.002j)
        self.assertEqual(sensor_data["listener_2"], 0.0005 + 0.001j)


class TestFEMWorker(unittest.TestCase):
    """Test FEM worker functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.worker = FEMWorker()
        
    def test_fem_worker_initialization(self):
        """Test FEM worker initialization."""
        self.assertIsNotNone(self.worker)
        self.assertIsNone(self.worker.solver)  # Should be None until first simulation
        
    def test_sensor_position_extraction(self):
        """Test sensor position extraction method."""
        request = create_test_request("Sensor Test")
        
        positions = self.worker._get_sensor_positions(request)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0], [0.5, 0.5, 0.5])
        
    def test_frequency_generation(self):
        """Test frequency range generation."""
        simulation_config = SimulationConfig(
            fmin=100.0,
            fmax=200.0,
            df=50.0
        )
        
        frequencies = self.worker._generate_frequencies(simulation_config)
        self.assertEqual(frequencies, [100.0, 150.0, 200.0])


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoint functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_url = "http://localhost:8000"
        
    def test_health_endpoint(self):
        """Test health check endpoint."""
        import requests
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
        except requests.exceptions.ConnectionError:
            self.skipTest("Backend not running")
            
    def test_job_submission_endpoint(self):
        """Test job submission endpoint."""
        import requests
        try:
            test_request = {
                "name": "API Test Job",
                "room": {"type": "box", "dimensions": [2.0, 2.0, 2.0], "center": [0.0, 0.0, 0.0]},
                "boundaries": {"walls": {"alpha": 0.1}},
                "sources": [{"id": "source_1", "position": [1.0, 1.0, 1.0], "signal": {"type": "sine", "f0": 100.0, "f1": 200.0}}],
                "mesh": {"element_order": 1},
                "simulation": {"fmin": 100.0, "fmax": 200.0, "df": 50.0},
                "output": {"sensors": [{"id": "listener_1", "position": [0.5, 0.5, 0.5]}]}
            }
            
            response = requests.post(f"{self.base_url}/api/simulate", json=test_request, timeout=10)
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn("job_id", data)
            self.assertEqual(data["status"], "pending")
            
        except requests.exceptions.ConnectionError:
            self.skipTest("Backend not running")


async def run_async_tests():
    """Run async tests."""
    print("🧪 Running Backend Component Unit Tests")
    print("=" * 50)
    
    # Test JobManager
    print("1️⃣ Testing JobManager...")
    job_manager = JobManager()
    
    job_id = str(uuid.uuid4())
    request = create_test_request("Unit Test Job")
    
    status = JobStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Unit test job",
        created_at="2024-01-01T00:00:00"
    )
    
    await job_manager.submit_job(job_id, request, status)
    retrieved = await job_manager.get_job_status(job_id)
    print(f"   ✅ Job submitted and retrieved: {retrieved.job_id}")
    
    await job_manager.cleanup()
    
    # Test PersistentJobManager
    print("2️⃣ Testing PersistentJobManager...")
    temp_dir = tempfile.mkdtemp()
    persistent_manager = PersistentJobManager(data_dir=temp_dir)
    
    await persistent_manager.submit_job(job_id, request, status)
    retrieved_request = await persistent_manager.get_job_request(job_id)
    print(f"   ✅ Persistent job request retrieved: {retrieved_request.name}")
    
    # Test ResultsIO
    print("3️⃣ Testing ResultsIO...")
    results_io = ResultsIO(data_dir=temp_dir)
    
    frequency_result = FrequencyResult(
        frequency=100.0,
        sensor_data={"listener_1": 0.001 + 0.002j},
        metadata={"test": "data"}
    )
    
    results = SimulationResult(
        job_id=job_id,
        config=create_test_request("Results Test"),
        frequencies=[frequency_result],
        metadata={"performance": {"solve_time": 1.5}}
    )
    
    await results_io.save_results(job_id, results)
    loaded_results = await results_io.load_results(job_id)
    print(f"   ✅ Results saved and loaded: {loaded_results.job_id}")
    
    # Test FEMWorker
    print("4️⃣ Testing FEMWorker...")
    worker = FEMWorker()
    
    sensors = [SensorConfig(id="listener_1", position=[1.0, 2.0, 3.0])]
    positions = worker._get_sensor_positions(sensors)
    print(f"   ✅ Sensor positions extracted: {positions}")
    
    simulation_config = SimulationConfig(fmin=100.0, fmax=200.0, df=50.0)
    frequencies = worker._generate_frequencies(simulation_config)
    print(f"   ✅ Frequencies generated: {frequencies}")
    
    print("\n✅ All unit tests completed successfully!")


def run_sync_tests():
    """Run synchronous tests."""
    print("🧪 Running Synchronous Unit Tests")
    print("=" * 50)
    
    # Test schemas
    print("1️⃣ Testing schemas...")
    try:
        request = create_test_request("Schema Test")
        print("   ✅ Schema validation passed")
    except Exception as e:
        print(f"   ❌ Schema validation failed: {e}")
        
    # Test FEMWorker
    print("2️⃣ Testing FEMWorker...")
    try:
        worker = FEMWorker()
        sensors = [SensorConfig(id="listener_1", position=[1.0, 2.0, 3.0])]
        positions = worker._get_sensor_positions(sensors)
        print(f"   ✅ FEMWorker initialized: {positions}")
    except Exception as e:
        print(f"   ❌ FEMWorker test failed: {e}")


if __name__ == "__main__":
    # Run synchronous tests
    run_sync_tests()
    
    # Run async tests
    asyncio.run(run_async_tests())