#!/usr/bin/env python3
"""Debug script for testing the acoustic simulator backend."""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_imports():
    """Test if all imports work correctly."""
    logger.info("Testing imports...")
    
    try:
        from backend.app.schemas import SimulationRequest, JobStatus
        logger.info("✓ Schemas imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import schemas: {e}")
        return False
    
    try:
        from backend.app.jobs.job_manager import JobManager
        logger.info("✓ JobManager imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import JobManager: {e}")
        return False
    
    try:
        from backend.app.io.results_io import ResultsIO
        logger.info("✓ ResultsIO imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import ResultsIO: {e}")
        return False
    
    try:
        from backend.app.workers.fem_worker import FEMWorker
        logger.info("✓ FEMWorker imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import FEMWorker: {e}")
        return False
    
    return True

async def test_job_manager():
    """Test the job manager."""
    logger.info("Testing JobManager...")
    
    try:
        from backend.app.jobs.job_manager import JobManager
        from backend.app.schemas import SimulationRequest, JobStatus
        
        # Create job manager
        job_manager = JobManager()
        
        # Create a simple simulation request
        request = SimulationRequest(
            room={"type": "box", "dimensions": [4.0, 3.0, 2.5]},
            boundaries={"walls": {"alpha": 0.15}},
            sources=[{
                "id": "test_source",
                "position": [0.0, 0.0, 1.0],
                "signal": {"type": "chirp", "f0": 20, "f1": 100}
            }],
            mesh={"element_order": 1, "target_h": 0.2},
            simulation={"fmin": 20, "fmax": 100, "df": 20},
            output={"points_of_interest": []}
        )
        
        # Create job status
        job_status = JobStatus(
            job_id="test_job",
            status="pending",
            progress=0.0,
            message="Test job",
            created_at="2024-01-01T00:00:00"
        )
        
        # Submit job
        await job_manager.submit_job("test_job", request, job_status)
        logger.info("✓ Job submitted successfully")
        
        # Get job status
        status = await job_manager.get_job_status("test_job")
        if status:
            logger.info(f"✓ Job status retrieved: {status.status}")
        else:
            logger.error("✗ Failed to retrieve job status")
            return False
        
        # List jobs
        jobs = await job_manager.list_jobs()
        logger.info(f"✓ Listed {len(jobs)} jobs")
        
        # Cleanup
        await job_manager.cleanup()
        logger.info("✓ JobManager test completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ JobManager test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def test_fem_worker():
    """Test the FEM worker (without FEniCS)."""
    logger.info("Testing FEMWorker...")
    
    try:
        from backend.app.workers.fem_worker import FEMWorker
        from backend.app.schemas import SimulationRequest
        
        # Create worker
        worker = FEMWorker()
        logger.info("✓ FEMWorker created successfully")
        
        # Test mesh generation
        request = SimulationRequest(
            room={"type": "box", "dimensions": [2.0, 2.0, 2.0]},
            boundaries={"walls": {"alpha": 0.15}},
            sources=[{
                "id": "test_source",
                "position": [0.0, 0.0, 1.0],
                "signal": {"type": "chirp", "f0": 20, "f1": 100}
            }],
            mesh={"element_order": 1, "target_h": 0.3},
            simulation={"fmin": 20, "fmax": 100, "df": 40},
            output={"points_of_interest": []}
        )
        
        # Test mesh generation
        mesh_file = await worker._generate_mesh(request)
        logger.info(f"✓ Mesh generated: {mesh_file}")
        
        # Test frequency generation
        frequencies = worker._generate_frequencies(request.simulation)
        logger.info(f"✓ Frequencies generated: {frequencies}")
        
        # Test boundary impedance
        impedance = worker._get_boundary_impedance(request)
        logger.info(f"✓ Boundary impedance: {impedance}")
        
        logger.info("✓ FEMWorker test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ FEMWorker test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def test_fenics_availability():
    """Test if FEniCS is available."""
    logger.info("Testing FEniCS availability...")
    
    try:
        import dolfinx
        import ufl
        import petsc4py
        logger.info("✓ FEniCS/dolfinx is available")
        return True
    except ImportError as e:
        logger.warning(f"⚠ FEniCS/dolfinx not available: {e}")
        logger.info("This is expected if FEniCS is not installed")
        return False

async def test_directory_structure():
    """Test if required directories exist."""
    logger.info("Testing directory structure...")
    
    required_dirs = ["data", "data/meshes", "data/results", "data/cache", "logs"]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created directory: {dir_path}")
        else:
            logger.info(f"✓ Directory exists: {dir_path}")
    
    return True

async def main():
    """Run all tests."""
    logger.info("Starting acoustic simulator debug tests...")
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Job Manager", test_job_manager),
        ("FEM Worker", test_fem_worker),
        ("FEniCS Availability", test_fenics_availability),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The system is ready to run.")
    else:
        logger.warning("⚠ Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
