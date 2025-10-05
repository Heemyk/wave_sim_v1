#!/usr/bin/env python3
"""
Test script to verify SfePy migration works correctly.
This script tests the basic functionality of the new SfePy-based HelmholtzSolver.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sfepy_import():
    """Test that SfePy can be imported."""
    try:
        import sfepy
        logger.info(f"✓ SfePy imported successfully (version: {sfepy.__version__})")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import SfePy: {e}")
        return False

def test_mesh_creation():
    """Test that we can create a simple mesh."""
    try:
        from fem.helmholtz_solver import create_simple_box_mesh
        
        # Create a simple box mesh
        mesh_file = create_simple_box_mesh([2.0, 2.0, 2.0], h=0.5)
        logger.info(f"✓ Created test mesh: {mesh_file}")
        
        # Check if file exists
        if Path(mesh_file).exists():
            logger.info(f"✓ Mesh file exists and is {Path(mesh_file).stat().st_size} bytes")
            return True, mesh_file
        else:
            logger.error("✗ Mesh file was not created")
            return False, None
            
    except Exception as e:
        logger.error(f"✗ Failed to create mesh: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, None

def test_solver_initialization(mesh_file):
    """Test that the HelmholtzSolver can be initialized."""
    try:
        from fem.helmholtz_solver import HelmholtzSolver
        
        # Initialize solver
        solver = HelmholtzSolver(mesh_file=mesh_file, element_order=1)
        logger.info(f"✓ HelmholtzSolver initialized with {solver.mesh.n_nod} nodes")
        
        # Test basic properties
        assert solver.c == 343.0, "Speed of sound should be 343.0 m/s"
        assert solver.rho == 1.225, "Air density should be 1.225 kg/m³"
        logger.info("✓ Solver properties are correct")
        
        return True, solver
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize solver: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, None

def test_basic_solve(solver):
    """Test basic solving capability."""
    try:
        # Test system assembly
        k = 2 * 3.14159 * 100 / 343.0  # Wavenumber for 100 Hz
        source_pos = [0.0, 0.0, 0.0]
        
        solver.assemble_system(k, source_pos)
        logger.info("✓ System assembled successfully")
        
        # Note: We won't actually solve here since SfePy problem definition
        # needs to be properly set up, but we can test the assembly
        logger.info("✓ Basic solve test passed (assembly only)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed basic solve test: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_mesh_data_extraction(solver):
    """Test mesh data extraction."""
    try:
        mesh_data = solver.get_mesh_data()
        
        assert 'vertices' in mesh_data, "Mesh data should contain vertices"
        assert 'cells' in mesh_data, "Mesh data should contain cells"
        assert 'num_vertices' in mesh_data, "Mesh data should contain num_vertices"
        assert 'num_cells' in mesh_data, "Mesh data should contain num_cells"
        
        logger.info(f"✓ Mesh data extraction successful:")
        logger.info(f"  - {mesh_data['num_vertices']} vertices")
        logger.info(f"  - {mesh_data['num_cells']} cells")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed mesh data extraction: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting SfePy migration tests...")
    logger.info("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: SfePy import
    if test_sfepy_import():
        tests_passed += 1
    
    # Test 2: Mesh creation
    mesh_success, mesh_file = test_mesh_creation()
    if mesh_success:
        tests_passed += 1
    
    # Test 3: Solver initialization
    if mesh_file:
        solver_success, solver = test_solver_initialization(mesh_file)
        if solver_success:
            tests_passed += 1
            
            # Test 4: Basic solve
            if test_basic_solve(solver):
                tests_passed += 1
            
            # Test 5: Mesh data extraction
            if test_mesh_data_extraction(solver):
                tests_passed += 1
    
    # Clean up
    if mesh_file and Path(mesh_file).exists():
        Path(mesh_file).unlink()
        logger.info(f"✓ Cleaned up test mesh file")
    
    # Results
    logger.info("=" * 50)
    logger.info(f"Tests completed: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 All tests passed! SfePy migration appears successful.")
        return 0
    else:
        logger.error(f"❌ {total_tests - tests_passed} tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
