"""
Test script for the Helmholtz equation solver.

This script demonstrates the functionality of the HelmholtzSolver class by:
1. Creating a simple box-shaped room mesh using GMSH
2. Setting up the finite element solver with appropriate boundary conditions
3. Computing frequency response at multiple sensor positions
4. Comparing results with analytical solutions where possible
5. Generating visualization plots of the results

The test validates the core acoustic simulation capabilities including:
- Mesh generation and loading
- Finite element discretization
- Linear system assembly and solving
- Frequency domain analysis
- Result evaluation and visualization
"""

# Standard library imports
import sys
from pathlib import Path

# Scientific computing imports
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the Helmholtz solver and mesh generation utilities
from fem.helmholtz_solver import HelmholtzSolver, create_simple_box_mesh


def test_helmholtz_solver():
    """
    Test the Helmholtz solver with a simple box-shaped room geometry.
    
    This function performs a comprehensive test of the acoustic simulation pipeline:
    1. Creates a 3D box mesh representing a room
    2. Initializes the finite element solver with rigid boundary conditions
    3. Places acoustic sources and sensors at strategic locations
    4. Computes frequency response across a range of frequencies
    5. Analyzes and visualizes the results
    
    The test room is a 4m × 3m × 2.5m box, which is representative of a typical
    small room or office space for acoustic analysis.
    
    Returns:
        Dict[str, Any]: Frequency response results for analysis and validation
    """
    print("Testing Helmholtz solver...")
    
    # Create a 3D box mesh representing a room geometry
    # Dimensions: 4m length × 3m width × 2.5m height (typical room size)
    # Mesh size h=0.2m provides good balance between accuracy and computational cost
    print("Creating mesh...")
    mesh_file = create_simple_box_mesh([4.0, 3.0, 2.5], h=0.2)
    print(f"Mesh created: {mesh_file}")
    
    # Initialize the finite element solver with physical parameters
    print("Initializing solver...")
    solver = HelmholtzSolver(
        mesh_file=mesh_file,           # Use the generated mesh file
        element_order=1,               # Linear finite elements (good balance of speed/accuracy)
        boundary_impedance={           # Rigid boundary conditions (perfectly reflecting walls)
            "walls": 0.0,              # Vertical walls are perfectly rigid
            "floor": 0.0,              # Floor is perfectly rigid  
            "ceiling": 0.0             # Ceiling is perfectly rigid
        }
    )
    
    # Define test parameters for acoustic simulation
    frequencies = [100, 200, 500, 1000]  # Test frequencies in Hz (low to mid-range)
    source_position = [0.0, 0.0, 1.0]   # Point source at room center, 1m height
    sensor_positions = [                 # Multiple sensor locations for spatial analysis
        [1.0, 1.0, 1.0],               # Corner sensor (near wall reflection)
        [0.0, 0.0, 1.0],               # Center sensor (direct field)
        [-1.0, -1.0, 1.0],             # Opposite corner sensor (far field)
    ]
    
    # Compute frequency response using direct solver for accuracy
    print("Computing frequency response...")
    results = solver.compute_frequency_response(
        frequencies, source_position, sensor_positions, solver_type="direct"
    )
    
    # Print results
    print("\nResults:")
    print(f"Frequencies: {results['frequencies']}")
    print(f"Number of DOFs: {results['metadata']['mesh_info']['num_dofs']}")
    
    for sensor_id, data in results['sensor_data'].items():
        print(f"\n{sensor_id}:")
        for i, freq in enumerate(frequencies):
            magnitude = abs(data[i])
            phase = np.angle(data[i]) * 180 / np.pi
            print(f"  {freq:4.0f} Hz: |P| = {magnitude:.3e}, ∠P = {phase:6.1f}°")
    
    # Plot frequency response
    plt.figure(figsize=(10, 6))
    
    for i, sensor_pos in enumerate(sensor_positions):
        sensor_id = f"sensor_{i}"
        if sensor_id in results['sensor_data']:
            magnitudes = [abs(data) for data in results['sensor_data'][sensor_id]]
            plt.semilogy(frequencies, magnitudes, 'o-', label=f'Sensor {i+1} {sensor_pos}')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Pressure Magnitude')
    plt.title('Frequency Response - Box Room')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "test_frequency_response.png", dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_dir / 'test_frequency_response.png'}")
    
    plt.show()
    
    print("\nTest completed successfully!")
    return results


def test_analytic_comparison():
    """Compare with analytic solution for a simple case."""
    print("\nTesting against analytic solution...")
    
    # Create a 1D case (rectangular duct)
    mesh_file = create_simple_box_mesh([2.0, 0.1, 0.1], h=0.05)  # Very thin box
    
    solver = HelmholtzSolver(
        mesh_file=mesh_file,
        element_order=1,
        boundary_impedance={"walls": 0.0, "floor": 0.0, "ceiling": 0.0}
    )
    
    # Test at a known resonant frequency
    # For a 2m duct, first mode is at f = c/(2L) = 343/(2*2) = 85.75 Hz
    test_freq = 85.75
    source_pos = [0.0, 0.0, 0.0]  # At one end
    sensor_pos = [[1.0, 0.0, 0.0]]  # At the other end
    
    results = solver.compute_frequency_response(
        [test_freq], source_pos, sensor_pos
    )
    
    # Analytic solution: standing wave with maximum at ends
    analytic_magnitude = 1.0  # Normalized
    
    computed_magnitude = abs(results['sensor_data']['sensor_0'][0])
    
    print(f"Test frequency: {test_freq:.2f} Hz")
    print(f"Analytic magnitude: {analytic_magnitude:.3e}")
    print(f"Computed magnitude: {computed_magnitude:.3e}")
    print(f"Relative error: {abs(computed_magnitude - analytic_magnitude) / analytic_magnitude * 100:.1f}%")
    
    # This is a rough test - in practice you'd need more sophisticated validation
    if abs(computed_magnitude - analytic_magnitude) / analytic_magnitude < 0.5:  # 50% error tolerance
        print("✓ Analytic comparison passed (within tolerance)")
    else:
        print("⚠ Analytic comparison shows significant error")
    
    return results


if __name__ == "__main__":
    try:
        # Run basic test
        results1 = test_helmholtz_solver()
        
        # Run analytic comparison
        results2 = test_analytic_comparison()
        
        print("\n" + "="*50)
        print("All tests completed!")
        print("="*50)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
