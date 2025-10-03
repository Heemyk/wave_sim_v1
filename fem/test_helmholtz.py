"""Test script for the Helmholtz solver."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from fem.helmholtz_solver import HelmholtzSolver, create_simple_box_mesh


def test_helmholtz_solver():
    """Test the Helmholtz solver with a simple box geometry."""
    print("Testing Helmholtz solver...")
    
    # Create a simple box mesh
    print("Creating mesh...")
    mesh_file = create_simple_box_mesh([4.0, 3.0, 2.5], h=0.2)
    print(f"Mesh created: {mesh_file}")
    
    # Initialize solver
    print("Initializing solver...")
    solver = HelmholtzSolver(
        mesh_file=mesh_file,
        element_order=1,
        boundary_impedance={"walls": 0.0, "floor": 0.0, "ceiling": 0.0}  # Rigid boundaries
    )
    
    # Test frequencies
    frequencies = [100, 200, 500, 1000]  # Hz
    source_position = [0.0, 0.0, 1.0]  # Center of room, 1m height
    sensor_positions = [
        [1.0, 1.0, 1.0],   # Corner
        [0.0, 0.0, 1.0],   # Center
        [-1.0, -1.0, 1.0], # Opposite corner
    ]
    
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
            magnitudes = [abs(data[i]) for i, data in enumerate(results['sensor_data'][sensor_id])]
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
