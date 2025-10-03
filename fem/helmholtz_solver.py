"""FEniCS-based Helmholtz equation solver for acoustic simulations."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

try:
    # Try dolfinx first (modern FEniCS)
    import dolfinx
    from dolfinx import mesh, fem, io
    from dolfinx.fem import Function, FunctionSpace, Constant
    from dolfinx.fem.petsc import LinearProblem
    import ufl
    # from mpi4py import MPI
    # import petsc4py
    # petsc4py.init()
    # from petsc4py import PETSc
    FENICS_AVAILABLE = True
    FENICS_VERSION = "dolfinx"
except ImportError:
    try:
        # Fall back to classic FEniCS
        import dolfin
        from dolfin import *
        import ufl
        from mpi4py import MPI
        import petsc4py
        petsc4py.init()
        from petsc4py import PETSc
        FENICS_AVAILABLE = True
        FENICS_VERSION = "classic"
    except ImportError:
        FENICS_AVAILABLE = False
        FENICS_VERSION = None
        logging.warning("FEniCS not available. Install with: conda install -c conda-forge fenics")
        
        # Define dummy types for when FEniCS is not available
        class Function:
            def __init__(self, *args, **kwargs):
                pass
        class FunctionSpace:
            def __init__(self, *args, **kwargs):
                pass

logger = logging.getLogger(__name__)


class HelmholtzSolver:
    """Solver for the Helmholtz equation using FEniCS/dolfinx."""
    
    def __init__(
        self,
        mesh_file: Optional[str] = None,
        mesh_obj: Optional[Any] = None,
        element_order: int = 1,
        boundary_impedance: Optional[Dict[str, complex]] = None,
        c: float = 343.0,  # Speed of sound (m/s)
        rho: float = 1.225,  # Air density (kg/m³)
    ):
        """
        Initialize the Helmholtz solver.
        
        Args:
            mesh_file: Path to mesh file (.msh, .xml, etc.)
            mesh_obj: Pre-loaded mesh object
            element_order: Order of Lagrange elements (1, 2, or 3)
            boundary_impedance: Dict mapping boundary markers to impedance values
            c: Speed of sound
            rho: Air density
        """
        # if not FENICS_AVAILABLE:
        #     raise ImportError("FEniCS/dolfinx is required but not available")
            
        self.c = c
        self.rho = rho
        self.element_order = element_order
        self.boundary_impedance = boundary_impedance or {}
        
        # Load or create mesh
        if mesh_obj is not None:
            self.mesh = mesh_obj
        elif mesh_file is not None:
            self.mesh = self._load_mesh(mesh_file)
        else:
            raise ValueError("Either mesh_file or mesh_obj must be provided")
            
        # Create function space
        self.V = FunctionSpace(self.mesh, ("Lagrange", element_order))
        
        # Boundary markers
        self.boundary_markers = self._setup_boundary_markers()
        
        # Pre-assembled forms (will be assembled when needed)
        self._a_form = None
        self._L_form = None
        self._assembled = False
        
        logger.info(f"Initialized Helmholtz solver with {self.V.dofmap.index_map.size_global} DOFs")
    
    def _load_mesh(self, mesh_file: str) -> Any:
        """Load mesh from file."""
        mesh_path = Path(mesh_file)
        
        if mesh_path.suffix == '.msh':
            # Use gmsh to load .msh files
            import gmsh
            gmsh.initialize()
            gmsh.open(str(mesh_path))
            mesh_obj, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
                gmsh.model, MPI.COMM_WORLD, 0, gdim=3
            )
            gmsh.finalize()
            return mesh_obj
        else:
            # Use dolfinx.io for other formats
            return dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, "r").read_mesh()
    
    def _setup_boundary_markers(self) -> Dict[int, str]:
        """Setup boundary markers for different boundary conditions."""
        # This is a simplified version - in practice you'd read from mesh
        # For now, assume all boundaries are walls
        return {1: "walls", 2: "floor", 3: "ceiling"}
    
    def assemble_system(self, k: float, source_position: List[float], source_amplitude: float = 1.0):
        """
        Assemble the linear system for the Helmholtz equation.
        
        Args:
            k: Wavenumber (omega/c)
            source_position: [x, y, z] position of point source
            source_amplitude: Amplitude of the source
        """
        # Define trial and test functions
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        
        # Define the bilinear form (left-hand side)
        # a(u,v) = ∫ ∇u · ∇v̄ dx - k² ∫ u v̄ dx + ∫ Z u v̄ ds
        dx = ufl.Measure("dx", domain=self.mesh)
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.boundary_markers)
        
        # Volume terms
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx - k**2 * ufl.inner(u, v) * dx
        
        # Boundary terms (impedance boundary conditions)
        for marker, boundary_type in self.boundary_markers.items():
            if boundary_type in self.boundary_impedance:
                Z = self.boundary_impedance[boundary_type]
                a += Z * ufl.inner(u, v) * ds(marker)
        
        # Define the linear form (right-hand side)
        # L(v) = ∫ S v̄ dx where S is the source term
        
        # Point source approximation (regularized delta function)
        source_radius = 0.1  # Small radius for point source approximation
        source_x, source_y, source_z = source_position
        
        # Create a localized source function
        source_expr = source_amplitude * ufl.exp(
            -((ufl.SpatialCoordinate(self.mesh)[0] - source_x)**2 +
              (ufl.SpatialCoordinate(self.mesh)[1] - source_y)**2 +
              (ufl.SpatialCoordinate(self.mesh)[2] - source_z)**2) / (2 * source_radius**2)
        )
        
        L = source_expr * v * dx
        
        # Store forms for later use
        self._a_form = a
        self._L_form = L
        self._assembled = True
        
        logger.debug(f"Assembled system for k={k:.2f}")
    
    def solve(self, solver_type: str = "direct", solver_params: Optional[Dict] = None) -> Function:
        """
        Solve the assembled system.
        
        Args:
            solver_type: "direct" or "iterative"
            solver_params: Additional solver parameters
            
        Returns:
            Solution function
        """
        if not self._assembled:
            raise RuntimeError("System must be assembled before solving")
        
        # Create solution function
        u_sol = Function(self.V)
        
        if solver_type == "direct":
            # Use direct solver (LU factorization)
            problem = LinearProblem(
                self._a_form, self._L_form, u=[u_sol],
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
            )
            problem.solve()
            
        elif solver_type == "iterative":
            # Use iterative solver with preconditioning
            petsc_options = {
                "ksp_type": "gmres",
                "pc_type": "hypre",
                "ksp_rtol": 1e-6,
                "ksp_max_it": 1000,
            }
            if solver_params:
                petsc_options.update(solver_params)
                
            problem = LinearProblem(
                self._a_form, self._L_form, u=[u_sol],
                petsc_options=petsc_options
            )
            problem.solve()
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
        
        logger.debug(f"Solved system using {solver_type} solver")
        return u_sol
    
    def evaluate_at_points(self, solution: Function, points: List[List[float]]) -> np.ndarray:
        """
        Evaluate solution at specific points.
        
        Args:
            solution: Solution function
            points: List of [x, y, z] coordinates
            
        Returns:
            Array of complex pressure values
        """
        points_array = np.array(points, dtype=np.float64)
        values = solution.eval(points_array, self.mesh)
        return values
    
    def compute_frequency_response(
        self,
        frequencies: List[float],
        source_position: List[float],
        sensor_positions: List[List[float]],
        solver_type: str = "direct"
    ) -> Dict[str, Any]:
        """
        Compute frequency response at multiple frequencies.
        
        Args:
            frequencies: List of frequencies (Hz)
            source_position: [x, y, z] position of source
            sensor_positions: List of [x, y, z] sensor positions
            solver_type: Solver type to use
            
        Returns:
            Dictionary with frequency response data
        """
        results = {
            "frequencies": frequencies,
            "sensor_data": {},
            "metadata": {
                "source_position": source_position,
                "sensor_positions": sensor_positions,
                "solver_type": solver_type,
                "mesh_info": {
                    "num_dofs": self.V.dofmap.index_map.size_global,
                    "element_order": self.element_order,
                }
            }
        }
        
        for i, freq in enumerate(frequencies):
            logger.info(f"Computing frequency {freq:.1f} Hz ({i+1}/{len(frequencies)})")
            
            # Compute wavenumber
            k = 2 * np.pi * freq / self.c
            
            # Assemble and solve
            self.assemble_system(k, source_position)
            solution = self.solve(solver_type)
            
            # Evaluate at sensor positions
            sensor_values = self.evaluate_at_points(solution, sensor_positions)
            
            # Store results
            for j, sensor_pos in enumerate(sensor_positions):
                sensor_id = f"sensor_{j}"
                if sensor_id not in results["sensor_data"]:
                    results["sensor_data"][sensor_id] = []
                results["sensor_data"][sensor_id].append(complex(sensor_values[j]))
        
        return results
    
    def save_solution(self, solution: Function, filename: str):
        """Save solution to file for visualization."""
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(solution)


def create_simple_box_mesh(
    dimensions: List[float],
    center: List[float] = [0.0, 0.0, 0.0],
    h: float = 0.1,
    filename: Optional[str] = None
) -> str:
    """
    Create a simple box mesh using gmsh.
    
    Args:
        dimensions: [length, width, height]
        center: [x, y, z] center position
        h: Target mesh size
        filename: Output filename (optional)
        
    Returns:
        Path to created mesh file
    """
    import gmsh
    
    gmsh.initialize()
    gmsh.model.add("box")
    
    # Box dimensions
    L, W, H = dimensions
    cx, cy, cz = center
    
    # Create box
    box = gmsh.model.occ.addBox(
        cx - L/2, cy - W/2, cz - H/2,
        L, W, H
    )
    
    # Synchronize
    gmsh.model.occ.synchronize()
    
    # Set mesh size
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    
    # Generate mesh
    gmsh.model.mesh.generate(3)
    
    # Save mesh
    if filename is None:
        filename = f"box_{L}x{W}x{H}_h{h:.3f}.msh"
    
    gmsh.write(filename)
    gmsh.finalize()
    
    return filename


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test case
    print("Creating test mesh...")
    mesh_file = create_simple_box_mesh([4.0, 3.0, 2.5], h=0.2)
    
    print("Initializing solver...")
    solver = HelmholtzSolver(mesh_file=mesh_file, element_order=1)
    
    print("Computing frequency response...")
    frequencies = [100, 200, 500, 1000]  # Hz
    source_pos = [0.0, 0.0, 1.0]  # Center of room, 1m height
    sensor_positions = [
        [1.0, 1.0, 1.0],  # Corner
        [0.0, 0.0, 1.0],  # Center
        [-1.0, -1.0, 1.0],  # Opposite corner
    ]
    
    results = solver.compute_frequency_response(
        frequencies, source_pos, sensor_positions
    )
    
    print("Results:")
    for sensor_id, data in results["sensor_data"].items():
        print(f"{sensor_id}: {len(data)} frequency points")
    
    print("Test completed successfully!")
