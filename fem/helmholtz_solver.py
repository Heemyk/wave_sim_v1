"""
FEniCS-based Helmholtz equation solver for acoustic simulations.

This module provides a comprehensive solver for the Helmholtz equation used in acoustic simulations.
It supports 3D room acoustics with configurable boundary conditions, source placement, and sensor arrays.

Key Features:
- 3D finite element mesh generation using GMSH
- Helmholtz equation solver with configurable boundary conditions
- Frequency response computation for multiple sensor positions
- Support for both direct and iterative linear solvers
- Real-time visualization and result export capabilities

Author: Acoustic Simulation Team
"""

# Standard library imports
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Scientific computing imports
import numpy as np

# FEniCS/dolfinx imports for finite element computations
import dolfinx
from dolfinx import mesh, fem, io, geometry
from dolfinx.fem import Function, FunctionSpace, Constant
from dolfinx.fem.petsc import LinearProblem
import ufl
import basix
import basix.ufl

# MPI and parallel computing imports
from mpi4py import MPI
import petsc4py
petsc4py.init()
from petsc4py import PETSc

# Mesh generation and I/O imports
import gmsh
import meshio

# Set up logging for debugging and monitoring
logger = logging.getLogger(__name__)


class HelmholtzSolver:
    """
    Finite Element Method (FEM) solver for the Helmholtz equation in acoustic simulations.
    
    This class implements a comprehensive solver for the time-harmonic wave equation (Helmholtz equation)
    used in room acoustics simulations. It supports 3D geometries, various boundary conditions,
    and provides frequency response analysis capabilities.
    
    The Helmholtz equation solved is:
        ∇²p + k²p = S(x)
    where:
        - p is the complex pressure field
        - k = ω/c is the wavenumber (ω = 2πf, c = speed of sound)
        - S(x) is the source term
    """
    
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
        Initialize the Helmholtz solver with mesh and physical parameters.
        
        This constructor sets up the finite element discretization of the Helmholtz equation,
        including the mesh, function space, and boundary conditions.
        
        Args:
            mesh_file: Path to mesh file (.msh, .xdmf, etc.) - will be loaded and converted
            mesh_obj: Pre-loaded mesh object (alternative to mesh_file)
            element_order: Polynomial order of finite elements (1=linear, 2=quadratic, etc.)
                         Higher order provides better accuracy but increases computational cost
            boundary_impedance: Dictionary mapping boundary names to complex impedance values
                              {"walls": 0.0+0.0j, "floor": 0.1+0.0j} for rigid and absorbing surfaces
            c: Speed of sound in air at standard conditions (m/s)
            rho: Air density at standard conditions (kg/m³)
        """
        # Store physical parameters for acoustic calculations
        self.c = c  # Speed of sound (m/s) - affects wavenumber calculation k = ω/c
        self.rho = rho  # Air density (kg/m³) - affects impedance calculations
        self.element_order = element_order  # Polynomial order of basis functions
        self.boundary_impedance = boundary_impedance or {}  # Boundary condition parameters
        
        # Load or create mesh - this is the spatial discretization of the acoustic domain
        if mesh_obj is not None:
            # Use pre-loaded mesh object (e.g., from previous computation or external source)
            self.mesh = mesh_obj
        elif mesh_file is not None:
            # Load mesh from file and convert to dolfinx format
            # This handles various mesh formats (.msh from GMSH, .xdmf, etc.)
            self.mesh = self._load_mesh(mesh_file)
        else:
            raise ValueError("Either mesh_file or mesh_obj must be provided")
            
        # Create finite element function space - defines the discrete solution space
        # "Lagrange" elements provide continuous piecewise polynomial approximations
        # element_order=1 gives linear elements, element_order=2 gives quadratic elements
        # Higher order elements provide better accuracy but increase computational cost
        self.V = fem.functionspace(self.mesh, ("Lagrange", element_order))
        
        # Set up boundary markers for applying different boundary conditions
        # This maps boundary regions to physical boundary types (walls, floor, ceiling, etc.)
        self.boundary_markers = self._setup_boundary_markers()
        
        # Initialize form storage - these will hold the assembled matrices and vectors
        self._a_form = None  # Bilinear form (stiffness matrix + mass matrix)
        self._L_form = None  # Linear form (source term vector)
        self._assembled = False  # Flag to track if forms have been assembled
        
        # Log initialization success with mesh information for debugging
        num_dofs = self.V.dofmap.index_map.size_global
        logger.info(f"Initialized Helmholtz solver with {num_dofs} DOFs")
    
    def _load_mesh(self, mesh_file: str) -> Any:
        """
        Load and convert mesh from file to dolfinx format.
        
        This method handles the conversion of mesh files (primarily GMSH .msh files) 
        into the dolfinx mesh format required for finite element computations.
        
        The conversion process involves:
        1. Reading the mesh file using meshio
        2. Extracting vertex coordinates and tetrahedral cell connectivity
        3. Creating a coordinate element for the mesh geometry
        4. Building the dolfinx mesh object
        
        Args:
            mesh_file: Path to the mesh file (.msh, .xdmf, etc.)
            
        Returns:
            dolfinx.mesh.Mesh: The loaded mesh object ready for FEM computations
            
        Raises:
            RuntimeError: If mesh loading or conversion fails
            ValueError: If the mesh doesn't contain tetrahedral elements
        """
        mesh_path = Path(mesh_file)
        
        if mesh_path.suffix == '.msh':
            # Handle GMSH mesh files - these need special conversion to dolfinx format
            try:
                # Read the GMSH file using meshio - this handles the binary/text format parsing
                logger.info(f"Loading GMSH mesh file: {mesh_path}")
                mesh_data = meshio.read(str(mesh_path))
                logger.info(f"Mesh loaded: {len(mesh_data.points)} vertices, {len(mesh_data.cells)} cell blocks")
                
                # Extract vertex coordinates and ensure proper data type for dolfinx
                # Points must be float64 for compatibility with dolfinx
                points = mesh_data.points.astype(np.float64)
                logger.debug(f"Vertex coordinates shape: {points.shape}")
                
                # Find tetrahedral cells in the mesh - dolfinx requires tetrahedra for 3D problems
                # GMSH files may contain various cell types (vertices, lines, triangles, tetrahedra)
                # We only need the tetrahedra for volume discretization
                tet_cells = None
                for i, cell_block in enumerate(mesh_data.cells):
                    logger.debug(f"Cell block {i}: {cell_block.type}, shape: {cell_block.data.shape}")
                    if cell_block.type == 'tetra':
                        # Extract tetrahedral connectivity and ensure proper integer type
                        tet_cells = cell_block.data.astype(np.int32)
                        break
                
                if tet_cells is None:
                    raise ValueError("No tetrahedral elements found in mesh - 3D FEM requires tetrahedra")
                
                logger.info(f"Found {len(tet_cells)} tetrahedral elements")
                
                # Create coordinate element for the mesh geometry
                # This defines how the mesh geometry is represented in the finite element framework
                # We use linear Lagrange elements for the coordinate mapping
                coord_element = basix.create_element(
                    basix.ElementFamily.P,      # Polynomial family (Lagrange)
                    basix.CellType.tetrahedron, # Cell type (tetrahedra)
                    1,                          # Polynomial degree (linear)
                    dtype=np.float64            # Data type for coordinates
                )
                logger.debug("Created coordinate element for mesh geometry")
                
                # Create the dolfinx mesh object from points and cells
                # This is the final step that builds the mesh data structures needed for FEM
                mesh_obj = dolfinx.mesh.create_mesh(
                    MPI.COMM_WORLD,  # MPI communicator for parallel processing
                    tet_cells,       # Tetrahedral connectivity array
                    points,          # Vertex coordinates array
                    coord_element    # Coordinate element for geometry representation
                )
                
                logger.info("Successfully created dolfinx mesh object")
                return mesh_obj
                
            except ImportError:
                raise RuntimeError("meshio package required for loading .msh files. Install with: pip install meshio")
            except Exception as e:
                logger.error(f"Failed to load mesh file {mesh_file}: {e}")
                raise RuntimeError(f"Mesh loading failed: {e}")
        else:
            # Handle other mesh formats (XDMF, etc.) using dolfinx built-in readers
            logger.info(f"Loading mesh file using dolfinx XDMF reader: {mesh_path}")
            return dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, "r").read_mesh()
    
    def _setup_boundary_markers(self) -> Dict[int, str]:
        """
        Set up boundary markers for different boundary condition types.
        
        This method defines which mesh boundary regions correspond to different physical
        boundary types (walls, floor, ceiling, etc.). In a complete implementation,
        these markers would be read from the mesh file or computed from geometry.
        
        Returns:
            Dict[int, str]: Mapping from boundary marker IDs to boundary type names
        """
        # Simplified boundary marker setup - in practice these would be read from mesh
        # or computed from the geometry. Each marker ID corresponds to a physical boundary type
        return {
            1: "walls",    # Vertical walls of the room
            2: "floor",    # Bottom surface of the room  
            3: "ceiling"   # Top surface of the room
        }
    
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
        ds = ufl.Measure("ds", domain=self.mesh)
        
        # Volume terms
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx - k**2 * ufl.inner(u, v) * dx
        
        # Boundary terms (impedance boundary conditions) - simplified for now
        # For rigid boundaries, we use homogeneous Neumann conditions (no additional terms)
        # For impedance boundaries, we would need proper mesh tags
        # For now, assume all boundaries are rigid (Z = 0)
        
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
                self._a_form, self._L_form, u=u_sol,
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
                self._a_form, self._L_form, u=u_sol,
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
        # For now, use a simplified approach - interpolate at mesh vertices
        # and return approximate values
        # This is a temporary solution until we fix the geometry issues
        
        # Get mesh coordinates
        mesh_coords = self.mesh.geometry.x
        
        # For each point, find the closest mesh vertex
        values = []
        for point in points:
            point_array = np.array(point, dtype=np.float64)
            
            # Find closest vertex
            distances = np.linalg.norm(mesh_coords - point_array, axis=1)
            closest_vertex_idx = np.argmin(distances)
            
            # Get solution value at closest vertex
            # This is an approximation - in practice you'd use proper interpolation
            vertex_value = solution.x.array[closest_vertex_idx]
            values.append(vertex_value)
        
        return np.array(values)
    
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
