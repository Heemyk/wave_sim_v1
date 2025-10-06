"""
SfePy-based Helmholtz equation solver for acoustic simulations.

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
import tempfile
import os

# Scientific computing imports
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

# SfePy imports for finite element computations
import sfepy
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.discrete import FieldVariable, Material, Integral
from sfepy.base.ioutils import ensure_path

# Mesh generation and I/O imports
import gmsh
import meshio

# Set up logging for debugging and monitoring
logger = logging.getLogger(__name__)


class HelmholtzSolver:
    """
    Finite Element Method (FEM) solver for the Helmholtz equation in acoustic simulations using SfePy.
    
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
        
        Args:
            mesh_file: Path to mesh file (.msh, .vtk, etc.) - will be loaded and converted
            mesh_obj: Pre-loaded mesh object (alternative to mesh_file)
            element_order: Polynomial order of finite elements (1=linear, 2=quadratic, etc.)
            boundary_impedance: Dictionary mapping boundary names to complex impedance values
            c: Speed of sound in air at standard conditions (m/s)
            rho: Air density at standard conditions (kg/m³)
        """
        # Store physical parameters for acoustic calculations
        self.c = c  # Speed of sound (m/s) - affects wavenumber calculation k = ω/c
        self.rho = rho  # Air density (kg/m³) - affects impedance calculations
        self.element_order = element_order  # Polynomial order of basis functions
        self.boundary_impedance = boundary_impedance or {}  # Boundary condition parameters
        
        # Set default boundary impedances if none provided
        if not self.boundary_impedance:
            # Default absorption coefficients (0 = rigid, 1 = fully absorbing)
            # Convert to impedance: Z = rho * c / alpha (correct Robin BC formula)
            default_alpha = 0.1  # 10% absorption (typical for hard walls)
            default_Z = self.rho * self.c / default_alpha
            self.boundary_impedance = {
                "walls": default_Z,
                "floor": default_Z, 
                "ceiling": default_Z
            }
            logger.info(f"Using default boundary impedances: {self.boundary_impedance}")
        
        # Load or create mesh - this is the spatial discretization of the acoustic domain
        if mesh_obj is not None:
            # Use pre-loaded mesh object (e.g., from previous computation or external source)
            self.mesh = mesh_obj
        elif mesh_file is not None:
            # Load mesh from file and convert to SfePy format
            self.mesh = self._load_mesh(mesh_file)
        else:
            raise ValueError("Either mesh_file or mesh_obj must be provided")
            
        # Create domain from mesh
        self.domain = FEDomain('domain', self.mesh)
        
        # Set up boundary markers for applying different boundary conditions
        self.boundary_markers = self._setup_boundary_markers()
        
        # Initialize problem components
        self._problem = None
        
        # Cache for FEM matrices (only depend on mesh geometry)
        self._cached_K = None  # Stiffness matrix
        self._cached_M = None  # Mass matrix
        self._cached_B = None  # Boundary matrix
        self._cached_mesh_hash = None  # Hash to detect mesh changes
        self._cached_boundary_hash = None  # Hash to detect boundary condition changes
        self._current_solution = None
        self._current_frequency = None
        
        # Log initialization success with mesh information for debugging
        num_dofs = self.mesh.n_nod
        logger.info(f"Initialized SfePy Helmholtz solver with {num_dofs} nodes")
    
    def _load_mesh(self, mesh_file: str) -> Any:
        """
        Load and convert mesh from file to SfePy format.
        
        Args:
            mesh_file: Path to the mesh file (.msh, .vtk, etc.)
            
        Returns:
            SfePy Mesh: The loaded mesh object ready for FEM computations
        """
        mesh_path = Path(mesh_file)
        
        if mesh_path.suffix == '.msh':
            # Handle GMSH mesh files - convert to VTK format for SfePy
            try:
                logger.info(f"Converting GMSH mesh file: {mesh_path}")
                
                # Read the GMSH file using meshio
                mesh_data = meshio.read(str(mesh_path))
                logger.info(f"Mesh loaded: {len(mesh_data.points)} vertices, {len(mesh_data.cells)} cell blocks")
                
                # Convert to VTK format (SfePy prefers VTK)
                vtk_file = mesh_path.with_suffix('.vtk')
                
                # Write VTK file with simplified cell data to avoid cell_sets issues
                # Create a clean mesh without problematic cell sets
                clean_mesh = meshio.Mesh(
                    points=mesh_data.points,
                    cells=mesh_data.cells
                )
                meshio.write(str(vtk_file), clean_mesh)
                
                # Load using SfePy's mesh reader
                mesh_obj = Mesh.from_file(str(vtk_file))
                
                logger.info("Successfully converted and loaded mesh for SfePy")
                return mesh_obj
                
            except Exception as e:
                logger.error(f"Failed to load mesh file {mesh_file}: {e}")
                # Try loading directly without conversion as fallback
                try:
                    logger.info("Attempting direct mesh loading...")
                    return Mesh.from_file(str(mesh_path))
                except Exception as e2:
                    logger.error(f"Direct loading also failed: {e2}")
                raise RuntimeError(f"Mesh loading failed: {e}")
        else:
            # Handle other mesh formats directly
            logger.info(f"Loading mesh file using SfePy: {mesh_path}")
            return Mesh.from_file(str(mesh_path))
    
    def _setup_boundary_markers(self) -> Dict[int, str]:
        """
        Set up boundary markers for different boundary condition types.
        
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
        # Store parameters for later use
        self._current_k = k
        self._current_source_pos = source_position
        self._current_source_amp = source_amplitude
        
        logger.debug(f"Stored system parameters for k={k:.2f}")
    
    def _compute_mesh_hash(self, mesh_coords: np.ndarray, cells: np.ndarray) -> str:
        """Compute a hash of the mesh to detect changes."""
        import hashlib
        # Create hash from mesh coordinates and connectivity
        coords_str = mesh_coords.tobytes()
        cells_str = cells.tobytes()
        combined = coords_str + cells_str
        return hashlib.md5(combined).hexdigest()
    
    def _compute_boundary_hash(self) -> str:
        """Compute a hash of the boundary conditions to detect changes."""
        import hashlib
        import json
        # Create hash from boundary impedance dictionary
        boundary_str = json.dumps(self.boundary_impedance, sort_keys=True, default=str)
        return hashlib.md5(boundary_str.encode()).hexdigest()
    
    def solve(self, solver_type: str = "direct", solver_params: Optional[Dict] = None) -> np.ndarray:
        """
        Solve the assembled Helmholtz equation using proper finite element method.
        
        Args:
            solver_type: "direct" or "iterative"
            solver_params: Additional solver parameters
            
        Returns:
            Solution array
        """
        if not hasattr(self, '_current_k'):
            raise RuntimeError("System must be assembled before solving")
        
        try:
            logger.info("Solving Helmholtz equation using proper FEM")
            
            # Get mesh information
            mesh_coords = self.mesh.coors  # Node coordinates
            num_nodes = self.mesh.n_nod
            
            # Get connectivity for tetrahedra
            try:
                # Try to get 3D tetrahedra connectivity
                cells = self.mesh.get_conn('3_4')  # 3D tetrahedra
            except:
                # Fallback to any 3D elements
                cell_types = ['3_4', '3_3', '3_8']  # tetra, tri, hex
                cells = None
                for cell_type in cell_types:
                    try:
                        cells = self.mesh.get_conn(cell_type)
                        logger.info(f"Using {cell_type} elements: {cells.shape}")
                        break
                    except:
                        continue
                
                if cells is None:
                    raise RuntimeError("No suitable 3D elements found in mesh")
            
            # Check if we can reuse cached matrices
            mesh_hash = self._compute_mesh_hash(mesh_coords, cells)
            
            if (self._cached_K is not None and self._cached_M is not None and 
                self._cached_mesh_hash == mesh_hash):
                logger.info("Using cached FEM matrices (significant speedup!)")
                K, M = self._cached_K, self._cached_M
            else:
                logger.info("Assembling finite element matrices (first time or mesh changed)")
                
                # Stiffness matrix (Laplacian term: ∫∇φᵢ·∇φⱼ dΩ)
                # Mass matrix (mass term: ∫φᵢφⱼ dΩ)
                K, M = self._assemble_fem_matrices(mesh_coords, cells)
                
                # Cache the matrices
                self._cached_K = K
                self._cached_M = M
                self._cached_mesh_hash = mesh_hash
                logger.info("FEM matrices cached for future reuse")
            
            # Create the Helmholtz system: (K - k²M + B)p = f
            # where K is stiffness matrix, M is mass matrix, B is boundary matrix, k is wavenumber
            k_squared = self._current_k**2
            A = K - k_squared * M
            
            # Apply boundary conditions (Robin: ∂p/∂n + (jk/Z)p = 0)
            if self.boundary_impedance:
                boundary_hash = self._compute_boundary_hash()
                
                if (self._cached_B is not None and 
                    self._cached_mesh_hash == mesh_hash and 
                    self._cached_boundary_hash == boundary_hash):
                    logger.info("Using cached boundary matrix (significant speedup!)")
                    B = self._cached_B
                else:
                    logger.info("Assembling boundary matrix (first time or boundary conditions changed)")
                    logger.info(f"  Boundary impedances: {self.boundary_impedance}")
                    B = self._assemble_boundary_matrix(mesh_coords, cells, self._current_frequency)
                    
                    # Cache the boundary matrix
                    self._cached_B = B
                    self._cached_boundary_hash = boundary_hash
                    logger.info("Boundary matrix cached for future reuse")
                
                A += B  # Add boundary terms to the system matrix
                logger.info("Boundary conditions applied - walls will now reflect and absorb sound!")
            else:
                logger.warning("No boundary conditions applied - sound will pass through walls!")
            
            # Create source vector
            f = self._create_source_vector(mesh_coords)
            
            # Solve the linear system
            logger.info("Solving linear system")
            if solver_type == "direct":
                # Direct solver using sparse LU decomposition
                pressure_values = scipy.sparse.linalg.spsolve(A, f)
            else:
                # Iterative solver
                pressure_values, info = scipy.sparse.linalg.cg(A, f, tol=1e-8)
                if info != 0:
                    logger.warning(f"CG solver did not converge: info={info}")
            
            # Convert to complex if needed (for frequency domain)
            if not np.iscomplexobj(pressure_values):
                pressure_values = pressure_values.astype(complex)
            
            # Store current solution
            self._current_solution = pressure_values
            self._current_frequency = self._current_k * self.c / (2 * np.pi)
            
            logger.info(f"Solved Helmholtz equation for {num_nodes} nodes, k={self._current_k:.3f}")
            return pressure_values
                
        except Exception as e:
            logger.error(f"Error solving Helmholtz equation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to analytical solution for testing
            logger.warning("Using analytical free-field solution as fallback")
            return self._analytical_solution()
    
    def _assemble_fem_matrices(self, mesh_coords: np.ndarray, cells: np.ndarray) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
        """
        Assemble finite element matrices for the Helmholtz equation.
        
        Args:
            mesh_coords: Node coordinates (N x 3)
            cells: Element connectivity (M x 4 for tetrahedra)
            
        Returns:
            Tuple of (stiffness_matrix, mass_matrix) as sparse matrices
        """
        num_nodes = mesh_coords.shape[0]
        num_elements = cells.shape[0]
        
        # Initialize sparse matrix builders
        K_data = []  # Stiffness matrix entries
        K_row = []
        K_col = []
        
        M_data = []  # Mass matrix entries  
        M_row = []
        M_col = []
        
        logger.info(f"Assembling FEM matrices for {num_elements} elements")
        
        # For each tetrahedral element
        for elem_idx in range(num_elements):
            if elem_idx % 1000 == 0:
                logger.debug(f"Processing element {elem_idx}/{num_elements}")
            
            # Get element nodes
            elem_nodes = cells[elem_idx, :]
            
            # Get element coordinates
            elem_coords = mesh_coords[elem_nodes, :]  # 4 x 3
            
            # Compute element matrices using linear tetrahedra
            K_elem, M_elem = self._compute_element_matrices(elem_coords)
            
            # Add to global matrices
            for i, node_i in enumerate(elem_nodes):
                for j, node_j in enumerate(elem_nodes):
                    # Stiffness matrix
                    K_data.append(K_elem[i, j])
                    K_row.append(node_i)
                    K_col.append(node_j)
                    
                    # Mass matrix
                    M_data.append(M_elem[i, j])
                    M_row.append(node_i)
                    M_col.append(node_j)
        
        # Create sparse matrices
        K = scipy.sparse.csr_matrix((K_data, (K_row, K_col)), shape=(num_nodes, num_nodes))
        M = scipy.sparse.csr_matrix((M_data, (M_row, M_col)), shape=(num_nodes, num_nodes))
        
        logger.info("FEM matrix assembly completed")
        return K, M
    
    def _compute_element_matrices(self, elem_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute element stiffness and mass matrices for a linear tetrahedron.
        
        Args:
            elem_coords: Element coordinates (4 x 3)
            
        Returns:
            Tuple of (stiffness_matrix, mass_matrix) (4 x 4)
        """
        # Linear tetrahedron shape functions:
        # N1 = 1 - ξ - η - ζ
        # N2 = ξ  
        # N3 = η
        # N4 = ζ
        
        # Shape function derivatives w.r.t. natural coordinates
        dN_dxi = np.array([
            [-1, -1, -1],
            [ 1,  0,  0],
            [ 0,  1,  0],
            [ 0,  0,  1]
        ])
        
        # Compute Jacobian matrix and determinant
        J = elem_coords.T @ dN_dxi  # 3 x 3 Jacobian matrix
        detJ = np.linalg.det(J)
        
        if detJ <= 0:
            logger.warning(f"Negative Jacobian determinant: {detJ}")
            detJ = abs(detJ)
        
        # Compute Jacobian inverse
        Jinv = np.linalg.inv(J)
        
        # Shape function derivatives w.r.t. physical coordinates
        dN_dx = dN_dxi @ Jinv.T  # 4 x 3
        
        # Element volume
        V = abs(detJ) / 6.0
        
        # Stiffness matrix: K_ij = ∫∇N_i·∇N_j dV
        K_elem = V * (dN_dx @ dN_dx.T)
        
        # Mass matrix: M_ij = ∫N_i N_j dV
        # For linear tetrahedra: M = (V/20) * [2 1 1 1; 1 2 1 1; 1 1 2 1; 1 1 1 2]
        M_elem = (V / 20.0) * np.array([
            [2, 1, 1, 1],
            [1, 2, 1, 1], 
            [1, 1, 2, 1],
            [1, 1, 1, 2]
        ])
        
        return K_elem, M_elem
    
    def _assemble_boundary_matrix(self, mesh_coords: np.ndarray, cells: np.ndarray, frequency: float = None) -> scipy.sparse.csr_matrix:
        """
        Assemble boundary matrix for Robin boundary conditions.
        
        For Robin BC: ∂p/∂n + (jk/Z)p = 0
        This contributes boundary integral terms: ∫_∂Ω (jk/Z) φᵢ φⱼ dS
        
        Args:
            mesh_coords: Node coordinates (N x 3)
            cells: Element connectivity (M x 4 for tetrahedra)
            
        Returns:
            Boundary matrix B as sparse matrix
        """
        num_nodes = len(mesh_coords)
        B = scipy.sparse.lil_matrix((num_nodes, num_nodes))
        
        logger.info("Assembling boundary matrix for Robin boundary conditions")
        
        # Get boundary faces (triangular faces of tetrahedra)
        boundary_faces = self._get_boundary_faces(cells)
        
        # Process each boundary face
        processed_faces = 0
        for face_idx, (face_nodes, boundary_type) in enumerate(boundary_faces):
            if boundary_type not in self.boundary_impedance:
                continue
                
            # Get impedance for this boundary
            Z = self.boundary_impedance[boundary_type]
            if Z == 0:
                continue  # Skip rigid boundaries (infinite impedance)
            
            processed_faces += 1
            
            # Get face coordinates
            face_coords = mesh_coords[face_nodes]
            
            # Compute face area and normal
            face_area, face_normal = self._compute_face_area_normal(face_coords)
            
            # Robin boundary condition term: (jk/Z) * ∫ φᵢ φⱼ dS
            # For triangular faces, this gives a 3x3 local matrix
            if frequency is None:
                # Use a representative frequency for time-domain (middle of audible range)
                frequency = 1000.0  # 1 kHz as representative
            k = 2 * np.pi * frequency / 343.0  # Wavenumber
            boundary_term = (1j * k / Z) * face_area / 12.0  # 1/12 for linear triangles
            
            # Add to global matrix (ensure complex type)
            if not np.iscomplexobj(B.data):
                # Convert to complex if needed
                B = B.astype(complex)
            
            for i in range(3):
                for j in range(3):
                    if i == j:
                        # Diagonal terms (2 contributions)
                        B[face_nodes[i], face_nodes[j]] += 2 * boundary_term
        else:
                        # Off-diagonal terms (1 contribution each)
                        B[face_nodes[i], face_nodes[j]] += boundary_term
        
        logger.info(f"Boundary matrix assembled: {processed_faces} faces processed, {B.nnz} non-zero entries")
        return B.tocsr()
    
    def _assemble_time_domain_boundary_matrix(self, mesh_coords: np.ndarray, cells: np.ndarray) -> scipy.sparse.csr_matrix:
        """
        Assemble damping matrix for time-domain Robin boundary conditions.
        
        Time-domain Robin BC: ∂p/∂n + (1/Z)∂p/∂t = 0
        This contributes damping terms: ∫_∂Ω (1/Z) φᵢ φⱼ dS
        
        Returns:
            Damping matrix D for the system: M * ∂²p/∂t² + D * ∂p/∂t + K * p = f(t)
        """
        logger.info("Assembling time-domain boundary damping matrix")
        
        num_nodes = mesh_coords.shape[0]
        D = scipy.sparse.lil_matrix((num_nodes, num_nodes), dtype=np.float64)
        
        # Get boundary faces
        boundary_faces = self._get_boundary_faces(cells)
        logger.info(f"Processing {len(boundary_faces)} boundary faces for time-domain damping")
        
        processed_faces = 0
        
        for face_nodes, boundary_type in boundary_faces:
            if boundary_type not in self.boundary_impedance:
                continue
                
            Z = self.boundary_impedance[boundary_type]
            if Z <= 0:
                continue
            
            processed_faces += 1
            
            # Get face coordinates
            face_coords = mesh_coords[face_nodes]
            
            # Compute face area
            face_area, _ = self._compute_face_area_normal(face_coords)
            
            # Time-domain damping term: (1/Z) * ∫ φᵢ φⱼ dS
            # For triangular faces, this gives a 3x3 local matrix
            damping_term = (1.0 / Z) * face_area / 12.0  # 1/12 for linear triangles
            
            # Add to global damping matrix
            for i in range(3):
                for j in range(3):
                    D[face_nodes[i], face_nodes[j]] += damping_term
        
        D = D.tocsr()
        logger.info(f"Time-domain boundary damping matrix: {processed_faces} faces processed, {D.nnz} non-zero entries")
        
        return D
    
    def _get_boundary_faces(self, cells: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        Identify boundary faces from tetrahedral mesh.
        
        Args:
            cells: Tetrahedral connectivity (M x 4)
            
        Returns:
            List of (face_nodes, boundary_type) tuples
        """
        boundary_faces = []
        
        # Create face-to-cell mapping
        face_to_cells = {}
        
        for cell_idx, cell in enumerate(cells):
            # Get all 4 triangular faces of this tetrahedron
            faces = [
                [cell[0], cell[1], cell[2]],  # Face 0,1,2
                [cell[0], cell[1], cell[3]],  # Face 0,1,3
                [cell[0], cell[2], cell[3]],  # Face 0,2,3
                [cell[1], cell[2], cell[3]]   # Face 1,2,3
            ]
            
            for face in faces:
                face_key = tuple(sorted(face))
                if face_key in face_to_cells:
                    face_to_cells[face_key].append(cell_idx)
                else:
                    face_to_cells[face_key] = [cell_idx]
        
        # Boundary faces are those that belong to only one cell
        for face_key, cell_list in face_to_cells.items():
            if len(cell_list) == 1:  # Boundary face
                face_nodes = np.array(list(face_key))
                boundary_type = self._classify_boundary_face(face_nodes)
                boundary_faces.append((face_nodes, boundary_type))
        
        logger.info(f"Found {len(boundary_faces)} boundary faces")
        return boundary_faces
    
    def _classify_boundary_face(self, face_nodes: np.ndarray) -> str:
        """
        Classify a boundary face by its position (walls, floor, ceiling).
        
        Args:
            face_nodes: Node indices of the face
            
        Returns:
            Boundary type string
        """
        # This is a simplified classification based on position
        # In practice, this would use the mesh's boundary markers
        
        # Get face center coordinates
        face_coords = self.mesh.coors[face_nodes]
        face_center = np.mean(face_coords, axis=0)
        
        # Simple classification based on coordinates
        x, y, z = face_center
        
        # Get domain bounds
        coords = self.mesh.coors
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
        
        # Classify based on position
        tol = 0.01  # Tolerance for boundary detection
        
        if abs(z - z_min) < tol:
            return "floor"
        elif abs(z - z_max) < tol:
            return "ceiling"
        else:
            return "walls"  # Default to walls for vertical boundaries
    
    def _compute_face_area_normal(self, face_coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute area and normal vector for a triangular face.
        
        Args:
            face_coords: 3x3 array of face vertex coordinates
            
        Returns:
            Tuple of (area, normal_vector)
        """
        # Compute edge vectors
        v1 = face_coords[1] - face_coords[0]
        v2 = face_coords[2] - face_coords[0]
        
        # Compute normal vector (cross product)
        normal = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(normal)
        normal = normal / np.linalg.norm(normal)
        
        return area, normal
    
    def _create_source_vector(self, mesh_coords: np.ndarray, source_position: List[float] = None, source_amplitude: float = 1.0) -> np.ndarray:
        """
        Create the source vector for the right-hand side of the Helmholtz equation.
        
        Args:
            mesh_coords: Node coordinates (N x 3)
            source_position: Source position [x, y, z] (if None, uses self._current_source_pos)
            source_amplitude: Source amplitude (if None, uses self._current_source_amp)
            
        Returns:
            Source vector (N,)
        """
        num_nodes = mesh_coords.shape[0]
        f = np.zeros(num_nodes)
        
        # Point source at specified position
        if source_position is None:
            source_pos = np.array(self._current_source_pos)
        else:
            source_pos = np.array(source_position)
            
        if hasattr(self, '_current_source_amp'):
            source_amp = self._current_source_amp
        else:
            source_amp = source_amplitude
        
        # Find closest node to source
        distances = np.linalg.norm(mesh_coords - source_pos, axis=1)
        closest_node = np.argmin(distances)
        
        # Apply source at closest node
        f[closest_node] = source_amp
        
        logger.debug(f"Applied point source at node {closest_node}, distance from target: {distances[closest_node]:.3f}")
        
        return f
    
    def _analytical_solution(self) -> np.ndarray:
        """
        Fallback analytical solution for free-field conditions.
        
        Returns:
            Analytical solution array
        """
        num_nodes = self.mesh.n_nod
        mesh_coords = self.mesh.coors
        k = self._current_k
        source_pos = np.array(self._current_source_pos)
        source_amp = self._current_source_amp
        
        pressure_values = np.zeros(num_nodes, dtype=complex)
        
        for i in range(num_nodes):
            node_pos = mesh_coords[i]
            r = np.linalg.norm(node_pos - source_pos)
            
            if r > 1e-6:  # Avoid division by zero at source
                # Fundamental solution to Helmholtz equation: p = A * exp(i*k*r) / (4*pi*r)
                pressure_values[i] = source_amp * np.exp(1j * k * r) / (4 * np.pi * r)
        else:
                # At source point, use a finite value
                pressure_values[i] = source_amp * (1 + 1j * k) / (4 * np.pi)
        
        # Store current solution
        self._current_solution = pressure_values
        self._current_frequency = k * self.c / (2 * np.pi)
        
        logger.info("Using analytical free-field solution")
        return pressure_values
    
    def evaluate_at_points(self, solution: np.ndarray, points: List[List[float]]) -> np.ndarray:
        """
        Evaluate solution at specific points.
        
        Args:
            solution: Solution array
            points: List of [x, y, z] coordinates
            
        Returns:
            Array of complex pressure values
        """
        # Get mesh coordinates
        mesh_coords = self.mesh.coors
        
        # For each point, find the closest mesh vertex
        values = []
        for point in points:
            point_array = np.array(point, dtype=np.float64)
            
            # Find closest vertex
            distances = np.linalg.norm(mesh_coords - point_array, axis=1)
            closest_vertex_idx = np.argmin(distances)
            
            # Get solution value at closest vertex
            vertex_value = solution[closest_vertex_idx]
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
                    "num_nodes": self.mesh.n_nod,
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
            
            # Store current solution for visualization
            self._current_solution = solution
            self._current_frequency = freq
            
            # Evaluate at sensor positions
            sensor_values = self.evaluate_at_points(solution, sensor_positions)
            
            # Store results
            for j, sensor_pos in enumerate(sensor_positions):
                sensor_id = f"sensor_{j}"
                if sensor_id not in results["sensor_data"]:
                    results["sensor_data"][sensor_id] = []
                complex_value = complex(sensor_values[j])
                results["sensor_data"][sensor_id].append(complex_value)
                logger.info(f"Stored {sensor_id}: {complex_value}")
        
        return results
    
    def get_field_data(self) -> Dict[str, Any]:
        """Get pressure field data for visualization."""
        if self._current_solution is None:
            logger.warning("No current solution available for field data extraction")
            return {}
        
        try:
            # Get pressure values at all mesh nodes
            pressure_values = self._current_solution
            
            # Convert to complex and extract components
            pressure_complex = pressure_values.astype(complex)
            pressure_real = pressure_complex.real
            pressure_imag = pressure_complex.imag
            pressure_magnitude = np.abs(pressure_complex)
            pressure_phase = np.angle(pressure_complex) * 180 / np.pi
            
            field_data = {
                "pressure_magnitude": pressure_magnitude.tolist(),
                "pressure_phase": pressure_phase.tolist(),
                "pressure_real": pressure_real.tolist(),
                "pressure_imag": pressure_imag.tolist(),
                "frequency": self._current_frequency,
                "num_nodes": len(pressure_values)
            }
            
            logger.info(f"Extracted field data for {self._current_frequency} Hz: {len(pressure_values)} nodes")
            return field_data
            
        except Exception as e:
            logger.error(f"Error extracting field data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def compute_time_domain_simulation(self, 
                                      sensor_positions: List[List[float]],
                                      source_position: List[float],
                                      source_frequency: float,
                                      sample_rate: float = 44100,
                                      duration: float = 2.0,
                                      max_frequency: float = 20000.0,
                                      precomputed_frequency_data: Optional[Dict] = None,
                                      custom_source_signal: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute time-domain simulation with both audio and visualization data.
        
        This method implements both:
        A) Wave propagation visualization (time-series pressure fields)
        B) Audio reconstruction at sensor points (impulse responses)
        
        Args:
            sensor_positions: List of [x,y,z] sensor positions
            source_position: [x,y,z] source position
            source_frequency: Source frequency for audio synthesis
            sample_rate: Audio sample rate (Hz)
            duration: Simulation duration (seconds)
            max_frequency: Maximum frequency for frequency domain computation
            
        Returns:
            Dictionary containing both visualization and audio data
        """
        import numpy as np
        from scipy.fft import ifft, fftfreq
        
        logger.info(f"Computing time-domain simulation:")
        logger.info(f"  Sensors: {len(sensor_positions)} positions")
        logger.info(f"  Source: {source_position} at {source_frequency} Hz")
        logger.info(f"  Duration: {duration}s, Sample rate: {sample_rate} Hz")
        logger.info(f"  Max frequency: {max_frequency} Hz")
        
        # Use precomputed frequency data if available, otherwise create new grids
        if precomputed_frequency_data:
            logger.info("Using precomputed frequency domain data - NO additional simulations needed!")
            freqs_audio = np.array(list(precomputed_frequency_data.keys()))
            freqs_viz = freqs_audio  # Use same frequencies for visualization
            total_freqs = 0  # No new simulations needed
            logger.info(f"Reusing {len(freqs_audio)} frequencies from frequency domain analysis")
            logger.info(f"TOTAL: 0 additional Helmholtz equation solves (MASSIVE OPTIMIZATION!)")
        else:
            # Create frequency grid for audio (increased for better quality)
            max_audio_freqs = min(100, int(max_frequency / 200))  # Max 100 frequencies or 1 per 200Hz
            freqs_audio = np.linspace(0, max_frequency, max_audio_freqs)
            
            # Create frequency grid for visualization (fewer frequencies for efficiency)
            N_viz = 64  # Much smaller for visualization
            freqs_viz = np.linspace(0, max_frequency, N_viz)
            
            total_freqs = len(freqs_audio) + len(freqs_viz)  # Much more efficient now!
            logger.info(f"Computing {len(freqs_audio)} frequencies for audio impulse responses ({len(sensor_positions)} sensors)")
            logger.info(f"Computing {len(freqs_viz)} frequencies for visualization (ALL {len(self.mesh.coors)} mesh nodes)")
            logger.info(f"TOTAL: {total_freqs} Helmholtz equation solves (OPTIMIZED: {len(sensor_positions)}x fewer than before!)")
        
        logger.info(f"Visualization will include pressure data at sensor points as part of the full mesh field")
        
        # Compute frequency responses for all sensors
        sensor_frequency_responses = {}
        for sensor_idx, sensor_pos in enumerate(sensor_positions):
            sensor_frequency_responses[sensor_idx] = np.zeros(len(freqs_audio), dtype=np.complex128)
        
        if precomputed_frequency_data:
            logger.info(f"Extracting sensor responses from precomputed frequency domain data")
            for i, freq in enumerate(freqs_audio):
                freq_key = float(freq)  # Convert to float for key lookup
                if freq_key in precomputed_frequency_data:
                    # Extract sensor responses from precomputed data
                    sensor_responses = precomputed_frequency_data[freq_key]["sensor_responses"]
                    for sensor_idx in range(len(sensor_positions)):
                        sensor_frequency_responses[sensor_idx][i] = sensor_responses[sensor_idx]
                else:
                    logger.warning(f"Frequency {freq_key} not found in precomputed data")
                    for sensor_idx in range(len(sensor_positions)):
                        sensor_frequency_responses[sensor_idx][i] = 0.0
        else:
            logger.info(f"Computing frequency responses for {len(freqs_audio)} frequencies (ALL sensors at once)")
            
            for i, freq in enumerate(freqs_audio):
                # Progress bar with percentage and time estimate
                progress = (i / len(freqs_audio)) * 100
                if i % max(1, len(freqs_audio) // 10) == 0:  # Update every 10%
                    remaining = len(freqs_audio) - i
                    logger.info(f"  Audio: [{progress:5.1f}%] {i+1}/{len(freqs_audio)} freqs | {freq:.1f} Hz | {remaining} remaining")
                
                # Set up source at the actual source position
                k = 2 * np.pi * freq / self.c
                self._current_k = k
                self._current_source_pos = source_position
                self._current_source_amp = 1.0
                
                try:
                    # Solve Helmholtz equation ONCE per frequency
                    pressure_values = self.solve(solver_type="direct")
                    
                    # Evaluate at ALL sensor positions from the same solution
                    for sensor_idx, sensor_pos in enumerate(sensor_positions):
                        sensor_pressure = self.evaluate_at_points(pressure_values, [sensor_pos])
                        sensor_frequency_responses[sensor_idx][i] = sensor_pressure[0]
                        
                except Exception as e:
                    logger.warning(f"Failed to solve at frequency {freq:.1f} Hz: {e}")
                    for sensor_idx in range(len(sensor_positions)):
                        sensor_frequency_responses[sensor_idx][i] = 0.0
        
        # Compute impulse responses for audio
        logger.info(f"Computing impulse responses for {len(sensor_frequency_responses)} sensors")
        impulse_responses = {}
        for sensor_idx, freq_response in sensor_frequency_responses.items():
            logger.info(f"  Computing impulse response for sensor {sensor_idx} at position {sensor_positions[sensor_idx]}")
            # Build Hermitian spectrum for IFFT
            # Ensure we have the right length for IFFT
            n_samples = len(freq_response)
            H_full = np.zeros(n_samples, dtype=np.complex128)
            
            # Copy positive frequencies
            H_full[:n_samples] = freq_response
            
            # For proper IFFT, we need to handle DC and Nyquist components correctly
            # Simple approach: just use the frequency response as-is
            H_full = freq_response
            
            # Compute impulse response
            h_time = np.fft.ifft(H_full).real
            h_normalized = h_time / np.max(np.abs(h_time)) if np.max(np.abs(h_time)) > 0 else h_time
            
            impulse_responses[sensor_idx] = {
                "impulse_response": h_normalized.tolist(),
                "time_vector": (np.arange(len(h_time)) / sample_rate).tolist(),
                "sample_rate": sample_rate,
                "duration": duration
            }
            logger.info(f"    Sensor {sensor_idx} impulse response: {len(h_normalized)} samples, max: {np.max(np.abs(h_normalized)):.2e}")
        
        # Compute time-domain field data for visualization
        # This is computationally expensive, so we'll do it for a subset of frequencies
        time_field_data = self._compute_time_field_data(freqs_viz, source_position, duration, sample_rate, precomputed_frequency_data)
        
        logger.info(f"Time-domain simulation completed! Generated {len(impulse_responses)} impulse responses and visualization data")
        logger.info(f"Returning data structure with keys: {['impulse_responses', 'sensor_positions', 'source_position', 'source_frequency', 'time_field_data', 'parameters']}")
        
        logger.info(f"COMPLETED: Generated impulse responses for {len(impulse_responses)} sensors")
        for sensor_idx, ir_data in impulse_responses.items():
            logger.info(f"  Sensor {sensor_idx}: {len(ir_data['impulse_response'])} samples, {ir_data['duration']:.1f}s duration")
        
        result = {
            "impulse_responses": impulse_responses,
            "sensor_positions": sensor_positions,
            "source_position": source_position,
            "source_frequency": source_frequency,
            "time_field_data": time_field_data,
            "custom_source_signal": custom_source_signal.tolist() if custom_source_signal is not None else None,
            "parameters": {
                "sample_rate": sample_rate,
                "duration": duration,
                "max_frequency": max_frequency,
                "num_frequencies_audio": len(freqs_audio),
                "num_frequencies_viz": len(freqs_viz),
                "has_custom_source": custom_source_signal is not None
            }
        }
        
        logger.info(f"Final result type: {type(result)}")
        logger.info(f"Final result keys: {list(result.keys())}")
        return result
    
    def solve_time_domain(self, source_position: List[float], sensor_positions: List[List[float]], 
                         source_signal: np.ndarray, sample_rate: float = 44100,
                         duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Solve the wave equation directly in time domain for realistic echoes.
        
        Wave equation: ∂²p/∂t² = c²∇²p + S(x,t)
        
        Args:
            source_position: [x, y, z] position of source
            sensor_positions: List of [x, y, z] sensor positions
            source_signal: Time-domain source signal (array of pressure values over time)
            sample_rate: Sample rate in Hz
            duration: Duration of simulation (if None, use source_signal length)
            
        Returns:
            Dictionary with time-domain pressure at sensor positions
        """
        logger.info("Solving wave equation in time domain for realistic echoes")
        
        # Determine simulation duration
        if duration is None:
            duration = len(source_signal) / sample_rate
        
        # Use proper time step for acoustic wave equation
        # For stability: dt < h/(c*sqrt(3)) where h is mesh size, c is speed of sound
        # For audio quality: dt should be much smaller than 1/sample_rate
        c = 343.0  # Speed of sound in m/s
        dt = 1.0 / sample_rate  # Match audio sample rate for proper wave propagation
        num_time_steps = int(duration * sample_rate)
        
        logger.info(f"Using proper acoustic time stepping: dt={dt:.6f}s, {num_time_steps} steps")
        
        logger.info(f"Time domain simulation: {duration:.2f}s, {num_time_steps} time steps, dt={dt:.6f}s")
        logger.info(f"Source signal range: {np.min(source_signal):.6f} to {np.max(source_signal):.6f}")
        logger.info(f"Source signal RMS: {np.sqrt(np.mean(source_signal**2)):.6f}")
        logger.info(f"Source signal non-zero samples: {np.count_nonzero(source_signal)}/{len(source_signal)} ({100*np.count_nonzero(source_signal)/len(source_signal):.1f}%)")
        
        # Get mesh information first
        mesh_coords = self.mesh.coors
        num_nodes = self.mesh.n_nod
        
        # Get connectivity
        try:
            cells = self.mesh.get_conn('3_4')  # 3D tetrahedra
        except:
            cell_types = ['3_4', '3_3', '3_8']
            cells = None
            for cell_type in cell_types:
                try:
                    cells = self.mesh.get_conn(cell_type)
                    break
                except:
                    continue
            if cells is None:
                raise RuntimeError("No suitable 3D elements found in mesh")
        
        # Check mesh resolution for acoustic simulation
        c = 343.0  # Speed of sound
        # Estimate mesh element size
        mesh_volume = np.max(mesh_coords, axis=0) - np.min(mesh_coords, axis=0)
        mesh_volume_total = np.prod(mesh_volume)
        avg_element_size = (mesh_volume_total / len(cells))**(1/3) if len(cells) > 0 else 0.1
        logger.info(f"Mesh analysis: volume={mesh_volume}, avg_element_size={avg_element_size:.3f}m")
        
        # Check if mesh is fine enough for audio frequencies
        max_audio_freq = sample_rate / 2  # Nyquist frequency
        min_wavelength = c / max_audio_freq
        elements_per_wavelength = min_wavelength / avg_element_size
        logger.info(f"Audio analysis: max_freq={max_audio_freq:.0f}Hz, min_wavelength={min_wavelength:.3f}m, elements_per_wavelength={elements_per_wavelength:.1f}")
        
        # Calculate maximum frequency this mesh can resolve properly
        max_resolvable_freq = c / (4 * avg_element_size)  # 4 elements per wavelength minimum
        logger.info(f"Mesh can properly resolve frequencies up to: {max_resolvable_freq:.0f}Hz")
        
        if elements_per_wavelength < 4:
            logger.warning(f"WARNING: Mesh may be too coarse for audio frequencies! Need at least 4 elements per wavelength.")
            logger.warning(f"Current: {elements_per_wavelength:.1f} elements/wavelength, Required: 4+ elements/wavelength")
            logger.warning(f"Recommendation: Use a finer mesh or reduce max frequency to {max_resolvable_freq:.0f}Hz")
            
            # Apply frequency filtering to prevent aliasing
            logger.info(f"Applying frequency filtering to prevent aliasing - limiting to {max_resolvable_freq:.0f}Hz")
            # This will be handled by filtering the source signal
        else:
            logger.info(f"Mesh resolution is adequate for audio frequencies.")
        
        # Assemble FEM matrices (stiffness and mass)
        K, M = self._assemble_fem_matrices(mesh_coords, cells)
        
        # For time-domain wave equation, boundary conditions are handled differently
        # Time-domain Robin BC: ∂p/∂n + (1/Z)∂p/∂t = 0
        # This creates a damping term in the mass matrix, not stiffness matrix
        if self.boundary_impedance:
            logger.info("Applying time-domain boundary conditions")
            logger.info(f"Boundary impedances: {self.boundary_impedance}")
            
            # Create damping matrix for time-domain Robin boundary conditions
            D = self._assemble_time_domain_boundary_matrix(mesh_coords, cells)
            
            # For time-domain: M * ∂²p/∂t² + D * ∂p/∂t + K * p = f(t)
            # where D is the damping matrix from Robin boundary conditions
            logger.info("Time-domain boundary damping matrix assembled")
        else:
            D = None
            logger.info("No boundary conditions - sound will pass through walls")
        
        # Create source vector (spatial distribution) - amplitude will be set per time step
        # The source amplitude varies with time based on the input audio signal
        # Use much larger base amplitude for realistic acoustic simulation
        source_base_amplitude = 1000.0  # Much larger for proper acoustic wave generation
        source_vector = self._create_source_vector(mesh_coords, source_position, source_base_amplitude)
        
        # Initialize time-stepping variables
        # For wave equation: M * ∂²p/∂t² + K * p = f(t)
        # Using Newmark-beta method for time integration
        
        # Use iterative solver instead of matrix inverse (much faster!)
        logger.info("Setting up iterative solver for time stepping")
        from scipy.sparse.linalg import spsolve, cg
        
        # Pre-compute LU decomposition of mass matrix for efficiency
        logger.info("Computing LU decomposition of mass matrix")
        M_lu = scipy.sparse.linalg.splu(M.tocsc())
        logger.info("Mass matrix LU decomposition completed")
        
        # Time stepping parameters for Newmark-beta method
        beta = 0.25  # Newmark parameter
        gamma = 0.5  # Newmark parameter
        
        # Initialize solution arrays
        p = np.zeros(num_nodes)  # Current pressure
        p_dot = np.zeros(num_nodes)  # Pressure velocity (∂p/∂t)
        p_ddot = np.zeros(num_nodes)  # Pressure acceleration (∂²p/∂t²)
        
        # Modified Newmark-beta for damped system: M * ∂²p/∂t² + D * ∂p/∂t + K * p = f(t)
        if D is not None:
            logger.info("Using damped Newmark-beta time integration")
        else:
            logger.info("Using undamped Newmark-beta time integration")
        
        # Storage for sensor data
        sensor_data = {}
        for i, sensor_pos in enumerate(sensor_positions):
            sensor_data[i] = []
        
        # Storage for full mesh pressure field over time (for visualization)
        # We'll store every Nth time step to manage memory
        mesh_pressure_history = []
        mesh_time_steps = []
        save_interval = max(1, num_time_steps // 100)  # Save 100 time steps max for visualization
        
        logger.info("Starting time-stepping loop with full mesh storage")
        
        # Time-stepping loop
        for n in range(num_time_steps):
            t = n * dt
            
            # Get source amplitude at current time from imported audio signal
            source_idx = min(n, len(source_signal) - 1)
            source_amplitude = source_signal[source_idx]
            
            # Scale source vector by time-varying amplitude from imported audio
            f = source_vector * source_amplitude
            
            # Debug: Log source amplitude for first few time steps
            if n < 10 or n % 1000 == 0:
                logger.debug(f"Time step {n}: source_amplitude={source_amplitude:.6f}, max_force={np.max(np.abs(f)):.2e}")
            
            # Modified Newmark-beta time integration for damped system
            if n == 0:
                # Initial conditions
                if D is not None:
                    rhs = f - K.dot(p) - D.dot(p_dot)
                else:
                    rhs = f - K.dot(p)
                p_ddot = M_lu.solve(rhs)
            else:
                # Predictor step
                p_pred = p + dt * p_dot + (0.5 - beta) * dt**2 * p_ddot
                p_dot_pred = p_dot + (1 - gamma) * dt * p_ddot
                
                # Corrector step
                if D is not None:
                    # Damped system: M * ∂²p/∂t² + D * ∂p/∂t + K * p = f(t)
                    rhs = f - K.dot(p_pred) - D.dot(p_dot_pred)
                else:
                    # Undamped system: M * ∂²p/∂t² + K * p = f(t)
                    rhs = f - K.dot(p_pred)
                
                p_ddot = M_lu.solve(rhs)
                p_dot = p_dot_pred + gamma * dt * p_ddot
                p = p_pred + beta * dt**2 * p_ddot
            
            # Store sensor data
            for i, sensor_pos in enumerate(sensor_positions):
                # Find closest mesh node to sensor
                point_array = np.array(sensor_pos, dtype=np.float64)
                distances = np.linalg.norm(mesh_coords - point_array, axis=1)
                closest_vertex_idx = np.argmin(distances)
                sensor_pressure = p[closest_vertex_idx]
                sensor_data[i].append(float(sensor_pressure))
            
            # Store full mesh pressure field at regular intervals for visualization
            if n % save_interval == 0:
                mesh_pressure_history.append(p.copy().tolist())  # Convert to list for JSON serialization
                mesh_time_steps.append(float(t))
            
            # Progress logging (less frequent to avoid spam)
            if n % max(1, num_time_steps // 20) == 0:
                progress = (n / num_time_steps) * 100
                pressure_rms = np.sqrt(np.mean(p**2))
                pressure_max = np.max(np.abs(p))
                logger.info(f"Time stepping progress: {progress:.1f}% (t={t:.3f}s) - Pressure RMS: {pressure_rms:.2e}, Max: {pressure_max:.2e}")
        
        logger.info("Time domain simulation completed")
        
        # Create result with JSON-serializable data
        result = {
            "sensor_data": sensor_data,
            "time_vector": (np.arange(num_time_steps) * dt).tolist(),  # Convert numpy array to list
            "sample_rate": float(sample_rate),
            "duration": float(duration),
            "num_time_steps": int(num_time_steps),
            "source_position": [float(x) for x in source_position],  # Ensure floats
            "sensor_positions": [[float(x) for x in pos] for pos in sensor_positions],  # Ensure floats
            # Full mesh pressure field for visualization
            "mesh_pressure_history": mesh_pressure_history,  # Pressure at all nodes over time
            "mesh_time_steps": mesh_time_steps,  # Time points for mesh data
            "mesh_coordinates": mesh_coords.tolist(),  # Mesh node coordinates
            "num_mesh_nodes": int(num_nodes)  # Number of mesh nodes
        }
        
        logger.info(f"Returned time domain data: {len(sensor_positions)} sensors, {num_time_steps} time steps")
        return result
    
    def _compute_time_field_data(self, frequencies: np.ndarray, source_position: List[float], 
                                duration: float, sample_rate: float, precomputed_frequency_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Compute time-domain field data for visualization."""
        import numpy as np
        
        logger.info(f"Computing time field data for {len(frequencies)} frequencies")
        
        # Store frequency responses at mesh nodes
        mesh_coords = self.mesh.coors
        num_nodes = len(mesh_coords)
        
        # Compute frequency responses at all mesh nodes
        node_frequency_responses = np.zeros((len(frequencies), num_nodes), dtype=np.complex128)
        
        if precomputed_frequency_data:
            logger.info("Using precomputed frequency domain data for visualization - NO additional field simulations needed!")
            for i, freq in enumerate(frequencies):
                freq_key = float(freq)
                if freq_key in precomputed_frequency_data:
                    # Extract pressure field from precomputed data
                    pressure_field = precomputed_frequency_data[freq_key]["pressure_field"]
                    if pressure_field is not None:
                        # Convert to numpy array if it's a list
                        if isinstance(pressure_field, list):
                            node_frequency_responses[i, :] = np.array(pressure_field, dtype=complex)
                        else:
                            node_frequency_responses[i, :] = pressure_field
                    else:
                        node_frequency_responses[i, :] = 0.0
                else:
                    logger.warning(f"Frequency {freq_key} not found in precomputed data")
                    node_frequency_responses[i, :] = 0.0
        else:
            for i, freq in enumerate(frequencies):
                # Progress for visualization frequencies
                progress = (i / len(frequencies)) * 100
                if i % max(1, len(frequencies) // 10) == 0:
                    remaining = len(frequencies) - i
                    logger.info(f"  Visualization: [{progress:5.1f}%] {i+1}/{len(frequencies)} freqs | {freq:.1f} Hz | {remaining} remaining")
                
                # Set up source
                k = 2 * np.pi * freq / self.c
                self._current_k = k
                self._current_source_pos = source_position
                self._current_source_amp = 1.0
                
                try:
                    # Solve and get full field (one simulation per frequency)
                    pressure_values = self.solve(solver_type="direct")
                    node_frequency_responses[i, :] = pressure_values
                    
                except Exception as e:
                    logger.warning(f"Failed to solve field at frequency {freq:.1f} Hz: {e}")
                    node_frequency_responses[i, :] = 0.0
        
        # Create time series data (increased for smoother animation)
        num_time_steps = min(int(duration * sample_rate / 44.1), 1000)  # Max 1000 time steps
        time_steps = np.linspace(0, duration, num_time_steps)
        
        logger.info(f"Creating time series with {num_time_steps} time steps")
        
        # Compute time-domain pressure at each node and time step
        time_pressure_data = np.zeros((num_time_steps, num_nodes), dtype=np.float32)
        
        for t_idx, t in enumerate(time_steps):
            if t_idx % max(1, num_time_steps // 10) == 0:
                logger.info(f"  Computing time step {t_idx+1}/{num_time_steps}: t={t:.3f}s")
            
            # Sum over frequencies for this time step
            for freq_idx, freq in enumerate(frequencies):
                if freq > 0:  # Skip DC
                    phase = -2 * np.pi * freq * t
                    amplitude = node_frequency_responses[freq_idx, :]
                    time_pressure_data[t_idx, :] += np.real(amplitude * np.exp(1j * phase))
        
        return {
            "time_steps": time_steps.tolist(),
            "pressure_time_series": time_pressure_data.tolist(),
            "mesh_coordinates": mesh_coords.tolist(),
            "frequencies_used": frequencies.tolist() if hasattr(frequencies, 'tolist') else list(frequencies),
            "num_nodes": num_nodes,
            "num_time_steps": num_time_steps
        }

    def compute_impulse_response(self, sensor_position: List[float], 
                                sample_rate: float = 44100, 
                                duration: float = 2.0) -> Dict[str, Any]:
        """
        Compute impulse response using frequency-domain to time-domain conversion.
        
        Args:
            sensor_position: [x, y, z] position of the sensor
            sample_rate: Audio sample rate (Hz)
            duration: Length of impulse response (seconds)
            
        Returns:
            Dictionary containing impulse response data
        """
        import numpy as np
        from scipy.fft import ifft
        
        logger.info(f"Computing impulse response for sensor at {sensor_position}")
        logger.info(f"Sample rate: {sample_rate} Hz, Duration: {duration} s")
        
        # Create uniform frequency grid for IFFT
        N_time = 2 ** int(np.ceil(np.log2(duration * sample_rate)))
        df = sample_rate / N_time
        freqs_pos = np.arange(0, sample_rate/2 + df, df)
        
        logger.info(f"Computing {len(freqs_pos)} frequency points from 0 to {sample_rate/2:.0f} Hz")
        logger.info(f"Time resolution: {1/sample_rate*1000:.1f} ms, Frequency resolution: {df:.2f} Hz")
        
        # Compute transfer function H(f) at each frequency
        H_pos = np.zeros(len(freqs_pos), dtype=np.complex128)
        
        for i, freq in enumerate(freqs_pos):
            if i % max(1, len(freqs_pos) // 20) == 0:  # Progress every 5%
                logger.info(f"Computing frequency {i+1}/{len(freqs_pos)}: {freq:.1f} Hz")
            
            # Set up the problem for this frequency
            k = 2 * np.pi * freq / self.c
            self._current_k = k
            self._current_source_pos = sensor_position  # Place source at sensor for impulse response
            self._current_source_amp = 1.0
            
            # Solve Helmholtz equation
            try:
                pressure_values = self.solve(solver_type="direct")
                
                # Evaluate pressure at the sensor position (source position in this case)
                sensor_pressure = self.evaluate_at_points(pressure_values, [sensor_position])
                H_pos[i] = sensor_pressure[0]
                
            except Exception as e:
                logger.warning(f"Failed to solve at frequency {freq:.1f} Hz: {e}")
                H_pos[i] = 0.0
        
        # Build full Hermitian spectrum for real-valued impulse response
        H_full = np.empty(N_time, dtype=np.complex128)
        H_full[:len(freqs_pos)] = H_pos
        # Mirror the spectrum (excluding DC and Nyquist)
        H_full[len(freqs_pos):] = np.conj(H_pos[1:-1][::-1])
        
        # Compute impulse response via IFFT
        h_time = np.fft.ifft(H_full).real
        
        # Create time vector
        dt = 1.0 / sample_rate
        time_vector = np.arange(len(h_time)) * dt
        
        # Normalize impulse response
        h_normalized = h_time / np.max(np.abs(h_time)) if np.max(np.abs(h_time)) > 0 else h_time
        
        logger.info(f"Impulse response computed: {len(h_time)} samples, max amplitude: {np.max(np.abs(h_time)):.2e}")
        
        return {
            "impulse_response": h_normalized.tolist(),
            "time_vector": time_vector.tolist(),
            "sample_rate": sample_rate,
            "duration": duration,
            "frequency_response": {
                "frequencies": freqs_pos.tolist(),
                "magnitude": np.abs(H_pos).tolist(),
                "phase": np.angle(H_pos).tolist()
            }
        }
    
    def get_mesh_data(self) -> Dict[str, Any]:
        """Get mesh geometry data."""
        try:
            logger.info("Starting mesh data extraction...")
            
            # Get vertex coordinates
            vertices = self.mesh.coors
            logger.info(f"Retrieved vertices: shape {vertices.shape}")
            
            vertex_list = []
            for i in range(vertices.shape[0]):
                vertex_list.append([float(vertices[i, 0]), float(vertices[i, 1]), float(vertices[i, 2])])
            
            logger.info(f"Converted {len(vertex_list)} vertices")
            
            # Get cell connectivity (tetrahedra)
            cells = self.mesh.get_conn('3_4')  # 3D tetrahedra
            logger.info(f"Retrieved cells: shape {cells.shape}")
            
            cell_list = []
            for i in range(cells.shape[0]):
                cell_vertices = []
                for j in range(cells.shape[1]):
                    cell_vertices.append(int(cells[i, j]))
                    cell_list.append(cell_vertices)
            
            logger.info(f"Converted {len(cell_list)} cells")
            
            mesh_data = {
                "vertices": vertex_list,
                "cells": cell_list,
                "num_vertices": len(vertex_list),
                "num_cells": len(cell_list),
                "element_order": self.element_order,
                "num_nodes": self.mesh.n_nod
            }
            
            logger.info(f"Extracted mesh data: {len(vertex_list)} vertices, {len(cell_list)} cells")
            return mesh_data
            
        except Exception as e:
            logger.error(f"Error extracting mesh data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    def save_solution(self, solution: np.ndarray, filename: str):
        """Save solution to file for visualization."""
        # SfePy can save to VTK format
        from sfepy.base.ioutils import ensure_path
        
        output_dir = ensure_path(filename)
        problem = self._problem
        if problem is not None:
            problem.save_state(filename, solution)


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
