"""FEM worker for acoustic simulations."""

import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from backend.app.schemas import SimulationRequest, SimulationResult, FrequencyResult
from fem.helmholtz_solver import HelmholtzSolver, create_simple_box_mesh

logger = logging.getLogger(__name__)


class FEMWorker:
    """Worker for running FEM acoustic simulations."""
    
    def __init__(self):
        self.solver: Optional[HelmholtzSolver] = None
    
    async def run_simulation(self, request: SimulationRequest) -> SimulationResult:
        """
        Run a complete acoustic simulation.
        
        Args:
            request: Simulation configuration
            
        Returns:
            Complete simulation results
        """
        logger.info(f"Starting simulation: {request.name or 'Unnamed'}")
        
        try:
            # Generate mesh
            mesh_file = await self._generate_mesh(request)
            
            # Initialize solver
            self.solver = HelmholtzSolver(
                mesh_file=mesh_file,
                element_order=request.mesh.element_order,
                boundary_impedance=self._get_boundary_impedance(request)
            )
            
            # Generate frequency list
            frequencies = self._generate_frequencies(request.simulation)
            
            # Run frequency-domain simulation
            frequency_results = []
            total_freqs = len(frequencies)
            
            for i, freq in enumerate(frequencies):
                logger.info(f"Computing frequency {freq:.1f} Hz ({i+1}/{total_freqs})")
                
                # Update progress
                progress = (i + 1) / total_freqs
                
                # Compute for this frequency
                freq_result = await self._compute_frequency(
                    freq, request, progress
                )
                frequency_results.append(freq_result)
            
            # Compute impulse responses if requested
            impulse_responses = None
            if request.output.impulse_response:
                impulse_responses = await self._compute_impulse_responses(
                    frequency_results, request
                )
            
            # Create result object
            result = SimulationResult(
                job_id="",  # Will be set by caller
                config=request,
                frequencies=frequency_results,
                impulse_responses=impulse_responses,
                metadata={
                    "mesh_file": mesh_file,
                    "num_frequencies": len(frequencies),
                    "solver_info": {
                        "num_dofs": self.solver.V.dofmap.index_map.size_global,
                        "element_order": request.mesh.element_order,
                    }
                },
                performance_stats={
                    "total_frequencies": len(frequencies),
                    "mesh_size": self.solver.V.dofmap.index_map.size_global,
                }
            )
            
            logger.info("Simulation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    
    async def _generate_mesh(self, request: SimulationRequest) -> str:
        """Generate mesh for the simulation."""
        mesh_dir = Path("data/meshes")
        mesh_dir.mkdir(parents=True, exist_ok=True)
        
        if request.room.type == "box":
            # Create box mesh
            dimensions = request.room.dimensions or [4.0, 3.0, 2.5]
            center = request.room.center or [0.0, 0.0, 0.0]
            h = request.mesh.target_h
            
            mesh_file = create_simple_box_mesh(
                dimensions, center, h,
                filename=str(mesh_dir / f"box_{request.name or 'sim'}.msh")
            )
        else:
            # Use provided geometry file
            if not request.room.geometry_file:
                raise ValueError("Geometry file required for custom rooms")
            mesh_file = request.room.geometry_file
        
        return mesh_file
    
    def _get_boundary_impedance(self, request: SimulationRequest) -> Dict[str, complex]:
        """Extract boundary impedance from request."""
        impedance = {}
        
        # Convert absorption coefficients to impedance
        # Z = rho * c / (1 + sqrt(1 - alpha))
        c = 343.0  # Speed of sound
        rho = 1.225  # Air density
        
        for boundary_name in ["walls", "floor", "ceiling"]:
            boundary_config = getattr(request.boundaries, boundary_name)
            if boundary_config.Z is not None:
                impedance[boundary_name] = boundary_config.Z
            else:
                # Convert absorption coefficient to impedance
                alpha = boundary_config.alpha
                if alpha > 0:
                    Z = rho * c / (1 + np.sqrt(1 - alpha))
                    impedance[boundary_name] = Z
                else:
                    impedance[boundary_name] = 0.0  # Rigid boundary
        
        return impedance
    
    def _generate_frequencies(self, sim_config) -> List[float]:
        """Generate frequency list from simulation config."""
        frequencies = []
        f = sim_config.fmin
        while f <= sim_config.fmax:
            frequencies.append(f)
            f += sim_config.df
        
        return frequencies
    
    async def _compute_frequency(
        self, 
        frequency: float, 
        request: SimulationRequest, 
        progress: float
    ) -> FrequencyResult:
        """Compute solution for a single frequency."""
        # Get source position (use first source)
        if not request.sources:
            raise ValueError("At least one source required")
        
        source = request.sources[0]
        source_pos = source.position
        
        # Get sensor positions
        sensor_positions = []
        sensor_ids = []
        
        for sensor in request.output.sensors:
            sensor_positions.append(sensor.position)
            sensor_ids.append(sensor.id)
        
        for poi in request.output.points_of_interest:
            sensor_positions.append(poi.position)
            sensor_ids.append(poi.id)
        
        # Compute frequency response
        results = self.solver.compute_frequency_response(
            [frequency], source_pos, sensor_positions,
            solver_type=request.simulation.solver_type
        )
        
        # Extract sensor data
        sensor_data = {}
        for i, sensor_id in enumerate(sensor_ids):
            sensor_key = f"sensor_{i}"
            if sensor_key in results["sensor_data"]:
                sensor_data[sensor_id] = results["sensor_data"][sensor_key][0]
        
        return FrequencyResult(
            frequency=frequency,
            sensor_data=sensor_data,
            metadata={
                "progress": progress,
                "source_position": source_pos,
                "num_sensors": len(sensor_positions)
            }
        )
    
    async def _compute_impulse_responses(
        self, 
        frequency_results: List[FrequencyResult], 
        request: SimulationRequest
    ) -> Dict[str, List[float]]:
        """Compute impulse responses from frequency data."""
        impulse_responses = {}
        
        # Get all sensor IDs
        sensor_ids = set()
        for freq_result in frequency_results:
            sensor_ids.update(freq_result.sensor_data.keys())
        
        # Compute IFFT for each sensor
        frequencies = [fr.frequency for fr in frequency_results]
        
        for sensor_id in sensor_ids:
            # Collect frequency response
            freq_response = []
            for freq_result in frequency_results:
                if sensor_id in freq_result.sensor_data:
                    freq_response.append(freq_result.sensor_data[sensor_id])
                else:
                    freq_response.append(0.0)
            
            # Convert to numpy array
            freq_array = np.array(freq_response, dtype=complex)
            
            # Pad for better time resolution
            padded_freq = np.zeros(len(freq_array) * 2, dtype=complex)
            padded_freq[:len(freq_array)] = freq_array
            
            # Compute inverse FFT
            impulse_response = np.fft.ifft(padded_freq).real
            
            # Store result
            impulse_responses[sensor_id] = impulse_response.tolist()
        
        return impulse_responses
