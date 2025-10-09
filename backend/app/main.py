"""FastAPI backend for acoustic room simulator."""

import asyncio
import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from backend.app.schemas import (
        SimulationRequest, JobStatus, SimulationResult, FrequencyResult
    )
    from backend.app.jobs.persistent_job_manager import PersistentJobManager
    from backend.app.io.results_io import ResultsIO
    from backend.app.workers.fem_worker import FEMWorker
    from fem.helmholtz_solver import HelmholtzSolver
except ImportError as e:
    logging.error(f"Import error: {e}")
    logging.error(f"Python path: {sys.path}")
    logging.error(f"Current working directory: {Path.cwd()}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/backend.log', mode='a') if Path('logs').exists() else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for app state
job_manager = None
results_io = None
fem_worker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global job_manager, results_io, fem_worker
    
    # Startup
    logger.info("Starting Acoustic Room Simulator backend...")
    
    # Create data directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "meshes").mkdir(exist_ok=True)
    (data_dir / "results").mkdir(exist_ok=True)
    (data_dir / "cache").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize components
    job_manager = PersistentJobManager()
    results_io = ResultsIO()
    fem_worker = FEMWorker()
    
    logger.info("Backend startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down backend...")
    if job_manager:
        await job_manager.cleanup()
    logger.info("Backend shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Acoustic Room Simulator",
    description="FEM-based acoustic simulation with geometric acoustics",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()


# Startup and shutdown are now handled by the lifespan manager


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Acoustic Room Simulator API", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global job_manager
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(job_manager.active_jobs) if job_manager else 0,
        "python_path": sys.path,
        "working_directory": str(Path.cwd())
    }


@app.post("/api/simulate", response_model=JobStatus)
async def submit_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
) -> JobStatus:
    """
    Submit a new simulation job.
    
    Args:
        request: Simulation configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Job status information
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create job status
        job_status = JobStatus(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Job submitted",
            created_at=datetime.now().isoformat()
        )
        
        # Submit job to manager
        global job_manager
        await job_manager.submit_job(job_id, request, job_status)
        
        # Note: Job processing will be started manually via /api/simulate/start/{job_id}
        
        logger.info(f"Submitted simulation job {job_id}")
        return job_status
        
    except Exception as e:
        logger.error(f"Error submitting simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate/start/{job_id}")
async def start_simulation(job_id: str, background_tasks: BackgroundTasks):
    """Start processing for a specific job."""
    try:
        global job_manager
        status = await job_manager.get_job_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if status.status != "pending":
            raise HTTPException(status_code=400, detail=f"Job is not pending (current status: {status.status})")
        
        # Get the job request from the job manager
        job_request = await job_manager.get_job_request(job_id)
        if job_request is None:
            raise HTTPException(status_code=404, detail="Job request not found")
        
        # Start job processing in background
        background_tasks.add_task(process_simulation_job, job_id, job_request)
        
        # Update job status
        await job_manager.update_job_status(job_id, "running", 0.0, "Processing started")
        
        logger.info(f"Started simulation job {job_id}")
        return {"message": "Job processing started", "job_id": job_id}
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """Get the status of a specific job."""
    try:
        global job_manager
        status = await job_manager.get_job_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return status
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}/results", response_model=SimulationResult)
async def get_job_results(job_id: str) -> SimulationResult:
    """Get the results of a completed job."""
    try:
        global results_io
        results = await results_io.load_results(job_id)
        if results is None:
            raise HTTPException(status_code=404, detail="Results not found")
        return results
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    try:
        global job_manager
        success = await job_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"message": "Job cancelled"}
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs."""
    try:
        global job_manager
        jobs = await job_manager.list_jobs()
        return {"jobs": jobs}
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}/mesh")
async def get_job_mesh(job_id: str):
    """Get mesh geometry data for a job."""
    try:
        global results_io
        results = await results_io.load_results(job_id)
        if results is None:
            raise HTTPException(status_code=404, detail="Job results not found")
        
        # Extract mesh data from first frequency result
        if results.frequencies and len(results.frequencies) > 0:
            freq_result = results.frequencies[0]
            visualization_data = freq_result.metadata.get("visualization_data", {})
            mesh_info = visualization_data.get("mesh_info", {})
            
            return {
                "job_id": job_id,
                "mesh": mesh_info,
                "metadata": {
                    "frequency": freq_result.frequency,
                    "num_frequencies": len(results.frequencies)
                }
            }
        else:
            raise HTTPException(status_code=404, detail="No frequency results found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting mesh data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}/field/{frequency}")
async def get_field_data(job_id: str, frequency: float):
    """Get pressure field data for a specific frequency."""
    try:
        global results_io
        results = await results_io.load_results(job_id)
        if results is None:
            raise HTTPException(status_code=404, detail="Job results not found")
        
        # Find the frequency result
        freq_result = None
        for fr in results.frequencies:
            if abs(fr.frequency - frequency) < 1e-6:  # Float comparison
                freq_result = fr
                break
        
        if freq_result is None:
            raise HTTPException(status_code=404, detail=f"Frequency {frequency} Hz not found")
        
        # Extract field data
        visualization_data = freq_result.metadata.get("visualization_data", {})
        field_data = visualization_data.get("field_data", {})
        
        # Convert complex numbers to JSON-serializable format
        def convert_complex(obj):
            if isinstance(obj, complex):
                return {"real": float(obj.real), "imag": float(obj.imag)}
            elif isinstance(obj, dict):
                return {k: convert_complex(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_complex(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return convert_complex(obj.tolist())
            else:
                return obj
        
        # Convert sensor data
        sensor_data = convert_complex(freq_result.sensor_data) if freq_result.sensor_data else {}
        
        return {
            "job_id": job_id,
            "frequency": frequency,
            "field_data": convert_complex(field_data),
            "sensor_data": sensor_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting field data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}/visualization")
async def get_visualization_data(job_id: str):
    """Get complete visualization data for a job."""
    try:
        global results_io
        results = await results_io.load_results(job_id)
        if results is None:
            raise HTTPException(status_code=404, detail="Job results not found")
        
        # Collect all visualization data
        visualization_summary = {
            "job_id": job_id,
            "frequencies": [],
            "mesh_info": {},
            "metadata": results.metadata
        }
        
        # Convert complex numbers to JSON-serializable format
        def convert_complex(obj):
            if isinstance(obj, complex):
                return {"real": float(obj.real), "imag": float(obj.imag)}
            elif isinstance(obj, dict):
                return {k: convert_complex(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_complex(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return convert_complex(obj.tolist())
            else:
                return obj
        
        for freq_result in results.frequencies:
            visualization_data = freq_result.metadata.get("visualization_data", {})
            
            freq_data = {
                "frequency": freq_result.frequency,
                "sensor_data": convert_complex(freq_result.sensor_data) if freq_result.sensor_data else {},
                "field_data": convert_complex(visualization_data.get("field_data", {})),
                "mesh_info": visualization_data.get("mesh_info", {})
            }
            visualization_summary["frequencies"].append(freq_data)
            
            # Use mesh info from first frequency
            if not visualization_summary["mesh_info"]:
                visualization_summary["mesh_info"] = visualization_data.get("mesh_info", {})
        
        return visualization_summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/unified_simulation")
async def compute_unified_simulation_direct(request: dict):
    """Compute both frequency domain and time domain simulations directly without requiring an existing job."""
    try:
        logger.info(f"DIRECT UNIFIED SIMULATION REQUEST")
        logger.info(f"Request data: {request}")
        
        # Get parameters from request
        sensor_positions = request.get("sensor_positions", [[0.0, 0.0, 0.0]])
        source_position = request.get("source_position", [1.0, 1.0, 1.0])
        source_frequency = request.get("source_frequency", 440.0)
        sample_rate = request.get("sample_rate", 44100)
        duration = request.get("duration", 2.0)
        max_frequency = request.get("max_frequency", 20000.0)
        num_frequencies = request.get("num_frequencies", 100)
        mesh_file = request.get("mesh_file", "data/meshes/box_Frontend Test Simulation.msh")
        element_order = request.get("element_order", 1)
        boundary_absorption = request.get("boundary_impedance", {})
        
        # Convert absorption coefficients to impedances
        import numpy as np
        c = 343.0  # Speed of sound
        rho = 1.225  # Air density
        boundary_impedance = {}
        
        for boundary_name, alpha in boundary_absorption.items():
            if alpha > 0:
                # Correct Robin boundary condition impedance for acoustics
                # Z = rho * c / alpha (for absorption-dominated boundaries)
                Z = rho * c / alpha
                boundary_impedance[boundary_name] = Z
                logger.info(f"Boundary {boundary_name}: alpha={alpha:.3f} -> Z={Z:.0f}")
            else:
                # Very high impedance for rigid boundaries (alpha=0)
                boundary_impedance[boundary_name] = 1e6  # Large but finite value
                logger.info(f"Boundary {boundary_name}: rigid (alpha=0) -> Z={1e6:.0e}")
        
        # Get custom audio data
        custom_audio_data = request.get("custom_audio_data", None)
        custom_audio_filename = request.get("custom_audio_filename", None)
        use_custom_audio = request.get("use_custom_audio", False)
        
        logger.info(f"Parameters: sensors={len(sensor_positions)}, source={source_position}")
        logger.info(f"Audio: {source_frequency}Hz, {sample_rate}Hz, {duration}s, max_freq={max_frequency}Hz")
        logger.info(f"Frequency analysis: {num_frequencies} frequencies")
        
        if use_custom_audio and custom_audio_data and custom_audio_filename:
            logger.info(f"Using custom audio file: {custom_audio_filename} ({len(custom_audio_data)} samples, {duration:.2f}s)")
        else:
            logger.info("Using default sine wave source signal")
        
        # Initialize solver with GPU acceleration
        solver = HelmholtzSolver(
            mesh_file=mesh_file,
            element_order=element_order,
            boundary_impedance=boundary_impedance,
            use_gpu=True  # Enable GPU acceleration
        )
        
        # Compute frequency domain analysis
        logger.info(f"Computing frequency domain analysis")
        import numpy as np
        frequency_data = {}
        
        if use_custom_audio and custom_audio_data:
            # Extract frequency content from custom audio
            logger.info("Analyzing custom audio frequency content")
            audio_array = np.array(custom_audio_data, dtype=np.float32)
            
            # Compute FFT to get frequency spectrum
            fft_result = np.fft.fft(audio_array)
            freqs = np.fft.fftfreq(len(audio_array), 1/sample_rate)
            
            # Get only positive frequencies
            positive_mask = freqs > 0
            positive_freqs = freqs[positive_mask]
            positive_fft = fft_result[positive_mask]
            
            # Get magnitude spectrum
            magnitude_spectrum = np.abs(positive_fft)
            
            # Find significant frequencies (above threshold)
            threshold = np.max(magnitude_spectrum) * 0.01  # 1% of max amplitude
            significant_mask = magnitude_spectrum > threshold
            significant_freqs = positive_freqs[significant_mask]
            significant_magnitudes = magnitude_spectrum[significant_mask]
            
            logger.info(f"Found {len(significant_freqs)} significant frequencies in audio")
            if len(significant_freqs) > 0:
                logger.info(f"Frequency range: {np.min(significant_freqs):.1f} - {np.max(significant_freqs):.1f} Hz")
            
            # Use significant frequencies for simulation (limit to num_frequencies)
            if len(significant_freqs) > 0:
                frequencies_to_simulate = significant_freqs[:num_frequencies]
                amplitudes_to_use = significant_magnitudes[:num_frequencies]
            else:
                # Fallback to default if no significant frequencies found
                frequencies_to_simulate = np.linspace(100, max_frequency, num_frequencies)
                amplitudes_to_use = np.ones(num_frequencies)
        else:
            # Use default frequency range
            frequencies_to_simulate = np.linspace(100, max_frequency, num_frequencies)
            amplitudes_to_use = np.ones(num_frequencies)  # Equal amplitude for all frequencies
        
        logger.info(f"Simulating {len(frequencies_to_simulate)} frequencies")
        
        # Use GPU batch processing for massive speedup
        try:
            logger.info("🚀 Using GPU batch processing for massive speedup!")
            batch_results = solver.solve_gpu_batch(frequencies_to_simulate.tolist())
            
            # Process results
            for i, freq in enumerate(frequencies_to_simulate):
                if i % 10 == 0:
                    logger.info(f"Processing frequency {i+1}/{len(frequencies_to_simulate)}: {freq:.1f} Hz")
                
                if freq in batch_results:
                    pressure_values = batch_results[freq]
                    
                    # Use amplitude from audio spectrum if available
                    source_amplitude = amplitudes_to_use[i] if i < len(amplitudes_to_use) else 1.0
                    
                    # Evaluate at sensor positions
                    sensor_responses = []
                    for sensor_pos in sensor_positions:
                        response = solver.evaluate_at_points(pressure_values, [sensor_pos])
                        sensor_responses.append(complex(response[0]).real) # Convert to real for JSON
                    
                    frequency_data[float(freq)] = {
                        "frequency": float(freq),
                        "pressure_field": pressure_values.real.tolist(),  # Convert complex to real
                        "sensor_responses": sensor_responses,
                        "source_amplitude": float(source_amplitude)
                    }
                else:
                    logger.warning(f"No result for frequency {freq:.1f} Hz")
                    frequency_data[float(freq)] = {
                        "frequency": float(freq),
                        "pressure_field": None,
                        "sensor_responses": [0.0] * len(sensor_positions),
                        "source_amplitude": 1.0
                    }
                    
        except Exception as e:
            logger.error(f"GPU batch processing failed: {e}")
            logger.info("Falling back to CPU processing...")
            
            # Fallback to CPU processing
            for i, freq in enumerate(frequencies_to_simulate):
                if i % 10 == 0:
                    logger.info(f"Computing frequency {i+1}/{len(frequencies_to_simulate)}: {freq:.1f} Hz")
                
                try:
                    # Set up the solver for this frequency
                    c = 343.0  # Speed of sound
                    k = 2 * np.pi * freq / c
                    solver._current_k = k
                    solver._current_source_pos = source_position
                    
                    # Use amplitude from audio spectrum if available
                    source_amplitude = amplitudes_to_use[i] if i < len(amplitudes_to_use) else 1.0
                    solver._current_source_amp = float(source_amplitude)
                    
                    # Solve Helmholtz equation for this frequency
                    pressure_values = solver.solve(solver_type="direct")
                    
                    # Evaluate at sensor positions
                    sensor_responses = []
                    for sensor_pos in sensor_positions:
                        response = solver.evaluate_at_points(pressure_values, [sensor_pos])
                        sensor_responses.append(complex(response[0]).real) # Convert to real for JSON
                    
                    frequency_data[float(freq)] = {
                        "frequency": float(freq),
                        "pressure_field": pressure_values.real.tolist(),  # Convert complex to real
                        "sensor_responses": sensor_responses,
                        "source_amplitude": float(source_amplitude)
                    }
                    
                except Exception as e2:
                    logger.warning(f"Failed to solve at frequency {freq:.1f} Hz: {e2}")
                    frequency_data[float(freq)] = {
                        "frequency": float(freq),
                        "pressure_field": None,
                        "sensor_responses": [0.0] * len(sensor_positions),
                    "source_amplitude": 0.0
                }
        
        # Compute time domain simulation using NEW independent solver
        logger.info("Computing time domain simulation with NEW independent solver")
        
        # Convert custom audio data to numpy array if available
        custom_audio_np = None
        if use_custom_audio and custom_audio_data:
            custom_audio_np = np.array(custom_audio_data, dtype=np.float32)
            logger.info(f"Using custom audio signal for time domain: {custom_audio_np.shape} samples")
        
        # Use NEW independent time-domain solver instead of old IFFT method
        logger.info("Computing time domain simulation with NEW independent solver")
        
        # Create source signal from custom audio or default
        source_signal_array = None
        if use_custom_audio and custom_audio_data:
            source_signal_array = custom_audio_np
            logger.info(f"Using custom audio: {len(source_signal_array)} samples")
        else:
            # Create default sine wave for testing
            num_samples = int(duration * sample_rate)
            t = np.linspace(0, duration, num_samples)
            source_signal_array = np.sin(2 * np.pi * source_frequency * t).astype(np.float32)
            logger.info(f"Using default sine wave: {source_frequency}Hz, {len(source_signal_array)} samples")
        
        # Normalize source signal for proper FEM simulation
        # FEM needs source amplitudes in a reasonable range for numerical stability
        source_max = np.max(np.abs(source_signal_array))
        if source_max > 0:
            # Normalize to reasonable FEM amplitude range (not too small, not too large)
            target_source_amplitude = 1.0  # Good range for FEM numerical stability
            source_signal_array = source_signal_array * (target_source_amplitude / source_max)
            logger.info(f"Normalized source signal: original max={source_max:.6f}, new max={target_source_amplitude}")
        else:
            logger.warning("Source signal is all zeros - FEM simulation may not work properly")
        
        # Use the NEW independent time-domain solver
        time_domain_data = solver.solve_time_domain(
            source_position=source_position,
            sensor_positions=sensor_positions,
            source_signal=source_signal_array,
            sample_rate=sample_rate,
            duration=duration
        )
        
        logger.info(f"Direct unified simulation completed successfully")
        
        # Get mesh data for visualization
        logger.info("Extracting mesh data for frontend visualization")
        mesh_data = solver.get_mesh_data()
        
        # Combine both results
        unified_data = {
            "frequency_data": frequency_data,
            "time_domain_data": time_domain_data,
            "mesh_data": mesh_data,  # Include mesh data directly
            "parameters": {
                "num_frequencies": num_frequencies,
                "frequency_range": [float(frequencies_to_simulate[0]), float(frequencies_to_simulate[-1])],
                "sensor_positions": sensor_positions,
                "source_position": source_position,
                "source_frequency": source_frequency,
                "sample_rate": sample_rate,
                "duration": duration,
                "max_frequency": max_frequency
            }
        }
        
        # Debug: Check for complex numbers before JSON serialization
        import json
        try:
            json.dumps(unified_data)
            logger.info("Unified data is JSON serializable")
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            # Find and fix complex numbers
            def fix_complex(obj):
                if isinstance(obj, dict):
                    return {k: fix_complex(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [fix_complex(item) for item in obj]
                elif hasattr(obj, 'real') and hasattr(obj, 'imag'):
                    return float(obj.real)
                else:
                    return obj
            
            unified_data = fix_complex(unified_data)
            logger.info("Fixed complex numbers in unified data")
        
        # Check if response is too large (>50MB)
        response_size = len(str(unified_data))
        if response_size > 50 * 1024 * 1024:  # 50MB
            logger.warning(f"Response too large ({response_size} bytes), saving to file instead")
            
            # Save to file and return file path
            import json
            import os
            
            # Create temporary results directory
            temp_job_id = str(uuid.uuid4())
            results_dir = f"data/results/{temp_job_id}"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save unified data to file
            unified_file = f"{results_dir}/unified_data.json"
            with open(unified_file, 'w') as f:
                json.dump(unified_data, f)
            
            logger.info(f"Saved unified data to {unified_file}")
            
            return {
                "job_id": temp_job_id,
                "unified_data": None,
                "unified_file": unified_file,
                "file_size": response_size
            }
        else:
            logger.info(f"Response size OK ({response_size} bytes), returning directly")
            return {
                "job_id": str(uuid.uuid4()),
                "unified_data": unified_data
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing direct unified simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/time_domain_simulation")
async def compute_time_domain_simulation_direct(request: dict):
    """Compute time-domain wave equation simulation directly for realistic echoes."""
    try:
        logger.info(f"TIME DOMAIN SIMULATION REQUEST")
        logger.info(f"Request data: {request}")
        
        # Get parameters from request
        sensor_positions = request.get("sensor_positions", [[0.0, 0.0, 0.0]])
        source_position = request.get("source_position", [1.0, 1.0, 1.0])
        sample_rate = request.get("sample_rate", 44100)
        duration = request.get("duration", 2.0)
        mesh_file = request.get("mesh_file", "data/meshes/box_Frontend Test Simulation.msh")
        element_order = request.get("element_order", 1)
        boundary_absorption = request.get("boundary_impedance", {})
        
        # Get source signal
        source_signal = request.get("source_signal", None)
        if source_signal is None:
            # Create default sine wave
            import numpy as np
            num_samples = int(duration * sample_rate)
            t = np.linspace(0, duration, num_samples)
            source_signal = np.sin(2 * np.pi * 440 * t).tolist()  # 440 Hz sine wave
        
        # Convert absorption coefficients to impedances
        import numpy as np
        c = 343.0  # Speed of sound
        rho = 1.225  # Air density
        boundary_impedance = {}
        
        for boundary_name, alpha in boundary_absorption.items():
            if alpha > 0:
                Z = rho * c / alpha
                boundary_impedance[boundary_name] = Z
                logger.info(f"Boundary {boundary_name}: alpha={alpha:.3f} -> Z={Z:.0f}")
            else:
                boundary_impedance[boundary_name] = 1e6
                logger.info(f"Boundary {boundary_name}: rigid (alpha=0)")
        
        logger.info(f"Parameters: sensors={len(sensor_positions)}, source={source_position}")
        logger.info(f"Audio: {sample_rate}Hz, {duration}s, {len(source_signal)} samples")
        
        # Initialize solver with GPU acceleration
        solver = HelmholtzSolver(
            mesh_file=mesh_file,
            element_order=element_order,
            boundary_impedance=boundary_impedance,
            use_gpu=True  # Enable GPU acceleration
        )
        
        # Convert source signal to numpy array
        source_signal_array = np.array(source_signal, dtype=np.float32)
        
        # Normalize source signal for proper FEM simulation
        source_max = np.max(np.abs(source_signal_array))
        if source_max > 0:
            target_source_amplitude = 1.0  # Good range for FEM numerical stability
            source_signal_array = source_signal_array * (target_source_amplitude / source_max)
            logger.info(f"Normalized source signal: original max={source_max:.6f}, new max={target_source_amplitude}")
        else:
            logger.warning("Source signal is all zeros - FEM simulation may not work properly")
        
        # Run time-domain simulation
        logger.info("Starting time-domain wave equation simulation")
        time_domain_result = solver.solve_time_domain(
            source_position=source_position,
            sensor_positions=sensor_positions,
            source_signal=source_signal_array,
            sample_rate=sample_rate,
            duration=duration
        )
        
        logger.info("Time-domain simulation completed successfully")
        
        # Ensure all data is JSON serializable
        def make_json_serializable(obj):
            """Recursively convert numpy arrays and other non-serializable objects to JSON-serializable formats."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        # Convert the result to ensure JSON serialization
        serializable_result = make_json_serializable(time_domain_result)
        
        return {
            "success": True,
            "time_domain_data": serializable_result,
            "parameters": {
                "source_position": source_position,
                "sensor_positions": sensor_positions,
                "sample_rate": sample_rate,
                "duration": duration,
                "num_samples": len(source_signal_array)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in time-domain simulation: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs/{job_id}/unified_simulation")
async def compute_unified_simulation(job_id: str, request: dict):
    """Compute both frequency domain and time domain simulations in one call."""
    try:
        logger.info(f"UNIFIED SIMULATION REQUEST for job {job_id}")
        logger.info(f"Request data: {request}")
        
        global results_io
        results = await results_io.load_results(job_id)
        if results is None:
            logger.error(f"Job results not found for {job_id}")
            raise HTTPException(status_code=404, detail="Job results not found")
        
        # Get parameters from request
        sensor_positions = request.get("sensor_positions", [[0.0, 0.0, 0.0]])
        source_position = request.get("source_position", [1.0, 1.0, 1.0])
        source_frequency = request.get("source_frequency", 440.0)
        sample_rate = request.get("sample_rate", 44100)
        duration = request.get("duration", 2.0)
        max_frequency = request.get("max_frequency", 20000.0)
        num_frequencies = request.get("num_frequencies", 100)  # New parameter
        
        logger.info(f"Parameters: sensors={len(sensor_positions)}, source={source_position}")
        logger.info(f"Audio: {source_frequency}Hz, {sample_rate}Hz, {duration}s, max_freq={max_frequency}Hz")
        logger.info(f"Frequency analysis: {num_frequencies} frequencies")
        
        # Get solver parameters from results
        mesh_file = results.metadata.get("mesh_file", "data/meshes/default.msh")
        element_order = results.metadata.get("element_order", 1)
        boundary_impedance = results.metadata.get("boundary_impedance", {})
        
        # Initialize solver with GPU acceleration
        solver = HelmholtzSolver(
            mesh_file=mesh_file,
            element_order=element_order,
            boundary_impedance=boundary_impedance,
            use_gpu=True  # Enable GPU acceleration
        )
        
        # Compute frequency domain analysis using GPU batch processing
        logger.info(f"Computing frequency domain analysis for {num_frequencies} frequencies using GPU batch processing")
        import numpy as np
        frequency_data = {}
        frequencies = np.linspace(100, max_frequency, num_frequencies)
        
        # Set up solver for batch processing
        solver._current_source_pos = source_position
        solver._current_source_amp = 1.0
        
        try:
            # Use GPU batch processing for all frequencies at once
            logger.info("🚀 Using GPU batch processing for massive speedup!")
            batch_results = solver.solve_gpu_batch(frequencies.tolist())
            
            # Process results
            for i, freq in enumerate(frequencies):
                if i % 10 == 0:
                    logger.info(f"Processing frequency {i+1}/{num_frequencies}: {freq:.1f} Hz")
                
                if freq in batch_results:
                    pressure_values = batch_results[freq]
                    
                    # Evaluate at sensor positions
                    sensor_responses = []
                    for sensor_pos in sensor_positions:
                        response = solver.evaluate_at_points(pressure_values, [sensor_pos])
                        sensor_responses.append(response[0])
                    
                    frequency_data[freq] = {
                        "frequency": freq,
                        "pressure_field": pressure_values.real.tolist(),  # Convert complex to real
                        "sensor_responses": [complex(resp).real for resp in sensor_responses]  # Convert complex to real
                    }
                else:
                    logger.warning(f"No result for frequency {freq:.1f} Hz")
                    frequency_data[freq] = {
                        "frequency": freq,
                        "pressure_field": None,
                        "sensor_responses": [0.0] * len(sensor_positions)
                    }
                    
        except Exception as e:
            logger.error(f"GPU batch processing failed: {e}")
            logger.info("Falling back to CPU processing...")
            
            # Fallback to CPU processing
            for i, freq in enumerate(frequencies):
                if i % 10 == 0:
                    logger.info(f"Computing frequency {i+1}/{num_frequencies}: {freq:.1f} Hz")
                
                try:
                    # Set up the solver for this frequency
                    c = 343.0  # Speed of sound
                    k = 2 * np.pi * freq / c
                    solver._current_k = k
                    solver._current_source_pos = source_position
                    solver._current_source_amp = 1.0
                    
                    # Solve Helmholtz equation for this frequency
                    pressure_values = solver.solve(solver_type="direct")
                    
                    # Evaluate at sensor positions
                    sensor_responses = []
                    for sensor_pos in sensor_positions:
                        response = solver.evaluate_at_points(pressure_values, [sensor_pos])
                        sensor_responses.append(response[0])
                    
                    frequency_data[freq] = {
                        "frequency": freq,
                        "pressure_field": pressure_values.real.tolist(),  # Convert complex to real
                        "sensor_responses": [complex(resp).real for resp in sensor_responses]  # Convert complex to real
                    }
                    
                except Exception as e2:
                    logger.warning(f"Failed to solve at frequency {freq:.1f} Hz: {e2}")
                    frequency_data[freq] = {
                        "frequency": freq,
                        "pressure_field": None,
                        "sensor_responses": [0.0] * len(sensor_positions)
                    }
        
        # Compute time domain simulation using NEW independent solver
        logger.info("Computing time domain simulation with independent solver")
        
        # Create source signal from custom audio or default
        source_signal_array = None
        if custom_audio_data:
            source_signal_array = np.array(custom_audio_data, dtype=np.float32)
            logger.info(f"Using custom audio: {len(source_signal_array)} samples")
        else:
            # Create default sine wave for testing
            num_samples = int(duration * sample_rate)
            t = np.linspace(0, duration, num_samples)
            source_signal_array = np.sin(2 * np.pi * source_frequency * t).astype(np.float32)
            logger.info(f"Using default sine wave: {source_frequency}Hz, {len(source_signal_array)} samples")
        
        # Normalize source signal for proper FEM simulation
        # FEM needs source amplitudes in a reasonable range for numerical stability
        source_max = np.max(np.abs(source_signal_array))
        if source_max > 0:
            # Normalize to reasonable FEM amplitude range (not too small, not too large)
            target_source_amplitude = 1.0  # Good range for FEM numerical stability
            source_signal_array = source_signal_array * (target_source_amplitude / source_max)
            logger.info(f"Normalized source signal: original max={source_max:.6f}, new max={target_source_amplitude}")
        else:
            logger.warning("Source signal is all zeros - FEM simulation may not work properly")
        
        # Use the NEW independent time-domain solver
        time_domain_data = solver.solve_time_domain(
            source_position=source_position,
            sensor_positions=sensor_positions,
            source_signal=source_signal_array,
            sample_rate=sample_rate,
            duration=duration
        )
        
        logger.info(f"Unified simulation completed successfully")
        
        # Ensure all data is JSON serializable
        def make_json_serializable(obj):
            """Recursively convert numpy arrays and other non-serializable objects to JSON-serializable formats."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        # Convert time domain data to ensure JSON serialization
        serializable_time_data = make_json_serializable(time_domain_data)
        
        # Combine both results
        unified_data = {
            "job_id": job_id,
            "frequency_data": frequency_data,
            "time_domain_data": serializable_time_data,
            "parameters": {
                "num_frequencies": num_frequencies,
                "frequency_range": [float(frequencies_to_simulate[0]), float(frequencies_to_simulate[-1])],
                "sensor_positions": sensor_positions,
                "source_position": source_position,
                "source_frequency": source_frequency,
                "sample_rate": sample_rate,
                "duration": duration,
                "max_frequency": max_frequency
            }
        }
        
        # Check if response is too large (>50MB)
        response_size = len(str(unified_data))
        if response_size > 50 * 1024 * 1024:  # 50MB
            logger.warning(f"Response too large ({response_size} bytes), saving to file instead")
            
            # Save to file and return file path
            import json
            import os
            
            # Create results directory if it doesn't exist
            results_dir = f"data/results/{job_id}"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save unified data to file
            unified_file = f"{results_dir}/unified_data.json"
            with open(unified_file, 'w') as f:
                json.dump(unified_data, f)
            
            logger.info(f"Saved unified data to {unified_file}")
            
            return {
                "job_id": job_id,
                "unified_data": None,
                "unified_file": unified_file,
                "file_size": response_size
            }
        else:
            logger.info(f"Response size OK ({response_size} bytes), returning directly")
            return {
                "job_id": job_id,
                "unified_data": unified_data
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing unified simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs/{job_id}/time_domain_simulation")
async def compute_time_domain_simulation(job_id: str, request: dict):
    """Compute comprehensive time-domain simulation with both audio and visualization."""
    try:
        logger.info(f"TIME DOMAIN SIMULATION REQUEST for job {job_id}")
        logger.info(f"Request data: {request}")
        
        global results_io
        results = await results_io.load_results(job_id)
        if results is None:
            logger.error(f"Job results not found for {job_id}")
            raise HTTPException(status_code=404, detail="Job results not found")
        
        # Get parameters from request
        sensor_positions = request.get("sensor_positions", [[0.0, 0.0, 0.0]])
        source_position = request.get("source_position", [1.0, 1.0, 1.0])
        source_frequency = request.get("source_frequency", 440.0)
        sample_rate = request.get("sample_rate", 44100)
        duration = request.get("duration", 2.0)
        max_frequency = request.get("max_frequency", 20000.0)
        
        logger.info(f"Parameters: sensors={len(sensor_positions)}, source={source_position}")
        logger.info(f"Audio: {source_frequency}Hz, {sample_rate}Hz, {duration}s, max_freq={max_frequency}Hz")
        
        # Get solver parameters from results
        mesh_file = results.metadata.get("mesh_file", "data/meshes/default.msh")
        element_order = results.metadata.get("element_order", 1)
        boundary_impedance = results.metadata.get("boundary_impedance", {})
        
        # Create solver and compute comprehensive time-domain simulation
        solver = HelmholtzSolver(
            mesh_file=mesh_file,
            element_order=element_order,
            boundary_impedance=boundary_impedance
        )
        
        # Use NEW independent time-domain solver instead of old IFFT method
        logger.info("Computing time domain simulation with NEW independent solver")
        
        # Create source signal from custom audio or default
        source_signal_array = None
        if use_custom_audio and custom_audio_data:
            source_signal_array = custom_audio_np
            logger.info(f"Using custom audio: {len(source_signal_array)} samples")
        else:
            # Create default sine wave for testing
            num_samples = int(duration * sample_rate)
            t = np.linspace(0, duration, num_samples)
            source_signal_array = np.sin(2 * np.pi * source_frequency * t).astype(np.float32)
            logger.info(f"Using default sine wave: {source_frequency}Hz, {len(source_signal_array)} samples")
        
        # Normalize source signal for proper FEM simulation
        # FEM needs source amplitudes in a reasonable range for numerical stability
        source_max = np.max(np.abs(source_signal_array))
        if source_max > 0:
            # Normalize to reasonable FEM amplitude range (not too small, not too large)
            target_source_amplitude = 1.0  # Good range for FEM numerical stability
            source_signal_array = source_signal_array * (target_source_amplitude / source_max)
            logger.info(f"Normalized source signal: original max={source_max:.6f}, new max={target_source_amplitude}")
        else:
            logger.warning("Source signal is all zeros - FEM simulation may not work properly")
        
        # Use the NEW independent time-domain solver
        time_domain_data = solver.solve_time_domain(
            source_position=source_position,
            sensor_positions=sensor_positions,
            source_signal=source_signal_array,
            sample_rate=sample_rate,
            duration=duration
        )
        
        logger.info(f"Time-domain simulation completed successfully")
        logger.info(f"Data size: {len(str(time_domain_data))} characters")
        
        # Check if response is too large (>50MB)
        response_size = len(str(time_domain_data))
        if response_size > 50 * 1024 * 1024:  # 50MB
            logger.warning(f"Response too large ({response_size} bytes), saving to file instead")
            
            # Save to file and return file path
            import json
            import os
            
            # Create results directory if it doesn't exist
            results_dir = f"data/results/{job_id}"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save time domain data to file
            time_domain_file = f"{results_dir}/time_domain_data.json"
            with open(time_domain_file, 'w') as f:
                json.dump(time_domain_data, f)
            
            logger.info(f"Saved time-domain data to {time_domain_file}")
            
            return {
                "job_id": job_id,
                "time_domain_simulation": None,
                "time_domain_file": time_domain_file,
                "file_size": response_size
            }
        else:
            logger.info(f"Response size OK ({response_size} bytes), returning directly")
            return {
                "job_id": job_id,
                "time_domain_simulation": time_domain_data
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing time-domain simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}/unified_data")
async def get_unified_data(job_id: str):
    """Get unified simulation data from file if it's too large for direct response."""
    try:
        import json
        import os
        
        unified_file = f"data/results/{job_id}/unified_data.json"
        
        if not os.path.exists(unified_file):
            raise HTTPException(status_code=404, detail="Unified data file not found")
        
        logger.info(f"Loading unified data from {unified_file}")
        
        # Check file size first
        file_size = os.path.getsize(unified_file)
        logger.info(f"File size: {file_size} bytes ({file_size / 1024 / 1024:.1f} MB)")
        
        with open(unified_file, 'r') as f:
            unified_data = json.load(f)
        
        logger.info(f"Loaded unified data: {len(str(unified_data))} characters")
        
        return {
            "job_id": job_id,
            "unified_data": unified_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading unified data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}/time_domain_data")
async def get_time_domain_data(job_id: str, chunk: int = 0, chunk_size: int = 50):
    """Get time-domain simulation data in chunks to avoid memory issues."""
    try:
        import json
        import os
        
        time_domain_file = f"data/results/{job_id}/time_domain_data.json"
        
        if not os.path.exists(time_domain_file):
            raise HTTPException(status_code=404, detail="Time domain data file not found")
        
        logger.info(f"Loading time-domain data chunk {chunk} from {time_domain_file}")
        
        # Check file size first
        file_size = os.path.getsize(time_domain_file)
        logger.info(f"File size: {file_size} bytes ({file_size / 1024 / 1024:.1f} MB)")
        
        with open(time_domain_file, 'r') as f:
            time_domain_data = json.load(f)
        
        # Extract time field data and chunk it
        if 'time_field_data' in time_domain_data:
            time_field = time_domain_data['time_field_data']
            total_time_steps = len(time_field['time_steps'])
            
            # Calculate chunk boundaries
            start_idx = chunk * chunk_size
            end_idx = min(start_idx + chunk_size, total_time_steps)
            
            logger.info(f"Returning time steps {start_idx} to {end_idx} of {total_time_steps}")
            
            # Create chunked response
            chunked_data = {
                'time_steps': time_field['time_steps'][start_idx:end_idx],
                'pressure_time_series': time_field['pressure_time_series'][start_idx:end_idx],
                'mesh_coordinates': time_field['mesh_coordinates'],
                'frequencies_used': time_field['frequencies_used'],
                'num_nodes': time_field['num_nodes'],
                'num_time_steps': total_time_steps,
                'chunk_info': {
                    'current_chunk': chunk,
                    'chunk_size': chunk_size,
                    'total_chunks': (total_time_steps + chunk_size - 1) // chunk_size,
                    'is_complete': end_idx >= total_time_steps
                }
            }
            
            return {
                "job_id": job_id,
                "time_domain_simulation": {
                    'time_field_data': chunked_data,
                    'impulse_responses': time_domain_data.get('impulse_responses', {}),
                    'sensor_positions': time_domain_data.get('sensor_positions', []),
                    'source_position': time_domain_data.get('source_position', []),
                    'source_frequency': time_domain_data.get('source_frequency', 0),
                    'parameters': time_domain_data.get('parameters', {})
                }
            }
        else:
            # Fallback to full data if no time_field_data
            return {
                "job_id": job_id,
                "time_domain_simulation": time_domain_data
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading time-domain data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs/{job_id}/impulse_response")
async def compute_impulse_response(job_id: str, request: dict):
    """Compute impulse response for a sensor position (legacy endpoint)."""
    try:
        # Use the new comprehensive time-domain simulation for backward compatibility
        sensor_position = request.get("sensor_position", [0.0, 0.0, 0.0])
        sample_rate = request.get("sample_rate", 44100)
        duration = request.get("duration", 2.0)
        
        # Convert to new format
        new_request = {
            "sensor_positions": [sensor_position],
            "source_position": sensor_position,  # Use sensor as source for impulse response
            "source_frequency": 1.0,  # Impulse has all frequencies
            "sample_rate": sample_rate,
            "duration": duration,
            "max_frequency": sample_rate / 2
        }
        
        response = await compute_time_domain_simulation(job_id, new_request)
        time_domain_data = response["time_domain_simulation"]
        
        # Extract impulse response for the first sensor
        sensor_0_response = time_domain_data["impulse_responses"].get(0, {})
        
        return {
            "job_id": job_id,
            "sensor_position": sensor_position,
            "impulse_response": sensor_0_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing impulse response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/jobs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates."""
    await manager.connect(websocket)
    last_status = None
    try:
        while True:
            # Send updates only when status changes
            global job_manager
            status = await job_manager.get_job_status(job_id)
            if status:
                # Only send if status has actually changed
                current_status_json = json.dumps(status.dict())
                if current_status_json != last_status:
                    logger.info(f"WebSocket sending status update for job {job_id}: {status.status}")
                    await manager.send_personal_message(current_status_json, websocket)
                    last_status = current_status_json
                    
                    # If job is completed or failed, close connection after sending
                    if status.status in ['completed', 'failed']:
                        logger.info(f"Job {job_id} finished with status {status.status}, closing WebSocket")
                        break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
        manager.disconnect(websocket)


@app.get("/api/examples")
async def get_examples():
    """Get available example configurations."""
    examples_dir = Path("examples")
    examples = []
    
    if examples_dir.exists():
        for example_dir in examples_dir.iterdir():
            if example_dir.is_dir():
                config_file = example_dir / "config.yaml"
                if config_file.exists():
                    examples.append({
                        "name": example_dir.name,
                        "path": str(example_dir),
                        "config_file": str(config_file)
                    })
    
    return {"examples": examples}


@app.get("/api/examples/{example_name}/config")
async def get_example_config(example_name: str):
    """Get configuration for a specific example."""
    config_file = Path(f"examples/{example_name}/config.yaml")
    if not config_file.exists():
        raise HTTPException(status_code=404, detail="Example not found")
    
    return FileResponse(config_file)


async def process_simulation_job(job_id: str, request: SimulationRequest):
    """Process a simulation job in the background."""
    global job_manager, results_io, fem_worker
    
    try:
        logger.info(f"Starting simulation job {job_id}")
        
        # Update job status
        await job_manager.update_job_status(
            job_id, "running", 0.0, "Initializing simulation"
        )
        
        # Use global FEM worker
        if fem_worker is None:
            fem_worker = FEMWorker()
        
        # Process simulation with real FEM computation
        logger.info(f"Running real FEM simulation for job {job_id}")
        results = await fem_worker.run_simulation(request)
        
        # Save results
        await results_io.save_results(job_id, results)
        
        # Update job status
        await job_manager.update_job_status(
            job_id, "completed", 1.0, "Simulation completed"
        )
        
        logger.info(f"Completed simulation job {job_id}")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        await job_manager.update_job_status(
            job_id, "failed", 0.0, f"Error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
