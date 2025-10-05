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


@app.websocket("/ws/jobs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            global job_manager
            status = await job_manager.get_job_status(job_id)
            if status:
                await manager.send_personal_message(
                    json.dumps(status.dict()), websocket
                )
            await asyncio.sleep(1)
    except WebSocketDisconnect:
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
