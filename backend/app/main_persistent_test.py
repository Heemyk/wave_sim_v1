"""FastAPI backend for acoustic room simulator - PERSISTENT TEST VERSION.

This version uses persistent job management to survive backend restarts.
"""

import asyncio
import json
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global job_manager, results_io
    
    # Startup
    logger.info("Starting Acoustic Room Simulator backend (PERSISTENT TEST MODE)...")
    
    # Create data directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "meshes").mkdir(exist_ok=True)
    (data_dir / "results").mkdir(exist_ok=True)
    (data_dir / "cache").mkdir(exist_ok=True)
    (data_dir / "jobs").mkdir(exist_ok=True)  # Job persistence directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize components with persistence
    job_manager = PersistentJobManager()
    results_io = ResultsIO()
    
    logger.info("Backend startup complete (PERSISTENT TEST MODE)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down backend...")
    if job_manager:
        await job_manager.cleanup()
    logger.info("Backend shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Acoustic Room Simulator (PERSISTENT TEST)",
    description="FEM-based acoustic simulation with persistent job management - TEST MODE",
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


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Acoustic Room Simulator API (PERSISTENT TEST MODE)", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global job_manager
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(job_manager.active_jobs) if job_manager else 0,
        "total_jobs": len(job_manager.active_jobs) + len(job_manager.job_history) if job_manager else 0,
        "python_path": sys.path,
        "working_directory": str(Path.cwd()),
        "mode": "PERSISTENT_TEST"
    }


@app.post("/api/simulate", response_model=JobStatus)
async def submit_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
) -> JobStatus:
    """
    Submit a new simulation job (PERSISTENT TEST MODE - no background processing).
    
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
            message="Job submitted (PERSISTENT TEST MODE - no processing)",
            created_at=datetime.now().isoformat()
        )
        
        # Submit job to manager (will persist to disk)
        global job_manager
        await job_manager.submit_job(job_id, request, job_status)
        
        # NOTE: In TEST MODE, we don't start background processing
        # This allows us to test job management without simulation complexity
        # background_tasks.add_task(process_simulation_job, job_id, request)
        
        logger.info(f"Submitted simulation job {job_id} (PERSISTENT TEST MODE)")
        return job_status
        
    except Exception as e:
        logger.error(f"Error submitting simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate/start/{job_id}")
async def start_simulation(job_id: str, background_tasks: BackgroundTasks):
    """
    Start processing for a specific job (PERSISTENT TEST MODE manual trigger).
    """
    try:
        global job_manager
        status = await job_manager.get_job_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if status.status != "pending":
            raise HTTPException(status_code=400, detail="Job is not in pending status")
        
        # Start background processing
        background_tasks.add_task(process_simulation_job, job_id, None)
        
        return {"message": f"Started processing job {job_id}"}
        
    except HTTPException:
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


@app.delete("/api/jobs/cleanup")
async def cleanup_old_jobs(days: int = 7):
    """Clean up old completed jobs."""
    try:
        global job_manager
        await job_manager.cleanup_old_jobs(days)
        return {"message": f"Cleaned up jobs older than {days} days"}
    except Exception as e:
        logger.error(f"Error cleaning up jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{job_id}")
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
    """Process a simulation job in the background (PERSISTENT TEST MODE)."""
    global job_manager, results_io
    
    try:
        logger.info(f"Starting simulation job {job_id} (PERSISTENT TEST MODE)")
        
        # Update job status
        await job_manager.update_job_status(
            job_id, "running", 0.0, "Initializing simulation (PERSISTENT TEST MODE)"
        )
        
        # Simulate some work
        await asyncio.sleep(2)
        
        await job_manager.update_job_status(
            job_id, "running", 0.5, "Processing simulation (PERSISTENT TEST MODE)"
        )
        
        # Simulate more work
        await asyncio.sleep(2)
        
        # Create mock results
        mock_result = SimulationResult(
            job_id=job_id,
            config=request or SimulationRequest(**{
                "name": "Test Job",
                "room": {"type": "box", "dimensions": [2.0, 2.0, 2.0], "center": [0.0, 0.0, 0.0]},
                "boundaries": {"walls": {"alpha": 0.0}, "floor": {"alpha": 0.0}, "ceiling": {"alpha": 0.0}},
                "sources": [],
                "mesh": {"element_order": 1, "target_h": 0.2, "refinement_level": 0, "adaptive": False, "quality_threshold": 0.3},
                "simulation": {"type": "frequency_domain", "fmin": 100.0, "fmax": 200.0, "df": 50.0, "solver_type": "direct", "tolerance": 1e-6, "max_iterations": 1000},
                "output": {"sensors": [], "points_of_interest": [], "field_snapshots": False, "frequency_response": True, "impulse_response": False, "visualization_data": False, "format": "json", "compression": False},
                "parallel_jobs": 1
            }),
            frequencies=[
                FrequencyResult(
                    frequency=100.0,
                    sensor_data={"test_sensor": 1.0 + 0.5j},
                    metadata={"dofs": 100}
                ),
                FrequencyResult(
                    frequency=150.0,
                    sensor_data={"test_sensor": 0.8 + 0.3j},
                    metadata={"dofs": 100}
                )
            ],
            metadata={
                "mesh_info": {"num_dofs": 100, "num_elements": 200},
                "performance": {"solve_time": 1.5},
                "persistent_test_mode": True
            }
        )
        
        # Save results
        await results_io.save_results(job_id, mock_result)
        
        # Update job status
        await job_manager.update_job_status(
            job_id, "completed", 1.0, "Simulation completed (PERSISTENT TEST MODE)"
        )
        
        logger.info(f"Completed simulation job {job_id} (PERSISTENT TEST MODE)")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        await job_manager.update_job_status(
            job_id, "failed", 0.0, f"Error: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main_persistent_test:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
