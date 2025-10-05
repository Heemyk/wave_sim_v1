"""Job management for acoustic simulations."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

from backend.app.schemas import SimulationRequest, JobStatus

logger = logging.getLogger(__name__)


class JobManager:
    """Manages simulation jobs and their status."""
    
    def __init__(self):
        self.active_jobs: Dict[str, JobStatus] = {}
        self.job_history: List[JobStatus] = []
        self._lock = asyncio.Lock()
    
    async def submit_job(self, job_id: str, request: SimulationRequest, status: JobStatus):
        """Submit a new job."""
        async with self._lock:
            self.active_jobs[job_id] = status
            logger.info(f"Job {job_id} submitted")
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status."""
        async with self._lock:
            # Check active jobs first
            if job_id in self.active_jobs:
                return self.active_jobs[job_id]
            
            # Check job history if not in active jobs
            for job in self.job_history:
                if job.job_id == job_id:
                    return job
            
            return None
    
    async def update_job_status(
        self, 
        job_id: str, 
        status: str, 
        progress: float, 
        message: Optional[str] = None
    ):
        """Update job status."""
        async with self._lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job.status = status
                job.progress = progress
                if message:
                    job.message = message
                
                # Update timestamps
                if status == "running" and not job.started_at:
                    job.started_at = datetime.now().isoformat()
                elif status in ["completed", "failed", "cancelled"]:
                    job.completed_at = datetime.now().isoformat()
                    # Move to history
                    self.job_history.append(job)
                    del self.active_jobs[job_id]
                
                logger.info(f"Job {job_id} status updated: {status} ({progress:.1%})")
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        async with self._lock:
            # Check active jobs first
            if job_id in self.active_jobs:
                # Update job status directly without recursive lock
                job = self.active_jobs[job_id]
                job.status = "cancelled"
                job.progress = 0.0
                job.message = "Job cancelled"
                job.completed_at = datetime.now().isoformat()
                
                # Move to history
                self.job_history.append(job)
                del self.active_jobs[job_id]
                
                logger.info(f"Job {job_id} cancelled")
                return True
            
            # Check if job is already in history (can't cancel completed jobs)
            for job in self.job_history:
                if job.job_id == job_id:
                    logger.info(f"Job {job_id} already completed, cannot cancel")
                    return False
            
            return False
    
    async def list_jobs(self) -> List[JobStatus]:
        """List all jobs (active and recent history)."""
        async with self._lock:
            all_jobs = list(self.active_jobs.values()) + self.job_history[-50:]  # Last 50
            return sorted(all_jobs, key=lambda x: x.created_at, reverse=True)
    
    async def cleanup(self):
        """Cleanup on shutdown."""
        async with self._lock:
            # Cancel all active jobs directly without recursive calls
            for job_id, job in list(self.active_jobs.items()):
                job.status = "cancelled"
                job.progress = 0.0
                job.message = "Job cancelled during shutdown"
                job.completed_at = datetime.now().isoformat()
                
                # Move to history
                self.job_history.append(job)
            
            # Clear active jobs
            self.active_jobs.clear()
            logger.info("Job manager cleanup complete")
