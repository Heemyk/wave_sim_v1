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
            return self.active_jobs.get(job_id)
    
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
            if job_id in self.active_jobs:
                await self.update_job_status(job_id, "cancelled", 0.0, "Job cancelled")
                return True
            return False
    
    async def list_jobs(self) -> List[JobStatus]:
        """List all jobs (active and recent history)."""
        async with self._lock:
            all_jobs = list(self.active_jobs.values()) + self.job_history[-50:]  # Last 50
            return sorted(all_jobs, key=lambda x: x.created_at, reverse=True)
    
    async def cleanup(self):
        """Cleanup on shutdown."""
        async with self._lock:
            # Cancel all active jobs
            for job_id in list(self.active_jobs.keys()):
                await self.cancel_job(job_id)
            logger.info("Job manager cleanup complete")
