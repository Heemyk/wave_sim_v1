"""Persistent job management for acoustic simulations."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pickle

from backend.app.schemas import SimulationRequest, JobStatus

logger = logging.getLogger(__name__)


class PersistentJobManager:
    """Manages simulation jobs with persistence to disk."""
    
    def __init__(self, data_dir: str = "data/jobs"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_jobs: Dict[str, JobStatus] = {}
        self.job_history: List[JobStatus] = []
        self.job_requests: Dict[str, SimulationRequest] = {}  # Store job requests
        self._lock = asyncio.Lock()
        
        # Load existing jobs from disk
        self._load_jobs_from_disk()
    
    def _get_job_file(self, job_id: str) -> Path:
        """Get the file path for a job."""
        return self.data_dir / f"{job_id}.json"
    
    def _save_job_to_disk(self, job: JobStatus):
        """Save a job to disk."""
        job_file = self._get_job_file(job.job_id)
        try:
            with open(job_file, 'w') as f:
                json.dump(job.dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save job {job.job_id} to disk: {e}")
    
    def _load_job_from_disk(self, job_id: str) -> Optional[JobStatus]:
        """Load a job from disk."""
        job_file = self._get_job_file(job_id)
        if not job_file.exists():
            return None
        
        try:
            with open(job_file, 'r') as f:
                job_data = json.load(f)
                return JobStatus(**job_data)
        except Exception as e:
            logger.error(f"Failed to load job {job_id} from disk: {e}")
            return None
    
    def _load_jobs_from_disk(self):
        """Load all jobs from disk on startup."""
        try:
            for job_file in self.data_dir.glob("*.json"):
                job_id = job_file.stem
                job = self._load_job_from_disk(job_id)
                if job:
                    if job.status in ["pending", "running"]:
                        self.active_jobs[job_id] = job
                    else:
                        self.job_history.append(job)
            
            logger.info(f"Loaded {len(self.active_jobs)} active jobs and {len(self.job_history)} completed jobs from disk")
        except Exception as e:
            logger.error(f"Failed to load jobs from disk: {e}")
    
    def _delete_job_from_disk(self, job_id: str):
        """Delete a job file from disk."""
        job_file = self._get_job_file(job_id)
        try:
            if job_file.exists():
                job_file.unlink()
        except Exception as e:
            logger.error(f"Failed to delete job {job_id} from disk: {e}")
    
    async def submit_job(self, job_id: str, request: SimulationRequest, status: JobStatus):
        """Submit a new job."""
        async with self._lock:
            self.active_jobs[job_id] = status
            self.job_requests[job_id] = request  # Store the request
            self._save_job_to_disk(status)
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
            
            # If not found in memory, try to load from disk
            job = self._load_job_from_disk(job_id)
            if job:
                if job.status in ["pending", "running"]:
                    self.active_jobs[job_id] = job
                else:
                    self.job_history.append(job)
                return job
            
            return None
    
    async def get_job_request(self, job_id: str) -> Optional[SimulationRequest]:
        """Get job request."""
        async with self._lock:
            return self.job_requests.get(job_id)
    
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
                
                # Save to disk
                self._save_job_to_disk(job)
                
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
                
                # Save to disk
                self._save_job_to_disk(job)
                
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
            # Cancel all active jobs
            for job_id, job in list(self.active_jobs.items()):
                job.status = "cancelled"
                job.progress = 0.0
                job.message = "Job cancelled during shutdown"
                job.completed_at = datetime.now().isoformat()
                
                # Move to history
                self.job_history.append(job)
                self._save_job_to_disk(job)
            
            # Clear active jobs
            self.active_jobs.clear()
            logger.info("Persistent job manager cleanup complete")
    
    async def cleanup_old_jobs(self, days: int = 7):
        """Clean up old completed jobs."""
        async with self._lock:
            cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
            
            jobs_to_remove = []
            for job in self.job_history:
                try:
                    job_date = datetime.fromisoformat(job.created_at.replace('Z', '+00:00')).timestamp()
                    if job_date < cutoff_date:
                        jobs_to_remove.append(job)
                except:
                    # If we can't parse the date, keep the job
                    pass
            
            for job in jobs_to_remove:
                self.job_history.remove(job)
                self._delete_job_from_disk(job.job_id)
            
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
