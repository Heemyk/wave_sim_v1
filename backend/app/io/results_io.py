
"""Results I/O for acoustic simulations."""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import h5py
import numpy as np
from typing import List
from backend.app.schemas import SimulationResult, SimulationRequest

logger = logging.getLogger(__name__)


class ResultsIO:
    """Handles saving and loading simulation results."""
    
    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_results(self, job_id: str, results: SimulationResult):
        """Save simulation results to disk."""
        try:
            # Set job ID
            results.job_id = job_id
            
            # Create job directory
            job_dir = self.results_dir / job_id
            job_dir.mkdir(exist_ok=True)
            
            # Save configuration
            config_file = job_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(results.config.dict(), f, indent=2, default=str)
            
            # Save metadata
            metadata_file = job_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(results.metadata, f, indent=2, default=str)
            
            # Save frequency results
            freq_file = job_dir / "frequencies.json"
            freq_data = []
            for freq_result in results.frequencies:
                freq_data.append({
                    "frequency": freq_result.frequency,
                    "sensor_data": {k: {"real": v.real, "imag": v.imag} if isinstance(v, complex) else v 
                                   for k, v in freq_result.sensor_data.items()},
                    "metadata": freq_result.metadata
                })
            
            with open(freq_file, 'w') as f:
                json.dump(freq_data, f, indent=2)
            
            # Save impulse responses if available
            if results.impulse_responses:
                ir_file = job_dir / "impulse_responses.json"
                with open(ir_file, 'w') as f:
                    json.dump(results.impulse_responses, f, indent=2)
            
            # Save performance stats
            if results.performance_stats:
                perf_file = job_dir / "performance.json"
                with open(perf_file, 'w') as f:
                    json.dump(results.performance_stats, f, indent=2)
            
            # Save HDF5 file for large data
            hdf5_file = job_dir / "results.h5"
            await self._save_hdf5(hdf5_file, results)
            
            logger.info(f"Results saved for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error saving results for job {job_id}: {e}")
            raise
    
    async def load_results(self, job_id: str) -> Optional[SimulationResult]:
        """Load simulation results from disk."""
        try:
            job_dir = self.results_dir / job_id
            if not job_dir.exists():
                return None
            
            # Load configuration
            config_file = job_dir / "config.json"
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Load metadata
            metadata_file = job_dir / "metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load frequency results
            freq_file = job_dir / "frequencies.json"
            with open(freq_file, 'r') as f:
                freq_data = json.load(f)
            
            frequency_results = []
            for freq_item in freq_data:
                # Deserialize complex numbers properly
                sensor_data = {}
                for k, v in freq_item["sensor_data"].items():
                    if isinstance(v, dict) and "real" in v and "imag" in v:
                        # Complex number stored as {"real": x, "imag": y}
                        sensor_data[k] = complex(v["real"], v["imag"])
                    elif isinstance(v, str):
                        # Handle string representation like "1+0.5j"
                        try:
                            sensor_data[k] = complex(v.replace('j', 'i'))
                        except:
                            sensor_data[k] = complex(v)
                    else:
                        # Already a number or complex
                        sensor_data[k] = v
                
                frequency_results.append({
                    "frequency": freq_item["frequency"],
                    "sensor_data": sensor_data,
                    "metadata": freq_item["metadata"]
                })
            
            # Load impulse responses if available
            impulse_responses = None
            ir_file = job_dir / "impulse_responses.json"
            if ir_file.exists():
                with open(ir_file, 'r') as f:
                    impulse_responses = json.load(f)
            
            # Load performance stats
            performance_stats = None
            perf_file = job_dir / "performance.json"
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    performance_stats = json.load(f)
            
            # Create result object
            result = SimulationResult(
                job_id=job_id,
                config=SimulationRequest(**config_data),
                frequencies=frequency_results,
                impulse_responses=impulse_responses,
                metadata=metadata,
                performance_stats=performance_stats
            )
            
            logger.info(f"Results loaded for job {job_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading results for job {job_id}: {e}")
            return None
    
    async def _save_hdf5(self, filename: Path, results: SimulationResult):
        """Save results to HDF5 file."""
        with h5py.File(filename, 'w') as f:
            # Save frequencies
            frequencies = [fr.frequency for fr in results.frequencies]
            f.create_dataset('frequencies', data=frequencies)
            
            # Save sensor data
            sensor_group = f.create_group('sensors')
            for freq_result in results.frequencies:
                freq_group = sensor_group.create_group(f'freq_{freq_result.frequency:.1f}')
                for sensor_id, value in freq_result.sensor_data.items():
                    freq_group.create_dataset(sensor_id, data=[value.real, value.imag])
            
            # Save impulse responses if available
            if results.impulse_responses:
                ir_group = f.create_group('impulse_responses')
                for sensor_id, ir_data in results.impulse_responses.items():
                    ir_group.create_dataset(sensor_id, data=ir_data)
            
            # Save metadata
            meta_group = f.create_group('metadata')
            for key, value in results.metadata.items():
                if isinstance(value, (str, int, float)):
                    meta_group.attrs[key] = value
    
    def list_results(self) -> List[Dict[str, Any]]:
        """List all available results."""
        results = []
        
        for job_dir in self.results_dir.iterdir():
            if job_dir.is_dir():
                metadata_file = job_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        results.append({
                            "job_id": job_dir.name,
                            "metadata": metadata,
                            "created_at": metadata.get("created_at", "unknown")
                        })
                    except Exception as e:
                        logger.warning(f"Error reading metadata for {job_dir.name}: {e}")
        
        return sorted(results, key=lambda x: x["created_at"], reverse=True)
    
    def delete_results(self, job_id: str) -> bool:
        """Delete results for a job."""
        try:
            job_dir = self.results_dir / job_id
            if job_dir.exists():
                import shutil
                shutil.rmtree(job_dir)
                logger.info(f"Deleted results for job {job_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting results for job {job_id}: {e}")
            return False
