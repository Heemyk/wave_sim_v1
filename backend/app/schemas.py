"""Configuration schemas and Pydantic models for the acoustic simulator."""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, validator
import numpy as np


class RoomGeometry(BaseModel):
    """Room geometry configuration."""
    geometry_file: Optional[str] = None
    type: Literal["box", "cylinder", "custom"] = "box"
    dimensions: Optional[List[float]] = Field(None, min_items=3, max_items=3)  # [length, width, height]
    center: Optional[List[float]] = Field(default=[0.0, 0.0, 0.0], min_items=3, max_items=3)


class ImpedanceConfig(BaseModel):
    """Acoustic impedance configuration for boundaries."""
    alpha: float = Field(0.15, ge=0.0, le=1.0)  # Absorption coefficient
    Z: Optional[complex] = None  # Complex impedance
    frequency_dependent: bool = False
    data: Optional[Dict[str, Any]] = None  # For frequency-dependent data


class BoundaryConditions(BaseModel):
    """Boundary conditions configuration."""
    walls: ImpedanceConfig = ImpedanceConfig()
    floor: ImpedanceConfig = ImpedanceConfig()
    ceiling: ImpedanceConfig = ImpedanceConfig()
    windows: Optional[List[Dict[str, Any]]] = None  # Open boundaries


class SourceConfig(BaseModel):
    """Acoustic source configuration."""
    id: str
    type: Literal["point", "line", "surface"] = "point"
    position: List[float] = Field(..., min_items=3, max_items=3)
    direction: Optional[List[float]] = Field(None, min_items=3, max_items=3)
    signal: "SignalConfig"
    amplitude: float = Field(1.0, gt=0.0)


class SignalConfig(BaseModel):
    """Signal configuration for sources."""
    type: Literal["chirp", "mls", "white_noise", "sine", "impulse"] = "chirp"
    f0: float = Field(20.0, gt=0.0)  # Start frequency (Hz)
    f1: float = Field(8000.0, gt=0.0)  # End frequency (Hz)
    duration: float = Field(1.0, gt=0.0)  # Duration (s)
    amplitude: float = Field(1.0, gt=0.0)
    phase: float = Field(0.0)  # Phase offset (rad)
    
    @validator('f1')
    def f1_greater_than_f0(cls, v, values):
        if 'f0' in values and v <= values['f0']:
            raise ValueError('f1 must be greater than f0')
        return v


class SensorConfig(BaseModel):
    """Sensor/measurement point configuration."""
    id: str
    position: List[float] = Field(..., min_items=3, max_items=3)
    type: Literal["point", "aperture"] = "point"
    aperture_size: Optional[float] = Field(None, gt=0.0)  # For aperture sensors
    noise_model: Optional[Dict[str, Any]] = None


class MeshConfig(BaseModel):
    """Mesh generation configuration."""
    element_order: int = Field(1, ge=1, le=3)
    target_h: float = Field(0.05, gt=0.0)  # Target element size (m)
    min_h: Optional[float] = Field(None, gt=0.0)
    max_h: Optional[float] = Field(None, gt=0.0)
    refinement_level: int = Field(0, ge=0, le=3)
    adaptive: bool = False
    quality_threshold: float = Field(0.3, ge=0.0, le=1.0)


class SimulationConfig(BaseModel):
    """Simulation configuration."""
    type: Literal["frequency_domain", "time_domain", "hybrid"] = "frequency_domain"
    fmin: float = Field(20.0, gt=0.0)
    fmax: float = Field(8000.0, gt=0.0)
    df: float = Field(20.0, gt=0.0)  # Frequency step (Hz)
    crossover_frequency: Optional[float] = Field(None, gt=0.0)  # For hybrid mode
    solver_type: Literal["direct", "iterative"] = "direct"
    preconditioner: Optional[str] = None
    tolerance: float = Field(1e-6, gt=0.0)
    max_iterations: int = Field(1000, gt=0)
    
    @validator('fmax')
    def fmax_greater_than_fmin(cls, v, values):
        if 'fmin' in values and v <= values['fmin']:
            raise ValueError('fmax must be greater than fmin')
        return v


class OutputConfig(BaseModel):
    """Output configuration."""
    points_of_interest: List[SensorConfig] = []
    sensors: List[SensorConfig] = []
    field_snapshots: bool = True
    frequency_response: bool = True
    impulse_response: bool = True
    visualization_data: bool = True
    format: Literal["hdf5", "xdmf", "json"] = "hdf5"
    compression: bool = True


class SimulationRequest(BaseModel):
    """Complete simulation request configuration."""
    room: RoomGeometry
    boundaries: BoundaryConditions
    sources: List[SourceConfig]
    mesh: MeshConfig
    simulation: SimulationConfig
    output: OutputConfig
    
    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    
    # Performance settings
    parallel_jobs: int = Field(4, ge=1, le=16)
    memory_limit: Optional[str] = None  # e.g., "8GB"
    timeout: Optional[int] = Field(None, gt=0)  # seconds


class JobStatus(BaseModel):
    """Job status information."""
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


class FrequencyResult(BaseModel):
    """Result for a single frequency."""
    frequency: float
    pressure_field: Optional[Dict[str, Any]] = None  # Field data
    sensor_data: Optional[Dict[str, complex]] = None  # Complex pressure at sensors
    metadata: Optional[Dict[str, Any]] = None


class SimulationResult(BaseModel):
    """Complete simulation result."""
    job_id: str
    config: SimulationRequest
    frequencies: List[FrequencyResult]
    impulse_responses: Optional[Dict[str, List[float]]] = None
    frequency_responses: Optional[Dict[str, List[complex]]] = None
    metadata: Dict[str, Any]
    performance_stats: Optional[Dict[str, Any]] = None


# Update forward references
SourceConfig.model_rebuild()
SignalConfig.model_rebuild()
