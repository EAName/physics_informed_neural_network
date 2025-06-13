from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
from datetime import datetime
import numpy as np

class PowerGridData(BaseModel):
    """Schema for power grid data."""
    voltage_magnitudes: List[float] = Field(..., min_items=1)
    voltage_angles: List[float] = Field(..., min_items=1)
    active_power: List[float] = Field(..., min_items=1)
    reactive_power: List[float] = Field(..., min_items=1)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('voltage_magnitudes', 'voltage_angles', 'active_power', 'reactive_power')
    def check_lengths_match(cls, v, values):
        if 'voltage_magnitudes' in values and len(v) != len(values['voltage_magnitudes']):
            raise ValueError('All input arrays must have the same length')
        return v

class RenewableData(BaseModel):
    """Schema for renewable energy data."""
    system_type: str = Field(..., regex='^(solar|wind)$')
    
    # Solar specific fields
    solar_radiation: Optional[float] = Field(None, ge=0, le=1000)
    temperature: Optional[float] = Field(None, ge=-20, le=60)
    time_of_day: Optional[float] = Field(None, ge=0, le=24)
    
    # Wind specific fields
    wind_speed: Optional[float] = Field(None, ge=0, le=30)
    wind_direction: Optional[float] = Field(None, ge=0, le=360)
    air_density: Optional[float] = Field(None, ge=1.0, le=1.3)
    
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('solar_radiation', 'temperature', 'time_of_day')
    def validate_solar_fields(cls, v, values):
        if values.get('system_type') == 'solar' and v is None:
            raise ValueError('Solar fields are required for solar system type')
        return v
    
    @validator('wind_speed', 'wind_direction', 'air_density')
    def validate_wind_fields(cls, v, values):
        if values.get('system_type') == 'wind' and v is None:
            raise ValueError('Wind fields are required for wind system type')
        return v

class ModelConfig(BaseModel):
    """Schema for model configuration."""
    layers: List[int] = Field(..., min_items=3)
    activation: str = Field(..., regex='^(tanh|relu|sigmoid)$')
    learning_rate: float = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    
    @validator('layers')
    def validate_layers(cls, v):
        if v[0] <= 0 or v[-1] <= 0:
            raise ValueError('Input and output layer sizes must be positive')
        return v

class TrainingConfig(BaseModel):
    """Schema for training configuration."""
    epochs: int = Field(..., gt=0)
    physics_weight: float = Field(..., ge=0)
    data_weight: float = Field(..., ge=0)
    validation_split: float = Field(..., ge=0, le=1)
    early_stopping_patience: int = Field(..., ge=0)

class GridConfig(BaseModel):
    """Schema for power grid configuration."""
    bus_count: int = Field(..., gt=0)
    line_count: int = Field(..., gt=0)
    base_mva: float = Field(..., gt=0)
    voltage_limits: Dict[str, float] = Field(..., min_items=2, max_items=2)
    line_limits: Dict[str, float] = Field(..., min_items=1)

class RenewableConfig(BaseModel):
    """Schema for renewable energy configuration."""
    system_type: str = Field(..., regex='^(solar|wind)$')
    capacity: float = Field(..., gt=0)
    location: Dict[str, float] = Field(..., min_items=2, max_items=2)
    timezone: str
    
    # Solar specific configuration
    solar: Optional[Dict[str, Union[float, str]]] = None
    
    # Wind specific configuration
    wind: Optional[Dict[str, Union[float, str]]] = None
    
    @validator('solar')
    def validate_solar_config(cls, v, values):
        if values.get('system_type') == 'solar' and v is None:
            raise ValueError('Solar configuration is required for solar system type')
        return v
    
    @validator('wind')
    def validate_wind_config(cls, v, values):
        if values.get('system_type') == 'wind' and v is None:
            raise ValueError('Wind configuration is required for wind system type')
        return v

class PredictionResult(BaseModel):
    """Schema for model prediction results."""
    predictions: List[float] = Field(..., min_items=1)
    confidence: List[float] = Field(..., min_items=1)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('confidence')
    def validate_confidence(cls, v, values):
        if 'predictions' in values and len(v) != len(values['predictions']):
            raise ValueError('Confidence scores must match predictions length')
        if not all(0 <= x <= 1 for x in v):
            raise ValueError('Confidence scores must be between 0 and 1')
        return v

class ModelMetrics(BaseModel):
    """Schema for model performance metrics."""
    loss: float = Field(..., ge=0)
    accuracy: float = Field(..., ge=0, le=1)
    physics_violation: float = Field(..., ge=0)
    training_time: float = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.now) 