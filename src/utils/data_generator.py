import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from ..schemas.data_schema import PowerGridData, RenewableData

class PowerGridDataGenerator:
    """Generator for synthetic power grid data."""
    
    def __init__(
        self,
        bus_count: int,
        line_count: int,
        base_mva: float = 100.0,
        voltage_limits: Tuple[float, float] = (0.95, 1.05)
    ):
        """Initialize the power grid data generator.
        
        Args:
            bus_count: Number of buses in the grid
            line_count: Number of transmission lines
            base_mva: Base MVA for the system
            voltage_limits: Tuple of (min, max) voltage limits
        """
        self.bus_count = bus_count
        self.line_count = line_count
        self.base_mva = base_mva
        self.voltage_limits = voltage_limits
        
    def generate_voltage_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic voltage magnitude and angle data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (voltage_magnitudes, voltage_angles)
        """
        v_min, v_max = self.voltage_limits
        voltage_magnitudes = np.random.uniform(
            v_min, v_max, (n_samples, self.bus_count)
        )
        voltage_angles = np.random.uniform(
            -np.pi, np.pi, (n_samples, self.bus_count)
        )
        return voltage_magnitudes, voltage_angles
        
    def generate_power_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic active and reactive power data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (active_power, reactive_power)
        """
        # Generate active power (MW)
        active_power = np.random.normal(
            0.5 * self.base_mva,
            0.2 * self.base_mva,
            (n_samples, self.bus_count)
        )
        
        # Generate reactive power (MVAr)
        reactive_power = np.random.normal(
            0.2 * self.base_mva,
            0.1 * self.base_mva,
            (n_samples, self.bus_count)
        )
        
        return active_power, reactive_power
        
    def generate_dataset(
        self,
        n_samples: int,
        timestamp: Optional[datetime] = None
    ) -> List[PowerGridData]:
        """Generate a complete dataset.
        
        Args:
            n_samples: Number of samples to generate
            timestamp: Base timestamp for the data
            
        Returns:
            List of PowerGridData objects
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        voltage_magnitudes, voltage_angles = self.generate_voltage_data(n_samples)
        active_power, reactive_power = self.generate_power_data(n_samples)
        
        dataset = []
        for i in range(n_samples):
            data = PowerGridData(
                voltage_magnitudes=voltage_magnitudes[i].tolist(),
                voltage_angles=voltage_angles[i].tolist(),
                active_power=active_power[i].tolist(),
                reactive_power=reactive_power[i].tolist(),
                timestamp=timestamp + timedelta(minutes=i)
            )
            dataset.append(data)
            
        return dataset

class RenewableDataGenerator:
    """Generator for synthetic renewable energy data."""
    
    def __init__(self, system_type: str):
        """Initialize the renewable energy data generator.
        
        Args:
            system_type: Type of renewable system ('solar' or 'wind')
        """
        if system_type not in ['solar', 'wind']:
            raise ValueError("system_type must be 'solar' or 'wind'")
        self.system_type = system_type
        
    def generate_solar_data(
        self,
        n_samples: int,
        timestamp: Optional[datetime] = None
    ) -> List[RenewableData]:
        """Generate synthetic solar data.
        
        Args:
            n_samples: Number of samples to generate
            timestamp: Base timestamp for the data
            
        Returns:
            List of RenewableData objects
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        dataset = []
        for i in range(n_samples):
            current_time = timestamp + timedelta(minutes=i)
            hour = current_time.hour + current_time.minute / 60
            
            # Generate solar radiation based on time of day
            radiation = self._generate_solar_radiation(hour)
            
            # Generate temperature (daily cycle)
            temperature = self._generate_temperature(hour)
            
            data = RenewableData(
                system_type='solar',
                solar_radiation=radiation,
                temperature=temperature,
                time_of_day=hour,
                timestamp=current_time
            )
            dataset.append(data)
            
        return dataset
        
    def generate_wind_data(
        self,
        n_samples: int,
        timestamp: Optional[datetime] = None
    ) -> List[RenewableData]:
        """Generate synthetic wind data.
        
        Args:
            n_samples: Number of samples to generate
            timestamp: Base timestamp for the data
            
        Returns:
            List of RenewableData objects
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        dataset = []
        for i in range(n_samples):
            # Generate wind speed (Weibull distribution)
            wind_speed = np.random.weibull(2.0) * 5.0
            
            # Generate wind direction (uniform)
            wind_direction = np.random.uniform(0, 360)
            
            # Generate air density (normal distribution)
            air_density = np.random.normal(1.225, 0.05)
            
            data = RenewableData(
                system_type='wind',
                wind_speed=float(wind_speed),
                wind_direction=float(wind_direction),
                air_density=float(air_density),
                timestamp=timestamp + timedelta(minutes=i)
            )
            dataset.append(data)
            
        return dataset
        
    def generate_dataset(
        self,
        n_samples: int,
        timestamp: Optional[datetime] = None
    ) -> List[RenewableData]:
        """Generate a complete dataset.
        
        Args:
            n_samples: Number of samples to generate
            timestamp: Base timestamp for the data
            
        Returns:
            List of RenewableData objects
        """
        if self.system_type == 'solar':
            return self.generate_solar_data(n_samples, timestamp)
        else:
            return self.generate_wind_data(n_samples, timestamp)
            
    def _generate_solar_radiation(self, hour: float) -> float:
        """Generate solar radiation based on time of day.
        
        Args:
            hour: Hour of the day (0-24)
            
        Returns:
            Solar radiation in W/m²
        """
        # Simple model for solar radiation
        peak_hour = 12.0
        max_radiation = 1000.0
        
        # Calculate radiation based on time of day
        if 6 <= hour <= 18:  # Daytime
            radiation = max_radiation * np.exp(-((hour - peak_hour) ** 2) / 8)
        else:
            radiation = 0.0
            
        return float(radiation)
        
    def _generate_temperature(self, hour: float) -> float:
        """Generate temperature based on time of day.
        
        Args:
            hour: Hour of the day (0-24)
            
        Returns:
            Temperature in Celsius
        """
        # Simple model for daily temperature cycle
        base_temp = 20.0
        daily_variation = 10.0
        peak_hour = 14.0
        
        temperature = base_temp + daily_variation * np.sin(
            (hour - peak_hour) * np.pi / 12
        )
        
        return float(temperature) 