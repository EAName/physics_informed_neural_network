import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from ..schemas.data_schema import (
    ModelConfig,
    TrainingConfig,
    GridConfig,
    RenewableConfig
)

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Any] = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a configuration file.
        
        Args:
            config_name: Name of the configuration file (without extension)
            
        Returns:
            Dictionary containing the configuration
        """
        if config_name in self._configs:
            return self._configs[config_name]
            
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self._configs[config_name] = config
        return config
        
    def get_model_config(self, config_name: str) -> ModelConfig:
        """Get and validate model configuration.
        
        Args:
            config_name: Name of the configuration file
            
        Returns:
            Validated ModelConfig object
        """
        config = self.load_config(config_name)
        return ModelConfig(**config['model_architecture'])
        
    def get_training_config(self, config_name: str) -> TrainingConfig:
        """Get and validate training configuration.
        
        Args:
            config_name: Name of the configuration file
            
        Returns:
            Validated TrainingConfig object
        """
        config = self.load_config(config_name)
        return TrainingConfig(**config['training_parameters'])
        
    def get_grid_config(self, config_name: str) -> GridConfig:
        """Get and validate grid configuration.
        
        Args:
            config_name: Name of the configuration file
            
        Returns:
            Validated GridConfig object
        """
        config = self.load_config(config_name)
        return GridConfig(**config['system_parameters'])
        
    def get_renewable_config(self, config_name: str) -> RenewableConfig:
        """Get and validate renewable energy configuration.
        
        Args:
            config_name: Name of the configuration file
            
        Returns:
            Validated RenewableConfig object
        """
        config = self.load_config(config_name)
        return RenewableConfig(**config['system_parameters'])
        
    def save_config(self, config_name: str, config: Dict[str, Any]) -> None:
        """Save a configuration to file.
        
        Args:
            config_name: Name of the configuration file
            config: Configuration dictionary to save
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        self._configs[config_name] = config
        
    def update_config(self, config_name: str, updates: Dict[str, Any]) -> None:
        """Update an existing configuration.
        
        Args:
            config_name: Name of the configuration file
            updates: Dictionary containing updates to apply
        """
        config = self.load_config(config_name)
        self._update_dict(config, updates)
        self.save_config(config_name, config)
        
    def _update_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """Recursively update a dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary containing updates
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_dict(d[k], v)
            else:
                d[k] = v
                
    def validate_configs(self) -> bool:
        """Validate all loaded configurations.
        
        Returns:
            True if all configurations are valid
        """
        try:
            for config_name in self._configs:
                if 'model_architecture' in self._configs[config_name]:
                    self.get_model_config(config_name)
                if 'training_parameters' in self._configs[config_name]:
                    self.get_training_config(config_name)
                if 'system_parameters' in self._configs[config_name]:
                    if 'bus_count' in self._configs[config_name]['system_parameters']:
                        self.get_grid_config(config_name)
                    else:
                        self.get_renewable_config(config_name)
            return True
        except Exception as e:
            print(f"Configuration validation failed: {str(e)}")
            return False 