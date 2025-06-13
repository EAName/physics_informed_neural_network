import argparse
import sys
from pathlib import Path
from typing import Optional

from ..config.config_manager import ConfigManager
from ..utils.logger import Logger
from ..models.power_grid_pinn import PowerGridPINN
from ..models.renewable_pinn import RenewablePINN

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PINN model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["power_grid", "renewable"],
        required=True,
        help="Type of model to train"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    return parser.parse_args()

def setup_logging(log_level: str) -> Logger:
    """Set up logging.
    
    Args:
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    return Logger(
        name="pinn_train",
        level=getattr(logging, log_level),
        console=True,
        file=True
    )

def train_power_grid(
    config_path: str,
    output_dir: str,
    logger: Logger
) -> None:
    """Train power grid PINN model.
    
    Args:
        config_path: Path to configuration file
        output_dir: Directory to save model
        logger: Logger instance
    """
    try:
        config_manager = ConfigManager()
        model_config = config_manager.get_model_config(config_path)
        training_config = config_manager.get_training_config(config_path)
        grid_config = config_manager.get_grid_config(config_path)
        
        logger.info("Initializing Power Grid PINN model")
        model = PowerGridPINN(
            layers=model_config.layers,
            activation=model_config.activation,
            bus_count=grid_config.bus_count,
            line_count=grid_config.line_count
        )
        
        logger.info("Starting training")
        model.train(
            epochs=training_config.epochs,
            physics_weight=training_config.physics_weight,
            data_weight=training_config.data_weight
        )
        
        output_path = Path(output_dir) / "power_grid_model.h5"
        logger.info(f"Saving model to {output_path}")
        model.save(output_path)
        
    except Exception as e:
        logger.exception("Error during power grid model training")
        raise

def train_renewable(
    config_path: str,
    output_dir: str,
    logger: Logger
) -> None:
    """Train renewable energy PINN model.
    
    Args:
        config_path: Path to configuration file
        output_dir: Directory to save model
        logger: Logger instance
    """
    try:
        config_manager = ConfigManager()
        model_config = config_manager.get_model_config(config_path)
        training_config = config_manager.get_training_config(config_path)
        renewable_config = config_manager.get_renewable_config(config_path)
        
        logger.info("Initializing Renewable Energy PINN model")
        model = RenewablePINN(
            layers=model_config.layers,
            activation=model_config.activation,
            system_type=renewable_config.system_type
        )
        
        logger.info("Starting training")
        model.train(
            epochs=training_config.epochs,
            physics_weight=training_config.physics_weight,
            data_weight=training_config.data_weight
        )
        
        output_path = Path(output_dir) / f"{renewable_config.system_type}_model.h5"
        logger.info(f"Saving model to {output_path}")
        model.save(output_path)
        
    except Exception as e:
        logger.exception("Error during renewable energy model training")
        raise

def main():
    args = parse_args()
    logger = setup_logging(args.log_level)
    
    try:
        if args.model_type == "power_grid":
            train_power_grid(args.config, args.output_dir, logger)
        else:
            train_renewable(args.config, args.output_dir, logger)
            
        logger.info("Training completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 