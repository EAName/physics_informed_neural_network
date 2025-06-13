import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import tensorflow as tf

from ..config.config_manager import ConfigManager
from ..utils.logger import Logger
from ..models.power_grid_pinn import PowerGridPINN
from ..models.renewable_pinn import RenewablePINN
from ..schemas.data_schema import PowerGridData, RenewableData, PredictionResult

def parse_args():
    parser = argparse.ArgumentParser(description="Make predictions using a trained PINN model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["power_grid", "renewable"],
        required=True,
        help="Type of model to use"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to input data file (JSON)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save predictions (JSON)"
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
        name="pinn_predict",
        level=getattr(logging, log_level),
        console=True,
        file=True
    )

def load_input_data(file_path: str, model_type: str) -> Dict[str, Any]:
    """Load and validate input data.
    
    Args:
        file_path: Path to input data file
        model_type: Type of model
        
    Returns:
        Validated input data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    if model_type == "power_grid":
        return PowerGridData(**data).dict()
    else:
        return RenewableData(**data).dict()

def predict_power_grid(
    model_path: str,
    input_data: Dict[str, Any],
    logger: Logger
) -> PredictionResult:
    """Make predictions using power grid model.
    
    Args:
        model_path: Path to trained model
        input_data: Input data
        logger: Logger instance
        
    Returns:
        Prediction results
    """
    try:
        logger.info("Loading Power Grid PINN model")
        model = PowerGridPINN.load(model_path)
        
        # Prepare input data
        v_mag = tf.convert_to_tensor(input_data["voltage_magnitudes"], dtype=tf.float32)
        v_ang = tf.convert_to_tensor(input_data["voltage_angles"], dtype=tf.float32)
        p_inj = tf.convert_to_tensor(input_data["active_power"], dtype=tf.float32)
        q_inj = tf.convert_to_tensor(input_data["reactive_power"], dtype=tf.float32)
        
        logger.info("Making predictions")
        predictions = model.predict_grid_state(v_mag, v_ang, p_inj, q_inj)
        
        # Calculate confidence scores (placeholder)
        confidence = [0.95] * len(predictions["voltage_magnitudes"])
        
        return PredictionResult(
            predictions=predictions["voltage_magnitudes"].tolist(),
            confidence=confidence
        )
        
    except Exception as e:
        logger.exception("Error during power grid prediction")
        raise

def predict_renewable(
    model_path: str,
    input_data: Dict[str, Any],
    logger: Logger
) -> PredictionResult:
    """Make predictions using renewable energy model.
    
    Args:
        model_path: Path to trained model
        input_data: Input data
        logger: Logger instance
        
    Returns:
        Prediction results
    """
    try:
        logger.info("Loading Renewable Energy PINN model")
        model = RenewablePINN.load(model_path)
        
        # Prepare input data based on system type
        if input_data["system_type"] == "solar":
            inputs = tf.convert_to_tensor([
                input_data["solar_radiation"],
                input_data["temperature"],
                input_data["time_of_day"]
            ], dtype=tf.float32)
        else:
            inputs = tf.convert_to_tensor([
                input_data["wind_speed"],
                input_data["wind_direction"],
                input_data["air_density"]
            ], dtype=tf.float32)
            
        logger.info("Making predictions")
        predictions = model.predict_power_generation(inputs)
        
        # Calculate confidence scores (placeholder)
        confidence = [0.95] * len(predictions)
        
        return PredictionResult(
            predictions=predictions.tolist(),
            confidence=confidence
        )
        
    except Exception as e:
        logger.exception("Error during renewable energy prediction")
        raise

def save_predictions(predictions: PredictionResult, output_file: str) -> None:
    """Save predictions to file.
    
    Args:
        predictions: Prediction results
        output_file: Path to save predictions
    """
    with open(output_file, 'w') as f:
        json.dump(predictions.dict(), f, indent=2)

def main():
    args = parse_args()
    logger = setup_logging(args.log_level)
    
    try:
        # Load and validate input data
        input_data = load_input_data(args.input_data, args.model_type)
        
        # Make predictions
        if args.model_type == "power_grid":
            predictions = predict_power_grid(args.model_path, input_data, logger)
        else:
            predictions = predict_renewable(args.model_path, input_data, logger)
            
        # Save predictions
        save_predictions(predictions, args.output_file)
        logger.info(f"Predictions saved to {args.output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 