import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..utils.data_generator import PowerGridDataGenerator, RenewableDataGenerator
from ..utils.logger import Logger
from ..schemas.data_schema import PowerGridData, RenewableData

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic data for PINN models")
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["power_grid", "renewable"],
        required=True,
        help="Type of data to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save generated data"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--system-type",
        type=str,
        choices=["solar", "wind"],
        help="Type of renewable system (required for renewable data)"
    )
    parser.add_argument(
        "--bus-count",
        type=int,
        default=14,
        help="Number of buses (for power grid data)"
    )
    parser.add_argument(
        "--line-count",
        type=int,
        default=20,
        help="Number of transmission lines (for power grid data)"
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
        name="data_generator",
        level=getattr(logging, log_level),
        console=True,
        file=True
    )

def save_dataset(
    dataset: List[Dict[str, Any]],
    output_dir: str,
    data_type: str,
    system_type: Optional[str] = None
) -> None:
    """Save generated dataset to file.
    
    Args:
        dataset: List of data points
        output_dir: Directory to save data
        data_type: Type of data
        system_type: Type of renewable system (if applicable)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if data_type == "power_grid":
        filename = f"power_grid_data_{timestamp}.json"
    else:
        filename = f"{system_type}_data_{timestamp}.json"
        
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump([data.dict() for data in dataset], f, indent=2, default=str)
        
    return output_path

def generate_power_grid_data(
    n_samples: int,
    bus_count: int,
    line_count: int,
    logger: Logger
) -> List[PowerGridData]:
    """Generate power grid data.
    
    Args:
        n_samples: Number of samples
        bus_count: Number of buses
        line_count: Number of transmission lines
        logger: Logger instance
        
    Returns:
        List of PowerGridData objects
    """
    logger.info(f"Generating power grid data with {n_samples} samples")
    generator = PowerGridDataGenerator(
        bus_count=bus_count,
        line_count=line_count
    )
    return generator.generate_dataset(n_samples)

def generate_renewable_data(
    n_samples: int,
    system_type: str,
    logger: Logger
) -> List[RenewableData]:
    """Generate renewable energy data.
    
    Args:
        n_samples: Number of samples
        system_type: Type of renewable system
        logger: Logger instance
        
    Returns:
        List of RenewableData objects
    """
    logger.info(f"Generating {system_type} data with {n_samples} samples")
    generator = RenewableDataGenerator(system_type=system_type)
    return generator.generate_dataset(n_samples)

def main():
    args = parse_args()
    logger = setup_logging(args.log_level)
    
    try:
        if args.data_type == "power_grid":
            dataset = generate_power_grid_data(
                args.n_samples,
                args.bus_count,
                args.line_count,
                logger
            )
            output_path = save_dataset(
                dataset,
                args.output_dir,
                args.data_type
            )
        else:
            if not args.system_type:
                logger.error("system_type is required for renewable data")
                return 1
                
            dataset = generate_renewable_data(
                args.n_samples,
                args.system_type,
                logger
            )
            output_path = save_dataset(
                dataset,
                args.output_dir,
                args.data_type,
                args.system_type
            )
            
        logger.info(f"Data saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.exception("Error generating data")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 