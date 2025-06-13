import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple

def plot_grid_state(grid_state: Dict[str, tf.Tensor], bus_count: int) -> None:
    """
    Plot the grid state including voltage magnitudes and angles.
    
    Args:
        grid_state: Dictionary containing voltage magnitudes and angles
        bus_count: Number of buses in the grid
    """
    v_mag = grid_state['voltage_magnitudes'].numpy()
    v_ang = grid_state['voltage_angles'].numpy()
    
    plt.figure(figsize=(12, 5))
    
    # Plot voltage magnitudes
    plt.subplot(1, 2, 1)
    plt.bar(range(bus_count), v_mag[0])
    plt.xlabel('Bus Number')
    plt.ylabel('Voltage Magnitude (p.u.)')
    plt.title('Voltage Magnitudes')
    plt.grid(True)
    
    # Plot voltage angles
    plt.subplot(1, 2, 2)
    plt.bar(range(bus_count), v_ang[0])
    plt.xlabel('Bus Number')
    plt.ylabel('Voltage Angle (radians)')
    plt.title('Voltage Angles')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_power_flow(power_flow: Dict[str, tf.Tensor], line_count: int) -> None:
    """
    Plot the power flow in transmission lines.
    
    Args:
        power_flow: Dictionary containing active and reactive power flows
        line_count: Number of transmission lines
    """
    p_flow = power_flow['active_power'].numpy()
    q_flow = power_flow['reactive_power'].numpy()
    
    plt.figure(figsize=(12, 5))
    
    # Plot active power flow
    plt.subplot(1, 2, 1)
    plt.bar(range(line_count), p_flow[0])
    plt.xlabel('Line Number')
    plt.ylabel('Active Power Flow (p.u.)')
    plt.title('Active Power Flow')
    plt.grid(True)
    
    # Plot reactive power flow
    plt.subplot(1, 2, 2)
    plt.bar(range(line_count), q_flow[0])
    plt.xlabel('Line Number')
    plt.ylabel('Reactive Power Flow (p.u.)')
    plt.title('Reactive Power Flow')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_power_generation(predictions: Dict[str, tf.Tensor], system_type: str) -> None:
    """
    Plot power generation predictions for renewable energy systems.
    
    Args:
        predictions: Dictionary containing power generation and efficiency/capacity factor
        system_type: Type of renewable system ('solar' or 'wind')
    """
    power = predictions['power_generation'].numpy()
    efficiency = predictions.get('efficiency', predictions.get('capacity_factor')).numpy()
    
    plt.figure(figsize=(12, 5))
    
    # Plot power generation
    plt.subplot(1, 2, 1)
    plt.plot(power)
    plt.xlabel('Time Step')
    plt.ylabel('Power Generation (MW)')
    plt.title(f'{system_type.capitalize()} Power Generation')
    plt.grid(True)
    
    # Plot efficiency/capacity factor
    plt.subplot(1, 2, 2)
    plt.plot(efficiency)
    plt.xlabel('Time Step')
    plt.ylabel('Efficiency' if system_type == 'solar' else 'Capacity Factor')
    plt.title(f'{system_type.capitalize()} System Efficiency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_efficiency(efficiency_data: Dict[str, np.ndarray], system_type: str) -> None:
    """
    Plot efficiency analysis for renewable energy systems.
    
    Args:
        efficiency_data: Dictionary containing efficiency data and parameters
        system_type: Type of renewable system ('solar' or 'wind')
    """
    if system_type == 'solar':
        plt.figure(figsize=(10, 8))
        plt.contourf(efficiency_data['temperature'],
                    efficiency_data['radiation'],
                    efficiency_data['efficiency'],
                    levels=20)
        plt.colorbar(label='Efficiency')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Solar Radiation (W/m²)')
        plt.title('Solar Panel Efficiency')
        
    else:  # wind
        plt.figure(figsize=(10, 8))
        plt.contourf(efficiency_data['wind_speed'],
                    efficiency_data['wind_direction'],
                    efficiency_data['capacity_factor'],
                    levels=20)
        plt.colorbar(label='Capacity Factor')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Wind Direction (degrees)')
        plt.title('Wind Turbine Capacity Factor')
    
    plt.show()

def plot_training_history(history: Dict[str, List[float]]) -> None:
    """
    Plot training history including loss and metrics.
    
    Args:
        history: Dictionary containing training metrics
    """
    plt.figure(figsize=(10, 6))
    
    for metric, values in history.items():
        plt.plot(values, label=metric)
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_prediction_vs_actual(predictions: np.ndarray,
                            actual: np.ndarray,
                            title: str = 'Prediction vs Actual') -> None:
    """
    Plot predicted values against actual values.
    
    Args:
        predictions: Array of predicted values
        actual: Array of actual values
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predictions, alpha=0.5)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True)
    plt.show() 