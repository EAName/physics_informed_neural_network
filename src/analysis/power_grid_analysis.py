#!/usr/bin/env python3
"""
Power Grid Analysis using Physics-Informed Neural Networks

This script demonstrates the application of Physics-Informed Neural Networks (PINNs)
for power grid analysis, including voltage stability prediction and power flow calculations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, optimizers

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Import project-specific modules
from src.models.power_grid_pinn import PowerGridPINN
from src.utils.data_generator import PowerGridDataGenerator
from src.config.config_manager import ConfigManager
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_power_flow_results

# Configure matplotlib
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

def setup_directories():
    """Create necessary directories for results and logs."""
    directories = ['results', 'logs', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def generate_data():
    """Generate synthetic power grid data for training and testing."""
    # Initialize data generator with realistic parameters
    data_gen = PowerGridDataGenerator(
        bus_count=5,          # Number of buses in the system
        line_count=7,         # Number of transmission lines
        base_mva=100,         # Base MVA for per-unit calculations
        voltage_limits=(0.95, 1.05)  # Voltage limits in per-unit
    )
    
    # Generate training and test datasets
    train_data = data_gen.generate_dataset(n_samples=1000)
    test_data = data_gen.generate_dataset(n_samples=200)
    
    # Display dataset information
    print("Training Data Summary:")
    for key, value in train_data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}, range [{value.min():.3f}, {value.max():.3f}]")
    
    print("\nTest Data Summary:")
    for key, value in test_data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}, range [{value.min():.3f}, {value.max():.3f}]")
    
    return train_data, test_data

def initialize_model():
    """Initialize the Power Grid PINN model with appropriate architecture."""
    # Load configuration
    config = ConfigManager().load_config('config/power_grid_config.yaml')
    
    # Initialize model with configuration parameters
    model = PowerGridPINN(
        layers=config['model']['layers'],
        learning_rate=config['training']['learning_rate'],
        activation=config['model']['activation'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Display model architecture
    print("Model Architecture:")
    model.summary()
    print(f"\nTotal trainable parameters: {model.count_params():,}")
    
    return model, config

def train_model(model, train_data, test_data, config):
    """Train the model using both data and physics-based losses."""
    # Training parameters
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    
    # Create TensorBoard callback for monitoring
    log_dir = f"logs/power_grid_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    # Train model
    history = model.train(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=test_data,
        callbacks=[tensorboard_callback]
    )
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('results/power_grid_training_history.csv', index=False)
    
    return history

def plot_training_history(history):
    """Plot training history with multiple loss components."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(history['total_loss'], label='Training')
    axes[0, 0].plot(history['val_total_loss'], label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Physics loss
    axes[0, 1].plot(history['physics_loss'], label='Training')
    axes[0, 1].plot(history['val_physics_loss'], label='Validation')
    axes[0, 1].set_title('Physics Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Data loss
    axes[1, 0].plot(history['data_loss'], label='Training')
    axes[1, 0].plot(history['val_data_loss'], label='Validation')
    axes[1, 0].set_title('Data Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    
    # Learning rate
    axes[1, 1].plot(history['lr'], label='Learning Rate')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('results/power_grid_training_history.png')
    plt.show()

def evaluate_model(model, test_data):
    """Evaluate the model's performance on the test dataset."""
    # Make predictions
    predictions = model.predict(test_data)
    
    # Calculate metrics
    metrics = calculate_metrics(test_data, predictions)
    
    # Display metrics
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save metrics to file
    pd.DataFrame([metrics]).to_csv('results/power_grid_metrics.csv', index=False)
    
    return predictions, metrics

def plot_predictions_vs_actual(test_data, predictions):
    """Plot predictions against actual values for voltage and power."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Voltage magnitude
    axes[0, 0].scatter(test_data['voltage_magnitude'], predictions['voltage_magnitude'], alpha=0.5)
    axes[0, 0].plot([0.9, 1.1], [0.9, 1.1], 'r--')
    axes[0, 0].set_xlabel('Actual Voltage Magnitude')
    axes[0, 0].set_ylabel('Predicted Voltage Magnitude')
    axes[0, 0].set_title('Voltage Magnitude Prediction')
    
    # Voltage angle
    axes[0, 1].scatter(test_data['voltage_angle'], predictions['voltage_angle'], alpha=0.5)
    axes[0, 1].plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--')
    axes[0, 1].set_xlabel('Actual Voltage Angle')
    axes[0, 1].set_ylabel('Predicted Voltage Angle')
    axes[0, 1].set_title('Voltage Angle Prediction')
    
    # Active power
    axes[1, 0].scatter(test_data['active_power'], predictions['active_power'], alpha=0.5)
    axes[1, 0].plot([-1, 1], [-1, 1], 'r--')
    axes[1, 0].set_xlabel('Actual Active Power')
    axes[1, 0].set_ylabel('Predicted Active Power')
    axes[1, 0].set_title('Active Power Prediction')
    
    # Reactive power
    axes[1, 1].scatter(test_data['reactive_power'], predictions['reactive_power'], alpha=0.5)
    axes[1, 1].plot([-1, 1], [-1, 1], 'r--')
    axes[1, 1].set_xlabel('Actual Reactive Power')
    axes[1, 1].set_ylabel('Predicted Reactive Power')
    axes[1, 1].set_title('Reactive Power Prediction')
    
    plt.tight_layout()
    plt.savefig('results/power_grid_predictions.png')
    plt.show()

def analyze_power_flow(model, test_data):
    """Analyze the power flow results and assess system stability."""
    # Perform power flow analysis
    flow_results = model.analyze_power_flow(test_data)
    
    # Plot power flow results
    plot_power_flow_results(flow_results)
    
    # Save results
    pd.DataFrame(flow_results).to_csv('results/power_flow_analysis.csv', index=False)
    
    return flow_results

def save_results(model, config, history):
    """Save the trained model and all results for future use."""
    # Save model
    model.save('models/power_grid_model.h5')
    print("Model saved successfully!")
    
    # Save configuration
    config.save('results/power_grid_config.yaml')
    print("Configuration saved successfully!")
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Total epochs: {config['training']['epochs']}")
    print(f"Final total loss: {history['total_loss'][-1]:.4f}")
    print(f"Final physics loss: {history['physics_loss'][-1]:.4f}")
    print(f"Final data loss: {history['data_loss'][-1]:.4f}")

def main():
    """Main function to run the power grid analysis."""
    # Setup directories
    setup_directories()
    
    # Generate data
    train_data, test_data = generate_data()
    
    # Initialize model
    model, config = initialize_model()
    
    # Train model
    history = train_model(model, train_data, test_data, config)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    predictions, metrics = evaluate_model(model, test_data)
    
    # Plot predictions
    plot_predictions_vs_actual(test_data, predictions)
    
    # Analyze power flow
    flow_results = analyze_power_flow(model, test_data)
    
    # Save results
    save_results(model, config, history)

if __name__ == "__main__":
    main() 