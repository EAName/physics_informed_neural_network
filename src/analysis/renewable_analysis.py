#!/usr/bin/env python3
"""
Renewable Energy Analysis using Physics-Informed Neural Networks

This script demonstrates the application of Physics-Informed Neural Networks (PINNs)
for renewable energy analysis, including solar and wind power prediction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras import layers, optimizers

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Import project-specific modules
from src.models.renewable_pinn import RenewablePINN
from src.utils.data_generator import RenewableDataGenerator
from src.config.config_manager import ConfigManager
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_renewable_results

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
    """Generate synthetic renewable energy data for training and testing."""
    # Initialize data generator for solar system
    solar_gen = RenewableDataGenerator(system_type='solar')
    
    # Generate training and test datasets
    solar_train_data = solar_gen.generate_dataset(n_samples=1000)
    solar_test_data = solar_gen.generate_dataset(n_samples=200)
    
    # Initialize data generator for wind system
    wind_gen = RenewableDataGenerator(system_type='wind')
    
    # Generate training and test datasets
    wind_train_data = wind_gen.generate_dataset(n_samples=1000)
    wind_test_data = wind_gen.generate_dataset(n_samples=200)
    
    # Display dataset information
    print("Solar Data Summary:")
    for key, value in solar_train_data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}, range [{value.min():.3f}, {value.max():.3f}]")
    
    print("\nWind Data Summary:")
    for key, value in wind_train_data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}, range [{value.min():.3f}, {value.max():.3f}]")
    
    return (solar_train_data, solar_test_data), (wind_train_data, wind_test_data)

def initialize_models():
    """Initialize the Renewable Energy PINN models for both solar and wind systems."""
    # Load configuration
    config = ConfigManager().load_config('config/renewable_config.yaml')
    
    # Initialize solar model
    solar_model = RenewablePINN(
        system_type='solar',
        layers=config['model']['layers'],
        learning_rate=config['training']['learning_rate'],
        activation=config['model']['activation'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Initialize wind model
    wind_model = RenewablePINN(
        system_type='wind',
        layers=config['model']['layers'],
        learning_rate=config['training']['learning_rate'],
        activation=config['model']['activation'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Display model architectures
    print("Solar Model Architecture:")
    solar_model.summary()
    print(f"\nTotal trainable parameters: {solar_model.count_params():,}")
    
    print("\nWind Model Architecture:")
    wind_model.summary()
    print(f"\nTotal trainable parameters: {wind_model.count_params():,}")
    
    return solar_model, wind_model, config

def train_models(solar_model, wind_model, solar_data, wind_data, config):
    """Train both models using data and physics-based losses."""
    # Training parameters
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    
    # Create TensorBoard callbacks
    solar_log_dir = f"logs/solar_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wind_log_dir = f"logs/wind_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    solar_tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=solar_log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    wind_tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=wind_log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    # Train solar model
    print("Training Solar Model...")
    solar_history = solar_model.train(
        solar_data[0],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=solar_data[1],
        callbacks=[solar_tensorboard]
    )
    
    # Train wind model
    print("\nTraining Wind Model...")
    wind_history = wind_model.train(
        wind_data[0],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=wind_data[1],
        callbacks=[wind_tensorboard]
    )
    
    # Save training histories
    pd.DataFrame(solar_history).to_csv('results/solar_training_history.csv', index=False)
    pd.DataFrame(wind_history).to_csv('results/wind_training_history.csv', index=False)
    
    return solar_history, wind_history

def plot_training_history(history, title):
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
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'results/{title.lower().replace(" ", "_")}_training_history.png')
    plt.show()

def evaluate_models(solar_model, wind_model, solar_test_data, wind_test_data):
    """Evaluate both models' performance on the test datasets."""
    # Make predictions
    solar_predictions = solar_model.predict(solar_test_data)
    wind_predictions = wind_model.predict(wind_test_data)
    
    # Calculate metrics
    solar_metrics = calculate_metrics(solar_test_data, solar_predictions)
    wind_metrics = calculate_metrics(wind_test_data, wind_predictions)
    
    # Display metrics
    print("Solar Model Performance Metrics:")
    for metric, value in solar_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nWind Model Performance Metrics:")
    for metric, value in wind_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save metrics to files
    pd.DataFrame([solar_metrics]).to_csv('results/solar_metrics.csv', index=False)
    pd.DataFrame([wind_metrics]).to_csv('results/wind_metrics.csv', index=False)
    
    return (solar_predictions, solar_metrics), (wind_predictions, wind_metrics)

def plot_predictions_vs_actual(test_data, predictions, title):
    """Plot predictions against actual values for power output."""
    plt.figure(figsize=(10, 6))
    
    # Plot actual vs predicted power output
    plt.scatter(test_data['power_output'], predictions['power_output'], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Power Output')
    plt.ylabel('Predicted Power Output')
    plt.title(f'{title} - Power Output Prediction')
    
    plt.tight_layout()
    plt.savefig(f'results/{title.lower().replace(" ", "_")}_predictions.png')
    plt.show()

def generate_forecasts(solar_model, wind_model):
    """Generate forecasts for both solar and wind power output."""
    # Initialize data generators
    solar_gen = RenewableDataGenerator(system_type='solar')
    wind_gen = RenewableDataGenerator(system_type='wind')
    
    # Generate forecast data
    forecast_hours = 24
    solar_forecast_data = solar_gen.generate_forecast_data(hours=forecast_hours)
    wind_forecast_data = wind_gen.generate_forecast_data(hours=forecast_hours)
    
    # Make forecasts
    solar_forecast = solar_model.predict(solar_forecast_data)
    wind_forecast = wind_model.predict(wind_forecast_data)
    
    return (solar_forecast_data, solar_forecast), (wind_forecast_data, wind_forecast)

def plot_forecast(forecast_data, forecast, title):
    """Plot power output forecast."""
    plt.figure(figsize=(12, 6))
    
    # Plot forecast
    plt.plot(forecast_data['timestamp'], forecast['power_output'], 'b-', label='Forecast')
    plt.fill_between(forecast_data['timestamp'], 
                    forecast['power_output'] - forecast['uncertainty'],
                    forecast['power_output'] + forecast['uncertainty'],
                    alpha=0.2, color='b')
    
    plt.xlabel('Time')
    plt.ylabel('Power Output')
    plt.title(f'{title} - 24-Hour Forecast')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/{title.lower().replace(" ", "_")}_forecast.png')
    plt.show()

def save_results(solar_model, wind_model, config, solar_history, wind_history):
    """Save the trained models and all results for future use."""
    # Save models
    solar_model.save('models/solar_model.h5')
    wind_model.save('models/wind_model.h5')
    print("Models saved successfully!")
    
    # Save configuration
    config.save('results/renewable_config.yaml')
    print("Configuration saved successfully!")
    
    # Print summary
    print("\nTraining Summary:")
    print("Solar Model:")
    print(f"Total epochs: {config['training']['epochs']}")
    print(f"Final total loss: {solar_history['total_loss'][-1]:.4f}")
    print(f"Final physics loss: {solar_history['physics_loss'][-1]:.4f}")
    print(f"Final data loss: {solar_history['data_loss'][-1]:.4f}")
    
    print("\nWind Model:")
    print(f"Total epochs: {config['training']['epochs']}")
    print(f"Final total loss: {wind_history['total_loss'][-1]:.4f}")
    print(f"Final physics loss: {wind_history['physics_loss'][-1]:.4f}")
    print(f"Final data loss: {wind_history['data_loss'][-1]:.4f}")

def main():
    """Main function to run the renewable energy analysis."""
    # Setup directories
    setup_directories()
    
    # Generate data
    solar_data, wind_data = generate_data()
    
    # Initialize models
    solar_model, wind_model, config = initialize_models()
    
    # Train models
    solar_history, wind_history = train_models(solar_model, wind_model, solar_data, wind_data, config)
    
    # Plot training histories
    plot_training_history(solar_history, 'Solar Model Training')
    plot_training_history(wind_history, 'Wind Model Training')
    
    # Evaluate models
    solar_results, wind_results = evaluate_models(solar_model, wind_model, solar_data[1], wind_data[1])
    
    # Plot predictions
    plot_predictions_vs_actual(solar_data[1], solar_results[0], 'Solar Model')
    plot_predictions_vs_actual(wind_data[1], wind_results[0], 'Wind Model')
    
    # Generate and plot forecasts
    solar_forecast, wind_forecast = generate_forecasts(solar_model, wind_model)
    plot_forecast(solar_forecast[0], solar_forecast[1], 'Solar')
    plot_forecast(wind_forecast[0], wind_forecast[1], 'Wind')
    
    # Save results
    save_results(solar_model, wind_model, config, solar_history, wind_history)

if __name__ == "__main__":
    main() 