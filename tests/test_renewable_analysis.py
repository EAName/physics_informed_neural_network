#!/usr/bin/env python3
"""
Tests for renewable energy analysis module.

This module contains test cases for the renewable energy analysis functionality,
including data generation, model initialization, training, and evaluation for both
solar and wind systems.
"""

import unittest
import numpy as np
import tensorflow as tf
from src.analysis.renewable_analysis import (
    setup_directories,
    generate_data,
    initialize_models,
    train_models,
    evaluate_models,
    generate_forecasts
)

class TestRenewableAnalysis(unittest.TestCase):
    """Test cases for renewable energy analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Create test directories
        setup_directories()

    def test_setup_directories(self):
        """Test directory creation."""
        import os
        directories = ['results', 'logs', 'models']
        for directory in directories:
            self.assertTrue(os.path.exists(directory))
            self.assertTrue(os.path.isdir(directory))

    def test_generate_data(self):
        """Test data generation functionality."""
        solar_data, wind_data = generate_data()
        
        # Check solar data structure
        solar_train, solar_test = solar_data
        required_solar_keys = ['solar_radiation', 'temperature', 'power_output']
        
        for data in [solar_train, solar_test]:
            for key in required_solar_keys:
                self.assertIn(key, data)
                self.assertIsInstance(data[key], np.ndarray)
        
        # Check wind data structure
        wind_train, wind_test = wind_data
        required_wind_keys = ['wind_speed', 'wind_direction', 'power_output']
        
        for data in [wind_train, wind_test]:
            for key in required_wind_keys:
                self.assertIn(key, data)
                self.assertIsInstance(data[key], np.ndarray)
        
        # Check data shapes
        self.assertEqual(solar_train['power_output'].shape[0], 1000)
        self.assertEqual(solar_test['power_output'].shape[0], 200)
        self.assertEqual(wind_train['power_output'].shape[0], 1000)
        self.assertEqual(wind_test['power_output'].shape[0], 200)

    def test_initialize_models(self):
        """Test model initialization."""
        solar_model, wind_model, config = initialize_models()
        
        # Check model structures
        self.assertIsNotNone(solar_model)
        self.assertIsNotNone(wind_model)
        self.assertIsNotNone(config)
        
        # Check configuration
        required_config_keys = ['model', 'training']
        for key in required_config_keys:
            self.assertIn(key, config)

    def test_train_models(self):
        """Test model training."""
        # Generate small dataset for quick testing
        solar_data, wind_data = generate_data()
        solar_model, wind_model, config = initialize_models()
        
        # Modify config for quick testing
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 32
        
        # Train models
        solar_history, wind_history = train_models(
            solar_model, wind_model, solar_data, wind_data, config
        )
        
        # Check training histories
        required_history_keys = ['total_loss', 'physics_loss', 'data_loss']
        for history in [solar_history, wind_history]:
            for key in required_history_keys:
                self.assertIn(key, history)
                self.assertIsInstance(history[key], list)
                self.assertEqual(len(history[key]), 2)  # 2 epochs

    def test_evaluate_models(self):
        """Test model evaluation."""
        # Generate test data
        solar_data, wind_data = generate_data()
        solar_model, wind_model, _ = initialize_models()
        
        # Evaluate models
        solar_results, wind_results = evaluate_models(
            solar_model, wind_model, solar_data[1], wind_data[1]
        )
        
        # Check predictions and metrics
        for results in [solar_results, wind_results]:
            predictions, metrics = results
            
            # Check predictions
            self.assertIsNotNone(predictions)
            self.assertIsInstance(predictions, dict)
            
            # Check metrics
            required_metrics = ['mse', 'mae', 'r2_score']
            for metric in required_metrics:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], float)

    def test_generate_forecasts(self):
        """Test forecast generation."""
        solar_model, wind_model, _ = initialize_models()
        
        # Generate forecasts
        solar_forecast, wind_forecast = generate_forecasts(solar_model, wind_model)
        
        # Check solar forecast
        solar_data, solar_pred = solar_forecast
        self.assertIn('timestamp', solar_data)
        self.assertIn('power_output', solar_pred)
        self.assertIn('uncertainty', solar_pred)
        
        # Check wind forecast
        wind_data, wind_pred = wind_forecast
        self.assertIn('timestamp', wind_data)
        self.assertIn('power_output', wind_pred)
        self.assertIn('uncertainty', wind_pred)
        
        # Check forecast length
        self.assertEqual(len(solar_data['timestamp']), 24)  # 24 hours
        self.assertEqual(len(wind_data['timestamp']), 24)  # 24 hours

if __name__ == '__main__':
    unittest.main() 