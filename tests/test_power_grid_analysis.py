#!/usr/bin/env python3
"""
Tests for power grid analysis module.

This module contains test cases for the power grid analysis functionality,
including data generation, model initialization, training, and evaluation.
"""

import unittest
import numpy as np
import tensorflow as tf
from src.analysis.power_grid_analysis import (
    setup_directories,
    generate_data,
    initialize_model,
    train_model,
    evaluate_model,
    analyze_power_flow
)

class TestPowerGridAnalysis(unittest.TestCase):
    """Test cases for power grid analysis functionality."""

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
        train_data, test_data = generate_data()
        
        # Check data structure
        required_keys = ['voltage_magnitude', 'voltage_angle', 
                        'active_power', 'reactive_power']
        
        for data in [train_data, test_data]:
            for key in required_keys:
                self.assertIn(key, data)
                self.assertIsInstance(data[key], np.ndarray)
        
        # Check data shapes
        self.assertEqual(train_data['voltage_magnitude'].shape[0], 1000)
        self.assertEqual(test_data['voltage_magnitude'].shape[0], 200)
        
        # Check value ranges
        self.assertTrue(np.all(train_data['voltage_magnitude'] >= 0.95))
        self.assertTrue(np.all(train_data['voltage_magnitude'] <= 1.05))

    def test_initialize_model(self):
        """Test model initialization."""
        model, config = initialize_model()
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertIsNotNone(config)
        
        # Check configuration
        required_config_keys = ['model', 'training']
        for key in required_config_keys:
            self.assertIn(key, config)

    def test_train_model(self):
        """Test model training."""
        # Generate small dataset for quick testing
        train_data, test_data = generate_data()
        model, config = initialize_model()
        
        # Modify config for quick testing
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 32
        
        # Train model
        history = train_model(model, train_data, test_data, config)
        
        # Check training history
        required_history_keys = ['total_loss', 'physics_loss', 'data_loss']
        for key in required_history_keys:
            self.assertIn(key, history)
            self.assertIsInstance(history[key], list)
            self.assertEqual(len(history[key]), 2)  # 2 epochs

    def test_evaluate_model(self):
        """Test model evaluation."""
        # Generate test data
        _, test_data = generate_data()
        model, _ = initialize_model()
        
        # Evaluate model
        predictions, metrics = evaluate_model(model, test_data)
        
        # Check predictions
        self.assertIsNotNone(predictions)
        self.assertIsInstance(predictions, dict)
        
        # Check metrics
        required_metrics = ['mse', 'mae', 'r2_score']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)

    def test_analyze_power_flow(self):
        """Test power flow analysis."""
        # Generate test data
        _, test_data = generate_data()
        model, _ = initialize_model()
        
        # Analyze power flow
        flow_results = analyze_power_flow(model, test_data)
        
        # Check results
        self.assertIsNotNone(flow_results)
        self.assertIsInstance(flow_results, dict)
        
        # Check required keys
        required_keys = ['voltage_stability', 'power_flow', 'line_loading']
        for key in required_keys:
            self.assertIn(key, flow_results)

if __name__ == '__main__':
    unittest.main() 