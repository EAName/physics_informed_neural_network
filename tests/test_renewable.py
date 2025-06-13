import unittest
import tensorflow as tf
import numpy as np
from src.models.renewable_pinn import RenewablePINN

class TestRenewablePINN(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.layers = [3, 32, 32, 1]  # Input: [env_params], Output: [power_generation]
        self.model = RenewablePINN(self.layers, system_type='solar')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def test_model_initialization(self):
        """Test if the model is initialized correctly."""
        self.assertEqual(self.model.system_type, 'solar')
        self.assertEqual(len(self.model.hidden_layers), len(self.layers) - 2)
        
        # Test wind system initialization
        wind_model = RenewablePINN(self.layers, system_type='wind')
        self.assertEqual(wind_model.system_type, 'wind')
        
        # Test invalid system type
        with self.assertRaises(ValueError):
            RenewablePINN(self.layers, system_type='invalid')
    
    def test_solar_physics_loss(self):
        """Test if solar physics loss computation works correctly."""
        batch_size = 4
        # Create input tensor with solar parameters
        radiation = tf.random.uniform((batch_size, 1), 0, 1000)  # W/m²
        temperature = tf.random.uniform((batch_size, 1), 0, 50)  # °C
        time = tf.random.uniform((batch_size, 1), 0, 24)  # hours
        
        inputs = tf.concat([radiation, temperature, time], axis=1)
        
        loss = self.model.compute_physics_loss(inputs)
        self.assertIsInstance(loss, tf.Tensor)
        self.assertGreaterEqual(loss, 0)
    
    def test_wind_physics_loss(self):
        """Test if wind physics loss computation works correctly."""
        # Create wind model
        wind_model = RenewablePINN(self.layers, system_type='wind')
        
        batch_size = 4
        # Create input tensor with wind parameters
        speed = tf.random.uniform((batch_size, 1), 0, 30)  # m/s
        direction = tf.random.uniform((batch_size, 1), 0, 360)  # degrees
        density = tf.random.uniform((batch_size, 1), 1.0, 1.3)  # kg/m³
        
        inputs = tf.concat([speed, direction, density], axis=1)
        
        loss = wind_model.compute_physics_loss(inputs)
        self.assertIsInstance(loss, tf.Tensor)
        self.assertGreaterEqual(loss, 0)
    
    def test_solar_constraints(self):
        """Test if solar system constraints are enforced correctly."""
        batch_size = 4
        # Create input with negative radiation (should be penalized)
        radiation = tf.random.uniform((batch_size, 1), -100, 0)
        temperature = tf.random.uniform((batch_size, 1), 0, 50)
        time = tf.random.uniform((batch_size, 1), 0, 24)
        
        inputs = tf.concat([radiation, temperature, time], axis=1)
        
        loss = self.model.compute_physics_loss(inputs)
        self.assertGreater(loss, 0)  # Should be positive for violations
    
    def test_wind_constraints(self):
        """Test if wind system constraints are enforced correctly."""
        wind_model = RenewablePINN(self.layers, system_type='wind')
        
        batch_size = 4
        # Create input with negative wind speed (should be penalized)
        speed = tf.random.uniform((batch_size, 1), -10, 0)
        direction = tf.random.uniform((batch_size, 1), 0, 360)
        density = tf.random.uniform((batch_size, 1), 1.0, 1.3)
        
        inputs = tf.concat([speed, direction, density], axis=1)
        
        loss = wind_model.compute_physics_loss(inputs)
        self.assertGreater(loss, 0)  # Should be positive for violations
    
    def test_power_generation_prediction(self):
        """Test if power generation prediction works correctly."""
        batch_size = 4
        # Create input data
        if self.model.system_type == 'solar':
            inputs = tf.random.uniform((batch_size, 3), 0, 1000)
        else:
            inputs = tf.random.uniform((batch_size, 3), 0, 30)
        
        predictions = self.model.predict_power_generation(inputs)
        
        self.assertIn('power_generation', predictions)
        self.assertIn('efficiency' if self.model.system_type == 'solar' else 'capacity_factor',
                     predictions)
        self.assertEqual(predictions['power_generation'].shape, (batch_size, 1))
    
    def test_system_efficiency(self):
        """Test if system efficiency computation works correctly."""
        batch_size = 4
        # Create input data and actual power
        if self.model.system_type == 'solar':
            inputs = tf.random.uniform((batch_size, 3), 0, 1000)
        else:
            inputs = tf.random.uniform((batch_size, 3), 0, 30)
        
        actual_power = tf.random.uniform((batch_size, 1), 0, 100)
        
        efficiency = self.model.compute_system_efficiency(inputs, actual_power)
        self.assertIsInstance(efficiency, tf.Tensor)
        self.assertGreaterEqual(efficiency, 0)
    
    def test_training_step(self):
        """Test if the training step works correctly."""
        batch_size = 4
        # Create input data
        if self.model.system_type == 'solar':
            inputs = tf.random.uniform((batch_size, 3), 0, 1000)
        else:
            inputs = tf.random.uniform((batch_size, 3), 0, 30)
        
        # Perform training step
        total_loss, physics_loss, data_loss = self.model.train_step(
            inputs, inputs, inputs, self.optimizer
        )
        
        self.assertIsInstance(total_loss, tf.Tensor)
        self.assertIsInstance(physics_loss, tf.Tensor)
        self.assertIsInstance(data_loss, tf.Tensor)
        self.assertGreaterEqual(total_loss, 0)
    
    def test_model_save_load(self):
        """Test if the model can be saved and loaded correctly."""
        # Save model
        self.model.save_weights('test_renewable_weights')
        
        # Create new model with same architecture
        new_model = RenewablePINN(self.layers, system_type=self.model.system_type)
        
        # Load weights
        new_model.load_weights('test_renewable_weights')
        
        # Test if weights are the same
        batch_size = 4
        inputs = tf.random.normal((batch_size, self.layers[0]))
        output1 = self.model(inputs)
        output2 = new_model(inputs)
        
        np.testing.assert_array_almost_equal(output1.numpy(), output2.numpy())

if __name__ == '__main__':
    unittest.main() 