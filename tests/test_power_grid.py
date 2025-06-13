import unittest
import tensorflow as tf
import numpy as np
from src.models.power_grid_pinn import PowerGridPINN

class TestPowerGridPINN(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.layers = [4, 32, 32, 2]  # Input: [v_mag, v_ang, p_inj, q_inj], Output: [v_mag, v_ang]
        self.bus_count = 14
        self.line_count = 20
        self.model = PowerGridPINN(self.layers, self.bus_count, self.line_count)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def test_model_initialization(self):
        """Test if the model is initialized correctly."""
        self.assertEqual(self.model.bus_count, self.bus_count)
        self.assertEqual(self.model.line_count, self.line_count)
        self.assertEqual(len(self.model.hidden_layers), len(self.layers) - 2)
    
    def test_physics_loss_computation(self):
        """Test if the physics loss computation works correctly."""
        batch_size = 4
        # Create input tensor with voltage magnitudes, angles, and power injections
        v_mag = tf.random.uniform((batch_size, self.bus_count), 0.9, 1.1)
        v_ang = tf.random.uniform((batch_size, self.bus_count), -np.pi, np.pi)
        p_inj = tf.random.uniform((batch_size, self.bus_count), -1.0, 1.0)
        q_inj = tf.random.uniform((batch_size, self.bus_count), -0.5, 0.5)
        
        inputs = tf.concat([v_mag, v_ang, p_inj, q_inj], axis=1)
        
        loss = self.model.compute_physics_loss(inputs)
        self.assertIsInstance(loss, tf.Tensor)
        self.assertGreaterEqual(loss, 0)
    
    def test_voltage_constraints(self):
        """Test if voltage constraints are enforced correctly."""
        batch_size = 4
        # Create input with voltage magnitudes outside limits
        v_mag = tf.random.uniform((batch_size, self.bus_count), 0.8, 1.2)
        v_ang = tf.random.uniform((batch_size, self.bus_count), -np.pi, np.pi)
        p_inj = tf.random.uniform((batch_size, self.bus_count), -1.0, 1.0)
        q_inj = tf.random.uniform((batch_size, self.bus_count), -0.5, 0.5)
        
        inputs = tf.concat([v_mag, v_ang, p_inj, q_inj], axis=1)
        
        loss = self.model.compute_physics_loss(inputs)
        self.assertGreater(loss, 0)  # Should be positive for violations
    
    def test_power_flow_equations(self):
        """Test if power flow equations are satisfied."""
        batch_size = 4
        # Create input with balanced power injections
        v_mag = tf.ones((batch_size, self.bus_count))
        v_ang = tf.zeros((batch_size, self.bus_count))
        p_inj = tf.zeros((batch_size, self.bus_count))
        q_inj = tf.zeros((batch_size, self.bus_count))
        
        inputs = tf.concat([v_mag, v_ang, p_inj, q_inj], axis=1)
        
        loss = self.model.compute_physics_loss(inputs)
        self.assertLess(loss, 1e-6)  # Should be close to zero for balanced case
    
    def test_grid_state_prediction(self):
        """Test if grid state prediction works correctly."""
        batch_size = 4
        # Create power injection inputs
        p_inj = tf.random.uniform((batch_size, self.bus_count), -1.0, 1.0)
        q_inj = tf.random.uniform((batch_size, self.bus_count), -0.5, 0.5)
        
        grid_state = self.model.predict_grid_state(tf.concat([p_inj, q_inj], axis=1))
        
        self.assertIn('voltage_magnitudes', grid_state)
        self.assertIn('voltage_angles', grid_state)
        self.assertEqual(grid_state['voltage_magnitudes'].shape, (batch_size, self.bus_count))
        self.assertEqual(grid_state['voltage_angles'].shape, (batch_size, self.bus_count))
    
    def test_training_step(self):
        """Test if the training step works correctly."""
        batch_size = 4
        # Create input data
        v_mag = tf.random.uniform((batch_size, self.bus_count), 0.9, 1.1)
        v_ang = tf.random.uniform((batch_size, self.bus_count), -np.pi, np.pi)
        p_inj = tf.random.uniform((batch_size, self.bus_count), -1.0, 1.0)
        q_inj = tf.random.uniform((batch_size, self.bus_count), -0.5, 0.5)
        
        inputs = tf.concat([v_mag, v_ang, p_inj, q_inj], axis=1)
        
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
        self.model.save_weights('test_power_grid_weights')
        
        # Create new model with same architecture
        new_model = PowerGridPINN(self.layers, self.bus_count, self.line_count)
        
        # Load weights
        new_model.load_weights('test_power_grid_weights')
        
        # Test if weights are the same
        batch_size = 4
        inputs = tf.random.normal((batch_size, self.layers[0]))
        output1 = self.model(inputs)
        output2 = new_model(inputs)
        
        np.testing.assert_array_almost_equal(output1.numpy(), output2.numpy())

if __name__ == '__main__':
    unittest.main() 