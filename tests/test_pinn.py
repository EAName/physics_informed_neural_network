import unittest
import tensorflow as tf
import numpy as np
from src.models.pinn import BasePINN

class TestPINN(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.layers = [2, 16, 16, 1]  # Simple architecture for testing
        self.model = BasePINN(self.layers)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    def test_model_initialization(self):
        """Test if the model is initialized correctly."""
        self.assertEqual(len(self.model.hidden_layers), len(self.layers) - 2)
        self.assertEqual(self.model.hidden_layers[0].units, self.layers[1])
        self.assertEqual(self.model.out_layer.units, self.layers[-1])
    
    def test_forward_pass(self):
        """Test if the forward pass works correctly."""
        batch_size = 4
        x = tf.random.normal((batch_size, self.layers[0]))
        output = self.model(x)
        
        self.assertEqual(output.shape, (batch_size, self.layers[-1]))
    
    def test_compute_data_loss(self):
        """Test if the data loss computation works correctly."""
        batch_size = 4
        x = tf.random.normal((batch_size, self.layers[0]))
        y = tf.random.normal((batch_size, self.layers[-1]))
        
        loss = self.model.compute_data_loss(x, y)
        self.assertIsInstance(loss, tf.Tensor)
        self.assertGreaterEqual(loss, 0)
    
    def test_compute_total_loss(self):
        """Test if the total loss computation works correctly."""
        batch_size = 4
        physics_inputs = tf.random.normal((batch_size, self.layers[0]))
        data_inputs = tf.random.normal((batch_size, self.layers[0]))
        data_targets = tf.random.normal((batch_size, self.layers[-1]))
        
        with self.assertRaises(NotImplementedError):
            self.model.compute_total_loss(physics_inputs, data_inputs, data_targets)
    
    def test_train_step(self):
        """Test if the training step works correctly."""
        batch_size = 4
        physics_inputs = tf.random.normal((batch_size, self.layers[0]))
        data_inputs = tf.random.normal((batch_size, self.layers[0]))
        data_targets = tf.random.normal((batch_size, self.layers[-1]))
        
        with self.assertRaises(NotImplementedError):
            self.model.train_step(physics_inputs, data_inputs, data_targets, self.optimizer)
    
    def test_activation_functions(self):
        """Test if different activation functions work correctly."""
        activations = ['tanh', 'relu', 'sigmoid']
        
        for activation in activations:
            model = BasePINN(self.layers, activation=activation)
            x = tf.random.normal((4, self.layers[0]))
            output = model(x)
            
            self.assertEqual(output.shape, (4, self.layers[-1]))
    
    def test_model_save_load(self):
        """Test if the model can be saved and loaded correctly."""
        # Save model
        self.model.save_weights('test_model_weights')
        
        # Create new model with same architecture
        new_model = BasePINN(self.layers)
        
        # Load weights
        new_model.load_weights('test_model_weights')
        
        # Test if weights are the same
        x = tf.random.normal((4, self.layers[0]))
        output1 = self.model(x)
        output2 = new_model(x)
        
        np.testing.assert_array_almost_equal(output1.numpy(), output2.numpy())

if __name__ == '__main__':
    unittest.main() 