import unittest
import numpy as np
import os
import sys
import tempfile
import shutil
from pathlib import Path
import tenseal as ts

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from crypto.homomorphic_engine import HomomorphicEngine

class TestHomomorphicEngine(unittest.TestCase):
    """
    Unit tests for the HomomorphicEngine class.
    """
    
    def setUp(self):
        """Set up the test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.keys_dir = Path(self.temp_dir) / "keys"
        self.keys_dir.mkdir(exist_ok=True)
        
        # Create engine with small parameters for faster tests
        self.engine = HomomorphicEngine(key_size=1024, security_level=128)
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test that the engine initializes correctly"""
        self.assertIsNotNone(self.engine.context)
        self.assertIsNotNone(self.engine.public_key)
        self.assertIsNotNone(self.engine.secret_key)
    
    def test_encrypt_decrypt_vector(self):
        """Test encryption and decryption of vectors"""
        original = [1.0, 2.0, 3.0, 4.0, 5.0]
        encrypted = self.engine.encrypt_vector(original)
        decrypted = self.engine.decrypt_vector(encrypted)
        
        # Check that decrypted values are close to original (within tolerance)
        for orig, dec in zip(original, decrypted):
            self.assertAlmostEqual(orig, dec, places=4)
    
    def test_homomorphic_addition(self):
        """Test homomorphic addition of encrypted vectors"""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [4.0, 5.0, 6.0]
        expected = [5.0, 7.0, 9.0]
        
        # Encrypt vectors
        enc_a = self.engine.encrypt_vector(vec_a)
        enc_b = self.engine.encrypt_vector(vec_b)
        
        # Perform homomorphic addition
        enc_result = self.engine.homomorphic_add(enc_a, enc_b)
        
        # Decrypt and verify
        result = self.engine.decrypt_vector(enc_result)
        
        for exp, res in zip(expected, result):
            self.assertAlmostEqual(exp, res, places=4)
    
    def test_homomorphic_multiply_plain(self):
        """Test homomorphic multiplication with plaintext scalar"""
        vec = [1.0, 2.0, 3.0]
        scalar = 2.5
        expected = [2.5, 5.0, 7.5]
        
        # Encrypt vector
        enc_vec = self.engine.encrypt_vector(vec)
        
        # Perform homomorphic multiplication
        enc_result = self.engine.homomorphic_multiply_plain(enc_vec, scalar)
        
        # Decrypt and verify
        result = self.engine.decrypt_vector(enc_result)
        
        for exp, res in zip(expected, result):
            self.assertAlmostEqual(exp, res, places=4)
    
    def test_weighted_aggregation(self):
        """Test weighted aggregation of encrypted vectors"""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [4.0, 5.0, 6.0]
        weight_a = 0.3
        weight_b = 0.7
        
        expected = [
            vec_a[0] * weight_a + vec_b[0] * weight_b,
            vec_a[1] * weight_a + vec_b[1] * weight_b,
            vec_a[2] * weight_a + vec_b[2] * weight_b
        ]
        
        # Encrypt vectors
        enc_a = self.engine.encrypt_vector(vec_a)
        enc_b = self.engine.encrypt_vector(vec_b)
        
        # Perform weighted aggregation
        enc_result = self.engine.weighted_aggregation([
            (enc_a, weight_a),
            (enc_b, weight_b)
        ])
        
        # Decrypt and verify
        result = self.engine.decrypt_vector(enc_result)
        
        for exp, res in zip(expected, result):
            self.assertAlmostEqual(exp, res, places=4)
    
    def test_encrypt_decrypt_model_parameters(self):
        """Test encryption and decryption of model parameters"""
        # Create sample parameters (weights)
        params = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),  # 1D array
            np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32),  # 2D array
            np.array([[[8.0, 9.0], [10.0, 11.0]]], dtype=np.float32)  # 3D array
        ]
        
        # Encrypt parameters
        encrypted_params = self.engine.encrypt_model_parameters(params)
        
        # Decrypt parameters
        decrypted_params = self.engine.decrypt_model_parameters(encrypted_params)
        
        # Verify each parameter
        for orig_param, dec_param in zip(params, decrypted_params):
            self.assertEqual(orig_param.shape, dec_param.shape)
            np.testing.assert_allclose(orig_param, dec_param, rtol=1e-3, atol=1e-3)
    
    def test_add_differential_privacy(self):
        """Test adding differential privacy to a vector"""
        original = [1.0, 2.0, 3.0, 4.0, 5.0]
        epsilon = 1.0
        
        # Add noise
        noisy = self.engine.add_differential_privacy(original, epsilon)
        
        # Verify the vector is changed but still close to original
        self.assertNotEqual(original, noisy)
        
        # Values should be within a reasonable range
        for orig, noisy_val in zip(original, noisy):
            self.assertLess(abs(orig - noisy_val), 5.0)  # Arbitrary threshold

if __name__ == '__main__':
    unittest.main()