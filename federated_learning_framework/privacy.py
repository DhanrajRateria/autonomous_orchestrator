import torch
import numpy as np
from typing import Dict, List, Any, Union, Optional
import logging
import tensorflow_privacy

class DifferentialPrivacy:
    """
    Implements differential privacy mechanisms for federated learning.
    Adds calibrated noise to protect privacy of individual samples.
    """
    
    def __init__(self, epsilon: float = 3.0, delta: float = 1e-5, 
                 noise_multiplier: float = 1.1, clipping_norm: float = 1.0):
        """
        Initialize the differential privacy engine.
        
        Args:
            epsilon: Privacy budget parameter (lower = more private)
            delta: Probability of privacy violation (lower = more private)
            noise_multiplier: Scale of noise to add
            clipping_norm: Gradient clipping threshold
        """
        self.logger = logging.getLogger("privacy.dp")
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        
        self.logger.info(f"Differential privacy initialized with ε={epsilon}, δ={delta}")
    
    def add_noise_to_model(self, model: torch.nn.Module) -> None:
        """
        Add calibrated noise to model parameters for differential privacy.
        
        Args:
            model: PyTorch model to add noise to
        """
        with torch.no_grad():
            for param in model.parameters():
                # Calculate noise scale based on parameter shape and sensitivity
                noise_scale = self.noise_multiplier * self.clipping_norm
                noise = torch.normal(0, noise_scale, param.shape, device=param.device)
                param.add_(noise)
    
    def clip_gradients(self, model: torch.nn.Module) -> None:
        """
        Clip gradients to bound sensitivity.
        
        Args:
            model: PyTorch model whose gradients to clip
        """
        # Calculate total gradient norm
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Apply clipping if norm exceeds threshold
        if total_norm > self.clipping_norm:
            scaling_factor = self.clipping_norm / (total_norm + 1e-6)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scaling_factor)
    
    def add_noise_to_gradients(self, model: torch.nn.Module) -> None:
        """
        Add noise to gradients before optimization step.
        
        Args:
            model: PyTorch model whose gradients to noise
        """
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    # Add noise calibrated to the clipping norm
                    noise_scale = self.noise_multiplier * self.clipping_norm / (len(param.view(-1)))
                    noise = torch.normal(0, noise_scale, param.grad.shape, device=param.grad.device)
                    param.grad.add_(noise)
    
    def calculate_privacy_spent(self, num_samples: int, batch_size: int, epochs: int) -> Dict[str, float]:
        """
        Calculate the actual privacy budget spent.
        
        Args:
            num_samples: Number of training samples
            batch_size: Batch size used in training
            epochs: Number of training epochs
            
        Returns:
            Dictionary with privacy budget spent
        """
        try:
            # Import RDP accountant from tensorflow privacy
            from tensorflow_privacy.privacy.analysis import rdp_accountant
            from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
            from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
            
            # Calculate number of steps
            steps = num_samples // batch_size * epochs
            
            # Compute RDP for Gaussian mechanism
            q = batch_size / num_samples  # Sampling rate
            orders = [1 + x / 10.0 for x in range(1, 100)]
            rdp = compute_rdp(q=q, noise_multiplier=self.noise_multiplier, steps=steps, orders=orders)
            
            # Convert to (ε, δ) guarantee
            eps, delta = get_privacy_spent(orders, rdp, target_delta=self.delta)
            
            return {
                "epsilon": eps,
                "delta": delta,
                "noise_multiplier": self.noise_multiplier,
                "steps": steps
            }
            
        except ImportError:
            self.logger.warning("tensorflow-privacy not available, cannot compute exact privacy budget")
            return {
                "epsilon": self.epsilon,
                "delta": self.delta,
                "noise_multiplier": self.noise_multiplier
            }