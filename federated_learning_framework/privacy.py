# privacy.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Union, Optional
import logging

# Try importing tensorflow_privacy, but make it optional
try:
    from tensorflow_privacy.privacy.analysis import compute_rdp
    from tensorflow_privacy.privacy.analysis import get_privacy_spent
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False
    logging.getLogger("privacy.dp").warning("TensorFlow Privacy library not found. Cannot compute precise RDP privacy budget.")


class DifferentialPrivacy:
    """
    Implements differential privacy mechanisms for federated learning.
    Focuses on adding noise, primarily intended for server-side aggregation.
    """

    def __init__(self, epsilon: float, delta: float,
                 noise_multiplier: float, clipping_norm: float):
        """
        Initialize the DP engine.
        Args:
            epsilon: Target privacy budget epsilon (used for info/reporting).
            delta: Target privacy budget delta.
            noise_multiplier: Controls the amount of noise added (sigma / sensitivity).
            clipping_norm: The L2 norm bound for sensitivity (e.g., gradients or updates).
        """
        self.logger = logging.getLogger("privacy.dp")
        self.target_epsilon = epsilon
        self.target_delta = delta
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm # Sensitivity S

        if self.noise_multiplier <= 0:
            self.logger.warning("Noise multiplier is zero or negative. DP noise will not be added.")
        if self.clipping_norm <= 0:
             self.logger.warning("Clipping norm is zero or negative. Sensitivity is assumed zero, noise might be ineffective.")

        self.logger.info(f"DP Engine Initialized: Target(ε={epsilon}, δ={delta}), NoiseMultiplier={noise_multiplier}, ClippingNorm={clipping_norm}")

    def add_noise_to_model(self, model: nn.Module):
        """
        Adds Gaussian noise directly to model parameters.
        NOTE: This applies noise to the *entire parameter set* after aggregation.
              A common alternative in FL is to noise the *aggregated update delta*.
              The noise level depends on the sensitivity of the *aggregation result*,
              which is related to the clipping norm applied to *client contributions*.

        Noise stddev = noise_multiplier * clipping_norm
        """
        if self.noise_multiplier <= 0 or self.clipping_norm <= 0:
            self.logger.debug("Skipping noise addition due to zero noise multiplier or clipping norm.")
            return

        # Sensitivity S is the clipping norm
        sensitivity = self.clipping_norm
        # Standard deviation of the Gaussian noise for DP
        noise_stddev = self.noise_multiplier * sensitivity

        if noise_stddev == 0:
             self.logger.warning("Calculated noise standard deviation is zero. No noise added.")
             return

        self.logger.debug(f"Adding Gaussian noise with stddev {noise_stddev:.4f} to model parameters.")

        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad: # Only add noise to trainable parameters
                    noise = torch.normal(0, noise_stddev, size=param.size(), device=param.device)
                    param.add_(noise)

    def clip_gradients(self, model: nn.Module):
        """
        Clips the L2 norm of gradients for the *entire model*.
        NOTE: This is global gradient clipping. For DP per-sample guarantees,
              clipping is usually done *per-sample* gradient before averaging
              within a batch (more complex to implement in standard PyTorch loops).
              This method clips the *accumulated* gradients for the batch/step.
        """
        if self.clipping_norm <= 0:
            self.logger.debug("Skipping gradient clipping due to zero clipping norm.")
            return

        # Use PyTorch's utility function for clipping grad norm
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), self.clipping_norm)
        self.logger.debug(f"Gradients clipped. Original Norm: {total_norm:.4f}, Clip Threshold: {self.clipping_norm}")


    def add_noise_to_gradients(self, model: nn.Module):
        """
        Adds noise directly to the *accumulated gradients* of the model parameters.
        NOTE: Similar to clip_gradients, this acts on the globally accumulated
              gradients, not per-sample. The noise level should be calibrated
              based on the sensitivity *after* clipping.

        Noise stddev = noise_multiplier * clipping_norm
        """
        if self.noise_multiplier <= 0 or self.clipping_norm <= 0:
            self.logger.debug("Skipping gradient noise addition due to zero noise multiplier or clipping norm.")
            return

        sensitivity = self.clipping_norm
        noise_stddev = self.noise_multiplier * sensitivity

        if noise_stddev == 0:
             self.logger.warning("Calculated noise standard deviation is zero. No noise added to gradients.")
             return

        self.logger.debug(f"Adding Gaussian noise with stddev {noise_stddev:.4f} to model gradients.")

        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.normal(0, noise_stddev, size=param.grad.size(), device=param.grad.device)
                    param.grad.add_(noise)


    def calculate_privacy_spent(self, num_samples: int, batch_size: int, epochs: int, rounds: int) -> Dict[str, float]:
        """
        Calculates the approximate (ε, δ) privacy budget spent using RDP accountant.
        Assumes DP Gaussian noise applied after aggregation on the server per round.

        Args:
            num_samples: Total number of samples across *all* participating clients in a round (approximate).
            batch_size: Effective batch size (relevant if noise added per batch, less so for post-aggregation).
                       Here, we consider the entire round's contribution as one "step" for simplicity.
            epochs: Local epochs per client (not directly used if noise is post-aggregation).
            rounds: Number of communication rounds DP was applied.

        Returns:
            Dictionary with calculated epsilon, delta, and parameters used.
        """
        if not TFP_AVAILABLE:
            self.logger.warning("Cannot calculate privacy spent, TensorFlow Privacy not available.")
            return {"error": "TFP library not found"}

        if self.noise_multiplier <= 0:
            self.logger.warning("Cannot calculate privacy, noise multiplier is zero.")
            return {"epsilon": float('inf'), "delta": 1.0, "noise_multiplier": self.noise_multiplier}


        # --- RDP Calculation for Gaussian Mechanism applied 'rounds' times ---
        # The 'sampling rate' q is tricky here. For post-aggregation noise, the effective
        # sensitivity depends on how many clients contribute and the clipping norm applied
        # to their contributions. Let's simplify: assume the Gaussian mechanism is applied
        # once per round with the given noise multiplier relative to the clipping norm.
        # The sampling probability q is effectively 1 if we consider the server mechanism itself.
        # More accurate analysis depends heavily on where noise is added and clipping happens.

        # Simplified RDP calculation assuming Gaussian mechanism applied `rounds` times.
        # q=1 (no subsampling in the server mechanism itself)
        q = 1.0
        steps = rounds # Each round is one step of the mechanism

        # RDP orders, standard range
        orders = list(np.arange(2.0, 32.0)) + [64.0, 128.0, 256.0, 512.0]

        try:
            rdp = compute_rdp(q=q, noise_multiplier=self.noise_multiplier, steps=steps, orders=orders)
            epsilon, best_alpha = get_privacy_spent(orders, rdp, target_delta=self.target_delta)

            return {
                "epsilon_spent": epsilon,
                "delta_target": self.target_delta,
                "noise_multiplier": self.noise_multiplier,
                "clipping_norm": self.clipping_norm,
                "mechanism_steps": steps, # Rounds
                "best_rdp_order": best_alpha
            }
        except Exception as e:
            self.logger.error(f"Failed to compute privacy spent using RDP: {e}", exc_info=True)
            return {"error": str(e)}