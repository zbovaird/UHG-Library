import torch
from typing import Optional, Callable
from .hyperbolic import HyperbolicSampler

class HyperbolicLangevin(HyperbolicSampler):
    """Langevin dynamics sampler in hyperbolic space.
    
    Implements Langevin dynamics directly in hyperbolic space
    using UHG operations, without tangent space mappings.
    
    Args:
        manifold: The hyperbolic manifold to sample on
        log_prob_fn: Function computing log probability of points
        step_size: Size of integration steps
        noise_std: Standard deviation of noise
        temperature: Temperature parameter for sampling
    """
    def __init__(
        self,
        manifold,
        log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
        step_size: float = 0.1,
        noise_std: float = 0.01,
        temperature: float = 1.0
    ):
        super().__init__(manifold, log_prob_fn, temperature)
        self.step_size = step_size
        self.noise_std = noise_std
        
    def sample(
        self,
        n_samples: int,
        initial_point: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate samples using hyperbolic Langevin dynamics.
        
        Args:
            n_samples: Number of samples to generate
            initial_point: Starting point for sampling
            **kwargs: Additional sampling parameters
            
        Returns:
            Tensor of sampled points
        """
        if initial_point is None:
            # Sample initial point uniformly from hyperbolic space
            initial_point = self.manifold.random_uniform(n_samples)
            
        samples = []
        current_x = initial_point
        
        for _ in range(n_samples):
            # Compute force at current point
            force = self._compute_force(current_x)
            
            # Sample noise in hyperbolic space
            noise = self.manifold.random_normal_like(current_x)
            noise = self.manifold.mobius_scalar_mul(
                self.noise_std,
                noise
            )
            
            # Update position using force and noise
            proposed_x = self.manifold.mobius_add(
                current_x,
                self.manifold.mobius_scalar_mul(
                    self.step_size,
                    force
                )
            )
            proposed_x = self.manifold.mobius_add(
                proposed_x,
                noise
            )
            
            # Ensure constraints are satisfied
            proposed_x = self._check_constraints(proposed_x)
            
            # Compute acceptance probability (Metropolis adjustment)
            current_energy = self._compute_energy(current_x)
            proposed_energy = self._compute_energy(proposed_x)
            
            # Accept or reject
            if torch.rand(1) < torch.exp(current_energy - proposed_energy):
                current_x = proposed_x
                
            samples.append(current_x)
            
        return torch.stack(samples) 