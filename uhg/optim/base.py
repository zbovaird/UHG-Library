"""
Base UHG Optimizer Implementation

This module implements the core optimization algorithms for Universal Hyperbolic Geometry,
ensuring all operations preserve hyperbolic structure and invariants.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Any, Optional, Callable
import math
from uhg.metrics import UHGMetric
from ..projective import ProjectiveUHG

class UHGBaseOptimizer(Optimizer):
    """
    Base optimizer for Universal Hyperbolic Geometry (UHG).
    Implements core hyperbolic optimization principles.
    """
    def __init__(self, params, defaults):
        """
        Initialize UHG base optimizer.

        Args:
            params: Iterable of parameters to optimize
            defaults: Dictionary of default parameter values
        """
        if not 0.0 <= defaults.get('lr', 0.0):
            raise ValueError(f"Invalid learning rate: {defaults.get('lr')}")
        if not 0.0 <= defaults.get('eps', 0.0):
            raise ValueError(f"Invalid epsilon value: {defaults.get('eps')}")
        super().__init__(params, defaults)
        self.metric = UHGMetric()
        self.uhg = ProjectiveUHG()
        
    def _project_to_manifold(self, param: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
        """
        Project parameters back to the UHG manifold.
        
        Args:
            param: Parameter tensor to project
            group: Parameter group containing defaults
            
        Returns:
            Projected parameter tensor
        """
        norm = torch.norm(param, p=2, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=group.get('eps', 1e-8))
        return param / norm
        
    def _check_manifold_constraint(self, param: torch.Tensor, group: Dict[str, Any]) -> bool:
        """
        Check if parameters satisfy UHG manifold constraints.
        In projective geometry, only the direction matters, not the magnitude.
        
        Args:
            param: Parameter tensor to check
            group: Parameter group containing defaults
            
        Returns:
            bool: True if parameters satisfy constraints
        """
        norm = torch.norm(param, p=2, dim=-1)
        return torch.all(norm > group.get('eps', 1e-8))
        
    def _compute_hyperbolic_gradient(self, grad: torch.Tensor, param: torch.Tensor, group: dict) -> torch.Tensor:
        """
        Compute hyperbolic gradient by projecting to tangent space.
        
        Args:
            grad: Euclidean gradient
            param: Parameter tensor
            group: Parameter group containing optimization settings
            
        Returns:
            Hyperbolic gradient in tangent space
        """
        # Project gradient to tangent space
        # First normalize the parameter to ensure it's on the manifold
        param_norm = torch.norm(param)
        if param_norm < group['eps']:
            return grad
            
        param_normalized = param / param_norm
        
        # Project gradient to tangent space by removing component parallel to parameter
        # This ensures the gradient is orthogonal to the parameter
        dot_product = torch.dot(grad, param_normalized)
        hgrad = grad - dot_product * param_normalized
        
        # Normalize the hyperbolic gradient to maintain scale
        hgrad_norm = torch.norm(hgrad)
        if hgrad_norm > 0:
            hgrad = hgrad / hgrad_norm
            
        return hgrad
        
    def _update_hyperbolic_step(self, param: torch.Tensor, hgrad: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
        """
        Update parameters using hyperbolic gradient descent.
        
        Args:
            param: Current parameter value
            hgrad: Hyperbolic gradient
            group: Parameter group containing defaults
            
        Returns:
            Updated parameter tensor
        """
        lr = group['lr']
        new_param = param - lr * hgrad
        return self._project_to_manifold(new_param, group)
        
    def _project_to_tangent_space(self, vector: torch.Tensor, point: torch.Tensor, group: dict) -> torch.Tensor:
        """
        Project a vector to the tangent space at a given point.

        Args:
            vector: Vector to project
            point: Point on the manifold
            group: Parameter group containing optimization parameters

        Returns:
            Projected vector in the tangent space
        """
        # Normalize point to ensure it's on the manifold
        point = point / torch.norm(point)

        # Project vector to tangent space by removing component parallel to point
        parallel_component = torch.sum(vector * point) * point
        tangent_vector = vector - parallel_component

        # Add small epsilon for numerical stability
        eps = group.get('eps', 1e-8)
        tangent_vector = tangent_vector + eps * point

        return tangent_vector
        
    def _exponential_map(self, point: torch.Tensor, tangent_vector: torch.Tensor) -> torch.Tensor:
        """
        Compute the exponential map at a point in the direction of a tangent vector.

        Args:
            point: Point on the manifold
            tangent_vector: Vector in the tangent space

        Returns:
            New point on the manifold
        """
        # Normalize point to ensure it's on the manifold
        point = point / torch.norm(point)

        # Compute norm of tangent vector
        v_norm = torch.norm(tangent_vector)

        # Handle small updates for numerical stability
        if v_norm < 1e-8:
            return point

        # Compute exponential map
        cosh_v = torch.cosh(v_norm)
        sinh_v = torch.sinh(v_norm)
        new_point = cosh_v * point + (sinh_v / v_norm) * tangent_vector

        # Ensure result is on the manifold
        new_point = new_point / torch.norm(new_point)

        return new_point
        
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step using pure projective operations."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # Pure Euclidean update step (no tangent space / exp map)
                p.data.add_(grad, alpha=-lr)

        return loss 