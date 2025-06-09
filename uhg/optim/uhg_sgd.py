"""
UHG-SGD Optimizer Implementation

Implements projected stochastic gradient descent for Universal Hyperbolic Geometry.
All updates are projected to the UHG manifold after each step.
References: UHG.pdf, Section on optimization in projective/hyperbolic space.
"""

import torch
import math
from typing import Dict, Any, List, Optional
from .base import UHGBaseOptimizer

class UHGSGD(UHGBaseOptimizer):
    """
    UHG-SGD optimizer with momentum.
    
    Implements stochastic gradient descent in Universal Hyperbolic Geometry (UHG) space
    with momentum support. All operations preserve the hyperbolic structure and ensure
    parameters remain on the UHG manifold.
    
    Args:
        params: Parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        eps: Epsilon for numerical stability (default: 1e-8)
    """
    
    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        eps: float = 1e-8
    ):
        """
        Initialize UHG-SGD optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            momentum: Momentum coefficient
            weight_decay: Weight decay (L2 penalty)
            eps: Term added to denominator to improve numerical stability
        """
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
            
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
        
    def _compute_hyperbolic_momentum(self, grad: torch.Tensor, param: torch.Tensor, group: dict) -> torch.Tensor:
        """
        Compute hyperbolic momentum update.
        
        Args:
            grad: Current gradient
            param: Parameter tensor
            group: Parameter group containing optimization settings
            
        Returns:
            Updated momentum in tangent space
        """
        # Get current momentum from state
        state = self.state[param]
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(param)
            
        # Update momentum using exponential moving average
        momentum = state['momentum_buffer']
        momentum.mul_(group['momentum']).add_(grad, alpha=1 - group['momentum'])
        
        # Project momentum to tangent space
        param_norm = torch.norm(param)
        if param_norm < group['eps']:
            return momentum
            
        param_normalized = param / param_norm
        dot_product = torch.dot(momentum, param_normalized)
        momentum = momentum - dot_product * param_normalized
        
        # Normalize momentum
        momentum_norm = torch.norm(momentum)
        if momentum_norm > 0:
            momentum = momentum / momentum_norm
            
        return momentum
        
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss if closure is provided, None otherwise.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get gradients and parameters
                grad = p.grad.data
                param = p.data

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(param, alpha=group['weight_decay'])

                # Clip gradients for numerical stability
                grad_norm = torch.norm(grad)
                if grad_norm > 1.0:
                    grad = grad / grad_norm

                # Compute hyperbolic momentum if enabled
                if group['momentum'] != 0:
                    momentum = self._compute_hyperbolic_momentum(grad, param, group)
                else:
                    momentum = grad

                # Update parameters using exponential map
                lr = group['lr']
                update = -lr * momentum

                # Ensure update is in tangent space
                update = self._project_to_tangent_space(update, param, group)

                # Apply update using exponential map
                new_param = self._exponential_map(param, update)

                # Ensure new parameter is on the manifold
                new_param = new_param / torch.norm(new_param)

                # Check for numerical stability
                if torch.isnan(new_param).any() or torch.isinf(new_param).any():
                    continue

                p.data.copy_(new_param)

        return loss 