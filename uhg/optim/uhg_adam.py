"""
UHG-Adam optimizer that includes momentum and adaptive learning rates while preserving hyperbolic structure.
"""

import torch
import math
from typing import Optional, Dict, Any, Callable
from .base import UHGBaseOptimizer

class UHGAdam(UHGBaseOptimizer):
    """UHG-Adam optimizer that includes momentum and adaptive learning rates while preserving hyperbolic structure."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        Initialize UHG-Adam optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to denominator to improve numerical stability
            weight_decay: Weight decay (L2 penalty)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def _compute_hyperbolic_momentum(self, grad: torch.Tensor, param: torch.Tensor, beta: float, group: dict) -> torch.Tensor:
        """
        Compute hyperbolic momentum update.
        
        Args:
            grad: Current gradient
            param: Parameter tensor
            beta: Momentum coefficient
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
        momentum.mul_(beta).add_(grad, alpha=1 - beta)
        
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
    
    def _compute_hyperbolic_second_moment(self, grad: torch.Tensor, param: torch.Tensor, beta: float, group: dict) -> torch.Tensor:
        """
        Compute hyperbolic second moment update.
        
        Args:
            grad: Current gradient
            param: Parameter tensor
            beta: Second moment coefficient
            group: Parameter group containing optimization settings
            
        Returns:
            Updated second moment in tangent space
        """
        # Get current second moment from state
        state = self.state[param]
        if 'second_moment_buffer' not in state:
            state['second_moment_buffer'] = torch.zeros_like(param)
            
        # Update second moment using exponential moving average
        second_moment = state['second_moment_buffer']
        second_moment.mul_(beta).addcmul_(grad, grad, value=1 - beta)
        
        # Project second moment to tangent space
        param_norm = torch.norm(param)
        if param_norm < group['eps']:
            return second_moment
            
        param_normalized = param / param_norm
        dot_product = torch.dot(second_moment, param_normalized)
        second_moment = second_moment - dot_product * param_normalized
        
        # Normalize second moment
        second_moment_norm = torch.norm(second_moment)
        if second_moment_norm > 0:
            second_moment = second_moment / second_moment_norm
            
        return second_moment
    
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Get gradient and current parameter
                grad = p.grad.data
                param = p.data
                
                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(param)
                    state['exp_avg_sq'] = torch.zeros_like(param)
                
                # Get state values
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(param, alpha=group['weight_decay'])
                
                # Clip gradients for numerical stability
                grad_norm = torch.norm(grad)
                if grad_norm > 1.0:
                    grad = grad / grad_norm
                
                # Update momentum and second moment
                exp_avg = self._compute_hyperbolic_momentum(grad, param, beta1, group)
                exp_avg_sq = self._compute_hyperbolic_second_moment(grad, param, beta2, group)
                
                # Update step count
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute step size
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # Compute update
                update = -step_size * (exp_avg / denom)
                
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