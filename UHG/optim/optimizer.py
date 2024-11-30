import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, Tuple, Union, Callable, List, Dict, Any
from ..manifolds.base import Manifold

class HyperbolicOptimizer(Optimizer):
    """Base class for optimizers operating in hyperbolic space.
    
    All optimization steps are performed directly in hyperbolic space
    using UHG principles, without tangent space mappings.
    
    Args:
        params: Iterable of parameters to optimize
        manifold: The hyperbolic manifold to operate on
        defaults: Default optimizer settings
    """
    def __init__(
        self,
        params: List[torch.Tensor],
        manifold: Manifold,
        defaults: Dict[str, Any]
    ):
        super().__init__(params, defaults)
        self.manifold = manifold
        
    def project_gradients(self, p: torch.Tensor) -> torch.Tensor:
        """Project gradients to preserve hyperbolic structure.
        
        Args:
            p: Parameter tensor
            
        Returns:
            Projected gradients
        """
        if p.grad is None:
            return p.grad
            
        # Project gradients directly in hyperbolic space
        return self.manifold.project(p, p.grad)
        
    def update_parameter(
        self,
        p: torch.Tensor,
        d_p: torch.Tensor,
        lr: float
    ) -> torch.Tensor:
        """Update parameter while preserving hyperbolic structure.
        
        Args:
            p: Current parameter value
            d_p: Update direction
            lr: Learning rate
            
        Returns:
            Updated parameter value
        """
        # Scale update in hyperbolic space
        scaled_d_p = self.manifold.mobius_scalar_mul(-lr, d_p)
        
        # Apply update directly in hyperbolic space
        p_new = self.manifold.mobius_add(p, scaled_d_p)
        
        # Ensure result satisfies hyperbolic constraints
        return self.manifold.normalize(p_new)
        
    def preserve_cross_ratio(
        self,
        group: Dict[str, Any],
        state: Dict[str, Any],
        p: torch.Tensor
    ) -> None:
        """Adjust optimizer state to preserve cross-ratios.
        
        Args:
            group: Optimizer parameter group
            state: Optimizer state
            p: Parameter tensor
        """
        # Initialize state if needed
        if len(state) == 0:
            state['step'] = 0
            state['prev_p'] = p.clone()
            return
            
        # Compute and preserve cross-ratio between steps
        prev_p = state['prev_p']
        cr = self.manifold.compute_cross_ratio(prev_p, p)
        p_adj = self.manifold.preserve_cross_ratio(p, cr)
        
        # Update state
        state['prev_p'] = p_adj.clone()
        state['step'] += 1
        
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step.
        
        Args:
            closure: Closure that reevaluates the model and returns loss
            
        Returns:
            Optional loss value from closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Get state for current parameter
                state = self.state[p]
                
                # Project gradients to hyperbolic space
                d_p = self.project_gradients(p)
                
                # Update parameter
                p_new = self.update_parameter(p, d_p, group['lr'])
                
                # Preserve cross-ratio
                self.preserve_cross_ratio(group, state, p_new)
                
                # Update parameter in-place
                p.data.copy_(p_new)
                
        return loss 