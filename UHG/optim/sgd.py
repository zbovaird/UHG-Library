import torch
from typing import List, Dict, Any, Optional, Callable
from .optimizer import HyperbolicOptimizer
from ..manifolds.base import Manifold

class HyperbolicSGD(HyperbolicOptimizer):
    """Hyperbolic version of Stochastic Gradient Descent.
    
    All operations are performed directly in hyperbolic space using UHG principles.
    Momentum is computed while preserving hyperbolic structure.
    
    Args:
        params: Iterable of parameters to optimize
        manifold: The hyperbolic manifold to operate on
        lr: Learning rate
        momentum: Momentum factor
        weight_decay: Weight decay factor
        nesterov: Whether to use Nesterov momentum
    """
    def __init__(
        self,
        params: List[torch.Tensor],
        manifold: Manifold,
        lr: float = 1e-3,
        momentum: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        super().__init__(params, manifold, defaults)
        
    def init_state(self, p: torch.Tensor, state: Dict[str, Any]) -> None:
        """Initialize optimizer state for parameter.
        
        Args:
            p: Parameter tensor
            state: State dictionary to initialize
        """
        state['step'] = 0
        state['prev_p'] = p.clone()
        
        # Initialize momentum buffer in hyperbolic space
        if self.defaults['momentum'] > 0:
            state['momentum_buffer'] = torch.zeros_like(p)
            
    def update_parameter(
        self,
        p: torch.Tensor,
        group: Dict[str, Any],
        state: Dict[str, Any]
    ) -> torch.Tensor:
        """Update parameter using hyperbolic SGD rule.
        
        Args:
            p: Parameter tensor
            group: Parameter group
            state: Optimizer state
            
        Returns:
            Updated parameter value
        """
        # Get gradients
        grad = self.project_gradients(p)
        
        if group['weight_decay'] != 0:
            grad = self.manifold.mobius_add(
                grad,
                self.manifold.mobius_scalar_mul(
                    group['weight_decay'],
                    p
                )
            )
            
        # Handle momentum in hyperbolic space
        if group['momentum'] > 0:
            buf = state['momentum_buffer']
            
            # Update momentum buffer
            buf_new = self.manifold.mobius_add(
                self.manifold.mobius_scalar_mul(group['momentum'], buf),
                grad
            )
            state['momentum_buffer'] = buf_new
            
            if group['nesterov']:
                # Nesterov momentum: grad += momentum * buf_new
                grad = self.manifold.mobius_add(
                    grad,
                    self.manifold.mobius_scalar_mul(group['momentum'], buf_new)
                )
            else:
                grad = buf_new
                
        # Apply update in hyperbolic space
        p_new = self.manifold.mobius_add(
            p,
            self.manifold.mobius_scalar_mul(-group['lr'], grad)
        )
        
        # Ensure result satisfies hyperbolic constraints
        return self.manifold.normalize(p_new)
        
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
                
                # Initialize state if needed
                if len(state) == 0:
                    self.init_state(p, state)
                    
                # Update parameter
                p_new = self.update_parameter(p, group, state)
                
                # Preserve cross-ratio
                self.preserve_cross_ratio(group, state, p_new)
                
                # Update parameter in-place
                p.data.copy_(p_new)
                
        return loss 