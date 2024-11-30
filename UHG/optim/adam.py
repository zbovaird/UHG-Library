import torch
from typing import List, Dict, Any, Optional, Callable
from .optimizer import HyperbolicOptimizer
from ..manifolds.base import Manifold

class HyperbolicAdam(HyperbolicOptimizer):
    """Hyperbolic version of Adam optimizer.
    
    All operations are performed directly in hyperbolic space using UHG principles.
    Momentum and adaptive learning rates are computed while preserving
    hyperbolic structure.
    
    Args:
        params: Iterable of parameters to optimize
        manifold: The hyperbolic manifold to operate on
        lr: Learning rate
        betas: Coefficients for computing running averages
        eps: Term added for numerical stability
        weight_decay: Weight decay factor
        amsgrad: Whether to use AMSGrad variant
    """
    def __init__(
        self,
        params: List[torch.Tensor],
        manifold: Manifold,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False
    ):
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
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
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
        
        # Initialize momentum and velocity in hyperbolic space
        state['exp_avg'] = torch.zeros_like(p)
        state['exp_avg_sq'] = torch.zeros_like(p)
        
        if self.defaults['amsgrad']:
            state['max_exp_avg_sq'] = torch.zeros_like(p)
            
    def update_parameter(
        self,
        p: torch.Tensor,
        group: Dict[str, Any],
        state: Dict[str, Any]
    ) -> torch.Tensor:
        """Update parameter using hyperbolic Adam rule.
        
        Args:
            p: Parameter tensor
            group: Parameter group
            state: Optimizer state
            
        Returns:
            Updated parameter value
        """
        # Get optimizer hyperparameters
        beta1, beta2 = group['betas']
        step = state['step'] + 1
        
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
            
        # Update momentum in hyperbolic space
        exp_avg = state['exp_avg']
        exp_avg_new = self.manifold.mobius_add(
            self.manifold.mobius_scalar_mul(beta1, exp_avg),
            self.manifold.mobius_scalar_mul(1 - beta1, grad)
        )
        state['exp_avg'] = exp_avg_new
        
        # Update velocity in hyperbolic space
        exp_avg_sq = state['exp_avg_sq']
        grad_sq = self.manifold.inner(grad, grad)
        exp_avg_sq_new = beta2 * exp_avg_sq + (1 - beta2) * grad_sq
        state['exp_avg_sq'] = exp_avg_sq_new
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        # Compute adaptive learning rate
        if group['amsgrad']:
            max_exp_avg_sq = state['max_exp_avg_sq']
            torch.maximum(max_exp_avg_sq, exp_avg_sq_new, out=max_exp_avg_sq)
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        else:
            denom = (exp_avg_sq_new.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            
        # Compute step size
        step_size = group['lr'] / bias_correction1
        
        # Apply update in hyperbolic space
        p_new = self.manifold.mobius_add(
            p,
            self.manifold.mobius_scalar_mul(
                -step_size,
                exp_avg_new / denom
            )
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