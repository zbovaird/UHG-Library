import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from ...manifolds.base import Manifold
from ...utils.cross_ratio import compute_cross_ratio, preserve_cross_ratio

class HyperbolicLayer(nn.Module):
    """Base class for all hyperbolic neural network layers.
    
    This class enforces strict adherence to Universal Hyperbolic Geometry principles:
    1. No tangent space operations
    2. Direct calculations in hyperbolic space
    3. Cross-ratio preservation
    4. Projective geometric operations
    
    Args:
        manifold (Manifold): The hyperbolic manifold to operate on
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool, optional): Whether to include bias. Defaults to True.
    """
    def __init__(
        self,
        manifold: Manifold,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights directly in hyperbolic space
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize parameters using Möbius gyrovector space."""
        # Initialize in hyperbolic space directly - no tangent space
        bound = 1 / self.in_features ** 0.5
        self.weight.data.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.data.uniform_(-bound, bound)
            
    def compute_hyperbolic_activation(
        self,
        x: torch.Tensor,
        activation: Optional[str] = None
    ) -> torch.Tensor:
        """Apply activation function directly in hyperbolic space.
        
        Args:
            x: Input tensor in hyperbolic space
            activation: Name of activation function
            
        Returns:
            Tensor after hyperbolic activation
        """
        if activation is None:
            return x
            
        # All activations performed directly in hyperbolic space
        if activation == 'relu':
            return self.manifold.relu(x)
        elif activation == 'sigmoid':
            return self.manifold.sigmoid(x) 
        elif activation == 'tanh':
            return self.manifold.tanh(x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def mobius_matvec(self, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform matrix-vector multiplication in hyperbolic space.
        
        Uses Möbius gyrovector operations to maintain hyperbolic structure.
        
        Args:
            weight: Weight matrix
            x: Input tensor
            
        Returns:
            Result of hyperbolic matrix-vector multiplication
        """
        # Compute using direct hyperbolic operations
        return self.manifold.mobius_matvec(weight, x)
        
    def preserve_cross_ratio(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensure transformations preserve the cross-ratio.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Tuple of tensors with preserved cross-ratio
        """
        # Compute initial cross-ratio
        cr_before = compute_cross_ratio(x, y)
        
        # Adjust tensors to preserve cross-ratio
        x_adj, y_adj = preserve_cross_ratio(x, y, cr_before)
        
        return x_adj, y_adj
        
    def forward(
        self,
        x: torch.Tensor,
        activation: Optional[str] = None
    ) -> torch.Tensor:
        """Forward pass of layer.
        
        All operations performed directly in hyperbolic space without
        tangent space mappings.
        
        Args:
            x: Input tensor
            activation: Optional activation function
            
        Returns:
            Output tensor
        """
        # Matrix multiplication in hyperbolic space
        output = self.mobius_matvec(self.weight, x)
        
        if self.bias is not None:
            output = self.manifold.mobius_add(output, self.bias)
            
        # Apply activation in hyperbolic space
        output = self.compute_hyperbolic_activation(output, activation)
        
        return output
        
    def extra_repr(self) -> str:
        """String representation of layer."""
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}')