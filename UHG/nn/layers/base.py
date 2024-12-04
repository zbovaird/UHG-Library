import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from ...projective import ProjectiveUHG

class ProjectiveLayer(nn.Module):
    """Base class for neural network layers using projective geometry.
    
    This class implements layers using pure projective geometry,
    following UHG principles without any manifold concepts.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include bias
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.uhg = ProjectiveUHG()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights as projective transformations
        self.weight = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize parameters as projective transformations."""
        # Create random projective transformations
        matrix = self.uhg.get_projective_matrix(self.in_features)
        self.weight.data.copy_(matrix[:-1])  # Remove last row for linear map
        
        if self.bias is not None:
            bound = 1 / self.in_features ** 0.5
            self.bias.data.uniform_(-bound, bound)
            
    def projective_transform(
        self,
        x: torch.Tensor,
        activation: Optional[str] = None
    ) -> torch.Tensor:
        """Apply projective transformation to input.
        
        Args:
            x: Input tensor
            activation: Optional activation function
            
        Returns:
            Transformed tensor
        """
        # Create projective transformation matrix
        matrix = torch.cat([
            self.weight,
            torch.ones(1, self.weight.size(1), device=self.weight.device)
        ])
        
        # Apply projective transformation
        out = self.uhg.transform(x, matrix)
        
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
            
        # Apply activation if specified
        if activation == 'relu':
            out = torch.relu(out)
        elif activation == 'sigmoid':
            out = torch.sigmoid(out)
        elif activation == 'tanh':
            out = torch.tanh(out)
            
        return out
        
    def forward(
        self,
        x: torch.Tensor,
        activation: Optional[str] = None
    ) -> torch.Tensor:
        """Forward pass using projective transformations.
        
        Args:
            x: Input tensor
            activation: Optional activation function
            
        Returns:
            Output tensor
        """
        return self.projective_transform(x, activation)
        
    def extra_repr(self) -> str:
        """String representation of layer."""
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}')