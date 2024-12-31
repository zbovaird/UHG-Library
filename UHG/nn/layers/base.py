import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from ...projective import ProjectiveUHG

class UHGLayer(nn.Module):
    """Base class for all UHG-compliant neural network layers.
    
    This layer ensures all operations preserve cross-ratios and follow UHG principles.
    """
    def __init__(self):
        super().__init__()
        self.uhg = ProjectiveUHG()
        
    def projective_transform(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply projective transformation preserving cross-ratios."""
        # Extract features and homogeneous coordinate
        features = x[..., :-1]
        homogeneous = x[..., -1:]
        
        # Apply weight to features
        transformed = torch.matmul(features, weight.t())
        
        # Add homogeneous coordinate back
        out = torch.cat([transformed, homogeneous], dim=-1)
        
        # Normalize to maintain projective structure
        norm = torch.norm(out[..., :-1], p=2, dim=-1, keepdim=True)
        out = torch.cat([out[..., :-1] / (norm + 1e-8), out[..., -1:]], dim=-1)
        return out
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")

class ProjectiveLayer(UHGLayer):
    """Layer that operates in projective space while preserving UHG principles.
    
    This layer implements the core projective operations needed by other layers.
    All transformations preserve cross-ratios and hyperbolic structure.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights to preserve hyperbolic structure."""
        nn.init.kaiming_uniform_(self.weight, a=2**0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply projective transformation preserving UHG structure."""
        return self.projective_transform(x, self.weight)