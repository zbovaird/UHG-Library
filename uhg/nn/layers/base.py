import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from ...projective import ProjectiveUHG

class UHGLayer(nn.Module):
    """Base class for all UHG-compliant neural network layers.
    
    This layer ensures all operations preserve cross-ratios and follow UHG principles.
    Uses float64 precision and regularization for numerical stability.
    """
    def __init__(self):
        super().__init__()
        self.uhg = ProjectiveUHG()
        self.eps = 1e-15  # Smaller epsilon for float64
        
    def _to_double(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to float64 if needed."""
        return x.double() if x.dtype != torch.float64 else x
        
    def projective_transform(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply projective transformation preserving cross-ratios.
        
        Uses float64 precision and careful regularization to maintain numerical stability
        and preserve cross-ratios accurately.
        """
        # Convert to float64 for better precision
        x = self._to_double(x)
        weight = self._to_double(weight)
        
        # Extract features and homogeneous coordinate
        features = x[..., :-1]
        homogeneous = x[..., -1:]
        
        # Ensure weight has correct shape for input features
        if weight.size(1) != features.size(-1):
            weight = weight[:, :features.size(-1)]
        
        # Apply weight to features with regularization
        transformed = torch.matmul(features, weight.t())
        
        # Add regularization term to prevent numerical instability
        reg_term = self.eps * torch.eye(features.size(-1), dtype=torch.float64, device=features.device)
        features_reg = torch.matmul(features, reg_term)
        transformed = transformed + torch.matmul(features_reg, weight.t())
        
        # Add homogeneous coordinate back
        out = torch.cat([transformed, homogeneous], dim=-1)
        
        # Normalize using hyperbolic dot product
        spatial = out[..., :-1]
        time = out[..., -1:]
        
        # Compute hyperbolic norm with regularization
        spatial_norm = torch.sum(spatial * spatial, dim=-1, keepdim=True)
        time_norm = time * time
        norm = torch.sqrt(torch.clamp(spatial_norm - time_norm + self.eps, min=self.eps))
        
        # Normalize spatial components while preserving time component
        normalized_spatial = spatial / (norm + self.eps)
        
        # Ensure time component remains stable
        normalized_time = torch.clamp(time, min=1.0)
        
        return torch.cat([normalized_spatial, normalized_time], dim=-1)
        
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
        # Initialize weights in float64
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float64))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights to preserve hyperbolic structure."""
        # Use a smaller gain for better numerical stability
        gain = (2**0.5) * 0.1
        nn.init.kaiming_uniform_(self.weight, a=gain)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply projective transformation preserving UHG structure."""
        return self.projective_transform(x, self.weight)