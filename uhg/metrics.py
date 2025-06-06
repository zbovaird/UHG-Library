"""Universal Hyperbolic Geometry metrics implementation."""

import torch
import math
from typing import Optional, Tuple

class UHGMetric:
    """
    Universal Hyperbolic Geometry (UHG) metric implementation.
    
    This class provides methods for computing hyperbolic distances, metrics,
    and related geometric operations in UHG space.
    """
    
    def __init__(self, eps: float = 1e-8):
        """
        Initialize UHG metric.
        
        Args:
            eps: Numerical stability constant
        """
        self.eps = eps
        
    def get_metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric tensor at point x in UHG space.
        
        Args:
            x: Point in UHG space
            
        Returns:
            torch.Tensor: Metric tensor at x
        """
        # In UHG, the metric tensor is the identity matrix
        # This preserves the hyperbolic structure
        return torch.eye(x.shape[-1], device=x.device)
        
    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the hyperbolic distance between points x and y.
        
        Args:
            x: First point in UHG space
            y: Second point in UHG space
            
        Returns:
            torch.Tensor: Hyperbolic distance between x and y
        """
        # Compute cross-ratio based distance
        # This preserves the hyperbolic structure
        dot_product = torch.dot(x, y)
        norm_x = torch.norm(x, p=2)
        norm_y = torch.norm(y, p=2)
        
        # Add epsilon for numerical stability
        denominator = (norm_x * norm_y + self.eps)
        cos_theta = dot_product / denominator
        
        # Clamp to valid range for arccos
        cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
        
        # Compute hyperbolic distance using cross-ratio
        return torch.acos(cos_theta)
        
    def project_to_tangent_space(
        self,
        v: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Project vector v to the tangent space at point x.
        
        Args:
            v: Vector to project
            x: Point in UHG space
            
        Returns:
            torch.Tensor: Projected vector in tangent space
        """
        # Project to tangent space by removing component parallel to x
        dot_product = torch.dot(v, x)
        norm_x = torch.norm(x, p=2)
        
        # Add epsilon for numerical stability
        denominator = (norm_x * norm_x + self.eps)
        parallel_component = (dot_product / denominator) * x
        
        return v - parallel_component
        
    def exponential_map(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        t: float = 1.0
    ) -> torch.Tensor:
        """
        Compute the exponential map at point x in direction v.
        
        Args:
            x: Base point in UHG space
            v: Tangent vector
            t: Scale factor (default: 1.0)
            
        Returns:
            torch.Tensor: Result of exponential map
        """
        # Project v to tangent space
        v = self.project_to_tangent_space(v, x)
        
        # Compute norm of tangent vector
        norm_v = torch.norm(v, p=2)
        
        # Add epsilon for numerical stability
        denominator = (norm_v + self.eps)
        
        # Compute exponential map
        return x * torch.cos(t * norm_v) + v * (torch.sin(t * norm_v) / denominator)
        
    def logarithmic_map(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the logarithmic map from x to y.
        
        Args:
            x: Base point in UHG space
            y: Target point in UHG space
            
        Returns:
            torch.Tensor: Tangent vector from x to y
        """
        # Compute hyperbolic distance
        d = self.hyperbolic_distance(x, y)
        
        # Project y - x to tangent space
        v = self.project_to_tangent_space(y - x, x)
        
        # Compute norm of projected vector
        norm_v = torch.norm(v, p=2)
        
        # Add epsilon for numerical stability
        denominator = (norm_v + self.eps)
        
        # Compute logarithmic map
        return v * (d / denominator) 