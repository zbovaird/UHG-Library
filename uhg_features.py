"""
UHG Feature Extraction Module

This module provides UHG-aware feature extraction capabilities, transforming raw network data
into a feature space that preserves hyperbolic geometric relationships.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from uhg.projective import ProjectiveUHG

@dataclass
class UHGFeatureConfig:
    """Configuration for UHG feature extraction."""
    input_dim: int
    output_dim: int
    preserve_cross_ratio: bool = True
    use_homogeneous_coords: bool = True
    eps: float = 1e-9

class UHGFeatureExtractor:
    """Extract UHG-aware features from network data."""
    
    def __init__(self, config: UHGFeatureConfig):
        """
        Initialize the UHG feature extractor.
        
        Args:
            config: Configuration parameters for feature extraction
        """
        self.config = config
        self.uhg = ProjectiveUHG()
        
        # Initialize transformation matrices
        self.init_transforms()
    
    def init_transforms(self):
        """Initialize UHG-aware transformation matrices."""
        # Create projective transformation matrix
        self.proj_transform = torch.eye(
            self.config.input_dim + 1 if self.config.use_homogeneous_coords 
            else self.config.input_dim
        )
        
        # Calculate feature dimensions
        if self.config.preserve_cross_ratio:
            cr_dim = self.config.input_dim - 2  # Number of cross-ratio features
        else:
            cr_dim = 0
            
        geom_dim = self.config.input_dim + 2  # Axis distances + norm + homogeneous ratio
        total_feature_dim = cr_dim + geom_dim
        
        # Initialize feature mapping to preserve geometric structure
        self.feature_map = nn.Parameter(
            self._init_geometric_feature_map(total_feature_dim, self.config.output_dim)
        )
        
        # Initialize for cross-ratio preservation
        if self.config.preserve_cross_ratio:
            self.cross_ratio_basis = self._init_cross_ratio_basis()
    
    def _init_geometric_feature_map(self, in_dim: int, out_dim: int) -> torch.Tensor:
        """Initialize feature map to preserve geometric structure."""
        # Create orthogonal matrix for better geometric preservation
        U, _, V = torch.linalg.svd(torch.randn(in_dim, out_dim))
        if in_dim >= out_dim:
            return U[:, :out_dim]
        else:
            return V[:in_dim, :]
    
    def _normalize_points(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize points to lie on the UHG manifold."""
        # Ensure points have unit norm in last coordinate
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / (norm + self.config.eps)
    
    def _init_cross_ratio_basis(self) -> torch.Tensor:
        """Initialize basis vectors that help preserve cross-ratio."""
        # Create basis vectors in projective space that are well-separated
        basis = torch.eye(self.config.input_dim + 1)
        # Add some random perturbation for better separation
        basis = basis + 0.1 * torch.randn_like(basis)
        # Ensure they satisfy cross-ratio preservation
        basis = self._normalize_points(basis)
        return basis
    
    def add_homogeneous_coordinate(self, x: torch.Tensor) -> torch.Tensor:
        """Add homogeneous coordinate to input tensor."""
        ones = torch.ones(*x.shape[:-1], 1, device=x.device)
        return torch.cat([x, ones], dim=-1)
    
    def _compute_quadrance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute quadrance between two points in UHG space."""
        # Compute dot product
        dot_prod = torch.sum(x * y, dim=-1)
        
        # Compute norms excluding homogeneous coordinate
        x_norm = torch.norm(x[..., :-1], dim=-1)
        y_norm = torch.norm(y[..., :-1], dim=-1)
        
        # Compute quadrance with better numerical stability
        numer = dot_prod ** 2
        denom = (x_norm ** 2 + self.config.eps) * (y_norm ** 2 + self.config.eps)
        return 1 - torch.clamp(numer / denom, min=0.0, max=1.0)
    
    def _compute_cross_ratio(self, p1: torch.Tensor, p2: torch.Tensor, 
                           p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        """Compute cross-ratio of four points with better numerical stability."""
        # Compute quadrances
        q12 = self._compute_quadrance(p1, p2)
        q34 = self._compute_quadrance(p3, p4)
        q13 = self._compute_quadrance(p1, p3)
        q24 = self._compute_quadrance(p2, p4)
        
        # Compute cross-ratio with clamping for stability
        numer = q12 * q34
        denom = q13 * q24 + self.config.eps
        ratio = numer / denom
        return torch.clamp(ratio, min=-10.0, max=10.0)
    
    def compute_uhg_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute UHG-aware features while preserving geometric structure.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of UHG-aware features of shape (batch_size, output_dim)
        """
        # Add homogeneous coordinate if needed
        if self.config.use_homogeneous_coords:
            x = self.add_homogeneous_coordinate(x)
        
        # Apply projective transformation
        x = torch.matmul(x, self.proj_transform)
        
        # Normalize in UHG space
        x = self._normalize_points(x)
        
        # Extract geometric features
        features = []
        
        # Compute cross-ratio based features if enabled
        if self.config.preserve_cross_ratio:
            cr_features = self._compute_cross_ratio_features(x)
            features.append(cr_features)
        
        # Compute basic geometric features
        geom_features = self._compute_geometric_features(x)
        features.append(geom_features)
        
        # Combine all features
        combined = torch.cat(features, dim=-1)
        
        # Project to desired output dimension while preserving structure
        output = torch.matmul(combined, self.feature_map)
        
        # Normalize output to maintain geometric relationships
        output = self._normalize_points(
            torch.cat([output, torch.ones_like(output[..., :1])], dim=-1)
        )[..., :-1]
        
        return output
    
    def _compute_cross_ratio_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute features based on cross-ratio preservation."""
        batch_size = x.shape[0]
        features = []
        
        # Use basis vectors as reference points
        for i in range(len(self.cross_ratio_basis) - 3):
            # Select four points for cross-ratio
            p1 = x
            p2 = self.cross_ratio_basis[i:i+1].expand(batch_size, -1)
            p3 = self.cross_ratio_basis[i+1:i+2].expand(batch_size, -1)
            p4 = self.cross_ratio_basis[i+2:i+3].expand(batch_size, -1)
            
            # Compute cross-ratio
            cr = self._compute_cross_ratio(p1, p2, p3, p4)
            features.append(cr)
        
        return torch.stack(features, dim=-1)
    
    def _compute_geometric_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute basic geometric features in UHG space."""
        # Compute distances to coordinate axes
        axis_distances = []
        for i in range(self.config.input_dim):
            axis = torch.zeros_like(x)
            axis[..., i] = 1
            dist = self._compute_quadrance(x, axis)
            axis_distances.append(dist)
        
        # Compute basic invariants with better stability
        norm_features = torch.norm(x[..., :-1], dim=-1, keepdim=True)
        norm_features = norm_features / (torch.max(norm_features) + self.config.eps)
        
        homogeneous_ratio = x[..., -1:] / (torch.norm(x, dim=-1, keepdim=True) + self.config.eps)
        homogeneous_ratio = torch.clamp(homogeneous_ratio, min=0.0, max=1.0)
        
        # Combine geometric features
        return torch.cat([
            torch.stack(axis_distances, dim=-1),
            norm_features,
            homogeneous_ratio
        ], dim=-1)
    
    def extract_features(self, 
                        data: torch.Tensor,
                        return_intermediate: bool = False
                        ) -> Dict[str, torch.Tensor]:
        """
        Main feature extraction method.
        
        Args:
            data: Input data tensor
            return_intermediate: Whether to return intermediate computations
            
        Returns:
            Dictionary containing extracted features and optionally intermediate results
        """
        results = {}
        
        # Ensure input is tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Compute UHG features
        uhg_features = self.compute_uhg_features(data)
        results['features'] = uhg_features
        
        if return_intermediate:
            # Store intermediate computations
            if self.config.use_homogeneous_coords:
                homogeneous = self.add_homogeneous_coordinate(data)
                results['homogeneous'] = homogeneous
            
            if self.config.preserve_cross_ratio:
                cr_features = self._compute_cross_ratio_features(
                    self.add_homogeneous_coordinate(data)
                )
                results['cross_ratio_features'] = cr_features
            
            geom_features = self._compute_geometric_features(
                self.add_homogeneous_coordinate(data)
            )
            results['geometric_features'] = geom_features
        
        return results

def create_feature_extractor(
    input_dim: int,
    output_dim: int,
    preserve_cross_ratio: bool = True,
    use_homogeneous_coords: bool = True
) -> UHGFeatureExtractor:
    """
    Factory function to create a UHG feature extractor.
    
    Args:
        input_dim: Dimension of input data
        output_dim: Desired dimension of output features
        preserve_cross_ratio: Whether to preserve cross-ratio in features
        use_homogeneous_coords: Whether to use homogeneous coordinates
        
    Returns:
        Configured UHGFeatureExtractor instance
    """
    config = UHGFeatureConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        preserve_cross_ratio=preserve_cross_ratio,
        use_homogeneous_coords=use_homogeneous_coords
    )
    return UHGFeatureExtractor(config) 