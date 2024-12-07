"""
UHG Anomaly Detection Module.

This module implements anomaly detection using pure UHG principles,
working directly in projective space without Euclidean assumptions.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

class UHGAnomalyScorer:
    """Detect anomalies using pure UHG principles."""
    
    def __init__(self, feature_dim: int):
        """Initialize UHG anomaly detector."""
        self.feature_dim = feature_dim
        self.baseline_features: List[torch.Tensor] = []
        self.ideal_points = torch.randn(2, feature_dim + 1)  # Two ideal points for cross-ratio
        self.is_baseline_ready = False
        self.eps = 1e-9
        
    def _cross_ratio(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        i1: torch.Tensor,
        i2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-ratio in projective space."""
        # Compute projective distances using determinants
        def proj_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            n = x.shape[-1]
            dist = 0.0
            for i in range(n-1):
                for j in range(i+1, n):
                    det = x[..., i] * y[..., j] - x[..., j] * y[..., i]
                    dist = dist + det * det
            return torch.sqrt(dist + self.eps)
        
        # Compute cross-ratio using projective distances
        ac = proj_dist(a, i1)
        bd = proj_dist(b, i2)
        ad = proj_dist(a, i2)
        bc = proj_dist(b, i1)
        
        return (ac * bd) / (ad * bc + self.eps)
        
    def _projective_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic distance using cross-ratio with ideal points."""
        # Get ideal points
        i1, i2 = self.ideal_points[0], self.ideal_points[1]
        
        # Compute distance using cross-ratio
        cr = self._cross_ratio(x, y, i1, i2)
        return torch.abs(torch.log(cr + self.eps))
        
    def _find_reference_points(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find stable reference points using projective geometry."""
        n = len(points)
        max_spread = -float('inf')
        best_pair = (0, 1)
        
        # Find pair of points with maximum projective spread
        for i in range(n-1):
            for j in range(i+1, n):
                spread = self._projective_distance(points[i], points[j])
                if spread > max_spread:
                    max_spread = spread
                    best_pair = (i, j)
                    
        return points[best_pair[0]], points[best_pair[1]]
        
    def add_baseline_sample(self, features: torch.Tensor):
        """Add baseline sample for anomaly detection."""
        # Add homogeneous coordinate if needed
        if features.shape[-1] == self.feature_dim:
            features = torch.cat([features, torch.ones_like(features[..., :1])], dim=-1)
            
        self.baseline_features.append(features)
        
    def finalize_baseline(self):
        """Finalize baseline by computing reference points."""
        if len(self.baseline_features) < 2:
            raise ValueError("Need at least 2 baseline samples")
            
        # Stack baseline features
        baseline_tensor = torch.stack(self.baseline_features)
        
        # Find reference points using projective geometry
        ref1, ref2 = self._find_reference_points(baseline_tensor)
        
        # Update ideal points using reference points
        self.ideal_points = torch.stack([ref1, ref2])
        self.is_baseline_ready = True
        
    def compute_score(self, features: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score using cross-ratio preservation."""
        if not self.is_baseline_ready:
            raise ValueError("Baseline not finalized")
            
        # Add homogeneous coordinate if needed
        if features.shape[-1] == self.feature_dim:
            features = torch.cat([features, torch.ones_like(features[..., :1])], dim=-1)
            
        # Stack baseline features
        baseline_tensor = torch.stack(self.baseline_features)
        
        # Compute distances to reference points
        scores = []
        for baseline in baseline_tensor:
            dist = self._projective_distance(features, baseline)
            scores.append(dist)
            
        # Convert to tensor and compute final score
        scores = torch.stack(scores)
        return torch.mean(scores)  # Higher score = more anomalous 