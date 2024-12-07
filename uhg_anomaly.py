"""
UHG Anomaly Scoring Module

This module provides anomaly scoring capabilities using UHG geometric principles.
All computations are performed in projective space to maintain UHG invariance.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from uhg.projective import ProjectiveUHG
from uhg_features import UHGFeatureExtractor, UHGFeatureConfig

@dataclass
class UHGAnomalyConfig:
    """Configuration for UHG anomaly detection."""
    feature_dim: int
    window_size: int = 100  # Window for baseline statistics
    alpha: float = 0.05  # Significance level for anomaly threshold
    use_cross_ratio: bool = True  # Whether to use cross-ratio in scoring
    eps: float = 1e-9  # Numerical stability
    min_baseline_size: int = 10  # Minimum number of baseline samples
    threshold_factor: float = 3.0  # Factor for threshold calculation

class UHGAnomalyScorer:
    """
    Detect anomalies using UHG geometric principles.
    
    This class computes anomaly scores based on:
    1. Geometric deviations in UHG space
    2. Cross-ratio preservation
    3. Hyperbolic invariants
    """
    
    def __init__(self, config: UHGAnomalyConfig):
        """
        Initialize UHG anomaly scorer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.uhg = ProjectiveUHG()
        
        # Initialize baseline storage
        self.baseline_features = []
        self.is_baseline_ready = False
        
        # Initialize baseline statistics
        self.mean_baseline = None
        self.mean_baseline_score = torch.tensor(0.0)
        self.std_baseline_score = torch.tensor(1.0)
        self.threshold = torch.tensor(float('inf'))
        
    def _update_baseline(self, features: torch.Tensor):
        """Update baseline with new features."""
        self.baseline_features.append(features)
        if len(self.baseline_features) > self.config.window_size:
            self.baseline_features.pop(0)
        self._compute_baseline_stats()
        
    def _compute_baseline_stats(self):
        """
        Compute baseline statistics for anomaly detection.
        """
        if len(self.baseline_features) < self.config.min_baseline_size:
            return
            
        # Add homogeneous coordinates
        homogeneous = torch.cat([
            torch.stack(self.baseline_features),
            torch.ones(len(self.baseline_features), 1)
        ], dim=1)
        
        # Compute mean in UHG space
        self.mean_baseline = self._uhg_mean(homogeneous)
        
        # Compute baseline scores
        baseline_scores = []
        for features in self.baseline_features:
            score_dict = self.compute_score(features)
            if not torch.isnan(score_dict['score']):
                baseline_scores.append(score_dict['score'])
        
        if baseline_scores:
            baseline_scores = torch.stack(baseline_scores)
            
            # Use robust statistics
            self.mean_baseline_score = torch.median(baseline_scores)
            self.std_baseline_score = torch.median(torch.abs(
                baseline_scores - self.mean_baseline_score
            ))
            
            # Set threshold using median absolute deviation
            self.threshold = (
                self.mean_baseline_score +
                self.config.threshold_factor * self.std_baseline_score
            )
            
            self.is_baseline_ready = True
    
    def _uhg_mean(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute the projective mean in UHG space.
        Uses proper projective averaging to preserve UHG structure.
        """
        # Convert list to tensor if needed
        if isinstance(points, list):
            points = torch.stack(points)
            
        # First normalize points to unit norm in last coordinate
        norm = torch.norm(points, dim=1, keepdim=True)
        normalized = points / (norm + self.config.eps)
        
        # Compute projective average using cross-ratio preservation
        n_points = len(points)
        if n_points == 1:
            return normalized[0]
            
        # Use iterative projective averaging
        current_mean = normalized[0]
        for i in range(1, n_points):
            # Weight for progressive averaging
            weight = torch.tensor(1.0 / (i + 1), device=points.device)
            
            # Compute projective interpolation
            dot_prod = torch.sum(current_mean * normalized[i])
            
            # Convert all operations to tensor operations
            one = torch.tensor(1.0, device=points.device)
            numerator = (one - weight)
            denominator = one + weight + 2 * torch.sqrt(one - weight) * dot_prod
            factor = torch.sqrt(numerator / (denominator + self.config.eps))
            
            # Update mean using projective combination
            current_mean = (
                torch.sqrt(one - weight) * factor * current_mean + 
                torch.sqrt(weight) * normalized[i] / factor
            )
            
            # Normalize to maintain projective structure
            current_mean = current_mean / (torch.norm(current_mean) + self.config.eps)
        
        return current_mean
    
    def _compute_deviation(self, x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """
        Compute geometric deviation in UHG space.
        Uses proper UHG quadrance for distance calculation.
        """
        # Ensure points are normalized in last coordinate
        x = x / (x[..., -1:] + self.config.eps)
        mean = mean / (mean[..., -1:] + self.config.eps)
        
        # Compute UHG quadrance using projective formula
        # Q(x,y) = sin²(d/2) where d is the hyperbolic distance
        dot_prod = torch.sum(x[..., :-1] * mean[..., :-1], dim=-1)
        x_norm = torch.norm(x[..., :-1], dim=-1)
        mean_norm = torch.norm(mean[..., :-1], dim=-1)
        
        # Compute using UHG formula from projective geometry
        cosh_dist = (1 + 2 * dot_prod) / (
            torch.sqrt((1 + x_norm ** 2) * (1 + mean_norm ** 2)) +
            self.config.eps
        )
        
        # Convert to quadrance (sin²(d/2) = (cosh(d) - 1)/(cosh(d) + 1))
        # Add small epsilon to ensure positive values
        quadrance = torch.abs((cosh_dist - 1) / (cosh_dist + 1 + self.config.eps))
        
        return quadrance
    
    def _compute_cross_ratio_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score based on cross-ratio preservation.
        Uses stable reference points to ensure reliable scoring.
        """
        if not self.is_baseline_ready:
            return torch.tensor(0.0)
        
        # Convert baseline features to tensor
        baseline_tensor = torch.stack(self.baseline_features)
        
        # Compute stable reference points using geometric structure
        # 1. Most central point
        mean = self._uhg_mean(baseline_tensor)
        central_idx = torch.argmin(torch.sum(
            (baseline_tensor - mean) ** 2,
            dim=1
        ))
        
        # 2. Most distant point from central
        dists_from_central = torch.sum(
            (baseline_tensor - baseline_tensor[central_idx]) ** 2,
            dim=1
        )
        distant_idx = torch.argmax(dists_from_central)
        
        # 3. Point maximizing determinant (generalization of triangle area)
        max_det = -float('inf')
        third_idx = None
        for i in range(len(baseline_tensor)):
            if i != central_idx and i != distant_idx:
                # Create matrix of differences
                v1 = baseline_tensor[distant_idx] - baseline_tensor[central_idx]
                v2 = baseline_tensor[i] - baseline_tensor[central_idx]
                
                # Compute determinant of 2x2 blocks
                det_sum = 0
                for j in range(len(v1)-1):
                    for k in range(j+1, len(v1)-1):
                        det = v1[j] * v2[k] - v1[k] * v2[j]
                        det_sum += det * det
                
                if det_sum > max_det:
                    max_det = det_sum
                    third_idx = i
        
        if third_idx is None:
            third_idx = (central_idx + 1) % len(baseline_tensor)
        
        # Get reference points with homogeneous coordinates
        ref_points = torch.stack([
            torch.cat([baseline_tensor[central_idx], torch.ones(1)]),
            torch.cat([baseline_tensor[distant_idx], torch.ones(1)]),
            torch.cat([baseline_tensor[third_idx], torch.ones(1)])
        ])
        
        # Add homogeneous coordinate to input
        x_homogeneous = torch.cat([x, torch.ones(1)])
        
        # Compute cross-ratio
        cr = self.uhg.cross_ratio(
            x_homogeneous,
            ref_points[0],
            ref_points[1],
            ref_points[2]
        )
        
        # Compare with baseline cross-ratios
        baseline_crs = []
        for i in range(len(baseline_tensor)):
            point = torch.cat([
                baseline_tensor[i],
                torch.ones(1)
            ])
            baseline_cr = self.uhg.cross_ratio(
                point,
                ref_points[0],
                ref_points[1],
                ref_points[2]
            )
            baseline_crs.append(baseline_cr)
        
        baseline_crs = torch.stack(baseline_crs)
        cr_mean = torch.mean(baseline_crs)
        cr_std = torch.std(baseline_crs)
        
        # Compute normalized cross-ratio deviation
        cr_score = torch.abs(cr - cr_mean) / (cr_std + self.config.eps)
        
        return cr_score
    
    def compute_score(self, features: torch.Tensor) -> Dict[str, Any]:
        """
        Compute anomaly score for input features.
        
        Args:
            features: Input feature vector
            
        Returns:
            Dictionary containing score and anomaly flag
        """
        if not self.is_baseline_ready:
            return {
                'score': torch.tensor(0.0),
                'is_anomaly': False
            }
        
        # Compute geometric deviation
        homogeneous = torch.cat([features, torch.ones(1)])
        deviation = self._compute_deviation(
            homogeneous,
            self.mean_baseline
        )
        
        # Compute cross-ratio score
        cross_ratio_score = self._compute_cross_ratio_score(features)
        
        # Combine scores with numerical stability
        combined_score = torch.clamp(
            deviation + cross_ratio_score,
            min=0.0,
            max=10.0  # Cap maximum score
        )
        
        # Normalize score using robust statistics
        normalized_score = (combined_score - self.mean_baseline_score) / (
            self.std_baseline_score + self.config.eps
        )
        
        # Use robust thresholding
        is_anomaly = normalized_score > self.threshold
        
        # Handle numerical instability
        if torch.isnan(normalized_score):
            normalized_score = torch.tensor(0.0)
            is_anomaly = False
        
        return {
            'score': normalized_score,
            'is_anomaly': is_anomaly
        }

def create_anomaly_scorer(
    feature_dim: int,
    window_size: int = 100,
    alpha: float = 0.05,
    use_cross_ratio: bool = True
) -> UHGAnomalyScorer:
    """
    Factory function to create a UHG anomaly scorer.
    
    Args:
        feature_dim: Dimension of input features
        window_size: Size of window for baseline statistics
        alpha: Significance level for anomaly threshold
        use_cross_ratio: Whether to use cross-ratio in scoring
        
    Returns:
        Configured UHGAnomalyScorer instance
    """
    config = UHGAnomalyConfig(
        feature_dim=feature_dim,
        window_size=window_size,
        alpha=alpha,
        use_cross_ratio=use_cross_ratio
    )
    return UHGAnomalyScorer(config) 