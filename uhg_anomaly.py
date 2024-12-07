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
    feature_dim: int  # Dimension of input features
    window_size: int = 100  # Size of sliding window
    min_baseline_size: int = 10  # Minimum samples needed for baseline
    threshold_factor: float = 2.5  # Factor for anomaly threshold
    use_cross_ratio: bool = True  # Whether to use cross-ratio in scoring
    eps: float = 1e-9  # Numerical stability
    warmup_period: int = 10  # Number of samples to warm up scoring
    min_anomaly_score: float = 2.0  # Minimum score to be considered anomalous
    pattern_threshold: float = 2.0  # Additional threshold for pattern anomalies
    zscore_threshold: float = 1.2  # Z-score threshold for consistent deviations

class UHGAnomalyScorer:
    """UHG-based anomaly scorer using projective geometry."""
    
    def __init__(self, config: UHGAnomalyConfig):
        """Initialize UHG anomaly scorer."""
        self.config = config
        self.uhg = ProjectiveUHG()
        
        # Initialize baseline storage
        self.baseline_features = []
        self.is_baseline_ready = False
        
        # Initialize baseline statistics
        self.mean_baseline = None
        self.baseline_scores = []
        self.mean_baseline_score = torch.tensor(0.0)
        self.std_baseline_score = torch.tensor(1.0)
        self.threshold = None
        
    def _update_baseline(self, features: torch.Tensor):
        """Update baseline with new features."""
        # Add to baseline features
        self.baseline_features.append(features)
        if len(self.baseline_features) > self.config.window_size:
            self.baseline_features.pop(0)
            
        # Update baseline statistics if we have enough samples
        if len(self.baseline_features) >= self.config.min_baseline_size:
            self._compute_baseline_stats()
            
    def _compute_raw_score(self, features: torch.Tensor) -> torch.Tensor:
        """Compute raw anomaly score."""
        # Add homogeneous coordinate
        x_homogeneous = torch.cat([features, torch.ones(1)])
        
        # If we don't have baseline yet, use simple deviation
        if self.mean_baseline is None:
            score = torch.norm(features) / torch.sqrt(torch.tensor(float(len(features))))
            return torch.clamp(score, min=0.0, max=10.0)  # Ensure finite score
            
        # Compute geometric deviation
        deviation = self._compute_deviation(
            x_homogeneous,
            self.mean_baseline
        )
        
        # Compute cross-ratio score if enabled
        if self.config.use_cross_ratio and len(self.baseline_features) >= 3:
            cross_ratio_score = self._compute_cross_ratio_score(features)
            score = (deviation + cross_ratio_score) / 2.0  # Average both scores
        else:
            score = deviation
            
        # Handle NaN scores
        if torch.isnan(score):
            score = torch.norm(features) / torch.sqrt(torch.tensor(float(len(features))))
            
        return torch.clamp(score, min=0.0, max=10.0)  # Ensure finite score
        
    def _compute_baseline_stats(self):
        """Compute baseline statistics."""
        if len(self.baseline_features) < self.config.min_baseline_size:
            return
            
        # Convert baseline features to tensor
        baseline_tensor = torch.stack(self.baseline_features)
        
        # Add homogeneous coordinates
        homogeneous = torch.cat([
            baseline_tensor,
            torch.ones(len(baseline_tensor), 1)
        ], dim=1)
        
        # Compute mean in UHG space
        self.mean_baseline = self._uhg_mean(homogeneous)
        
        # Compute baseline scores
        self.baseline_scores = []
        for features in baseline_tensor:
            score = self._compute_raw_score(features)
            if not torch.isnan(score):
                self.baseline_scores.append(score)
                
        if self.baseline_scores:
            scores = torch.stack(self.baseline_scores)
            
            # Update score statistics using robust estimators
            self.mean_baseline_score = torch.median(scores)
            self.std_baseline_score = torch.median(torch.abs(
                scores - self.mean_baseline_score
            )) + self.config.eps
            
            # Use robust statistics for threshold
            self.threshold = max(
                self.mean_baseline_score + (
                    self.std_baseline_score * self.config.threshold_factor
                ),
                self.config.min_anomaly_score
            )
            
            self.is_baseline_ready = True
            
    def _detect_pattern_anomaly(self, features: torch.Tensor) -> bool:
        """Detect pattern-based anomalies using distribution analysis."""
        if not self.is_baseline_ready:
            return False
            
        # Convert to tensor for computation
        baseline_tensor = torch.stack(self.baseline_features)
        
        # Compute feature-wise statistics
        baseline_mean = torch.mean(baseline_tensor, dim=0)
        baseline_std = torch.std(baseline_tensor, dim=0) + self.config.eps
        
        # Compute z-scores for each feature
        z_scores = torch.abs(features - baseline_mean) / baseline_std
        
        # Check for extreme deviations in any feature
        max_zscore = torch.max(z_scores)
        mean_zscore = torch.mean(z_scores)
        
        # Check for pattern changes
        feature_diffs = torch.abs(features[1:] - features[:-1])
        baseline_diffs = torch.abs(baseline_tensor[:, 1:] - baseline_tensor[:, :-1])
        mean_baseline_diff = torch.mean(baseline_diffs)
        std_baseline_diff = torch.std(baseline_diffs) + self.config.eps
        
        # Compute pattern deviation score
        pattern_score = torch.mean(torch.abs(feature_diffs - mean_baseline_diff)) / std_baseline_diff
        
        # Check for unusual patterns in feature relationships
        feature_ratios = features[1:] / (features[:-1] + self.config.eps)
        baseline_ratios = baseline_tensor[:, 1:] / (baseline_tensor[:, :-1] + self.config.eps)
        mean_baseline_ratio = torch.mean(baseline_ratios)
        std_baseline_ratio = torch.std(baseline_ratios) + self.config.eps
        
        ratio_score = torch.mean(torch.abs(feature_ratios - mean_baseline_ratio)) / std_baseline_ratio
        
        # Combine different pattern indicators
        is_pattern_anomaly = (
            max_zscore > self.config.pattern_threshold or  # Individual feature anomaly
            pattern_score > self.config.pattern_threshold or  # Sequential pattern anomaly
            ratio_score > self.config.pattern_threshold  # Feature relationship anomaly
        )
        
        # Additional check for subtle variations
        if not is_pattern_anomaly:
            # Check if all features are consistently higher/lower than baseline
            consistent_deviation = torch.all(z_scores > 0.8) or torch.all(z_scores < -0.8)
            is_pattern_anomaly = consistent_deviation and mean_zscore > self.config.zscore_threshold
        
        return is_pattern_anomaly
        
    def compute_score(self, features: torch.Tensor) -> Dict[str, Any]:
        """Compute anomaly score for input features."""
        # Compute raw score
        score = self._compute_raw_score(features)
        
        # Update baseline
        self._update_baseline(features)
        
        # Determine if anomalous
        if self.is_baseline_ready and self.threshold is not None:
            # Check for volume-based anomalies
            volume_anomaly = score > self.threshold
            
            # Check for pattern-based anomalies
            pattern_anomaly = self._detect_pattern_anomaly(features)
            
            # Combine both types of detection
            is_anomaly = volume_anomaly or pattern_anomaly
            
            # Additional check for subtle variations within normal range
            if not is_anomaly and score > self.threshold * 0.8:
                # Check if the score is consistently high but not quite anomalous
                recent_scores = torch.tensor(self.baseline_scores[-10:])
                if len(recent_scores) >= 10:
                    recent_mean = torch.mean(recent_scores)
                    recent_std = torch.std(recent_scores) + self.config.eps
                    score_zscore = (score - recent_mean) / recent_std
                    is_anomaly = score_zscore > self.config.zscore_threshold
        else:
            # During warmup, use a simple threshold based on feature dimension
            warmup_threshold = max(
                torch.sqrt(torch.tensor(float(self.config.feature_dim))),
                self.config.min_anomaly_score
            )
            is_anomaly = score > warmup_threshold
            
        return {
            'score': score,
            'is_anomaly': is_anomaly,
            'is_warmup': not self.is_baseline_ready
        }

    def _uhg_mean(self, points: torch.Tensor) -> torch.Tensor:
        """Compute the projective mean in UHG space."""
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
        """Compute geometric deviation in UHG space."""
        # Ensure points are normalized in last coordinate
        x = x / (x[..., -1:] + self.config.eps)
        mean = mean / (mean[..., -1:] + self.config.eps)
        
        # Compute UHG quadrance using projective formula
        dot_prod = torch.sum(x[..., :-1] * mean[..., :-1], dim=-1)
        x_norm = torch.norm(x[..., :-1], dim=-1)
        mean_norm = torch.norm(mean[..., :-1], dim=-1)
        
        # Compute using UHG formula from projective geometry
        cosh_dist = (1 + 2 * dot_prod) / (
            torch.sqrt((1 + x_norm ** 2) * (1 + mean_norm ** 2)) +
            self.config.eps
        )
        
        # Convert to quadrance (sin²(d/2) = (cosh(d) - 1)/(cosh(d) + 1))
        quadrance = torch.abs((cosh_dist - 1) / (cosh_dist + 1 + self.config.eps))
        
        return quadrance
        
    def _compute_cross_ratio_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score based on cross-ratio preservation."""
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
        
        # 3. Point maximizing determinant
        max_det = -float('inf')
        third_idx = None
        for i in range(len(baseline_tensor)):
            if i != central_idx and i != distant_idx:
                v1 = baseline_tensor[distant_idx] - baseline_tensor[central_idx]
                v2 = baseline_tensor[i] - baseline_tensor[central_idx]
                
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

def create_anomaly_scorer(
    feature_dim: int,
    window_size: int = 100,
    threshold_factor: float = 3.0,
    use_cross_ratio: bool = True
) -> UHGAnomalyScorer:
    """
    Factory function to create a UHG anomaly scorer.
    
    Args:
        feature_dim: Dimension of input features
        window_size: Size of window for baseline statistics
        threshold_factor: Factor for anomaly threshold
        use_cross_ratio: Whether to use cross-ratio in scoring
        
    Returns:
        Configured UHGAnomalyScorer instance
    """
    config = UHGAnomalyConfig(
        feature_dim=feature_dim,
        window_size=window_size,
        threshold_factor=threshold_factor,
        use_cross_ratio=use_cross_ratio
    )
    
    return UHGAnomalyScorer(config) 