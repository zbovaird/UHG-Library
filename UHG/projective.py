"""
Universal Hyperbolic Geometry implementation based on projective geometry.

This implementation strictly follows UHG principles:
- Works directly with cross-ratios
- Uses projective transformations
- No differential geometry or manifold concepts
- No curvature parameters
- Pure projective operations

References:
    - UHG.pdf Chapter 3: Projective Geometry
    - UHG.pdf Chapter 4: Cross-ratios and Invariants
    - UHG.pdf Chapter 5: The Fundamental Operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

class ProjectiveUHG:
    """Universal Hyperbolic Geometry operations using projective geometry.
    
    All operations are performed using pure projective geometry,
    without any manifold concepts or tangent space mappings.
    """
    def __init__(self):
        pass
        
    def get_projective_matrix(self, dim: int) -> torch.Tensor:
        """Get a random projective transformation matrix.
        
        Args:
            dim: Dimension of the projective space
            
        Returns:
            Projective transformation matrix
        """
        # Create random matrix
        matrix = torch.randn(dim + 1, dim + 1)
        
        # Make it orthogonal (preserves cross-ratios)
        q, r = torch.linalg.qr(matrix)
        
        # Ensure determinant is positive
        if torch.det(q) < 0:
            q = -q
            
        return q
        
    def transform(self, points: torch.Tensor, matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply projective transformation to points.
        
        Args:
            points: Points to transform [B, N, D+1] or [N, D+1]
            matrix: Optional transformation matrix [D+1, D+1]
            
        Returns:
            Transformed points [B, N, D+1] or [N, D+1]
        """
        # Handle unbatched input
        if points.dim() == 2:
            points = points.unsqueeze(0)  # [N, D+1] -> [1, N, D+1]
            
        # Default to identity if no matrix provided
        if matrix is None:
            matrix = torch.eye(points.shape[-1], device=points.device)
            
        # Ensure matrix has correct dimensions
        if points.shape[-1] != matrix.shape[-1]:
            # Add row and column for homogeneous coordinate
            pad_size = points.shape[-1] - matrix.shape[-1]
            if pad_size > 0:
                pad = torch.zeros(matrix.shape[0] + pad_size, matrix.shape[1] + pad_size, device=points.device)
                pad[:matrix.shape[0], :matrix.shape[1]] = matrix
                pad[-1, -1] = 1.0
                matrix = pad
            else:
                # Truncate matrix if needed
                matrix = matrix[:points.shape[-1], :points.shape[-1]]
            
        # Store original cross-ratio if enough points
        has_cr = points.size(1) >= 4
        if has_cr:
            cr_before = self.cross_ratio(points[:, 0], points[:, 1], points[:, 2], points[:, 3])
            
        # Apply transformation
        transformed = torch.matmul(points, matrix.t())
        
        # Normalize features
        features = transformed[..., :-1]  # [B, N, D]
        homogeneous = transformed[..., -1:]  # [B, N, 1]
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        features = features / (norm + 1e-8)
        transformed = torch.cat([features, homogeneous], dim=-1)
        
        # Restore cross-ratio if needed
        if has_cr:
            cr_after = self.cross_ratio(transformed[:, 0], transformed[:, 1], transformed[:, 2], transformed[:, 3])
            valid_cr = ~torch.isnan(cr_after) & ~torch.isnan(cr_before) & (cr_after != 0)
            if valid_cr.any():
                # Compute scale factor in log space for better numerical stability
                log_scale = 0.5 * (torch.log(cr_before[valid_cr] + 1e-8) - torch.log(cr_after[valid_cr] + 1e-8))
                scale = torch.exp(log_scale)
                
                # Apply scale to features
                features = transformed[..., :-1]
                features = features * scale.view(-1, 1, 1)
                
                # Re-normalize features
                norm = torch.norm(features, p=2, dim=-1, keepdim=True)
                features = features / (norm + 1e-8)
                transformed = torch.cat([features, transformed[..., -1:]], dim=-1)
                
        # Remove batch dimension if input was unbatched
        if points.size(0) == 1:
            transformed = transformed.squeeze(0)
            
        return transformed
        
    def normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        """Normalize points to lie in projective space.
        
        Args:
            points: Points to normalize
            
        Returns:
            Normalized points
        """
        norm = torch.norm(points, p=2, dim=-1, keepdim=True)
        return points / (norm + 1e-8)
        
    def join(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Join two points to create a line.
        
        Args:
            p1: First point
            p2: Second point
            
        Returns:
            Line through the points
        """
        # Extract 3D coordinates for cross product
        p1_3d = p1[..., :3]
        p2_3d = p2[..., :3]
        
        # Use cross product to get line coefficients
        line = torch.linalg.cross(p1_3d, p2_3d, dim=-1)
        
        # Add homogeneous coordinate
        if p1.size(-1) > 3:
            line = torch.cat([line, torch.ones_like(line[..., :1])], dim=-1)
            
        return self.normalize_points(line)
        
    def meet(self, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        """Meet two lines to get their intersection point.
        
        Args:
            l1: First line
            l2: Second line
            
        Returns:
            Intersection point
        """
        # Extract 3D coordinates for cross product
        l1_3d = l1[..., :3]
        l2_3d = l2[..., :3]
        
        # Use cross product to get intersection point
        point = torch.linalg.cross(l1_3d, l2_3d, dim=-1)
        
        # Add homogeneous coordinate
        if l1.size(-1) > 3:
            point = torch.cat([point, torch.ones_like(point[..., :1])], dim=-1)
            
        return self.normalize_points(point)
        
    def get_ideal_points(self, line: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the ideal points on a line.
        
        These are the points where the line intersects the absolute conic.
        
        Args:
            line: Line to find ideal points on
            
        Returns:
            Tuple of two ideal points
        """
        # Get coefficients of line
        a, b, c = line[..., :3].unbind(-1)
        
        # Solve quadratic equation ax^2 + by^2 + c = 0
        discriminant = torch.sqrt(b**2 - 4*a*c + 1e-8)
        x1 = (-b + discriminant) / (2*a + 1e-8)
        x2 = (-b - discriminant) / (2*a + 1e-8)
        
        # Create homogeneous coordinates
        p1 = torch.stack([x1, torch.ones_like(x1), torch.zeros_like(x1)], dim=-1)
        p2 = torch.stack([x2, torch.ones_like(x2), torch.zeros_like(x2)], dim=-1)
        
        # Add homogeneous coordinate if needed
        if line.size(-1) > 3:
            p1 = torch.cat([p1, torch.ones_like(p1[..., :1])], dim=-1)
            p2 = torch.cat([p2, torch.ones_like(p2[..., :1])], dim=-1)
            
        return self.normalize_points(p1), self.normalize_points(p2)
        
    def cross_ratio(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        """Compute the cross-ratio of four points.
        
        Args:
            p1: First point
            p2: Second point
            p3: Third point
            p4: Fourth point
            
        Returns:
            Cross-ratio value
        """
        # Compute distances using dot products
        d12 = torch.sum(p1 * p2, dim=-1)
        d34 = torch.sum(p3 * p4, dim=-1)
        d13 = torch.sum(p1 * p3, dim=-1)
        d24 = torch.sum(p2 * p4, dim=-1)
        
        # Compute cross-ratio
        return (d12 * d34) / (d13 * d24 + 1e-8)
        
    def projective_average(self, points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted average of points in projective space.
        
        Args:
            points: Points to average [B, N, D+1] or [N, D+1]
            weights: Weights for averaging [N]
            
        Returns:
            Averaged point [B, D+1] or [D+1]
        """
        # Handle unbatched input
        if points.dim() == 2:
            points = points.unsqueeze(0)  # [N, D+1] -> [1, N, D+1]
            
        # Extract features and homogeneous coordinates
        features = points[..., :-1]  # [B, N, D]
        homogeneous = points[..., -1:]  # [B, N, 1]
        
        # Reshape weights for broadcasting
        weights = weights.view(1, -1, 1)  # [N] -> [1, N, 1]
        
        # Apply weights
        weighted_features = torch.sum(features * weights, dim=1)  # [B, D]
        weighted_homogeneous = torch.sum(homogeneous * weights, dim=1)  # [B, 1]
        
        # Normalize features
        norm = torch.norm(weighted_features, p=2, dim=-1, keepdim=True)
        weighted_features = weighted_features / (norm + 1e-8)
        
        # Combine features and homogeneous coordinates
        out = torch.cat([weighted_features, weighted_homogeneous], dim=-1)  # [B, D+1]
        
        # Remove batch dimension if input was unbatched
        if points.size(0) == 1:
            out = out.squeeze(0)  # [1, D+1] -> [D+1]
            
        return out
        
    def absolute_polar(self, line: torch.Tensor) -> torch.Tensor:
        """Get the polar of a line with respect to the absolute conic.
        
        Args:
            line: Line to find polar of
            
        Returns:
            Polar point
        """
        # For the absolute conic x^2 + y^2 = z^2, the polar is simple
        return self.normalize_points(line)
        
    def normalize(self, points: torch.Tensor) -> torch.Tensor:
        """Normalize points to lie on the unit sphere while preserving cross-ratios.
        
        Args:
            points: Points to normalize [N, D] or [B, N, D]
            
        Returns:
            Normalized points with same shape
        """
        # Handle unbatched input
        if points.dim() == 2:
            points = points.unsqueeze(0)  # [N, D] -> [1, N, D]
            
        # Store original cross-ratio if enough points
        has_cr = points.size(1) > 3
        if has_cr:
            cr_before = self.cross_ratio(points[:, 0], points[:, 1], points[:, 2], points[:, 3])
            
        # Compute norms
        norms = torch.norm(points, p=2, dim=-1, keepdim=True)
        
        # Normalize points
        normalized = points / (norms + 1e-8)
        
        # Restore cross-ratio if needed
        if has_cr:
            cr_after = self.cross_ratio(normalized[:, 0], normalized[:, 1], normalized[:, 2], normalized[:, 3])
            valid_cr = ~torch.isnan(cr_after) & ~torch.isnan(cr_before) & (cr_after != 0)
            if valid_cr.any():
                # Compute scale factor in log space for better numerical stability
                log_scale = 0.5 * (torch.log(cr_before[valid_cr] + 1e-8) - torch.log(cr_after[valid_cr] + 1e-8))
                scale = torch.exp(log_scale)
                
                # Apply scale while preserving unit norm
                scaled = normalized * scale.view(-1, 1, 1)
                norms = torch.norm(scaled, p=2, dim=-1, keepdim=True)
                normalized = scaled / (norms + 1e-8)
                
        # Remove batch dimension if input was unbatched
        if points.size(0) == 1:
            normalized = normalized.squeeze(0)
            
        return normalized
        
    def distance(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Compute projective distance between two points.
        
        Args:
            p1, p2: Points in projective space
            
        Returns:
            Distance value
        """
        # Extract features (ignore homogeneous coordinate)
        p1_feat = p1[:-1]
        p2_feat = p2[:-1]
        
        # Normalize features
        p1_feat = p1_feat / (torch.norm(p1_feat, p=2) + 1e-8)
        p2_feat = p2_feat / (torch.norm(p2_feat, p=2) + 1e-8)
        
        # Compute distance using cosine similarity
        cos_sim = torch.sum(p1_feat * p2_feat)
        # Map from [-1, 1] to [0, 2] for better numerical stability
        return torch.sqrt(2 * (1 - cos_sim.clamp(-1 + 1e-8, 1 - 1e-8)))
        
    def cross_ratio(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        """Compute cross-ratio of four points.
        
        Args:
            p1, p2, p3, p4: Points in projective space
            
        Returns:
            Cross-ratio value
        """
        # Extract features (ignore homogeneous coordinate)
        p1_feat = p1[:-1]
        p2_feat = p2[:-1]
        p3_feat = p3[:-1]
        p4_feat = p4[:-1]
        
        # Normalize features
        p1_feat = p1_feat / (torch.norm(p1_feat, p=2) + 1e-8)
        p2_feat = p2_feat / (torch.norm(p2_feat, p=2) + 1e-8)
        p3_feat = p3_feat / (torch.norm(p3_feat, p=2) + 1e-8)
        p4_feat = p4_feat / (torch.norm(p4_feat, p=2) + 1e-8)
        
        # Compute distances
        d13 = torch.norm(p1_feat - p3_feat, p=2)
        d24 = torch.norm(p2_feat - p4_feat, p=2)
        d14 = torch.norm(p1_feat - p4_feat, p=2)
        d23 = torch.norm(p2_feat - p3_feat, p=2)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        
        # Compute cross-ratio with numerical stability
        # Use log-space computation to prevent overflow/underflow
        log_cr = torch.log(d13 + eps) + torch.log(d24 + eps) - torch.log(d14 + eps) - torch.log(d23 + eps)
        cr = torch.exp(log_cr)
        
        return cr
        
    def scale(self, points: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Scale points while preserving projective structure.
        
        Args:
            points: Points to scale
            scale: Scale factor
            
        Returns:
            Scaled points
        """
        # Store original cross-ratio if enough points
        has_cr = points.size(0) > 3
        if has_cr:
            cr_before = self.cross_ratio(points[0], points[1], points[2], points[3])
            
        # Extract features and homogeneous coordinate
        features = points[..., :-1]
        homogeneous = points[..., -1:]
        
        # Apply scale to features
        scaled_features = features * scale
        
        # Normalize features
        norm = torch.norm(scaled_features, p=2, dim=-1, keepdim=True)
        scaled_features = scaled_features / (norm + 1e-8)
        scaled = torch.cat([scaled_features, homogeneous], dim=-1)
        
        # Restore cross-ratio if needed
        if has_cr:
            cr_after = self.cross_ratio(scaled[0], scaled[1], scaled[2], scaled[3])
            if not torch.isnan(cr_after) and not torch.isnan(cr_before) and cr_after != 0:
                scale = torch.sqrt(torch.abs(cr_before / cr_after))
                features = scaled[..., :-1] * scale
                # Re-normalize after scaling
                norm = torch.norm(features, p=2, dim=-1, keepdim=True)
                features = features / (norm + 1e-8)
                scaled = torch.cat([features, scaled[..., -1:]], dim=-1)
                
        return scaled
        
    def aggregate(self, points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Aggregate points using weighted average.
        
        Args:
            points: Points to aggregate
            weights: Aggregation weights
            
        Returns:
            Aggregated points
        """
        # Store original cross-ratio if enough points
        has_cr = points.size(0) > 3
        if has_cr:
            cr_before = self.cross_ratio(points[0], points[1], points[2], points[3])
            
        # Apply weights
        weighted = torch.matmul(weights, points)
        
        # Normalize features
        features = weighted[..., :-1]
        homogeneous = weighted[..., -1:]
        norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        features = features / (norm + 1e-8)
        
        return torch.cat([features, homogeneous], dim=-1)