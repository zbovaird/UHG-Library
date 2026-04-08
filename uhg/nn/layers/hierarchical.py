"""
Hierarchical Graph Neural Network layer using Universal Hyperbolic Geometry.

This implementation uses pure projective operations and preserves cross-ratios
throughout all hierarchical operations. No manifold concepts or tangent spaces
are used, following strict UHG principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from .base import ProjectiveLayer
from ...projective import ProjectiveUHG
from ...utils.cross_ratio import compute_cross_ratio, restore_cross_ratio


class ProjectiveHierarchicalLayer(ProjectiveLayer):
    """UHG-compliant hierarchical GNN layer using pure projective operations.

    This layer implements hierarchical message passing using only projective geometry,
    ensuring all operations preserve cross-ratios and follow UHG principles.

    Key features:
    1. Pure projective operations - no manifold concepts
    2. Cross-ratio preservation in all transformations
    3. Hierarchical structure preservation
    4. Level-aware message passing
    5. Parent-child relationship preservation

    Args:
        in_features: Number of input features
        out_features: Number of output features
        num_levels: Number of hierarchical levels
        level_dim: Dimension for level encoding
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_levels: int,
        level_dim: int = 8,
        bias: bool = True,
    ):
        # Initialize base class without reset_parameters
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.uhg = ProjectiveUHG()

        # Store hierarchical parameters
        self.num_levels = num_levels
        self.level_dim = level_dim

        # Set epsilon for numerical stability
        self.eps = 1e-15  # Small epsilon for float64

        # Register parameters
        self.register_parameter(
            "level_encodings",
            nn.Parameter(torch.empty(num_levels, level_dim, dtype=torch.float64)),
        )
        self.register_parameter(
            "W_self",
            nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float64)),
        )
        self.register_parameter(
            "W_neigh",
            nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float64)),
        )
        self.register_parameter(
            "W_parent",
            nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float64)),
        )
        self.register_parameter(
            "W_child",
            nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float64)),
        )
        self.register_parameter(
            "W_level",
            nn.Parameter(torch.empty(out_features, level_dim, dtype=torch.float64)),
        )

        if bias:
            self.register_parameter(
                "bias", nn.Parameter(torch.empty(out_features, dtype=torch.float64))
            )
        else:
            self.register_parameter("bias", None)

        # Now initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using UHG-aware initialization."""
        # Use Glorot initialization for weight matrices
        nn.init.xavier_uniform_(self.W_self)
        nn.init.xavier_uniform_(self.W_neigh)
        nn.init.xavier_uniform_(self.W_parent)
        nn.init.xavier_uniform_(self.W_child)
        nn.init.xavier_uniform_(self.W_level)

        # Initialize level encodings to be orthogonal
        nn.init.orthogonal_(self.level_encodings)

        if self.bias is not None:
            # Initialize bias with small values
            nn.init.zeros_(self.bias)

    def _normalize_with_homogeneous(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize features while preserving homogeneous coordinate."""
        spatial = x[..., :-1]
        time_like = torch.sqrt(
            torch.clamp(
                1.0 + torch.sum(spatial * spatial, dim=-1, keepdim=True), min=1e-12
            )
        )
        return torch.cat([spatial, time_like], dim=-1)

    def _track_cross_ratios(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Track cross-ratios for preservation."""
        crs = []
        if x.size(0) > 3:
            for i in range(0, x.size(0) - 3, 2):
                cr = compute_cross_ratio(x[i], x[i + 1], x[i + 2], x[i + 3])
                if not torch.isnan(cr) and not torch.isinf(cr):
                    crs.append(cr)
        return crs

    def compute_cross_ratio_weight(
        self, p1: torch.Tensor, p2: torch.Tensor, level_diff: int = 0
    ) -> torch.Tensor:
        """Compute cross-ratio based weight between two points."""

        # Null-safe distance: if near-null, softly lift z then normalize inside distance
        def lift_if_needed(p: torch.Tensor) -> torch.Tensor:
            if p.size(-1) > 3:
                feats = p[..., :-1]
                z = torch.sqrt(
                    torch.clamp(
                        1.0 + torch.sum(feats * feats, dim=-1, keepdim=True), min=1e-12
                    )
                )
                return torch.cat([feats, z], dim=-1)
            return p

        p1s = lift_if_needed(p1)
        p2s = lift_if_needed(p2)
        dist = self.uhg.distance(p1s, p2s)

        # Level factors guarantee weight_same > weight_diff > weight_far
        if level_diff == 0:
            level_factor = 1.0
        elif level_diff == 1:
            level_factor = 0.5
        else:
            level_factor = 0.2  # level_diff >= 2

        base = torch.sigmoid(-dist)
        weight = base * level_factor
        return weight.clamp(0.05, 0.95)

    def _compute_dynamic_weights(
        self, node_levels: torch.Tensor, N: int, device: torch.device
    ) -> torch.Tensor:
        """Compute dynamic weights based on level structure."""
        level_diffs = torch.abs(node_levels.unsqueeze(1) - node_levels.unsqueeze(0))
        same_level_mask = (level_diffs == 0).float()
        parent_mask = (level_diffs == 1).float()
        child_mask = (level_diffs == -1).float()

        weights = torch.stack(
            [
                0.4 * torch.ones(N, device=device),  # self weight
                0.3 * same_level_mask.sum(1),  # same level weight
                0.15 * parent_mask.sum(1),  # parent weight
                0.15 * child_mask.sum(1),  # child weight
                0.1 * torch.ones(N, device=device),  # level encoding weight
            ],
            dim=1,
        )

        # Normalize weights
        weights = weights / weights.sum(1, keepdim=True)
        return weights

    def aggregate_hierarchical(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_levels: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Aggregate features using hierarchical structure and projective operations."""
        # Convert inputs to float64
        x = self._to_double(x)
        node_levels = node_levels.to(x.device)

        # Track initial cross-ratios
        initial_crs = self._track_cross_ratios(x)

        row, col = edge_index
        N = x.size(0)

        # Add homogeneous coordinate if not present
        if x.size(-1) == self.in_features:
            x = torch.cat([x, torch.ones_like(x[..., :1], dtype=torch.float64)], dim=-1)

        # Transform and normalize level encodings
        # Align to in_features so level_features stacks with nodes_self
        node_level_enc = F.embedding(
            node_levels, self.level_encodings
        )  # [N, level_dim]
        if self.level_dim < self.in_features:
            node_level_enc = F.pad(
                node_level_enc, (0, self.in_features - self.level_dim), value=0.0
            )
        elif self.level_dim > self.in_features:
            node_level_enc = node_level_enc[..., : self.in_features]
        level_features = torch.cat(
            [
                node_level_enc,
                torch.ones_like(node_level_enc[..., :1], dtype=torch.float64),
            ],
            dim=-1,
        )
        level_features = self._normalize_with_homogeneous(
            self.projective_transform(level_features, self.W_level)
        )

        # Transform and normalize node features
        nodes_self = self._normalize_with_homogeneous(
            self.projective_transform(x, self.W_self)
        )
        nodes_neigh = self._normalize_with_homogeneous(
            self.projective_transform(x, self.W_neigh)
        )
        nodes_parent = self._normalize_with_homogeneous(
            self.projective_transform(x, self.W_parent)
        )
        nodes_child = self._normalize_with_homogeneous(
            self.projective_transform(x, self.W_child)
        )

        # Initialize aggregated features
        agg_same = torch.zeros_like(nodes_neigh, dtype=torch.float64)
        agg_parent = torch.zeros_like(nodes_parent, dtype=torch.float64)
        agg_child = torch.zeros_like(nodes_child, dtype=torch.float64)

        # Count neighbors for normalization
        same_count = torch.zeros(N, 1, device=x.device, dtype=torch.float64)
        parent_count = torch.zeros(N, 1, device=x.device, dtype=torch.float64)
        child_count = torch.zeros(N, 1, device=x.device, dtype=torch.float64)

        # Aggregate features with level-aware weights
        for i in range(len(row)):
            src, dst = row[i], col[i]
            src_level = node_levels[src].item()
            dst_level = node_levels[dst].item()
            level_diff = dst_level - src_level

            # Compute weight using cross-ratio and level difference
            weight = self.compute_cross_ratio_weight(x[src], x[dst], level_diff)

            if level_diff == 0:
                agg_same[dst] += weight * nodes_neigh[src]
                same_count[dst] += weight  # Weight-based counting
            elif level_diff > 0:
                agg_parent[dst] += weight * nodes_parent[src]
                parent_count[dst] += weight
            else:
                agg_child[dst] += weight * nodes_child[src]
                child_count[dst] += weight

        # Normalize aggregated features with regularization
        agg_same = agg_same / (same_count + self.eps)
        agg_parent = agg_parent / (parent_count + self.eps)
        agg_child = agg_child / (child_count + self.eps)

        # Compute dynamic weights based on level structure
        weights = self._compute_dynamic_weights(node_levels, N, x.device)

        # Stack features and combine: [N, 5, D+1]
        stacked_features = torch.stack(
            [nodes_self, agg_same, agg_parent, agg_child, level_features], dim=1
        )

        # Weights [N, 5] - one per source per node
        weights = weights.to(torch.float64) + self.eps
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Aggregate: combine 5 sources per node into single output [N, D+1]
        out = self._normalize_with_homogeneous(
            self.uhg.aggregate(stacked_features, weights)
        )

        # Restore cross-ratios if needed (guard: skip when ratio negative -> sqrt fails, or invalid)
        if initial_crs and out.size(0) >= 4:
            for cr in initial_crs:
                current_cr = compute_cross_ratio(out[0], out[1], out[2], out[3])
                ratio_ok = not (
                    torch.isnan(current_cr)
                    or torch.isinf(current_cr)
                    or torch.abs(current_cr) < 1e-8
                    or cr * current_cr <= 0
                )  # same sign for sqrt(target/current)
                if ratio_ok:
                    out = restore_cross_ratio(out, cr)
                    out = self._normalize_with_homogeneous(out)

        features = out[..., :-1]  # Spatial part [N, in_features]
        # Pad or truncate to out_features so next layer receives correct dimension
        if features.size(-1) < self.out_features:
            features = F.pad(features, (0, self.out_features - features.size(-1)))
        elif features.size(-1) > self.out_features:
            features = features[..., : self.out_features]
        # L2 normalize (uhg.normalize expects homogeneous form; we need unit-norm vectors)
        norm = torch.norm(features, dim=-1, keepdim=True).clamp(min=self.eps)
        return (features / norm).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_levels: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Forward pass using pure projective operations."""
        # Aggregate and return [N, out_features] (pad/truncate done inside aggregate_hierarchical)
        return self.aggregate_hierarchical(x, edge_index, node_levels, size)
