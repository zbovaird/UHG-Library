import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union, List
from .base import UHGLayer
from ...projective import ProjectiveUHG


class ProjectiveSAGEConv(UHGLayer):
    """UHG-compliant GraphSAGE convolution layer using pure projective operations.

    This layer implements the GraphSAGE convolution using only projective geometry,
    ensuring all operations preserve cross-ratios and follow UHG principles.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        num_samples: Number of neighbors to sample. If None, use all neighbors
        aggregator: Aggregation method ('mean', 'max', 'lstm')
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_samples: Optional[int] = None,
        aggregator: str = "mean",
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_samples = num_samples
        self.aggregator = aggregator.lower()

        if self.aggregator not in ["mean", "max", "lstm"]:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        # Initialize projective transformations
        self.W_self = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_neigh = nn.Parameter(torch.Tensor(out_features, in_features))

        # LSTM aggregator if specified
        if self.aggregator == "lstm":
            self.lstm = nn.LSTM(
                input_size=in_features, hidden_size=in_features, batch_first=True
            )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using UHG-aware initialization."""
        nn.init.orthogonal_(self.W_self)
        nn.init.orthogonal_(self.W_neigh)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def sample_neighbors(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Sample a fixed number of neighbors for each node."""
        if self.num_samples is None:
            return edge_index

        row, col = edge_index

        # Create adjacency list representation
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(len(row)):
            adj_list[row[i].item()].append(col[i].item())

        # Sample neighbors
        sampled_rows = []
        sampled_cols = []

        for node, neighbors in enumerate(adj_list):
            if len(neighbors) == 0:
                continue

            # Sample with replacement if we need more neighbors than available
            num_to_sample = min(self.num_samples, len(neighbors))
            sampled = torch.tensor(neighbors)[
                torch.randperm(len(neighbors))[:num_to_sample]
            ]

            sampled_rows.extend([node] * len(sampled))
            sampled_cols.extend(sampled.tolist())

        return torch.tensor([sampled_rows, sampled_cols], device=edge_index.device)

    def aggregate_neighbors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Aggregate neighborhood features using projective operations."""
        # Get sizes
        num_nodes = x.size(0)
        if size is not None:
            out_size = size[1]
        else:
            out_size = num_nodes

        # Initialize output with correct dimensions (match x dtype for downstream)
        out = torch.zeros(
            out_size, self.out_features + 1, device=x.device, dtype=x.dtype
        )
        out[..., -1] = 1.0  # Set homogeneous coordinate

        # Sample neighbors if specified using full node count
        edge_index = self.sample_neighbors(edge_index, num_nodes)
        if edge_index.numel() == 0:
            return out
        row, col = edge_index

        # Clamp indices to valid ranges
        row = torch.clamp(row, 0, out_size - 1)
        col = torch.clamp(col, 0, num_nodes - 1)

        # Get neighbor features (col = source nodes, row = destination nodes)
        neigh_features = x[col]
        transformed = self.projective_transform(neigh_features, self.W_neigh).to(
            x.dtype
        )
        # projective_transform preserves input dim (in_features+1); we need out_features+1
        target_dim = self.out_features + 1
        if transformed.size(-1) != target_dim:
            if transformed.size(-1) < target_dim:
                n_pad = target_dim - transformed.size(-1)
                pad = torch.zeros(
                    transformed.size(0), n_pad, device=x.device, dtype=x.dtype
                )
                transformed = torch.cat(
                    [transformed[..., :-1], pad, transformed[..., -1:]], dim=-1
                )
            else:
                transformed = transformed[..., :target_dim]
            transformed[..., -1] = 1.0

        if self.aggregator == "mean":
            # Vectorized mean aggregation with cross-ratio weights (normalize per dst)
            x_src = x[col]  # source node features
            x_dst = x[row]  # destination node features (for weight)
            weights = self._compute_cross_ratio_weight_batch(x_dst, x_src)
            sum_w = torch.zeros(out_size, device=x.device, dtype=weights.dtype)
            sum_w.index_add_(0, row, weights)
            weights = weights / (sum_w[row] + 1e-8)
            weighted_messages = transformed * weights.unsqueeze(-1)
            out.index_add_(0, row, weighted_messages)
            out[..., -1] = 1.0  # Restore homogeneous coordinate

        elif self.aggregator == "max":
            # Vectorized max aggregation using scatter_reduce
            spatial = transformed[..., :-1]
            fill_val = (
                torch.finfo(spatial.dtype).min
                if spatial.is_floating_point()
                else -(2**31)
            )
            out_spatial = spatial.new_full((out_size, spatial.size(-1)), fill_val)
            out_spatial.scatter_reduce_(
                0,
                row.unsqueeze(-1).expand(-1, spatial.size(-1)),
                spatial,
                reduce="amax",
            )
            # Replace uninitialized nodes with zeros
            is_init = (out_spatial != fill_val).any(dim=-1, keepdim=True)
            out[..., :-1] = torch.where(
                is_init.expand_as(out[..., :-1]),
                out_spatial,
                torch.zeros_like(out_spatial),
            )

        else:  # LSTM - keep loop for LSTM (requires variable-length sequences)
            for node in range(out_size):
                mask = row == node
                if mask.any():
                    node_neighs = transformed[mask][..., :-1]
                    node_neighs = node_neighs.unsqueeze(0)
                    lstm_out, _ = self.lstm(node_neighs)
                    out[node, :-1] = lstm_out[0, -1]

        return out

    def compute_cross_ratio_weight(
        self, p1: torch.Tensor, p2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-ratio based weight between two points."""
        # Extract features
        p1_feat = p1[:-1]
        p2_feat = p2[:-1]

        # Compute cosine similarity
        dot_product = torch.sum(p1_feat * p2_feat)
        norm_p1 = torch.norm(p1_feat)
        norm_p2 = torch.norm(p2_feat)
        cos_sim = dot_product / (norm_p1 * norm_p2 + 1e-8)

        # Map to [0, 1] range
        weight = (cos_sim + 1) / 2
        return weight.clamp(0.1, 0.9)  # Prevent extreme values

    def _compute_cross_ratio_weight_batch(
        self, p1: torch.Tensor, p2: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized cross-ratio weight for batches of (p1, p2) pairs."""
        p1_feat = p1[..., :-1]
        p2_feat = p2[..., :-1]
        dot_product = torch.sum(p1_feat * p2_feat, dim=-1)
        norm_p1 = torch.norm(p1_feat, dim=-1)
        norm_p2 = torch.norm(p2_feat, dim=-1)
        cos_sim = dot_product / (norm_p1 * norm_p2 + 1e-8)
        weight = (cos_sim + 1) / 2
        return weight.clamp(0.1, 0.9)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Forward pass using pure projective operations."""
        # Add homogeneous coordinate if not present
        if x.size(-1) == self.in_features:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)

        # Get output size
        if size is not None:
            out_size = size[1]
        else:
            out_size = x.size(0)

        # Transform self features
        self_trans = self.projective_transform(x, self.W_self).to(x.dtype)
        target_dim = self.out_features + 1
        if self_trans.size(-1) != target_dim:
            if self_trans.size(-1) < target_dim:
                n_pad = target_dim - self_trans.size(-1)
                pad = torch.zeros(
                    self_trans.size(0), n_pad, device=x.device, dtype=x.dtype
                )
                self_trans = torch.cat(
                    [self_trans[..., :-1], pad, self_trans[..., -1:]], dim=-1
                )
            else:
                self_trans = self_trans[..., :target_dim]
            self_trans[..., -1] = 1.0

        # Ensure self_trans has correct size
        if self_trans.size(0) < out_size:
            pad_size = out_size - self_trans.size(0)
            self_trans = torch.cat(
                [
                    self_trans,
                    torch.zeros(pad_size, self_trans.size(1), device=x.device),
                ],
                dim=0,
            )
            self_trans[self_trans.size(0) - pad_size :, -1] = 1.0

        # Aggregate and transform neighbor features
        neigh_trans = self.aggregate_neighbors(x, edge_index, size)

        # Ensure neigh_trans has correct size
        if neigh_trans.size(0) < out_size:
            pad_size = out_size - neigh_trans.size(0)
            neigh_trans = torch.cat(
                [
                    neigh_trans,
                    torch.zeros(pad_size, neigh_trans.size(1), device=x.device),
                ],
                dim=0,
            )
            neigh_trans[neigh_trans.size(0) - pad_size :, -1] = 1.0

        # Combine using projective average with equal weights
        points = torch.stack(
            [self_trans[:out_size], neigh_trans[:out_size]], dim=-2
        )  # [N, 2, D]
        weights = torch.tensor([0.5, 0.5], device=x.device)  # [2]
        out = self.uhg.projective_average(points, weights)

        if self.bias is not None:
            # Add bias in projective space
            bias_point = torch.cat([self.bias, torch.ones_like(self.bias[:1])], dim=0)
            bias_point = bias_point / torch.norm(bias_point)
            out = self.uhg.projective_average(
                torch.stack([out, bias_point.expand_as(out)], dim=1),
                torch.tensor([0.9, 0.1], device=x.device),
            )

        # Construct output projective vector of size [N, out_features]
        # Take first (out_features - 1) as spatial, compute time-like to get Minkowski norm -1
        spatial_dim = self.out_features - 1
        spatial = out[..., :-1]
        if spatial.size(-1) >= spatial_dim:
            spatial_sel = spatial[..., :spatial_dim]
        else:
            pad = spatial_dim - spatial.size(-1)
            spatial_sel = torch.cat(
                [
                    spatial,
                    torch.zeros(
                        spatial.size(0), pad, device=spatial.device, dtype=spatial.dtype
                    ),
                ],
                dim=-1,
            )
        # Bound spatial part before mapping to the upper hyperboloid sheet; large
        # activations here explode float32 when forming t = sqrt(1 + ||s||^2) and
        # break Minkowski inner products on the next layer.
        spatial_sel = torch.tanh(spatial_sel)
        time_like = torch.sqrt(
            1.0 + torch.sum(spatial_sel * spatial_sel, dim=-1, keepdim=True)
        )
        out_proj = torch.cat([spatial_sel, time_like], dim=-1)
        return self.uhg.normalize_points(out_proj)
