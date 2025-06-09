import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..projective import ProjectiveUHG
from ..utils.cross_ratio import compute_cross_ratio, restore_cross_ratio
from .layers.linear import HyperbolicLinear

class HyperbolicAttention(nn.Module):
    """UHG-compliant attention mechanism (no tangent space, pure projective geometry).
    Supports edge features and masking. All operations preserve UHG invariants.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True
    ):
        super().__init__()
        self.uhg = ProjectiveUHG()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat

        # Projection matrices
        self.query = nn.Linear(in_features, out_features * num_heads)
        self.key = nn.Linear(in_features, out_features * num_heads)
        self.value = nn.Linear(in_features, out_features * num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(out_features * num_heads, out_features)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.edge_mlp = None
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize linear layers
        self.query.reset_parameters()
        self.key.reset_parameters()
        self.value.reset_parameters()
        self.out_proj.reset_parameters()
        
        if self.edge_mlp is not None:
            nn.init.xavier_uniform_(self.edge_mlp.weight)
            if self.edge_mlp.bias is not None:
                nn.init.zeros_(self.edge_mlp.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """UHG-compliant forward pass (no tangent space, pure projective geometry).
        Args:
            x: Node features [N, in_features]
            edge_index: [2, E]
            edge_attr: [E, edge_dim] (optional)
            mask: [E] or [N, N] (optional)
        Returns:
            Updated node features [N, out_features]
        """
        N = x.size(0)
        
        # Store initial cross-ratio if possible
        has_cr = x.size(0) > 3
        if has_cr:
            cr_initial = compute_cross_ratio(x[0], x[1], x[2], x[3])
        
        # Projective normalization (homogeneous coordinates)
        x_proj = self.uhg.normalize_points(x)
        
        # Linear projections (per head)
        queries = self.query(x_proj).view(N, self.num_heads, self.out_features)
        keys = self.key(x_proj).view(N, self.num_heads, self.out_features)
        values = self.value(x_proj).view(N, self.num_heads, self.out_features)
        
        row, col = edge_index
        
        # Compute attention scores using UHG cross-ratio
        attn_scores = self._compute_attention_scores(queries[row], keys[col])  # [E, num_heads, 1]
        attn_scores = attn_scores.squeeze(-1)  # [E, num_heads]

        # Edge features
        if edge_attr is not None:
            if edge_attr.shape[-1] != self.in_features:
                if self.edge_mlp is None or self.edge_mlp.in_features != edge_attr.shape[-1]:
                    self.edge_mlp = nn.Linear(edge_attr.shape[-1], self.in_features).to(edge_attr.device)
                edge_proj = self.edge_mlp(edge_attr)  # [E, in_features]
            else:
                edge_proj = edge_attr  # [E, in_features]
            
            # Expand edge_proj to [E, num_heads, in_features] for value update
            edge_proj_heads = edge_proj.unsqueeze(1).expand(-1, self.num_heads, -1)  # [E, num_heads, in_features]
            # For attention scores, project edge features to a scalar per head
            edge_proj_score = edge_proj_heads.sum(dim=-1)  # [E, num_heads]
            
            attn_scores = attn_scores + edge_proj_score
        else:
            edge_proj_heads = None

        # Masking
        if mask is not None:
            # Expand mask to match attention heads
            mask = mask.unsqueeze(-1).expand(-1, self.num_heads)  # [E, num_heads]
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # Softmax
        alpha = F.softmax(attn_scores, dim=0)  # [E, num_heads]
        alpha = self.dropout_layer(alpha)

        # Aggregate values
        out = torch.zeros(N, self.num_heads, self.out_features, device=x.device)
        value_messages = values[col]  # [E, num_heads, out_features]
        if edge_proj_heads is not None:
            value_messages = value_messages + edge_proj_heads  # [E, num_heads, out_features]
        
        # Multiply by attention weights
        value_messages = value_messages * alpha.unsqueeze(-1)  # [E, num_heads, out_features]
        
        # Scatter add to nodes
        out.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.out_features), value_messages)

        # Concatenate or average heads
        if self.concat:
            out = out.view(N, self.num_heads * self.out_features)
        else:
            out = out.mean(dim=1)

        # Output projection
        out = self.out_proj(out)

        # Restore cross-ratio if possible
        if has_cr:
            out = restore_cross_ratio(out, cr_initial)
        
        return out

    def _compute_attention_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention scores using pure projective operations."""
        # Normalize queries and keys
        q = self.uhg.normalize_points(q)
        k = self.uhg.normalize_points(k)
        
        # Compute hyperbolic inner product
        scores = self.uhg.inner_product(q, k)
        
        # Scale scores
        scores = scores / math.sqrt(self.in_features)
        
        return scores

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'num_heads={self.num_heads}, dropout={self.dropout}, concat={self.concat}')

def test_hyperbolic_attention():
    """Test function for HyperbolicAttention module."""
    print("\n=== Testing HyperbolicAttention ===")
    
    # Test parameters
    batch_size = 4
    num_nodes = 10
    in_features = 8
    num_heads = 2
    edge_dim = 6
    
    # Initialize attention module
    attention = HyperbolicAttention(
        in_features=in_features,
        out_features=in_features,
        num_heads=num_heads,
        dropout=0.1,
        concat=True
    )
    
    # Create random input data
    x = torch.randn(num_nodes, in_features)  # Node features
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # source nodes
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]   # target nodes
    ])
    edge_attr = torch.randn(edge_index.size(1), edge_dim)  # Edge features
    mask = torch.ones(edge_index.size(1), dtype=torch.bool)  # All edges active
    
    print("\nInput shapes:")
    print(f"x: {x.shape}")
    print(f"edge_index: {edge_index.shape}")
    print(f"edge_attr: {edge_attr.shape}")
    print(f"mask: {mask.shape}")
    
    # Forward pass
    try:
        out = attention(x, edge_index, edge_attr, mask)
        print("\nForward pass successful!")
        print(f"Output shape: {out.shape}")
        print(f"Output norm: {torch.norm(out, dim=-1).mean():.4f}")
    except Exception as e:
        print(f"\nError during forward pass: {str(e)}")
        raise e
    
    # Test without edge features
    try:
        out_no_edge = attention(x, edge_index, None, mask)
        print("\nForward pass without edge features successful!")
        print(f"Output shape: {out_no_edge.shape}")
        print(f"Output norm: {torch.norm(out_no_edge, dim=-1).mean():.4f}")
    except Exception as e:
        print(f"\nError during forward pass without edge features: {str(e)}")
        raise e
    
    # Test without mask
    try:
        out_no_mask = attention(x, edge_index, edge_attr, None)
        print("\nForward pass without mask successful!")
        print(f"Output shape: {out_no_mask.shape}")
        print(f"Output norm: {torch.norm(out_no_mask, dim=-1).mean():.4f}")
    except Exception as e:
        print(f"\nError during forward pass without mask: {str(e)}")
        raise e
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_hyperbolic_attention() 