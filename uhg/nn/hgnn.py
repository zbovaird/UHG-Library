import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..projective import ProjectiveUHG

class HyperbolicGATLayer(nn.Module):
    """
    A single layer of UHG-compliant Hyperbolic Graph Attention Network (HGAT).
    Uses projective geometry and preserves UHG invariants.
    """
    def __init__(self, in_features: int, out_features: int, num_heads: int = 1, dropout: float = 0.0, concat: bool = True):
        super().__init__()
        self.uhg = ProjectiveUHG()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        print(f"[DEBUG][HGATLayer] __init__: in_features={in_features}, out_features={out_features}, num_heads={num_heads}, concat={concat}")
        # Attention parameters
        self.query = nn.Linear(in_features, out_features * num_heads)
        self.key = nn.Linear(in_features, out_features * num_heads)
        self.value = nn.Linear(in_features, out_features * num_heads)
        # Output projection
        if concat:
            self.out_proj = nn.Linear(out_features * num_heads, out_features)
            print(f"[DEBUG][HGATLayer] __init__: out_proj in_features={out_features * num_heads}, out_features={out_features}")
        else:
            self.out_proj = nn.Linear(out_features, out_features)
            print(f"[DEBUG][HGATLayer] __init__: out_proj in_features={out_features}, out_features={out_features}")
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query.weight)
        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)
        nn.init.xavier_uniform_(self.key.weight)
        if self.key.bias is not None:
            nn.init.zeros_(self.key.bias)
        nn.init.xavier_uniform_(self.value.weight)
        if self.value.bias is not None:
            nn.init.zeros_(self.value.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, edge_index, edge_attr=None, mask=None):
        print(f"[DEBUG][HGATLayer] forward: x.shape={x.shape}")
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        print(f"[DEBUG][HGATLayer] forward: q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
        h = self.uhg.attn(q, k, v, edge_index, edge_attr, mask)
        print(f"[DEBUG][HGATLayer] forward: attn output shape={h.shape}")
        out = self.out_proj(h)
        print(f"[DEBUG][HGATLayer] forward: out_proj output shape={out.shape}")
        return out

class HyperbolicGAT(nn.Module):
    """
    Multi-layer UHG-compliant Hyperbolic Graph Attention Network (HGAT).
    Stackable, supports edge features and masking.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, num_layers: int = 2, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        curr_in = in_features
        # All layers except the last
        for i in range(num_layers - 1):
            self.layers.append(HyperbolicGATLayer(curr_in, hidden_features, num_heads=num_heads, dropout=dropout, concat=True))
            curr_in = hidden_features
        # Last layer
        self.layers.append(HyperbolicGATLayer(curr_in, out_features, num_heads=1, dropout=dropout, concat=False))

    def forward(self, x, edge_index, edge_attr=None, mask=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr, mask)
            if i < len(self.layers) - 1:
                x = self.dropout(torch.relu(x))
        return x

# --- TEST FUNCTION ---
def test_hgnn():
    print("\n=== Testing HyperbolicGAT ===")
    num_nodes = 10
    in_channels = 8
    hidden_channels = 16
    out_channels = 4
    num_heads = 2
    num_layers = 3
    edge_dim = 6
    model = HyperbolicGAT(
        in_features=in_channels,
        hidden_features=hidden_channels,
        out_features=out_channels,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.1
    )
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    ])
    edge_attr = torch.randn(edge_index.size(1), edge_dim)
    mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    print(f"Input x: {x.shape}, edge_index: {edge_index.shape}, edge_attr: {edge_attr.shape}, mask: {mask.shape}")
    try:
        out = model(x, edge_index, edge_attr, mask)
        print("Output shape:", out.shape)
        print("Output (first 2 rows):\n", out[:2])
    except Exception as e:
        print("Error during HGNN forward pass:", str(e))
        raise e
    print("=== HGNN Test Complete ===\n")

if __name__ == "__main__":
    test_hgnn() 