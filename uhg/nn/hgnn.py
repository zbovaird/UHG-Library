import torch
import torch.nn as nn
from typing import Optional
from .attention import HyperbolicAttention
from uhg.manifolds import HyperbolicManifold

class HyperbolicGATLayer(nn.Module):
    """
    A single layer of UHG-compliant Hyperbolic Graph Attention Network (HGAT).
    Uses projective geometry and preserves UHG invariants.
    """
    def __init__(self, manifold: HyperbolicManifold, in_channels: int, out_channels: int, num_heads: int = 1, dropout: float = 0.0, concat: bool = True):
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        print(f"[DEBUG][HGATLayer] __init__: in_channels={in_channels}, out_channels={out_channels}, num_heads={num_heads}, concat={concat}")
        # Attention module
        self.attn = HyperbolicAttention(
            manifold=manifold,
            in_channels=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            concat=concat
        )
        # Output projection always matches attention output shape
        if concat:
            self.lin = nn.Linear(num_heads * in_channels, out_channels)
            print(f"[DEBUG][HGATLayer] __init__: lin in_features={num_heads * in_channels}, out_features={out_channels}")
        else:
            self.lin = nn.Linear(in_channels, out_channels)
            print(f"[DEBUG][HGATLayer] __init__: lin in_features={in_channels}, out_features={out_channels}")
        self.reset_parameters()

    def reset_parameters(self):
        self.attn.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x, edge_index, edge_attr=None, mask=None):
        print(f"[DEBUG][HGATLayer] forward: x.shape={x.shape}")
        h = self.attn(x, edge_index, edge_attr, mask)
        print(f"[DEBUG][HGATLayer] forward: attn output shape={h.shape}")
        out = self.lin(h)
        print(f"[DEBUG][HGATLayer] forward: lin output shape={out.shape}")
        return out

class HyperbolicGAT(nn.Module):
    """
    Multi-layer UHG-compliant Hyperbolic Graph Attention Network (HGAT).
    Stackable, supports edge features and masking.
    """
    def __init__(self, manifold: HyperbolicManifold, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        curr_in = in_channels
        # All layers except the last
        for i in range(num_layers - 1):
            self.layers.append(HyperbolicGATLayer(manifold, curr_in, hidden_channels, num_heads=num_heads, dropout=dropout, concat=True))
            curr_in = hidden_channels
        # Last layer
        self.layers.append(HyperbolicGATLayer(manifold, curr_in, out_channels, num_heads=1, dropout=dropout, concat=False))

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
    manifold = HyperbolicManifold()
    model = HyperbolicGAT(
        manifold=manifold,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
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