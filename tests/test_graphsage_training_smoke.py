"""One training step on a tiny graph — guards ProjectiveGraphSAGE + projective_average."""

from __future__ import annotations

import torch

from uhg.nn.models.sage import ProjectiveGraphSAGE


def test_graphsage_forward_backward_one_step() -> None:
    """Regression guard for broadcast in projective_average (GraphSAGE layer)."""
    torch.manual_seed(0)
    n, in_ch, k = 32, 8, 4
    x = torch.randn(n, in_ch)
    rows = torch.arange(n, dtype=torch.long).repeat_interleave(k)
    cols = (
        torch.arange(n, dtype=torch.long).unsqueeze(1) + 1 + torch.arange(k)
    ).flatten() % n
    edge_index = torch.stack([rows, cols], dim=0)

    model = ProjectiveGraphSAGE(
        in_channels=in_ch,
        hidden_channels=16,
        out_channels=8,
        num_layers=2,
        dropout=0.0,
    )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    out = model(x, edge_index)
    loss = out.pow(2).mean()
    loss.backward()
    opt.step()
    assert out.shape == (n, 8)
    assert torch.isfinite(loss)


def test_projective_average_broadcast() -> None:
    from uhg.projective import ProjectiveUHG

    uhg = ProjectiveUHG()
    # [N, K, D] with K=2 (matches GraphSAGE stack of self + neighbor)
    p = torch.randn(5, 2, 7)
    w = torch.tensor([0.5, 0.5], dtype=p.dtype)
    avg = uhg.projective_average(p, w)
    assert avg.shape == (5, 7)
    assert torch.isfinite(avg).all()
