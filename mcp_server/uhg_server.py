"""
UHG MCP Server - Model Context Protocol server for UHG Library workflows.

Runs over stdio. Provides tools that offload computation, improve accuracy,
and save time compared to AI doing operations manually.

Usage:
  python -m mcp_server.uhg_server

Or with uv:
  uv run python -m mcp_server.uhg_server

Add to Cursor MCP config (~/.cursor/mcp.json):
{
  "mcpServers": {
    "uhg": {
      "command": "python",
      "args": ["-m", "mcp_server.uhg_server"],
      "cwd": "/path/to/UHG-Library"
    }
  }
}
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

# Resolve workspace root (parent of mcp_server/)
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

mcp = FastMCP("UHG")


@mcp.tool()
def run_tests(
    pattern: str = "",
    verbose: bool = True,
    collect_only: bool = False,
) -> dict[str, Any]:
    """Run UHG unit tests via pytest. Returns pass/fail counts and any failures.

    Args:
        pattern: Optional pytest pattern (e.g. 'test_sage' or 'tests/test_projective.py').
        verbose: If True, use -v flag for verbose output.
        collect_only: If True, only collect tests without running them.

    Returns:
        Dict with keys: passed, failed, skipped, errors, total, output, failures (list of failure messages).
    """
    args = [sys.executable, "-m", "pytest", "tests/", "--tb=short", "-q"]
    if pattern:
        args.extend(["-k", pattern])
    if verbose and not collect_only:
        args.insert(-1, "-v")
    if collect_only:
        args.extend(["--collect-only", "-q"])

    try:
        result = subprocess.run(
            args,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        out = result.stdout + result.stderr

        # Parse pytest output for counts (e.g. "7 passed in 1.20s" or "5 passed, 2 failed")
        passed = failed = skipped = errors = 0
        failures: list[str] = []
        import re
        for m in re.finditer(r"(\d+)\s+(passed|failed|skipped|error)", out, re.I):
            n, label = int(m.group(1)), m.group(2).lower()
            if label == "passed":
                passed = n
            elif label == "failed":
                failed = n
            elif label == "skipped":
                skipped = n
            elif label == "error":
                errors = n
        for line in out.splitlines():
            if "FAILED" in line:
                failures.append(line.strip())

        total = passed + failed + skipped + errors
        return {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "total": total if total > 0 else None,
            "returncode": result.returncode,
            "output": out[-4000:] if len(out) > 4000 else out,
            "failures": failures[:20],
        }
    except subprocess.TimeoutExpired:
        return {"error": "Tests timed out after 300s", "output": ""}
    except Exception as e:
        return {"error": str(e), "output": ""}


@mcp.tool()
def uhg_quadrance(
    point_a: list[float],
    point_b: list[float],
    epsilon: float = 1e-9,
) -> dict[str, Any]:
    """Compute UHG quadrance between two points in projective/homogeneous coordinates.

    Quadrance is the hyperbolic analog of squared distance. Uses the projective form:
    q(a,b) = 1 - (⟨a,b⟩²) / (⟨a,a⟩⟨b,b⟩). Points should be inside the null cone
    (e.g. [0.3, 0.2, 1.0] where 0.3²+0.2² < 1²).

    Args:
        point_a: Homogeneous coordinates [x, y, z] or [x1, x2, ..., z].
        point_b: Homogeneous coordinates for second point.
        epsilon: Numerical stability epsilon.

    Returns:
        Dict with quadrance value and status.
    """
    try:
        import torch
        from uhg.projective import ProjectiveUHG

        uhg = ProjectiveUHG(epsilon=epsilon)
        a = torch.tensor(point_a, dtype=torch.float64)
        b = torch.tensor(point_b, dtype=torch.float64)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        # Ensure points are proper hyperbolic (inside null cone)
        for t, name in [(a, "a"), (b, "b")]:
            ip = uhg.inner_product(t, t)
            if torch.any(torch.abs(ip) < epsilon):
                return {"quadrance": None, "status": "error", "error": f"Point {name} is null (on cone)"}
        q = uhg.quadrance(a, b)
        val = float(q.item() if q.numel() == 1 else q[0].item())
        return {"quadrance": val, "status": "ok", "epsilon": epsilon}
    except Exception as e:
        return {"quadrance": None, "status": "error", "error": str(e)}


@mcp.tool()
def uhg_cross_ratio(
    p1: list[float],
    p2: list[float],
    p3: list[float],
    p4: list[float],
    epsilon: float = 1e-9,
) -> dict[str, Any]:
    """Compute UHG cross-ratio of four collinear points. A projective invariant.

    Args:
        p1, p2, p3, p4: Homogeneous coordinates for each point.
        epsilon: Numerical stability epsilon.

    Returns:
        Dict with cross_ratio value and status.
    """
    try:
        import torch
        from uhg.utils.cross_ratio import compute_cross_ratio

        pts = [torch.tensor(p, dtype=torch.float64) for p in (p1, p2, p3, p4)]
        cr = compute_cross_ratio(*pts)
        val = float(cr.item() if cr.numel() == 1 else cr[0].item())
        return {"cross_ratio": val, "status": "ok", "epsilon": epsilon}
    except Exception as e:
        return {"cross_ratio": None, "status": "error", "error": str(e)}


@mcp.tool()
def workspace_status() -> dict[str, Any]:
    """Get UHG workspace status: version, test counts, key paths.

    Returns:
        Dict with version, workspace path, test file count, and env info.
    """
    info: dict[str, Any] = {
        "workspace": str(WORKSPACE_ROOT),
        "uhg_version": None,
        "test_files": 0,
        "python": sys.version.split()[0],
    }
    try:
        import uhg

        info["uhg_version"] = getattr(uhg, "__version__", "unknown")
    except ImportError:
        info["uhg_version"] = "not installed (development mode?)"

    test_dir = WORKSPACE_ROOT / "tests"
    if test_dir.is_dir():
        info["test_files"] = len(list(test_dir.glob("test_*.py")))

    return info


@mcp.tool()
def run_benchmark(
    data_path: Optional[str] = None,
    epochs: int = 10,
) -> dict[str, Any]:
    """Run a quick benchmark of ProjectiveGraphSAGE (reduced epochs for speed).

    Uses synthetic data if data_path is not provided or file is missing.
    For full CIC benchmark, run benchmarks/intrusion_detection_cic10.py directly.

    Args:
        data_path: Path to CIC_data.csv. If None or missing, uses synthetic data.
        epochs: Number of training epochs (default 10 for quick check).

    Returns:
        Dict with timing, final accuracy, and status.
    """
    try:
        import time

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from sklearn.datasets import make_classification
        from sklearn.neighbors import kneighbors_graph
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse import coo_matrix
        import numpy as np

        from uhg.projective import ProjectiveUHG
        from uhg.nn.layers.sage import ProjectiveSAGEConv
        from torch_geometric.data import Data

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def projective_normalize(x: torch.Tensor) -> torch.Tensor:
            spatial = x[..., :-1]
            time_like = torch.sqrt(
                torch.clamp(1.0 + (spatial * spatial).sum(dim=-1, keepdim=True), min=1e-9)
            )
            return torch.cat([spatial, time_like], dim=-1)

        def make_synthetic_graph(n_samples: int = 2000, k: int = 2):
            X, y = make_classification(
                n_samples=n_samples,
                n_features=20,
                n_informative=10,
                n_classes=5,
                random_state=42,
            )
            X = StandardScaler().fit_transform(X)
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            knn = kneighbors_graph(X.numpy(), k, mode="connectivity", include_self=False)
            coo = coo_matrix(knn)
            edge_index = torch.from_numpy(np.vstack((coo.row, coo.col))).long().to(device)
            x = torch.cat([X.to(device), torch.ones(X.size(0), 1, device=device)], dim=1)
            x = projective_normalize(x)
            N = x.size(0)
            idx = torch.randperm(N)
            train = torch.zeros(N, dtype=torch.bool, device=device)
            val = torch.zeros(N, dtype=torch.bool, device=device)
            test = torch.zeros(N, dtype=torch.bool, device=device)
            train[idx[: int(0.7 * N)]] = True
            val[idx[int(0.7 * N) : int(0.85 * N)]] = True
            test[idx[int(0.85 * N) :]] = True
            return Data(
                x=x, edge_index=edge_index, y=y.to(device),
                train_mask=train, val_mask=val, test_mask=test
            ).to(device)

        class ProjectiveGraphSAGE(nn.Module):
            def __init__(self, in_channels, hidden, out_channels, num_layers=2, dropout=0.2):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                self.layers = nn.ModuleList()
                actual_in = in_channels - 1
                self.layers.append(ProjectiveSAGEConv(actual_in, hidden))
                for _ in range(num_layers - 2):
                    self.layers.append(ProjectiveSAGEConv(hidden, hidden))
                self.layers.append(ProjectiveSAGEConv(hidden, out_channels))

            def forward(self, x, edge_index):
                h = x
                for layer in self.layers[:-1]:
                    h = layer(h, edge_index)
                    spatial = F.relu(h[:, :-1])
                    h = torch.cat([spatial, h[:, -1:]], dim=1)
                    h = self.dropout(h)
                h = self.layers[-1](h, edge_index)
                return h[:, :-1]

        data = make_synthetic_graph(n_samples=2000)
        n_classes = int(data.y.max().item()) + 1
        model = ProjectiveGraphSAGE(
            in_channels=data.x.size(1),
            hidden=64,
            out_channels=n_classes,
            num_layers=2,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        crit = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        t0 = time.perf_counter()
        use_amp = torch.cuda.is_available()
        for ep in range(epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                logits = model(data.x, data.edge_index)
                loss = crit(logits[data.train_mask], data.y[data.train_mask])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        with torch.no_grad():
            model.eval()
            logits = model(data.x, data.edge_index)
            pred = logits[data.test_mask].argmax(dim=1)
            acc = (pred == data.y[data.test_mask]).float().mean().item()

        elapsed = time.perf_counter() - t0
        return {
            "status": "ok",
            "epochs": epochs,
            "elapsed_s": round(elapsed, 3),
            "sec_per_epoch": round(elapsed / epochs, 4),
            "test_accuracy": round(acc, 4),
            "device": str(device),
            "note": "Synthetic data (data_path not used)",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def run_anomaly_smoke(
    n_samples: int = 1000,
    k: int = 2,
    epochs: int = 5,
) -> dict[str, Any]:
    """Run UHGUnsupervisedAnomalyDetector end-to-end on synthetic data. Quick smoke test.

    Args:
        n_samples: Number of synthetic samples.
        k: k for kNN graph.
        epochs: Training epochs.

    Returns:
        Dict with status, summary_keys, total_s, and summary excerpt.
    """
    try:
        import time
        import numpy as np
        from uhg.anomaly.unsupervised import UHGUnsupervisedAnomalyDetector

        X = np.random.RandomState(42).randn(n_samples, 5) * 0.5
        t0 = time.perf_counter()
        det = UHGUnsupervisedAnomalyDetector(hidden=16, embedding_dim=8)
        det.fit(X, k=k, epochs=epochs, seed=42)
        det.cluster(eps=0.8, min_samples=2)
        summary = det.summarize(topk=3)
        elapsed = time.perf_counter() - t0
        return {
            "status": "ok",
            "n_samples": n_samples,
            "total_s": round(elapsed, 3),
            "summary_keys": list(summary.keys()),
            "n_nodes": summary.get("n_nodes"),
            "top_entity_score": summary.get("top_entities", [{}])[0].get("score") if summary.get("top_entities") else None,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
