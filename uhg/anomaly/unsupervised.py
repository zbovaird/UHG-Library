"""Unsupervised anomaly detector using UHG and GraphSAGE."""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from uhg import __version__ as uhg_version
from uhg.anomaly.report import aggregate_by_entity, rank_topk
from uhg.anomaly.scores import centroid_quadrance, composite_score, neighbor_quadrance
from uhg.cluster.dbscan import run_dbscan
from uhg.cluster.metrics import calinski_harabasz, davies_bouldin, silhouette
from uhg.graph.build import build_knn_graph
from uhg.nn.early_stopping import EarlyStopping
from uhg.nn.losses import UHGAnomalyLoss
from uhg.nn.models.sage import ProjectiveGraphSAGE
from uhg.utils.timing import TimingsDict, time_block


class UHGUnsupervisedAnomalyDetector:
    """First-class unsupervised anomaly detector: fit, cluster, score, summarize, export."""

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        embedding_dim: int = 32,
    ):
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.config: Dict[str, Any] = {}
        self.embeddings: Optional[torch.Tensor] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.timings: Dict[str, float] = {}
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[ProjectiveGraphSAGE] = None
        self._cluster_result: Optional[dict] = None
        self._X_fit: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[torch.Tensor, np.ndarray],
        k: int = 2,
        data_fraction: float = 1.0,
        *,
        metric: str = "euclidean",
        edge_cache: Optional[str] = None,
        seed: Optional[int] = None,
        epochs: int = 50,
    ) -> "UHGUnsupervisedAnomalyDetector":
        """Fit the detector: standardize, build graph, train GraphSAGE."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        X_np = np.asarray(X, dtype=np.float64)
        if X_np.size == 0:
            raise ValueError("X is empty")
        if X_np.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_np.shape}")
        n, d = X_np.shape
        if k >= n:
            raise ValueError(f"k must be < n_samples ({n}), got k={k}")
        if np.any(~np.isfinite(X_np)):
            raise ValueError("X contains nan or inf")

        timings = TimingsDict()
        if data_fraction < 1.0:
            idx = np.random.choice(n, int(n * data_fraction), replace=False)
            X_np = X_np[idx]
            n = X_np.shape[0]

        with time_block("data_load_s", timings):
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_np)
            X_t = torch.from_numpy(X_scaled).float()

        with time_block("knn_build_s", timings):
            edge_index = build_knn_graph(
                X_scaled, k, metric=metric, cache_key=edge_cache
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = X_t.to(device)
        edge_index = edge_index.to(device)

        in_channels = X_t.size(1)
        self.model = ProjectiveGraphSAGE(
            in_channels=in_channels,
            hidden_channels=self.hidden,
            out_channels=self.embedding_dim,
            num_layers=self.layers,
            dropout=self.dropout,
        ).to(device)
        criterion = UHGAnomalyLoss(spread_weight=0.1, quad_weight=1.0, margin=1.0)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=1e-5
        )
        use_amp = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        with time_block("train_s", timings):
            self.model.train()
            for ep in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp):
                    z = self.model(X_t, edge_index)
                    z_h = torch.cat(
                        [z, torch.ones(z.size(0), 1, device=device)], dim=-1
                    )
                    loss = criterion(z_h, edge_index, n)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        with torch.no_grad():
            self.model.eval()
            z = self.model(X_t, edge_index)
            self.embeddings = torch.cat(
                [z, torch.ones(z.size(0), 1, device=device)], dim=-1
            ).cpu()
        self.edge_index = edge_index.cpu()
        self._X_fit = X_scaled
        self.timings = dict(timings)
        self.timings["total_s"] = sum(self.timings.values())
        self.config = {
            "k": k,
            "n_nodes": n,
            "n_edges": edge_index.size(1),
            "in_channels": in_channels,
            "embedding_dim": self.embedding_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "dropout": self.dropout,
        }
        return self

    def cluster(
        self,
        method: str = "dbscan",
        **kwargs,
    ) -> dict:
        """Run clustering on embeddings. Returns labels, params, metrics."""
        if self.embeddings is None:
            raise RuntimeError("Must call fit() before cluster()")
        emb = self.embeddings.numpy()
        if method == "dbscan":
            eps = kwargs.get("eps", 0.5)
            min_samples = kwargs.get("min_samples", 3)
            out = run_dbscan(emb, eps, min_samples)
            labels = out["labels"]
            core_mask = out["core_mask"]
            n_clusters = len(set(labels) - {-1})
            metrics = None
            if n_clusters >= 2:
                try:
                    metrics = {
                        "davies_bouldin": davies_bouldin(emb, labels),
                        "silhouette": silhouette(emb, labels),
                        "calinski_harabasz": calinski_harabasz(emb, labels),
                    }
                except Exception:
                    pass
            self._cluster_result = {
                "labels": labels,
                "core_mask": core_mask,
                "params": {"eps": eps, "min_samples": min_samples},
                "metrics": metrics,
            }
            return self._cluster_result
        raise NotImplementedError(f"method={method}")

    def score(
        self,
        method: str = "centroid_quadrance",
        **kwargs,
    ) -> torch.Tensor:
        """Compute anomaly scores. Higher = more anomalous."""
        if self.embeddings is None:
            raise RuntimeError("Must call fit() before score()")
        if method == "centroid_quadrance":
            return centroid_quadrance(self.embeddings, **kwargs)
        if method == "neighbor_quadrance":
            return neighbor_quadrance(self.embeddings, **kwargs)
        if method == "composite":
            scores = kwargs.get("scores", {})
            weights = kwargs.get("weights", {})
            return composite_score(scores, weights)
        raise ValueError(f"Unknown method={method}")

    def summarize(
        self,
        entity_ids: Optional[np.ndarray] = None,
        topk: int = 20,
        score_method: str = "centroid_quadrance",
    ) -> dict:
        """Aggregate stats, timings, cluster metrics, top entities."""
        if self.embeddings is None:
            raise RuntimeError("Must call fit() before summarize()")
        scores = self.score(method=score_method)
        summary = {
            "n_nodes": self.config["n_nodes"],
            "n_edges": self.config["n_edges"],
            "k": self.config["k"],
            "timings": self.timings,
        }
        if self._cluster_result and self._cluster_result.get("metrics"):
            summary["cluster_metrics"] = self._cluster_result["metrics"]
        top = rank_topk(scores, k=topk)
        summary["top_entities"] = top
        if entity_ids is not None:
            summary["entity_stats"] = aggregate_by_entity(
                scores.numpy(), entity_ids, stats=("mean", "p95", "count")
            )
        return summary

    def export(self, path: str) -> None:
        """Save detector state (config, model, embeddings, etc.)."""
        state = {
            "uhg_version": uhg_version,
            "config_version": 1,
            "config": self.config,
            "scaler_mean": self.scaler.mean_ if self.scaler else None,
            "scaler_scale": self.scaler.scale_ if self.scaler else None,
            "model_state": self.model.state_dict() if self.model else None,
            "embeddings": self.embeddings,
            "edge_index": self.edge_index,
            "timings": self.timings,
            "cluster_result": self._cluster_result,
        }
        torch.save(state, path)

    @classmethod
    def from_export(cls, path: str) -> "UHGUnsupervisedAnomalyDetector":
        """Load detector from exported state."""
        state = torch.load(path, weights_only=False)
        if isinstance(state, dict) and "config" in state:
            cfg = state["config"]
            det = cls(
                hidden=cfg.get("hidden", 64),
                layers=cfg.get("layers", 2),
                dropout=cfg.get("dropout", 0.2),
                embedding_dim=cfg.get("embedding_dim", 32),
            )
            det.config = state["config"]
            det.embeddings = state.get("embeddings")
            det.edge_index = state.get("edge_index")
            det.timings = state.get("timings", {})
            det._cluster_result = state.get("cluster_result")
            if state.get("scaler_mean") is not None:
                det.scaler = StandardScaler()
                det.scaler.mean_ = state["scaler_mean"]
                det.scaler.scale_ = state["scaler_scale"]
            if state.get("model_state") is not None and det.config:
                in_ch = det.config.get("in_channels", 1)
                det.model = ProjectiveGraphSAGE(
                    in_channels=in_ch,
                    hidden_channels=det.config.get("hidden", 64),
                    out_channels=det.config.get("embedding_dim", 32),
                    num_layers=det.config.get("layers", 2),
                    dropout=det.config.get("dropout", 0.2),
                )
                det.model.load_state_dict(state["model_state"], strict=True)
            if state.get("uhg_version") and state["uhg_version"] != uhg_version:
                import warnings

                warnings.warn(
                    f"Loading from uhg {state['uhg_version']}, current is {uhg_version}"
                )
            return det
        raise ValueError(f"Invalid export format: {type(state)}")

    def predict(
        self,
        scores: Optional[torch.Tensor] = None,
        percentile: float = 0.95,
        threshold: Optional[float] = None,
        score_method: str = "centroid_quadrance",
    ) -> tuple:
        """Return (scores, binary_labels). Labels 1 = anomaly (top percentile or above threshold)."""
        if self.embeddings is None:
            raise RuntimeError("Must call fit() before predict()")
        if scores is None:
            scores = self.score(method=score_method)
        if threshold is not None:
            labels = (scores >= threshold).long()
        else:
            k = max(1, int((1 - percentile) * scores.size(0)))
            _, top_idx = torch.topk(scores, k)
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            labels[top_idx] = 1
        return scores, labels

    def score_new(
        self,
        X_new: Union[torch.Tensor, np.ndarray],
        k: Optional[int] = None,
        score_method: str = "centroid_quadrance",
    ) -> torch.Tensor:
        """Score new points without retraining. Embeds via k-NN subgraph to training nodes."""
        if self.embeddings is None or self.model is None or self._X_fit is None:
            raise RuntimeError("Must call fit() before score_new()")
        k = k or self.config.get("k", 5)
        X_new = np.asarray(X_new, dtype=np.float64)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        X_new_scaled = self.scaler.transform(X_new)
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self._X_fit)
        neigh_idx = nn.kneighbors(X_new_scaled, return_distance=False)
        device = next(self.model.parameters()).device
        scores_list = []
        for i in range(X_new_scaled.shape[0]):
            idx = neigh_idx[i]
            X_sub = np.vstack([X_new_scaled[i : i + 1], self._X_fit[idx]])
            x_t = torch.from_numpy(X_sub).float().to(device)
            row = torch.zeros(k, dtype=torch.long, device=device)
            col = torch.arange(1, k + 1, device=device)
            ei = torch.stack([row, col])
            self.model.eval()
            with torch.no_grad():
                z = self.model(x_t, ei)
                z_h = torch.cat([z, torch.ones(z.size(0), 1, device=device)], dim=-1)
            s = centroid_quadrance(z_h, eps=1e-9)
            scores_list.append(s[0].item())
        return torch.tensor(scores_list)

    def fit_from_dataframe(
        self, df, feature_columns=None, **fit_kwargs
    ) -> "UHGUnsupervisedAnomalyDetector":
        """Convenience: fit from DataFrame. Uses enforce_numeric, drops label column if present."""
        from uhg.utils.schema import detect_label_column, enforce_numeric

        label_col = detect_label_column(df)
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != label_col]
        X = df[feature_columns]
        X = enforce_numeric(X, fill="mean", replace_inf=True)
        return self.fit(X.values, **fit_kwargs)
