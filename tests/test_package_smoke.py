"""Minimal CI checks: install, import, and stable public API surface."""

from __future__ import annotations


def test_uhg_version() -> None:
    import uhg

    assert uhg.__version__
    assert isinstance(uhg.__version__, str)


def test_uhg_public_exports() -> None:
    import uhg

    for name in (
        "ProjectiveUHG",
        "build_knn_graph",
        "UHGUnsupervisedAnomalyDetector",
        "centroid_quadrance",
        "run_dbscan",
        "__version__",
    ):
        assert hasattr(uhg, name), f"missing uhg.{name}"


def test_uhg_nn_sage_import() -> None:
    from uhg.nn.models.sage import ProjectiveGraphSAGE

    assert ProjectiveGraphSAGE is not None
