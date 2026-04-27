"""Minimal CI checks: install, import, and stable public API surface."""

from __future__ import annotations


def test_core_imports_do_not_load_sklearn() -> None:
    import subprocess
    import sys

    code = (
        "import sys; "
        "import uhg, uhg.layers, uhg.manifolds, uhg.nn; "
        "assert uhg.ProjectiveUHG is not None; "
        "assert 'sklearn' not in sys.modules; "
        "assert 'scipy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


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
