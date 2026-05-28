"""Tests for bdi.bct module."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for testing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from bdi.bct import eder_bct, plot_bct
from bdi.metrics import cosine, minmax, euclidean


@pytest.fixture
def simple_data():
    """Create simple test data with 3 clusters."""
    rng = np.random.default_rng(42)
    n_per_cluster = 20
    n_features = 10

    # 3 clusters with distinct feature profiles
    X = pd.DataFrame(
        np.vstack(
            [
                rng.normal(0, 0.5, (n_per_cluster, n_features)),
                rng.normal(3, 0.5, (n_per_cluster, n_features)),
                rng.normal(6, 0.5, (n_per_cluster, n_features)),
            ]
        )
    )
    y = (
        [f"work_a - {i}" for i in range(n_per_cluster)]
        + [f"work_b - {i}" for i in range(n_per_cluster)]
        + [f"work_c - {i}" for i in range(n_per_cluster)]
    )
    work_names = (
        ["work_a"] * n_per_cluster
        + ["work_b"] * n_per_cluster
        + ["work_c"] * n_per_cluster
    )

    return X, y, work_names


@pytest.fixture
def graph_trim(simple_data):
    """Run eder_bct on simple data."""
    X, y, _ = simple_data
    return eder_bct(X, y, n=100, rng=np.random.default_rng(42), metric=cosine)


class TestEderBct:
    def test_returns_dataframe(self, graph_trim):
        assert isinstance(graph_trim, pd.DataFrame)

    def test_has_required_columns(self, graph_trim):
        assert set(graph_trim.columns) == {"from", "to", "weight"}

    def test_weights_are_positive(self, graph_trim):
        assert (graph_trim["weight"] > 0).all()

    def test_fewer_edges_than_possible(self, simple_data):
        X, y, _ = simple_data
        result = eder_bct(X, y, n=100, keep_pct=0.2, rng=np.random.default_rng(42))
        n_possible = len(y) * (len(y) - 1) / 2
        assert len(result) < n_possible

    def test_different_metrics(self, simple_data):
        X, y, _ = simple_data
        for metric in [cosine, minmax, euclidean]:
            result = eder_bct(X, y, n=50, rng=np.random.default_rng(42), metric=metric)
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0

    def test_input_validation(self, simple_data):
        X, y, _ = simple_data
        with pytest.raises(ValueError, match="X has .* rows but y has .* labels"):
            eder_bct(X, y[:-1], n=10)

    def test_keep_pct(self, simple_data):
        X, y, _ = simple_data
        result_20 = eder_bct(X, y, n=100, keep_pct=0.2, rng=np.random.default_rng(42))
        result_80 = eder_bct(X, y, n=100, keep_pct=0.8, rng=np.random.default_rng(42))
        assert len(result_20) < len(result_80)


class TestPlotBct:
    def test_returns_fig_ax(self, simple_data, graph_trim):
        X, y, work_names = simple_data
        fig, ax = plot_bct(graph_trim, work_names, y)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_with_highlights(self, simple_data, graph_trim):
        X, y, work_names = simple_data
        fig, ax = plot_bct(graph_trim, work_names, y, highlight_works=["work_a"])
        assert fig is not None
        plt.close(fig)

    def test_with_min_weight(self, simple_data, graph_trim):
        X, y, work_names = simple_data
        fig, ax = plot_bct(graph_trim, work_names, y, min_weight=50)
        assert fig is not None
        plt.close(fig)

    def test_dodge_labels_false(self, simple_data, graph_trim):
        X, y, work_names = simple_data
        fig, ax = plot_bct(graph_trim, work_names, y, dodge_labels=False)
        assert fig is not None
        plt.close(fig)

    def test_ax_parameter(self, simple_data, graph_trim):
        """Test passing an existing axes object."""
        X, y, work_names = simple_data
        fig, axes = plt.subplot_mosaic([["A", "B"]], figsize=(12, 6))
        returned_fig, returned_ax = plot_bct(
            graph_trim, work_names, y, ax=axes["A"], min_weight=50
        )
        assert returned_fig is fig
        assert returned_ax is axes["A"]
        plt.close(fig)

    def test_single_work(self):
        """Edge case: all nodes belong to the same work."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(0, 1, (20, 5)))
        y = [f"only_work - {i}" for i in range(20)]
        work_names = ["only_work"] * 20
        graph = eder_bct(X, y, n=50, rng=np.random.default_rng(42))
        if len(graph) > 0:
            fig, ax = plot_bct(graph, work_names, y)
            assert fig is not None
            plt.close(fig)

    def test_two_works(self):
        """Edge case: only two works."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            np.vstack(
                [
                    rng.normal(0, 0.5, (10, 5)),
                    rng.normal(3, 0.5, (10, 5)),
                ]
            )
        )
        y = [f"a - {i}" for i in range(10)] + [f"b - {i}" for i in range(10)]
        work_names = ["a"] * 10 + ["b"] * 10
        graph = eder_bct(X, y, n=50, rng=np.random.default_rng(42))
        if len(graph) > 0:
            fig, ax = plot_bct(graph, work_names, y)
            assert fig is not None
            plt.close(fig)

    def test_custom_palette(self, simple_data, graph_trim):
        X, y, work_names = simple_data
        custom_palette = {"work_a": "#ff0000", "work_b": "#00ff00", "work_c": "#0000ff"}
        fig, ax = plot_bct(graph_trim, work_names, y, palette=custom_palette)
        assert fig is not None
        plt.close(fig)

    def test_title(self, simple_data, graph_trim):
        X, y, work_names = simple_data
        fig, ax = plot_bct(graph_trim, work_names, y, title="Custom Title")
        assert ax.get_title() == "Custom Title"
        plt.close(fig)
