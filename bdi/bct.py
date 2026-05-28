"""Bootstrap Consensus Tree (BCT) visualization.

Implementation of the Eder BCT algorithm for authorship verification, with
networkx/matplotlib visualization.

References:
    - Eder, M. (2017). Visualization in stylometry: cluster analysis using
      networks. Digital Scholarship in the Humanities, 32(1), 50-64.
    - https://computationalstylistics.github.io/projects/bootstrap-networks/

Example:
    >>> from bdi import eder_bct, plot_bct
    >>> from bdi.metrics import cosine
    >>> graph_trim = eder_bct(X, y, n=1000, metric=cosine)
    >>> fig, ax = plot_bct(graph_trim, work_names, y)

    # Post-hoc customization via fig/ax:
    >>> ax.legend().remove()
    >>> ax.set_facecolor("#f5f5f5")
    >>> for text in ax.texts:
    ...     text.set_fontweight("normal")
    >>> plt.show()
"""

import warnings
from typing import Any, Optional, cast

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from sklearn.neighbors import KNeighborsClassifier

__all__ = ["eder_bct", "plot_bct"]


def _dodge_labels(
    labels: list[str],
    positions: dict[str, tuple[float, float]],
    ax: Axes,
    fontsize: int = 9,
    fontfamily: str = "Roboto Condensed",
    fontweight: str = "bold",
    pad: float = 0.35,
    iterations: int = 50,
    repulsion: float = 0.08,
    attraction: float = 0.02,
) -> dict[str, tuple[float, float]]:
    """Dodge overlapping label positions using force-directed repulsion.

    For each pair of labels whose bounding boxes overlap, applies a repulsion
    force pushing them apart. Also applies a weak attraction force pulling
    each label back toward its original centroid to prevent excessive drift.

    Returns:
        Dict mapping label -> (x, y) dodged position.
    """
    if len(labels) <= 1:
        return positions

    temp_texts = {}
    for w in labels:
        t = ax.text(
            positions[w][0],
            positions[w][1],
            w,
            fontsize=fontsize,
            fontfamily=fontfamily,
            fontweight=fontweight,
            ha="center",
            va="center",
            alpha=0,
        )
        t.set_bbox(
            dict(boxstyle=f"round,pad={pad}", facecolor="none", edgecolor="none")
        )
        temp_texts[w] = t

    fig = ax.figure
    fig.canvas.draw()

    bboxes = {}
    for w, t in temp_texts.items():
        bbox = t.get_window_extent()
        bboxes[w] = bbox.transformed(ax.transData.inverted())

    pos = {w: np.array(p, dtype=float) for w, p in positions.items()}
    orig_pos = {w: np.array(p, dtype=float) for w, p in positions.items()}

    for _ in range(iterations):
        forces = {w: np.zeros(2) for w in labels}

        for i, w1 in enumerate(labels):
            for w2 in labels[i + 1 :]:
                b1, b2 = bboxes[w1], bboxes[w2]
                if b1.overlaps(b2):
                    c1 = np.array([b1.x0 + b1.width / 2, b1.y0 + b1.height / 2])
                    c2 = np.array([b2.x0 + b2.width / 2, b2.y0 + b2.height / 2])
                    diff = c1 - c2
                    dist = np.linalg.norm(diff)
                    if dist < 1e-9:
                        diff = np.random.randn(2) * 0.01
                        dist = np.linalg.norm(diff)
                    force = repulsion * diff / dist
                    forces[w1] += force
                    forces[w2] -= force

        for w in labels:
            forces[w] += attraction * (orig_pos[w] - pos[w])

        max_move = 0.0
        for w in labels:
            pos[w] += forces[w]
            max_move = max(max_move, float(np.linalg.norm(forces[w])))

        for w, t in temp_texts.items():
            t.set_position((float(pos[w][0]), float(pos[w][1])))
        fig.canvas.draw()
        for w, t in temp_texts.items():
            bboxes[w] = t.get_window_extent().transformed(ax.transData.inverted())

        if max_move < 1e-4:
            break

    for t in temp_texts.values():
        t.remove()

    return {w: (float(p[0]), float(p[1])) for w, p in pos.items()}


def eder_bct(
    X: pd.DataFrame,
    y: list[str],
    n: int = 500,
    keep_pct: float = 0.2,
    feats_pct: float = 0.3,
    rng: np.random.Generator = np.random.default_rng(),
    metric: Any = "cosine",
) -> pd.DataFrame:
    """Bootstrap Consensus Tree (Eder et al.).

    At each iteration, samples a random subset of features and computes
    3-nearest-neighbors. Edge weights are the inverse rank (3 for nearest,
    2 for second, 1 for third), aggregated across all iterations.

    Returns:
        DataFrame with columns ['from', 'to', 'weight'].

    Raises:
        ValueError: If X and y have different lengths.
    """
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} labels.")

    dfs = []
    cl = KNeighborsClassifier(n_neighbors=3, metric=metric)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The number of unique classes is greater than 50%",
        )
        for _ in range(n):
            this_X = X.sample(int(X.shape[1] * feats_pct), axis=1, random_state=rng)
            cl.fit(this_X, y)
            dists, indices = cl.kneighbors(n_neighbors=3)
            dicts = []
            for i, node_ary in enumerate(indices):
                for j, node in enumerate(node_ary):
                    dicts.append(
                        {"from": y[i], "to": y[node], "weight": len(node_ary) - j}
                    )
            dfs.append(pd.DataFrame(dicts))
    graph = pd.concat(dfs).groupby(["from", "to"], as_index=False).agg("sum")
    n_keep = int(len(graph) * keep_pct)
    # nlargest signature in pandas stubs is narrower than runtime behavior
    result = cast(pd.DataFrame, graph.nlargest(n_keep, "weight"))  # pyright: ignore
    return result


def plot_bct(
    graph_trim: pd.DataFrame,
    work_names: list[str],
    y: list[str],
    highlight_works: Optional[list[str]] = None,
    palette: Optional[dict[str, Any]] = None,
    figsize: tuple[float, float] = (14, 10),
    seed: int = 42,
    min_weight: Optional[float] = None,
    edge_alpha_range: tuple[float, float] = (0.3, 0.85),
    edge_width_range: tuple[float, float] = (0.7, 6.0),
    node_size: int = 45,
    highlight_size: int = 140,
    label_fontsize: int = 9,
    legend_fontsize: int = 7,
    title: Optional[str] = None,
    layout_k: float = 0.5,
    layout_iterations: int = 100,
    curvature: float = 0.18,
    dodge_labels: bool = True,
    dodge_iterations: int = 50,
    dodge_repulsion: float = 0.08,
    ax: Optional[Axes] = None,
    dpi: int = 144,
) -> tuple[Figure, Axes]:
    """Plot BCT using networkx with colored curved edges and clean styling.

    Produces a force-directed graph with gradient-colored curved edges,
    optional edge weight filtering, highlighted nodes as triangles,
    cluster labels with tinted backgrounds, and a legend.

    Pass an existing ``ax`` to draw into a subplot (e.g. from
    ``plt.subplot_mosaic``)::

        fig, axes = plt.subplot_mosaic([["A", "B"], ["A", "C"]])
        plot_bct(graph_trim, work_names, y, ax=axes["A"])
        plot_bct(graph_trim, work_names, y, ax=axes["B"], min_weight=100)

    For post-hoc customization, use the returned fig and ax::

        fig, ax = plot_bct(graph_trim, work_names, y)
        ax.legend().remove()
        ax.set_facecolor("#f5f5f5")
        for text in ax.texts:
            text.set_fontweight("normal")
        plt.show()

    Args:
        graph_trim (pd.DataFrame): BCT edge DataFrame with columns
            'from', 'to', 'weight'.
        work_names (list[str]): Work names (one per node, chunk numbers
            stripped).
        y (list[str]): Node labels (with chunk numbers).
        highlight_works (list[str] | None): Work names to highlight as
            triangles. Default: None.
        palette (dict[str, Any] | None): Mapping from work name to color.
            Auto-generated from tab20 colormap if None. Default: None.
        figsize (tuple[float, float]): Figure size in inches. Ignored if
            ``ax`` is provided. Default: (14, 10).
        seed (int): Random seed for layout. Default: 42.
        min_weight (float | None): Minimum edge weight to display. If None,
            auto-set to 25th percentile. Default: None.
        edge_alpha_range (tuple[float, float]): (min, max) alpha for edges.
            Default: (0.3, 0.85).
        edge_width_range (tuple[float, float]): (min, max) linewidth for
            edges. Default: (0.7, 6.0).
        node_size (int): Size of normal nodes. Default: 45.
        highlight_size (int): Size of highlighted triangle nodes.
            Default: 140.
        label_fontsize (int): Font size for cluster labels. Default: 9.
        legend_fontsize (int): Font size for legend. Default: 7.
        title (str | None): Plot title. Auto-generated if None. Default: None.
        layout_k (float): Optimal distance between nodes in spring layout
            (smaller = tighter). Default: 0.5.
        layout_iterations (int): Number of spring layout iterations.
            Default: 100.
        curvature (float): Edge curvature factor (higher = more curved).
            Default: 0.18.
        dodge_labels (bool): Whether to apply force-directed label dodging.
            Default: True.
        dodge_iterations (int): Number of dodge iterations. Default: 50.
        dodge_repulsion (float): Repulsion strength between overlapping
            labels. Default: 0.08.
        ax (Axes | None): Existing matplotlib Axes to draw into. If None,
            creates a new figure. Default: None.
        dpi (int): Resolution for the created figure. Ignored if ``ax`` is
            provided. Default: 144.

    Returns:
        tuple[Figure, Axes]: The figure and axes objects.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
        assert fig is not None

    G = nx.Graph()
    for _, row in graph_trim.iterrows():
        G.add_edge(row["from"], row["to"], weight=row["weight"])

    node_work = dict(zip(y, work_names))

    # Layout uses ALL edges; min_weight only affects rendering, not positions.
    pos = nx.spring_layout(G, k=layout_k, iterations=layout_iterations, seed=seed)

    edges = list(G.edges(data=True))
    weights = [d["weight"] for _, _, d in edges]
    w_min, w_max = min(weights), max(weights)

    if min_weight is None:
        min_weight = float(np.percentile(weights, 25))
    filtered_edges = [(u, v, d) for u, v, d in edges if d["weight"] >= min_weight]

    unique_works = sorted(set(work_names))
    if palette is None:
        cmap = plt.get_cmap("tab20")
        _palette: dict[str, Any] = {w: cmap(i % 20) for i, w in enumerate(unique_works)}
    else:
        _palette = palette

    if highlight_works is None:
        highlight_works = []
    highlighted_nodes = [
        n for n in G.nodes() if node_work.get(n, "") in highlight_works
    ]
    normal_nodes = [n for n in G.nodes() if node_work.get(n, "") not in highlight_works]

    ax.set_facecolor("white")

    # --- Edges with gradient coloring ---
    # Each edge is drawn as a single polyline. We create individual 2-point
    # segments with linearly interpolated colors, using butt caps to avoid
    # circular artifacts at joints.
    edge_segments: list[list[np.ndarray]] = []
    edge_colors: list[tuple[float, ...]] = []
    edge_lws_list: list[float] = []

    for u, v, d in filtered_edges:
        w_norm = (d["weight"] - w_min) / (w_max - w_min + 1e-9)
        lw = edge_width_range[0] + w_norm * (edge_width_range[1] - edge_width_range[0])
        a = edge_alpha_range[0] + w_norm * (edge_alpha_range[1] - edge_alpha_range[0])

        cu = to_rgba(_palette.get(node_work.get(u, ""), "#888888"))
        cv = to_rgba(_palette.get(node_work.get(v, ""), "#888888"))

        s, e = np.array(pos[u]), np.array(pos[v])
        direction = e - s
        length = np.linalg.norm(direction)
        if length > 0:
            perp = np.array([-direction[1], direction[0]])
            perp = perp / np.linalg.norm(perp)
            ctrl = (s + e) / 2 + perp * curvature * length
            t = np.linspace(0, 1, 80)
            pts = (
                np.outer((1 - t) ** 2, s)
                + np.outer(2 * (1 - t) * t, ctrl)
                + np.outer(t**2, e)
            )
            for i in range(len(pts) - 1):
                frac = i / (len(pts) - 1)
                r = (1 - frac) * cu[0] + frac * cv[0]
                g = (1 - frac) * cu[1] + frac * cv[1]
                b = (1 - frac) * cu[2] + frac * cv[2]
                seg_color: tuple[float, float, float, float] = (r, g, b, a)
                edge_segments.append([pts[i], pts[i + 1]])
                edge_colors.append(seg_color)
                edge_lws_list.append(lw)

    if edge_segments:
        lc = LineCollection(
            edge_segments,
            colors=edge_colors,
            linewidths=edge_lws_list,
            capstyle="butt",
            zorder=1,
        )
        ax.add_collection(lc)

    # --- Normal nodes ---
    if normal_nodes:
        nc = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=normal_nodes,
            node_color=[
                _palette.get(node_work.get(n, ""), "#888888") for n in normal_nodes
            ],
            node_size=node_size,
            alpha=0.9,
            ax=ax,
        )
        nc.set_zorder(2)

    # --- Highlighted nodes ---
    if highlighted_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=highlighted_nodes,
            node_color="black",
            node_size=highlight_size + 20,
            node_shape="^",
            ax=ax,
        ).set_zorder(3)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=highlighted_nodes,
            node_color=[
                _palette.get(node_work.get(n, ""), "#888888") for n in highlighted_nodes
            ],
            node_size=highlight_size,
            node_shape="^",
            edgecolors="black",
            linewidths=0.5,
            ax=ax,
        ).set_zorder(4)

    # --- Labels ---
    work_nodes_map: dict[str, list[str]] = {}
    for n in G.nodes():
        w = node_work.get(n, "")
        work_nodes_map.setdefault(w, []).append(n)

    centroids: dict[str, tuple[float, float]] = {}
    for w, nodes in work_nodes_map.items():
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        centroids[w] = (float(np.mean(xs)), float(np.mean(ys)))

    if dodge_labels and len(centroids) > 1:
        label_pos = _dodge_labels(
            list(centroids.keys()),
            centroids,
            ax,
            fontsize=label_fontsize,
            pad=0.35,
            iterations=dodge_iterations,
            repulsion=dodge_repulsion,
        )
    else:
        label_pos = centroids

    for w, (cx, cy) in label_pos.items():
        bc = to_rgba(_palette.get(w, "#888888"))
        bg = tuple(0.85 * c + 0.15 for c in bc[:3]) + (0.9,)
        tc = "white" if sum(bc[:3]) < 1.5 else "black"
        t = ax.text(
            cx,
            cy,
            w,
            fontsize=label_fontsize,
            fontfamily="Roboto Condensed",
            fontweight="bold",
            color=tc,
            ha="center",
            va="center",
            zorder=5,
        )
        t.set_bbox(
            dict(boxstyle="round,pad=0.35", facecolor=bg, edgecolor="none", linewidth=0)
        )

    # --- Legend ---
    legend_handles = []
    for w in unique_works:
        hi = w in highlight_works
        legend_handles.append(
            plt.scatter(
                [],
                [],
                marker="^" if hi else "o",
                s=(7 if hi else 6) ** 2,
                color=_palette.get(w, "#888888"),
                edgecolors="black" if hi else "none",
                linewidths=0.5 if hi else 0,
                label=w,
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.82, 0.98),
        fontsize=legend_fontsize,
        framealpha=0.9,
        ncol=3,
        title="Works",
        title_fontsize=legend_fontsize + 1,
    )

    if title is None:
        title = f"Bootstrap Consensus Tree ({len(G.nodes())} nodes, {len(filtered_edges)} edges shown)"
    ax.set_title(title, fontsize=14, fontfamily="Roboto Condensed")
    ax.axis("off")

    real_fig = cast(Figure, fig)
    real_fig.tight_layout()
    return real_fig, ax
