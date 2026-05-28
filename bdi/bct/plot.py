"""BCT plotting and visualization."""

from typing import Any, Optional, cast

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from .fonts import ensure_fonts
from .labels import dodge_labels


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
    dodge: bool = True,
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
        dodge (bool): Whether to apply force-directed label dodging.
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

    ensure_fonts()

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

    if dodge and len(centroids) > 1:
        label_pos = dodge_labels(
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
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.82, 0.98),
        fontsize=legend_fontsize,
        framealpha=0.9,
        ncol=3,
        title="Works",
        title_fontsize=legend_fontsize + 1,
    )
    legend.get_title().set_fontfamily("Roboto Condensed")
    for text in legend.get_texts():
        text.set_fontfamily("Roboto Condensed")

    if title is None:
        title = f"Bootstrap Consensus Tree ({len(G.nodes())} nodes, {len(filtered_edges)} edges shown)"
    ax.set_title(title, fontsize=14, fontfamily="Roboto Condensed")
    ax.axis("off")

    real_fig = cast(Figure, fig)
    real_fig.tight_layout()
    return real_fig, ax
