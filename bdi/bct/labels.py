"""Force-directed label dodging for BCT plots."""

import numpy as np
from matplotlib.axes import Axes


def dodge_labels(
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
