"""Bootstrap Consensus Tree (Eder et al.) algorithm."""

import warnings
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


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
