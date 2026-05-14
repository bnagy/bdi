#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BDIVerifier: Authorship Verification in the General Imposters Framework.

This module implements the Burrows-Distance Imposters (BDI) verification algorithm
for authorship attribution, following the General Imposters framework.

References:
    - M. Koppel and Y. Winter (2014), Determining if Two Documents are
      by the Same Author, JASIST, 65(1): 178-187
    - Stover, J. A., Y. Winter, M. Koppel, M. Kestemont (2015).
      Computational Authorship Verification Method Attributes New Work
      to Major 2nd Century African Author, JASIST, doi: 10.1002/asi.23460
"""

import heapq
import logging

import numpy as np
import pandas as pd
import scipy as sp
import numpy.typing as npt
from typing import Collection, Callable

from .metrics import minmax, manhattan, euclidean, common_ngrams, cosine, nini

logger = logging.getLogger("bdi")

CPU_METRICS: dict[str, Callable] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "minmax": minmax,
    "cng": common_ngrams,
    "cosine": cosine,
    "nini": nini,
}


class BDIVerifier:
    """
    Authorship verification using the Bootstrap Distance Imposters algorithm.

    This class implements the General Imposters framework for authorship verification.
    It follows sklearn-like conventions with `fit()` and `predict_proba()` methods.

    The algorithm works by:
    1. Bootstrapping random feature subsets
    2. Comparing test document distances to candidate vs imposter documents
    3. Calculating verification scores based on distance differences

    Example:
        >>> verifier = BDIVerifier(metric="manhattan", method="ranked")
        >>> verifier.fit(X_train, y_train)
        >>> scores = verifier.predict_proba(X_test, y_test)
    """

    def __init__(
        self,
        metric: str = "manhattan",
        method: str = "ranked",
        nb_bootstrap_iter: int = 100,
        random_state: int = 1066,
        rnd_prop: float = 0.35,
        balance: bool = False,
    ) -> None:
        """
        Initialize the BDIVerifier.

        Args:
            metric: Distance metric to use. One of 'manhattan', 'euclidean',
                'minmax', 'cng', 'cosine', or 'nini'. Defaults to 'manhattan'.
            method: Bootstrapping method. One of 'ranked' (default), 'random',
                or 'closest'. See class docstring for details.
            nb_bootstrap_iter: Number of bootstrap iterations. Defaults to 100.
            random_state: Random seed for reproducibility. Defaults to 1066.
            rnd_prop: Proportion of features to sample per iteration. Must be
                between 0 and 1. Defaults to 0.35.
            balance: If True, balance classes to smallest class size each
                iteration. Defaults to False.

        Raises:
            ValueError: If metric or method is not supported.
            AssertionError: If rnd_prop is not between 0 and 1.
        """
        assert 0.0 < rnd_prop < 1.0, "rnd_prop must be between 0 and 1"
        if method not in ["ranked", "random", "closest"]:
            raise ValueError(
                f"Unsupported method '{method}'. Valid options: random, closest, ranked"
            )

        self.rnd = np.random.default_rng(seed=random_state)
        self.method = method
        self.nb_bootstrap_iter = nb_bootstrap_iter
        self.rnd_prop = rnd_prop
        self.balance = balance
        self.fitted: bool = False
        self.train_X: npt.NDArray
        self.train_y: npt.NDArray

        try:
            self.metric_fn = CPU_METRICS[metric]
        except KeyError:
            raise ValueError(
                f"Unknown metric '{metric}'. Valid options: {list(CPU_METRICS.keys())}"
            )

    def fit(
        self, X: Collection[Collection[float]], y: Collection[int]
    ) -> "BDIVerifier":
        """
        Fit the verifier by storing training data.

        Args:
            X: Training vectors, shape (n_documents, n_features).
            y: Author labels for training documents, shape (n_documents,).

        Returns:
            self: The fitted verifier instance.
        """
        logger.info(f"Fitting on {len(y)} documents...")
        self.train_X = np.array(X, dtype=np.float64)
        self.train_y = np.array(y, dtype="int")
        self.fitted = True
        return self

    def _balanced_subsample(
        self, y: pd.Series, rng: np.random.Generator, size: int = 0
    ) -> list[int]:
        """
        Generate a balanced subsample of indices.

        Samples equally from each class, down to the size of the smallest class.

        Args:
            y: Series of class labels.
            rng: Random number generator.
            size: Total sample size. If 0, uses smallest class size * n_classes.

        Returns:
            List of sampled indices.
        """
        subsample = []

        if size == 0:
            n_smp = y.value_counts().min()
        else:
            n_smp = int(size / len(y.value_counts().index))

        for label in y.value_counts().index:
            idx = pd.Series(y[y == label]).index
            samples = np.asarray(idx)
            indexes = rng.choice(len(samples), size=n_smp, replace=False)
            subsample += list(samples[indexes])

        return subsample

    def _bootstrap_imposters(
        self, test_vec: npt.NDArray[np.float64], target_int: int, nb_imposters: int
    ) -> list[float]:
        """
        Run the bootstrap imposters algorithm for a single test vector.

        Args:
            test_vec: The test document vector.
            target_int: The target author label.
            nb_imposters: Number of imposters to sample.

        Returns:
            List of distance differences from bootstrap iterations.

        Raises:
            ValueError: If too many ZeroDivisionErrors occur (data too sparse).
        """
        candidates = self.train_X[(self.train_y == target_int).nonzero()]
        others = self.train_X[(self.train_y != target_int).nonzero()]
        differences: list[float] = []
        cand_samps: npt.NDArray[np.float64] = np.array([], dtype="float64")
        other_samps: npt.NDArray[np.float64] = np.array([], dtype="float64")
        pdy = pd.Series(self.train_y)

        if self.method == "random":
            cand_samps = candidates[
                self.rnd.choice(
                    candidates.shape[0], self.nb_bootstrap_iter, replace=True
                ),
                :,
            ]
            other_samps = others[
                self.rnd.choice(others.shape[0], self.nb_bootstrap_iter, replace=True),
                :,
            ]

        for i in range(self.nb_bootstrap_iter * 2):
            if self.balance:
                ss = self._balanced_subsample(pdy, rng=self.rnd)
                candidates = self.train_X[np.where(self.train_y[ss] == target_int)]
                others = self.train_X[np.where(self.train_y[ss] != target_int)]

            try:
                ridx = self.rnd.choice(
                    self.train_X.shape[1],
                    int(self.train_X.shape[1] * self.rnd_prop),
                    replace=False,
                )

                if self.method == "random":
                    in_dist = self.metric_fn(test_vec[ridx], cand_samps[i][ridx])
                    out_dist = self.metric_fn(test_vec[ridx], other_samps[i][ridx])
                    differences.append(out_dist - in_dist)

                elif self.method in ("ranked", "closest"):
                    in_dists = [
                        self.metric_fn(test_vec[ridx], cand_samp[ridx])
                        for cand_samp in candidates
                    ]

                    if nb_imposters > 0:
                        n_imposters = min(nb_imposters, others.shape[0])
                        this_others = others[
                            self.rnd.choice(
                                others.shape[0], n_imposters, replace=False
                            ),
                            :,
                        ]
                    else:
                        this_others = others

                    out_dists = [
                        self.metric_fn(test_vec[ridx], other_samp[ridx])
                        for other_samp in this_others
                    ]

                    top_in = heapq.nsmallest(3, in_dists)
                    top_out = heapq.nsmallest(3, out_dists)
                    d = 0.0
                    depth = 1 if self.method == "closest" else 3

                    for nn in range(depth):
                        try:
                            d += (top_out[nn] - top_in[nn]) / (nn + 1)
                        except IndexError:
                            d += (top_out[-1] - top_in[-1]) / (nn + 1)

                    differences.append(d)

                if len(differences) >= self.nb_bootstrap_iter:
                    return differences[: self.nb_bootstrap_iter]

            except ZeroDivisionError:
                continue

        raise ValueError("Too many ZeroDivisionErrors. Data too sparse?")

    def predict_proba(
        self,
        test_X: Collection[Collection[float]],
        test_y: Collection[int],
        nb_imposters: int = 30,
    ) -> npt.NDArray[np.float64]:
        """
        Predict verification probabilities for test documents.

        Args:
            test_X: Test vectors, shape (n_test_problems, n_features).
            test_y: Target author labels for each test problem.
            nb_imposters: Number of imposters to sample per iteration.
                Use -1 to consider all imposters. Defaults to 30.

        Returns:
            Array of verification scores in [0, 1], where higher values
            indicate stronger evidence for same authorship.

        Raises:
            RuntimeError: If called before fitting.
        """
        if not self.fitted:
            raise RuntimeError("Cannot predict without training. Call fit() first.")

        dist_arrays = []
        logger.info(f"Predicting on {len(test_y)} documents")

        for vec, candidate_int in zip(np.array(test_X), np.array(test_y)):
            dist_arrays.append(
                self._bootstrap_imposters(vec, candidate_int, nb_imposters)
            )

        self._dist_arrays = dist_arrays
        probas = [(100 - sp.stats.percentileofscore(x, 0)) / 100.0 for x in dist_arrays]
        return np.array(probas, dtype="float64")
