#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Score shifting utilities for authorship verification.

This module provides the ScoreShifter class for optimizing verification
scores to better align with PAN competition metrics (AUC x c@1).
"""

import warnings
from itertools import permutations
from typing import Callable, Collection
from typing_extensions import Self

import numpy as np

from .evaluation import auc, c_at_1

EPSILON = 1e-6


def rescale(
    value: float, orig_min: float, orig_max: float, new_min: float, new_max: float
) -> float:
    """
    Rescale a value from one range to another.

    Args:
        value: The value to rescale.
        orig_min: Minimum of the original range.
        orig_max: Maximum of the original range.
        new_min: Minimum of the new range.
        new_max: Maximum of the new range.

    Returns:
        The rescaled value.
    """
    orig_span = orig_max - orig_min
    new_span = new_max - new_min
    scaled_value = (value - orig_min) / (orig_span + EPSILON)
    return new_min + (scaled_value * new_span)


def correct_scores(
    scores: Collection[float], p1: float = 0.25, p2: float = 0.75
) -> list[float]:
    """
    Rescale scores to three intervals for PAN metrics.

    Scores are rescaled to:
    - [0, p1] -> [0, 0.499] (negative attribution)
    - [p1, p2] -> 0.5 (unanswered)
    - [p2, 1] -> [0.501, 1] (positive attribution)

    Args:
        scores: Input scores in [0, 1].
        p1: Lower threshold. Defaults to 0.25.
        p2: Upper threshold. Defaults to 0.75.

    Returns:
        Rescaled scores.
    """
    if min(scores) < 0.0 - EPSILON or max(scores) > 1.0 + EPSILON:
        warnings.warn(
            "Warning: scores are expected to be in [0,1], shifting may not work properly."
        )
    new_scores = []
    for score in scores:
        if score <= p1:
            new_scores.append(
                rescale(score, orig_min=0, orig_max=p1, new_min=0.0, new_max=0.499)
            )
        elif score >= p2:
            new_scores.append(
                rescale(score, orig_min=p2, orig_max=1, new_min=0.501, new_max=1.0)
            )
        else:
            new_scores.append(0.5)
    return new_scores


def _auc_c_at_1(predicted, gt: Collection[float]) -> float:
    """Objective function combining AUC and c@1."""
    return auc(predicted, gt) * c_at_1(predicted, gt)


class ScoreShifter:
    """
    Shift verification scores to optimize PAN metrics.

    This class finds optimal thresholds p1 and p2 that maximize
    AUC x c@1 through grid search, then applies score correction.

    Example:
        >>> shifter = ScoreShifter()
        >>> shifter.fit(predicted_scores, ground_truth)
        >>> corrected = shifter.transform(new_scores)
    """

    def __init__(
        self,
        grid_size: int = 100,
        min_val: float = 0.2,
        max_val: float = 0.8,
        min_spread: float = 0.0,
    ):
        """
        Initialize the ScoreShifter.

        Args:
            grid_size: Number of points in grid search. Defaults to 100.
            min_val: Minimum value for p1. Defaults to 0.2.
            max_val: Maximum value for p2. Defaults to 0.8.
            min_spread: Minimum distance between p1 and p2. Defaults to 0.0.
        """
        self.optimal_p1: float
        self.optimal_p2: float
        self.min = min_val
        self.max = max_val
        self.min_spread = min_spread
        self.grid_size = grid_size
        self.fitted: bool = False

    def manual_fit(self, p1: float, p2: float) -> Self:
        """
        Manually set the threshold values.

        Args:
            p1: Lower threshold.
            p2: Upper threshold.

        Returns:
            self: The fitted ScoreShifter.

        Raises:
            ValueError: If p1 > p2 or values outside [0, 1].
        """
        if 0 <= p1 <= p2 <= 1:
            self.optimal_p1 = p1
            self.optimal_p2 = p2
        else:
            raise ValueError("Bad values. Need 0 <= p1 <= p2 <= 1")
        self.fitted = True
        return self

    def fit(
        self,
        predicted_scores: Collection[float],
        ground_truth_scores: Collection[float],
        obj_func: Callable[[Collection[float], Collection[float]], float] = _auc_c_at_1,
    ) -> Self:
        """
        Fit the score shifter by grid search.

        Args:
            predicted_scores: Predicted scores in [0, 1].
            ground_truth_scores: True labels (0 or 1).
            obj_func: Objective function to maximize. Defaults to AUC x c@1.

        Returns:
            self: The fitted ScoreShifter.
        """
        thresholds = np.around(
            np.linspace(self.min, self.max, num=self.grid_size, endpoint=False), 6
        )
        nb_thresholds = thresholds.shape[0]

        objective_scores = np.zeros((nb_thresholds, nb_thresholds))

        gt = np.array(ground_truth_scores)
        for i, j in permutations(range(nb_thresholds), 2):
            p1, p2 = thresholds[i], thresholds[j]

            if (p1 <= p2) and (p2 - p1 >= self.min_spread):
                corrected_scores = np.array(correct_scores(predicted_scores, p1, p2))
                objective_scores[i][j] = obj_func(corrected_scores, ground_truth_scores)

        opt_p1_idx, opt_p2_idx = np.unravel_index(
            objective_scores.argmax(), objective_scores.shape
        )
        self.optimal_p1 = thresholds[opt_p1_idx]
        self.optimal_p2 = thresholds[opt_p2_idx]

        self.fitted = True
        return self

    def transform(self, scores: Collection[float]) -> list[float]:
        """
        Apply score shifting to new scores.

        Args:
            scores: Scores to transform.

        Returns:
            Transformed scores.

        Raises:
            RuntimeError: If shifter has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("Must fit before transforming.")

        return correct_scores(scores, p1=self.optimal_p1, p2=self.optimal_p2)
