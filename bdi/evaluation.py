#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation metrics for authorship verification.

This module provides PAN competition-specific evaluation metrics:
- accuracy: Standard classification accuracy
- auc: Area Under the ROC Curve
- c_at_1: PAN-specific metric that rewards unanswered problems

References:
    - Stamatatos, E. et al. Overview of the Author Identification
      Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
    - Peñas, A. and Rodrigo, A. A Simple Measure to Assess Nonresponse.
      Proc. of the 49th Annual Meeting of the Association for
      Computational Linguistics, Vol. 1, pages 1415-1424, 2011.
"""

import numba
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Collection


def accuracy(
    prediction_scores: Collection[float], ground_truth_scores: Collection[float]
) -> float:
    """
    Calculate verification accuracy.

    A prediction is correct if both prediction and ground truth are on the
    same side of the 0.5 threshold.

    Args:
        prediction_scores: Predicted scores in [0, 1].
        ground_truth_scores: True labels (0 or 1).

    Returns:
        Accuracy as a fraction of correct predictions.
    """
    acc = 0.0
    assert len(ground_truth_scores) == len(prediction_scores)

    for gt_score, pred_score in zip(ground_truth_scores, prediction_scores):
        if (pred_score >= 0.5) == (gt_score >= 0.5):
            acc += 1.0
    return acc / float(len(prediction_scores))


def auc(
    prediction_scores: Collection[float], ground_truth_scores: Collection[float]
) -> float:
    """
    Calculate the Area Under the ROC Curve.

    Args:
        prediction_scores: Predicted scores in [0, 1].
        ground_truth_scores: True labels (0 or 1).

    Returns:
        AUC score.
    """
    return float(roc_auc_score(ground_truth_scores, prediction_scores))


@numba.jit(nopython=True)
def c_at_1(prediction_scores, ground_truth_scores) -> float:
    """
    Calculate the c@1 score.

    This PAN-specific metric rewards systems that leave uncertain problems
    unanswered (score = 0.5).

    Args:
        prediction_scores: Predicted scores in [0, 1].
        ground_truth_scores: True labels (0 or 1).

    Returns:
        c@1 score.
    """
    n = float(len(prediction_scores))
    nc, nu = 0.0, 0.0
    for i in range(len(prediction_scores)):
        pred_score = prediction_scores[i]
        gt_score = ground_truth_scores[i]
        if pred_score == 0.5:
            nu += 1.0
        elif (pred_score > 0.5) == (gt_score > 0.5):
            nc += 1.0
    return (1 / n) * (nc + (nu * nc / n))


def pan_metrics(
    prediction_scores: Collection[float], ground_truth_scores: Collection[float]
) -> tuple[float, float, float]:
    """
    Calculate all three PAN evaluation metrics.

    Args:
        prediction_scores: Predicted scores in [0, 1].
        ground_truth_scores: True labels (0 or 1).

    Returns:
        Tuple of (accuracy, auc, c@1) scores.
    """
    return (
        accuracy(prediction_scores, ground_truth_scores),
        auc(prediction_scores, ground_truth_scores),
        c_at_1(np.array(prediction_scores), np.array(ground_truth_scores)),
    )
