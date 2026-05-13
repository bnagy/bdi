#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distance metrics for authorship verification.

This module provides optimized distance functions used by BDIVerifier.
All functions are JIT-compiled with numba for performance.

References:
    - Koppel, M. and Winter, Y. (2014), Determining if Two Documents are
      by the Same Author, JASIST, 65(1): 178-187
    - Nini, A. (2023). A Theory of Linguistic Individuality for Authorship
      Analysis (Elements in Forensic Linguistics). Cambridge University Press.
"""

import math

import numba
import numpy as np
from numpy.typing import NDArray

TARGET = "cpu"


@numba.jit(nopython=True)
def minmax(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """
    Calculate the Ružička (minmax) distance between two vectors.

    This is a symmetric distance measure defined as 1 - (sum of mins / sum of maxs).
    Range is [0, 1] where 0 means identical vectors.

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        The minmax distance between x and y.

    References:
        - Cha SH. Comprehensive Survey on Distance/Similarity Measures
          between Probability Density Functions. International Journ.
          of Math. Models and Methods in Applied Sciences. 2007; 1(4):300–307.
    """
    assert x.shape == y.shape
    mins, maxs = 0.0, 0.0
    a, b = 0.0, 0.0
    for i in range(x.shape[0]):
        a, b = x[i], y[i]
        if a >= b:
            maxs += a
            mins += b
        else:
            maxs += b
            mins += a

    if maxs > 0.0:
        return 1.0 - (mins / maxs)
    return 0.0


@numba.jit(nopython=True)
def nini(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """
    Calculate the Nini distance between two vectors.

    This distance is based on Pearson's correlation applied to binary indicator
    vectors (non-zero values are converted to 1). Range is [0, 2].

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        The Nini distance between x and y.

    References:
        - Nini, A. (2023). A Theory of Linguistic Individuality for Authorship
          Analysis (Elements in Forensic Linguistics). Cambridge University Press.
    """
    xn, ny, xy, nn = 0, 0, 0, 0
    assert x.shape == y.shape
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            if y[i] > 0.0:
                xy += 1
            else:
                xn += 1
        else:
            if y[i] > 0.0:
                ny += 1
            else:
                nn += 1

    length = xn + xy + ny + nn
    xbar = (xn + xy) / length
    ybar = (ny + xy) / length

    top = (
        xn * (1.0 - xbar) * (0.0 - ybar)
        + ny * (0.0 - xbar) * (1.0 - ybar)
        + nn * (0.0 - xbar) * (0.0 - ybar)
        + xy * (1.0 - xbar) * (1.0 - ybar)
    )

    bottom = math.sqrt(
        ((1.0 - xbar) * (1.0 - xbar) * (xn + xy))
        + ((0.0 - xbar) * (0.0 - xbar) * (ny + nn))
    ) * math.sqrt(
        ((1.0 - ybar) * (1.0 - ybar) * (ny + xy))
        + ((0.0 - ybar) * (0.0 - ybar) * (xn + nn))
    )

    if bottom == 0.0:
        return 2.0

    return 1.0 - (top / bottom)


@numba.jit(nopython=True)
def manhattan(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """
    Calculate the Manhattan (city block) distance between two vectors.

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        The Manhattan distance between x and y.
    """
    diff, z = 0.0, 0.0
    assert x.shape == y.shape
    for i in range(x.shape[0]):
        z = x[i] - y[i]
        if z < 0.0:
            z = -z
        diff += z
    return diff


@numba.jit(nopython=True)
def euclidean(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """
    Calculate the Euclidean distance between two vectors.

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        The Euclidean distance between x and y.
    """
    diff, z = 0.0, 0.0
    assert x.shape == y.shape
    for i in range(x.shape[0]):
        z = x[i] - y[i]
        diff += z * z
    return math.sqrt(diff)


@numba.jit(nopython=True)
def common_ngrams(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """
    Calculate the common n-grams distance between two vectors.

    This distance considers only positions where y has non-zero values.

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        The common n-grams distance between x and y.
    """
    diff, z = 0.0, 0.0
    assert x.shape == y.shape
    for i in range(x.shape[0]):
        if y[i] > 0.0:
            z = (2.0 * (x[i] - y[i])) / (x[i] + y[i])
            diff += z * z
    return diff


@numba.jit(nopython=True)
def cosine(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """
    Calculate the cosine distance between two vectors.

    Cosine distance = 1 - cosine_similarity.

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        The cosine distance between x and y.
    """
    numerator, denom_a, denom_b = 0.0, 0.0, 0.0
    assert x.shape == y.shape
    for i in range(x.shape[0]):
        numerator += x[i] * y[i]
        denom_a += x[i] * x[i]
        denom_b += y[i] * y[i]

    return 1.0 - (numerator / (math.sqrt(denom_a) * math.sqrt(denom_b)))
