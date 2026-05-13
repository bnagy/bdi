#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for distance metrics in bdi.metrics.

These tests verify the correctness of all distance functions used by BDIVerifier.
Each metric is tested with known inputs and expected outputs.
"""

import pytest
import numpy as np

from bdi.metrics import manhattan, euclidean, minmax, common_ngrams, cosine, nini


class TestManhattan:
    """Tests for the Manhattan distance metric."""

    def test_identical_vectors(self):
        """Identical vectors should have distance 0."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        assert manhattan(x, y) == 0.0

    def test_simple_difference(self):
        """Simple case with known result."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        # |1-2| + |2-3| + |3-4| = 1 + 1 + 1 = 3
        assert manhattan(x, y) == 3.0

    def test_negative_values(self):
        """Test with negative values."""
        x = np.array([-1.0, 0.0, 1.0])
        y = np.array([1.0, 0.0, -1.0])
        # |-1-1| + |0-0| + |1-(-1)| = 2 + 0 + 2 = 4
        assert manhattan(x, y) == 4.0

    def test_zeros(self):
        """Test with zero vectors."""
        x = np.zeros(5)
        y = np.zeros(5)
        assert manhattan(x, y) == 0.0


class TestEuclidean:
    """Tests for the Euclidean distance metric."""

    def test_identical_vectors(self):
        """Identical vectors should have distance 0."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        assert euclidean(x, y) == 0.0

    def test_simple_difference(self):
        """Simple case: sqrt(1^2 + 1^2 + 1^2) = sqrt(3)."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        assert np.isclose(euclidean(x, y), np.sqrt(3))

    def test_orthogonal(self):
        """Orthogonal vectors: sqrt(1 + 1) = sqrt(2)."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        assert np.isclose(euclidean(x, y), np.sqrt(2))


class TestMinmax:
    """Tests for the Ružička (minmax) distance metric."""

    def test_identical_vectors(self):
        """Identical vectors should have distance 0."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        assert minmax(x, y) == 0.0

    def test_disjoint_vectors(self):
        """Disjoint vectors should have distance 1."""
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 0.0, 1.0])
        # mins = 0, maxs = 1, so 1 - 0/1 = 1
        assert minmax(x, y) == 1.0

    def test_partial_overlap(self):
        """Partial overlap case."""
        x = np.array([1.0, 2.0, 0.0])
        y = np.array([2.0, 1.0, 0.0])
        # mins = 1+1+0 = 2, maxs = 2+2+0 = 4, so 1 - 2/4 = 0.5
        assert minmax(x, y) == 0.5

    def test_equal_vectors(self):
        """Equal vectors should have distance 0."""
        x = np.array([0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5])
        assert minmax(x, y) == 0.0


class TestCommonNgrams:
    """Tests for the common n-grams distance metric."""

    def test_identical_vectors(self):
        """Identical vectors should have distance 0."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        assert common_ngrams(x, y) == 0.0

    def test_different_vectors(self):
        """Test with different vectors."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        # Only y[i] > 0 matters: (1-0)/1 = 1, (0-1)/1 = -1, sum = 2
        result = common_ngrams(x, y)
        assert result > 0

    def test_zeros(self):
        """Test with zero vectors - should handle gracefully."""
        x = np.zeros(3)
        y = np.zeros(3)
        # All y[i] = 0, so no contribution, result = 0
        assert common_ngrams(x, y) == 0.0


class TestCosine:
    """Tests for the cosine distance metric."""

    def test_identical_vectors(self):
        """Identical vectors should have distance 0."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        assert cosine(x, y) == 0.0

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have distance 1."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        assert cosine(x, y) == 1.0

    def test_opposite_vectors(self):
        """Opposite vectors should have distance 2 (cosine distance can exceed 1)."""
        x = np.array([1.0, 0.0])
        y = np.array([-1.0, 0.0])
        # cosine distance = 1 - (-1) = 2 for opposite vectors
        assert cosine(x, y) == 2.0

    def test_similar_vectors(self):
        """Similar vectors should have small distance."""
        x = np.array([1.0, 1.0])
        y = np.array([1.0, 1.1])
        assert cosine(x, y) < 0.2


class TestNini:
    """Tests for the Nini distance metric."""

    def test_identical_binary(self):
        """Identical binary vectors should have distance 0."""
        x = np.array([1.0, 1.0, 0.0, 0.0])
        y = np.array([1.0, 1.0, 0.0, 0.0])
        assert nini(x, y) == 0.0

    def test_disjoint_binary(self):
        """Disjoint binary vectors should have distance > 1."""
        x = np.array([1.0, 0.0, 0.0, 0.0])
        y = np.array([0.0, 0.0, 0.0, 1.0])
        # No overlap, should be high distance
        result = nini(x, y)
        assert result > 1.0

    def test_partial_overlap(self):
        """Partial overlap case."""
        x = np.array([1.0, 1.0, 0.0, 0.0])
        y = np.array([1.0, 0.0, 1.0, 0.0])
        # 50% overlap
        result = nini(x, y)
        assert 0.0 <= result <= 2.0


class TestMetricsBenchmark:
    """Benchmark tests for distance metrics."""

    @pytest.mark.parametrize(
        "metric_func", [manhattan, euclidean, minmax, common_ngrams, cosine, nini]
    )
    def test_metric_performance(self, metric_func):
        """Test each metric with 1000-dimensional vectors."""
        np.random.seed(42)
        x = np.random.rand(1000)
        y = np.random.rand(1000)

        # Run multiple iterations to test performance
        for _ in range(100):
            result = metric_func(x, y)

        # Verify result is valid
        assert isinstance(result, float)
        assert not np.isnan(result)
