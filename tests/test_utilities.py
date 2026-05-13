#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for utilities in bdi.utilities.

These tests verify the correctness of utility functions.
"""

import numpy as np

from bdi.utilities import binarize, make_up_lies


class TestBinarize:
    """Tests for binarize function."""

    def test_binarize_below_threshold(self):
        """Scores below 0.5 should become 'N'."""
        scores = [0.1, 0.3, 0.49]
        result = binarize(scores)
        assert result == ["N", "N", "N"]

    def test_binarize_above_threshold(self):
        """Scores above 0.5 should become 'Y'."""
        scores = [0.51, 0.7, 0.99]
        result = binarize(scores)
        assert result == ["Y", "Y", "Y"]

    def test_binarize_at_threshold(self):
        """Scores at 0.5 should become 'X'."""
        scores = [0.5, 0.5, 0.5]
        result = binarize(scores)
        assert result == ["X", "X", "X"]

    def test_binarize_mixed(self):
        """Test mixed scores."""
        scores = [0.1, 0.5, 0.9]
        result = binarize(scores)
        assert result == ["N", "X", "Y"]


class TestMakeUpLies:
    """Tests for make_up_lies function."""

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = [0, 1, 2]

        ret_X, ret_y, ground_truth = make_up_lies(X, y)

        assert ret_X.shape[0] == 6  # Doubled
        assert len(ret_y) == 6
        assert len(ground_truth) == 6

    def test_ground_truth_values(self):
        """Test that ground truth has correct values."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = [0, 1]

        ret_X, ret_y, ground_truth = make_up_lies(X, y)

        # First half should be 1.0 (correct), second half 0.0 (lies)
        assert all(ground_truth[:2] == 1.0)
        assert all(ground_truth[2:] == 0.0)

    def test_lies_are_different(self):
        """Test that lies have different labels."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = [0, 1, 2]

        ret_X, ret_y, ground_truth = make_up_lies(X, y)

        # Lies should have different labels than original
        for i in range(3):
            assert ret_y[i] != ret_y[i + 3]
