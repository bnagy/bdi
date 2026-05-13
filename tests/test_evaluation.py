#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for evaluation metrics in bdi.evaluation.

These tests verify the correctness of accuracy, auc, c_at_1, and pan_metrics.
"""

import pytest
import numpy as np

from bdi import accuracy, auc, c_at_1, pan_metrics


class TestAccuracy:
    """Tests for accuracy function."""

    def test_perfect_accuracy(self):
        """Perfect predictions should give accuracy 1.0."""
        predictions = [0.0, 0.0, 1.0, 1.0]
        ground_truth = [0.0, 0.0, 1.0, 1.0]
        assert accuracy(predictions, ground_truth) == 1.0

    def test_zero_accuracy(self):
        """Completely wrong predictions should give accuracy 0.0."""
        predictions = [1.0, 1.0, 0.0, 0.0]
        ground_truth = [0.0, 0.0, 1.0, 1.0]
        assert accuracy(predictions, ground_truth) == 0.0

    def test_half_accuracy(self):
        """Half correct predictions should give accuracy 0.5."""
        predictions = [0.0, 1.0, 0.0, 1.0]
        ground_truth = [0.0, 0.0, 1.0, 1.0]
        assert accuracy(predictions, ground_truth) == 0.5

    def test_with_threshold(self):
        """Test that 0.5 threshold is used correctly."""
        predictions = [0.4, 0.6, 0.4, 0.6]
        ground_truth = [0.0, 1.0, 0.0, 1.0]
        assert accuracy(predictions, ground_truth) == 1.0


class TestAuc:
    """Tests for auc function."""

    def test_perfect_auc(self):
        """Perfect separation should give AUC 1.0."""
        predictions = [0.1, 0.2, 0.8, 0.9]
        ground_truth = [0.0, 0.0, 1.0, 1.0]
        assert auc(predictions, ground_truth) == 1.0

    def test_random_auc(self):
        """Random predictions should give AUC around 0.5."""
        predictions = [0.5, 0.5, 0.5, 0.5]
        ground_truth = [0.0, 0.0, 1.0, 1.0]
        assert auc(predictions, ground_truth) == 0.5


class TestCAt1:
    """Tests for c_at_1 function."""

    def test_all_correct(self):
        """All correct predictions should give c@1 of 1.0."""
        predictions = [0.1, 0.2, 0.8, 0.9]
        ground_truth = [0.0, 0.0, 1.0, 1.0]
        assert c_at_1(predictions, ground_truth) == 1.0

    def test_all_wrong(self):
        """All wrong predictions should give c@1 of 0.0."""
        predictions = [0.9, 0.8, 0.2, 0.1]
        ground_truth = [0.0, 0.0, 1.0, 1.0]
        assert c_at_1(predictions, ground_truth) == 0.0

    def test_unanswered_problems(self):
        """Unanswered problems (score = 0.5) should affect c@1."""
        predictions = [0.5, 0.5, 0.5, 0.5]
        ground_truth = [0.0, 0.0, 1.0, 1.0]
        # All unanswered, so c@1 should be 0
        assert c_at_1(predictions, ground_truth) == 0.0


class TestPanMetrics:
    """Tests for pan_metrics function."""

    def test_returns_tuple(self):
        """Should return a tuple of three values."""
        predictions = [0.1, 0.2, 0.8, 0.9]
        ground_truth = [0.0, 0.0, 1.0, 1.0]
        result = pan_metrics(predictions, ground_truth)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_all_correct(self):
        """Perfect predictions should give high scores."""
        predictions = [0.1, 0.2, 0.8, 0.9]
        ground_truth = [0.0, 0.0, 1.0, 1.0]
        acc, auc_score, c1 = pan_metrics(predictions, ground_truth)

        assert acc == 1.0
        assert auc_score == 1.0
        assert c1 == 1.0
