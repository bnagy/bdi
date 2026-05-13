#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for ScoreShifter in bdi.score_shifting.

These tests verify the correctness of the score shifting functionality.
"""

import pytest
import numpy as np

from bdi import ScoreShifter


class TestScoreShifterInit:
    """Tests for ScoreShifter initialization."""

    def test_default_parameters(self):
        """Test default parameter values."""
        shifter = ScoreShifter()
        assert shifter.grid_size == 100
        assert shifter.min == 0.2
        assert shifter.max == 0.8
        assert shifter.min_spread == 0.0
        assert shifter.fitted is False

    def test_custom_parameters(self):
        """Test custom parameter values."""
        shifter = ScoreShifter(grid_size=50, min_val=0.1, max_val=0.9, min_spread=0.1)
        assert shifter.grid_size == 50
        assert shifter.min == 0.1
        assert shifter.max == 0.9
        assert shifter.min_spread == 0.1


class TestScoreShifterManualFit:
    """Tests for manual_fit method."""

    def test_manual_fit_valid(self):
        """Test manual fit with valid values."""
        shifter = ScoreShifter()
        result = shifter.manual_fit(0.25, 0.75)
        assert shifter.optimal_p1 == 0.25
        assert shifter.optimal_p2 == 0.75
        assert shifter.fitted is True
        assert result is shifter

    def test_manual_fit_invalid_order(self):
        """Test manual fit with p1 > p2 raises error."""
        shifter = ScoreShifter()
        with pytest.raises(ValueError, match="Bad values"):
            shifter.manual_fit(0.75, 0.25)

    def test_manual_fit_out_of_range(self):
        """Test manual fit with out of range values."""
        shifter = ScoreShifter()
        with pytest.raises(ValueError, match="Bad values"):
            shifter.manual_fit(-0.1, 0.5)


class TestScoreShifterFit:
    """Tests for fit method."""

    def test_fit_basic(self):
        """Test basic fit operation."""
        shifter = ScoreShifter(grid_size=20)
        predicted = np.array([0.1, 0.2, 0.8, 0.9])
        ground_truth = np.array([0.0, 0.0, 1.0, 1.0])

        result = shifter.fit(predicted, ground_truth)

        assert shifter.fitted is True
        assert 0.0 <= shifter.optimal_p1 <= shifter.optimal_p2 <= 1.0
        assert result is shifter

    def test_fit_returns_self(self):
        """Test that fit returns self for chaining."""
        shifter = ScoreShifter(grid_size=20)
        predicted = np.array([0.1, 0.9])
        ground_truth = np.array([0.0, 1.0])

        result = shifter.fit(predicted, ground_truth)

        assert result is shifter


class TestScoreShifterTransform:
    """Tests for transform method."""

    def test_transform_not_fitted(self):
        """Transform without fit should raise RuntimeError."""
        shifter = ScoreShifter()
        with pytest.raises(RuntimeError, match="Must fit before transforming"):
            shifter.transform([0.5, 0.6, 0.7])

    def test_transform_basic(self):
        """Test basic transform operation."""
        shifter = ScoreShifter()
        shifter.manual_fit(0.25, 0.75)

        scores = [0.1, 0.5, 0.9]
        result = shifter.transform(scores)

        assert len(result) == 3
        assert result[0] < 0.5  # Below p1 -> rescaled to [0, 0.499]
        assert result[1] == 0.5  # Between p1 and p2 -> 0.5
        assert result[2] > 0.5  # Above p2 -> rescaled to [0.501, 1]

    def test_transform_output_range(self):
        """Test that transform output is in valid range."""
        shifter = ScoreShifter()
        shifter.manual_fit(0.3, 0.7)

        scores = np.random.rand(100)
        result = shifter.transform(scores)

        assert all(0.0 <= r <= 1.0 for r in result)


class TestScoreShifterCorrectScores:
    """Tests for correct_scores helper function."""

    def test_correct_scores_below_p1(self):
        """Scores below p1 should be rescaled to [0, 0.499]."""
        from bdi.score_shifting import correct_scores

        scores = [0.1, 0.2]
        result = correct_scores(scores, p1=0.25, p2=0.75)

        assert all(r < 0.5 for r in result)

    def test_correct_scores_above_p2(self):
        """Scores above p2 should be rescaled to [0.501, 1]."""
        from bdi.score_shifting import correct_scores

        scores = [0.8, 0.9]
        result = correct_scores(scores, p1=0.25, p2=0.75)

        assert all(r > 0.5 for r in result)

    def test_correct_scores_between(self):
        """Scores between p1 and p2 should become 0.5."""
        from bdi.score_shifting import correct_scores

        scores = [0.3, 0.5, 0.7]
        result = correct_scores(scores, p1=0.25, p2=0.75)

        assert all(r == 0.5 for r in result)


class TestScoreShifterEndToEnd:
    """End-to-end tests for ScoreShifter."""

    def test_full_workflow(self):
        """Test full fit-transform workflow."""
        shifter = ScoreShifter(grid_size=30)

        # Create some predicted scores with clear separation
        predicted = np.array([0.1, 0.15, 0.85, 0.9])
        ground_truth = np.array([0.0, 0.0, 1.0, 1.0])

        shifter.fit(predicted, ground_truth)

        # Transform new scores
        new_scores = [0.2, 0.5, 0.8]
        corrected = shifter.transform(new_scores)

        assert len(corrected) == 3
        assert all(isinstance(c, float) for c in corrected)
