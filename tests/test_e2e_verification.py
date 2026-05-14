#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end verification test for BDIVerifier.

This test uses the PAN 2014 du_essays corpus to verify that the BDIVerifier
produces consistent results with a seeded RNG.
"""

import numpy as np
import pytest

from bdi import BDIVerifier  # type: ignore

# Load gold results from numpy file
GOLD_SCORES = np.load("tests/gold_e2e_results.npy")
GOLD_TRAIN_X = np.load("tests/gold_train_X.npy")
GOLD_TEST_X = np.load("tests/gold_test_X.npy")
GOLD_TRAIN_Y = np.load("tests/gold_train_y.npy")
GOLD_TEST_Y = np.load("tests/gold_test_y.npy")


class TestE2EVerification:
    """End-to-end verification tests for BDIVerifier."""

    def test_new_verifier_matches_gold_results(self):
        """Test that BDIVerifier produces gold standard results.

        These gold results were generated using the same parameters
        (minmax metric, ranked method, 10 bootstrap iterations,
        random_state=1066, rnd_prop=0.35, nb_imposters=30).
        """
        # Use pre-vectorized data for reproducibility across platforms
        train_X = GOLD_TRAIN_X
        train_y = GOLD_TRAIN_Y
        test_X = GOLD_TEST_X
        test_y = GOLD_TEST_Y

        # Run BDIVerifier with same parameters as gold results
        verifier = BDIVerifier(
            metric="minmax",
            method="ranked",
            nb_bootstrap_iter=10,
            random_state=1066,
            rnd_prop=0.35,
        )
        verifier.fit(train_X, train_y)
        scores = verifier.predict_proba(test_X, test_y, nb_imposters=30)

        # Debug output for CI troubleshooting
        import sys

        print(f"\nDEBUG: scores[:10] = {scores[:10]}", file=sys.stderr)
        print(f"DEBUG: GOLD_SCORES[:10] = {GOLD_SCORES[:10]}", file=sys.stderr)
        print(
            f"DEBUG: Max diff = {np.max(np.abs(scores - GOLD_SCORES))}", file=sys.stderr
        )
        print(
            f"DEBUG: scipy version = {__import__('scipy').__version__}", file=sys.stderr
        )
        print(f"DEBUG: numpy version = {np.__version__}", file=sys.stderr)

        # Verify against gold standard
        assert np.allclose(
            scores, GOLD_SCORES, rtol=1e-10, atol=1e-10
        ), f"Scores don't match gold standard! Max diff: {np.max(np.abs(scores - GOLD_SCORES))}"


class TestPercentileOfScoreConsistency:
    """Tests for scipy.stats.percentileofscore cross-platform consistency."""

    def test_percentileofscore_basic(self):
        """Test percentileofscore with basic data."""
        import scipy.stats as sp

        # Test with known data
        x = np.array([-0.5, -0.3, 0.0, 0.1, 0.3])
        result = sp.stats.percentileofscore(x, 0)
        # 2 values below 0, 1 equal to 0, 2 above 0
        # With 'rank' kind: (2 + 3 + 1) * 50 / 5 = 60
        assert result == 60.0, f"Expected 60.0, got {result}"

    def test_percentileofscore_all_zeros(self):
        """Test percentileofscore with all zeros."""
        import scipy.stats as sp

        x = np.array([0.0, 0.0, 0.0, 0.0])
        result = sp.stats.percentileofscore(x, 0)
        # With 'rank' kind: (0 + 4 + 1) * 50 / 4 = 62.5
        assert result == 62.5, f"Expected 62.5, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
