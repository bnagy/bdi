#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for BDIVerifier.

These tests verify the correctness of the BDIVerifier class including:
- Initialization and parameter validation
- Fit and predict_proba methods
- All three methods: ranked, random, closest
- All six metrics
- Edge cases and error conditions
"""

import pytest
import numpy as np

from bdi import BDIVerifier  # type: ignore


class TestBDIVerifierInit:
    """Tests for BDIVerifier initialization."""

    def test_default_parameters(self):
        """Test default parameter values."""
        verifier = BDIVerifier()
        assert verifier.metric_fn.__name__ == "manhattan"
        assert verifier.method == "ranked"
        assert verifier.nb_bootstrap_iter == 100
        assert verifier.rnd_prop == 0.35
        assert verifier.balance is False
        assert verifier.fitted is False

    def test_custom_parameters(self):
        """Test custom parameter values."""
        verifier = BDIVerifier(
            metric="cosine",
            method="random",
            nb_bootstrap_iter=50,
            random_state=42,
            rnd_prop=0.5,
            balance=True,
        )
        assert verifier.method == "random"
        assert verifier.nb_bootstrap_iter == 50
        assert verifier.rnd_prop == 0.5
        assert verifier.balance is True

    def test_invalid_metric(self):
        """Invalid metric should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            BDIVerifier(metric="invalid_metric")

    def test_invalid_method(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported method"):
            BDIVerifier(method="invalid_method")

    def test_invalid_rnd_prop_zero(self):
        """rnd_prop of 0 should raise AssertionError."""
        with pytest.raises(AssertionError):
            BDIVerifier(rnd_prop=0.0)

    def test_invalid_rnd_prop_one(self):
        """rnd_prop of 1 should raise AssertionError."""
        with pytest.raises(AssertionError):
            BDIVerifier(rnd_prop=1.0)


class TestBDIVerifierFit:
    """Tests for BDIVerifier.fit method."""

    def test_fit_basic(self):
        """Test basic fit operation."""
        verifier = BDIVerifier()
        X = np.random.rand(10, 5)
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

        result = verifier.fit(X, y)

        assert verifier.fitted is True
        assert np.array_equal(verifier.train_X, X)
        assert np.array_equal(verifier.train_y, y)
        assert result is verifier  # Should return self

    def test_fit_single_class(self):
        """Test fit with single class (edge case)."""
        verifier = BDIVerifier()
        X = np.random.rand(5, 3)
        y = np.array([0, 0, 0, 0, 0])

        verifier.fit(X, y)
        assert verifier.fitted is True


class TestBDIVerifierPredict:
    """Tests for BDIVerifier.predict_proba method."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data with clear class separation."""
        np.random.seed(42)
        # Class 0: centered around [1, 1]
        X_train_0 = np.random.randn(10, 5) + 1.0
        # Class 1: centered around [-1, -1]
        X_train_1 = np.random.randn(10, 5) - 1.0
        X_train = np.vstack([X_train_0, X_train_1])
        y_train = np.array([0] * 10 + [1] * 10)

        # Test: one from class 0 region, one from class 1 region
        X_test = np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]])
        y_test = np.array([0, 1])

        return X_train, y_train, X_test, y_test

    def test_predict_not_fitted(self):
        """Predict without fit should raise RuntimeError."""
        verifier = BDIVerifier()
        X_test = np.random.rand(2, 5)
        y_test = np.array([0, 1])

        with pytest.raises(RuntimeError, match="Cannot predict without training"):
            verifier.predict_proba(X_test, y_test)

    def test_predict_proba_output_shape(self, simple_data):
        """Test that predict_proba returns correct shape."""
        X_train, y_train, X_test, y_test = simple_data
        verifier = BDIVerifier(nb_bootstrap_iter=50, random_state=42)
        verifier.fit(X_train, y_train)

        probas = verifier.predict_proba(X_test, y_test)

        assert probas.shape == (2,)
        assert all(0.0 <= p <= 1.0 for p in probas)

    def test_predict_proba_high_confidence(self, simple_data):
        """Test that well-separated data gives high confidence scores."""
        X_train, y_train, X_test, y_test = simple_data
        verifier = BDIVerifier(nb_bootstrap_iter=100, random_state=42)
        verifier.fit(X_train, y_train)

        probas = verifier.predict_proba(X_test, y_test)

        # Both should have reasonably high confidence (> 0.7) for correct class
        assert probas[0] > 0.7  # Test[0] vs class 0
        assert probas[1] > 0.7  # Test[1] vs class 1

    def test_predict_proba_all_methods(self, simple_data):
        """Test predict_proba with all supported methods."""
        X_train, y_train, X_test, y_test = simple_data

        for method in ["ranked", "random", "closest"]:
            verifier = BDIVerifier(method=method, nb_bootstrap_iter=50, random_state=42)
            verifier.fit(X_train, y_train)
            probas = verifier.predict_proba(X_test, y_test)
            assert probas.shape == (2,)

    def test_predict_proba_stores_dist_arrays(self, simple_data):
        """Test that _dist_arrays is populated after predict."""
        X_train, y_train, X_test, y_test = simple_data
        verifier = BDIVerifier(nb_bootstrap_iter=50, random_state=42)
        verifier.fit(X_train, y_train)

        verifier.predict_proba(X_test, y_test)

        assert hasattr(verifier, "_dist_arrays")
        assert len(verifier._dist_arrays) == 2
        assert all(len(arr) == 50 for arr in verifier._dist_arrays)

    def test_predict_proba_nb_imposters(self, simple_data):
        """Test nb_imposters parameter."""
        X_train, y_train, X_test, y_test = simple_data

        verifier = BDIVerifier(nb_bootstrap_iter=50, random_state=42)
        verifier.fit(X_train, y_train)

        # Test with different nb_imposters values
        probas_default = verifier.predict_proba(X_test, y_test, nb_imposters=30)
        probas_all = verifier.predict_proba(X_test, y_test, nb_imposters=-1)

        assert len(probas_default) == 2
        assert len(probas_all) == 2


class TestBDIVerifierEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_small_feature_space(self):
        """Test with very small feature space."""
        verifier = BDIVerifier(nb_bootstrap_iter=10, random_state=42)
        X_train = np.random.rand(10, 3)
        y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        X_test = np.random.rand(2, 3)
        y_test = np.array([0, 1])

        verifier.fit(X_train, y_train)
        probas = verifier.predict_proba(X_test, y_test)

        assert probas.shape == (2,)

    def test_single_candidate(self):
        """Test with only one candidate per class."""
        verifier = BDIVerifier(nb_bootstrap_iter=10, random_state=42)
        X_train = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_train = np.array([0, 1])
        X_test = np.array([[1.0, 0.0]])
        y_test = np.array([0])

        verifier.fit(X_train, y_train)
        probas = verifier.predict_proba(X_test, y_test)

        assert probas.shape == (1,)

    def test_balance_parameter(self):
        """Test the balance parameter."""
        verifier = BDIVerifier(balance=True, nb_bootstrap_iter=50, random_state=42)
        # Unbalanced classes
        X_train = np.random.rand(20, 5)
        y_train = np.array([0] * 18 + [1] * 2)  # Highly unbalanced
        X_test = np.random.rand(2, 5)
        y_test = np.array([0, 1])

        verifier.fit(X_train, y_train)
        probas = verifier.predict_proba(X_test, y_test)

        assert probas.shape == (2,)

    def test_balanced_subsample_with_size(self):
        """Test _balanced_subsample with explicit size parameter."""
        import pandas as pd

        verifier = BDIVerifier(balance=True, nb_bootstrap_iter=50, random_state=42)
        # Use equal class sizes so size parameter works
        y = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2])  # 3 of each class
        rng = np.random.default_rng(42)

        # Test with size parameter - 6 samples = 2 per class
        result = verifier._balanced_subsample(y, rng, size=6)
        assert len(result) == 6  # 2 per class (6 / 3 classes)

    def test_closest_method(self):
        """Test the closest method (depth=1)."""
        verifier = BDIVerifier(method="closest", nb_bootstrap_iter=50, random_state=42)
        X_train = np.random.rand(20, 5)
        y_train = np.array([0] * 10 + [1] * 10)
        X_test = np.random.rand(2, 5)
        y_test = np.array([0, 1])

        verifier.fit(X_train, y_train)
        probas = verifier.predict_proba(X_test, y_test)

        assert probas.shape == (2,)

    def test_index_error_handling(self):
        """Test IndexError handling in _bootstrap_imposters for ranked method.

        This tests the edge case where there are fewer than 3 candidates or
        imposters, triggering the IndexError exception handler.
        """
        verifier = BDIVerifier(method="ranked", nb_bootstrap_iter=50, random_state=42)
        # Only 2 candidates per class - will trigger IndexError handling
        X_train = np.random.rand(4, 5)
        y_train = np.array([0, 0, 1, 1])
        X_test = np.random.rand(1, 5)
        y_test = np.array([0])

        verifier.fit(X_train, y_train)
        probas = verifier.predict_proba(X_test, y_test)

        assert probas.shape == (1,)

    def test_zero_division_error_handling(self):
        """Test ZeroDivisionError handling in _bootstrap_imposters.

        This tests the edge case where minmax distance encounters zero vectors,
        causing division by zero. The exception handler should catch this and
        continue to the next iteration.
        """
        verifier = BDIVerifier(
            metric="minmax", method="ranked", nb_bootstrap_iter=50, random_state=42
        )
        # Create data where test vector is non-zero, but some candidates are zero.
        # ZeroDivisionError occurs when comparing two all-zero vectors with minmax.
        # We need non-zero candidates so the test can succeed after catching errors.
        np.random.seed(42)
        X_train = np.random.rand(20, 5)
        # Add some zero vectors as candidates (class 0) - these will cause
        # ZeroDivisionError when the random feature subset happens to be all zeros
        X_train[0] = [0, 0, 0, 0, 0]
        X_train[1] = [0, 0, 0, 0, 0]
        # Keep most candidates non-zero so test can succeed
        y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # Test vector is non-zero - comparing with zero candidates may cause ZeroDivisionError
        # depending on random feature selection
        X_test = np.random.rand(1, 5)
        y_test = np.array([0])

        verifier.fit(X_train, y_train)
        # This should handle ZeroDivisionError and eventually succeed
        probas = verifier.predict_proba(X_test, y_test)

        assert probas.shape == (1,)


class TestBDIVerifierEndToEnd:
    """End-to-end tests with known expected behavior."""

    def test_synthetic_identical_authors(self):
        """Test with synthetic data where same-author should score high."""
        np.random.seed(123)

        # Create training data: two authors with distinct styles
        # Author 0: high values in first half of features
        X_author0 = np.random.rand(15, 10) * 0.8 + 0.2
        X_author0[:, 5:] *= 0.1  # Low values in second half

        # Author 1: high values in second half of features
        X_author1 = np.random.rand(15, 10) * 0.8 + 0.2
        X_author1[:, :5] *= 0.1  # Low values in first half

        X_train = np.vstack([X_author0, X_author1])
        y_train = np.array([0] * 15 + [1] * 15)

        # Test documents: one from each author's style
        X_test = np.vstack(
            [
                np.random.rand(3, 10) * 0.8 + 0.2,  # Author 0 style
                np.random.rand(3, 10) * 0.8 + 0.2,  # Author 1 style
            ]
        )
        X_test[:3, 5:] *= 0.1
        X_test[3:, :5] *= 0.1

        y_test = np.array([0, 0, 0, 1, 1, 1])

        verifier = BDIVerifier(nb_bootstrap_iter=100, random_state=42)
        verifier.fit(X_train, y_train)
        probas = verifier.predict_proba(X_test, y_test)

        # First 3 should have high confidence for author 0
        # Last 3 should have high confidence for author 1
        assert all(probas[:3] > 0.7)
        assert all(probas[3:] > 0.7)
