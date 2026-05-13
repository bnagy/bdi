#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for Vectorizer in bdi.vectorizer.

These tests verify the correctness of text vectorization functionality.
"""

import pytest
import numpy as np

from bdi import Vectorizer


class TestVectorizerInit:
    """Tests for Vectorizer initialization."""

    def test_default_parameters(self):
        """Test default parameter values."""
        vectorizer = Vectorizer()
        assert vectorizer.params["max_features"] == 100
        assert vectorizer.params["ngram_range"] == (1, 1)
        assert vectorizer.params["lowercase"] is True

    def test_custom_parameters(self):
        """Test custom parameter values."""
        vectorizer = Vectorizer(
            mfi=50,
            ngram_type="char",
            ngram_size=2,
            vector_space="tf_idf",
            lowercase=False,
        )
        assert vectorizer.params["max_features"] == 50
        assert vectorizer.params["ngram_range"] == (2, 2)

    def test_invalid_vector_space(self):
        """Invalid vector_space should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported vector space"):
            Vectorizer(vector_space="invalid")


class TestVectorizerFit:
    """Tests for Vectorizer.fit method."""

    def test_fit_basic(self):
        """Test basic fit operation."""
        vectorizer = Vectorizer(mfi=10)
        texts = ["hello world", "hello there", "goodbye world"]

        result = vectorizer.fit(texts)

        assert hasattr(vectorizer, "feature_names")
        assert result is vectorizer

    def test_fit_returns_feature_names(self):
        """Test that fit populates feature_names."""
        vectorizer = Vectorizer(mfi=10)
        texts = ["the quick brown fox", "the lazy dog"]

        vectorizer.fit(texts)

        assert len(vectorizer.feature_names) > 0

    def test_fit_tf_std(self):
        """Test tf_std vector space."""
        vectorizer = Vectorizer(mfi=10, vector_space="tf_std")
        texts = ["hello world", "hello there"]

        vectorizer.fit(texts)
        result = vectorizer.transform(texts)

        assert result.shape[0] == 2

    def test_fit_bin(self):
        """Test bin vector space."""
        vectorizer = Vectorizer(mfi=10, vector_space="bin")
        texts = ["hello world", "hello there"]

        vectorizer.fit(texts)
        result = vectorizer.transform(texts)

        assert result.shape[0] == 2


class TestVectorizerTransform:
    """Tests for Vectorizer.transform method."""

    def test_transform_output_shape(self):
        """Test that transform returns correct shape."""
        vectorizer = Vectorizer(mfi=10)
        texts = ["hello world", "hello there", "goodbye world"]

        vectorizer.fit(texts)
        result = vectorizer.transform(texts)

        assert result.shape == (3, len(vectorizer.feature_names))

    def test_transform_before_fit(self):
        """Transform without fit should raise error."""
        vectorizer = Vectorizer()
        texts = ["hello world"]

        with pytest.raises(AttributeError):
            vectorizer.transform(texts)


class TestVectorizerFitTransform:
    """Tests for Vectorizer.fit_transform method."""

    def test_fit_transform_basic(self):
        """Test fit_transform in one step."""
        vectorizer = Vectorizer(mfi=10)
        texts = ["hello world", "hello there", "goodbye world"]

        result = vectorizer.fit_transform(texts)

        assert result.shape[0] == 3
        assert hasattr(vectorizer, "feature_names")


class TestStdDevScaler:
    """Tests for StdDevScaler class."""

    def test_scaler_basic(self):
        """Test basic scaling operation."""
        from bdi.vectorizer import StdDevScaler

        scaler = StdDevScaler()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = scaler.fit_transform(X)

        assert result.shape == X.shape

    def test_scaler_returns_self(self):
        """Test that fit returns self."""
        from bdi.vectorizer import StdDevScaler

        scaler = StdDevScaler()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = scaler.fit(X)

        assert result is scaler
