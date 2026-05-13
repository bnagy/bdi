#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Text vectorization utilities for authorship verification.

This module provides the Vectorizer class for converting text documents
into numerical feature vectors suitable for use with BDIVerifier.
"""

from __future__ import annotations

import scipy.sparse as sp
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from typing import Collection, cast
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, Normalizer


def identity(y: str) -> list[str]:
    """
    Simple identity tokenizer that splits on whitespace.

    Args:
        y: Input string to tokenize.

    Returns:
        List of tokens.
    """
    return y.split()


class StdDevScaler(BaseEstimator):
    """
    Scale features by dividing by column-wise standard deviations.

    This scaling gives more weight to features with lower standard deviation,
    which is useful for calculating Burrows's Delta.

    Reference:
        Argamon, S. 'Interpreting Burrows's Delta: Geometric and Probabilistic
        Foundations', LLC 23:3 (2008).
    """

    weights_: np.ndarray

    def fit(self, X: sp.spmatrix | NDArray[np.float64], y=None) -> StdDevScaler:  # type: ignore[override]
        """
        Compute column-wise standard deviations.

        Args:
            X: Input data of shape (n_samples, n_features).
            y: Ignored, present for API compatibility.

        Returns:
            self: The fitted scaler.
        """
        scaler = StandardScaler(with_mean=False).fit(X)
        self.weights_ = np.asarray(scaler.scale_)
        return self

    def transform(self, X: sp.spmatrix | NDArray[np.float64]) -> sp.spmatrix:  # type: ignore[override]
        """
        Scale data by dividing by standard deviations.

        Args:
            X: Input data to transform.

        Returns:
            Transformed data with same shape as input.
        """
        if not sp.isspmatrix_csr(X):
            X_csr = sp.csr_matrix(X, dtype=np.float64)
        else:
            X_csr = cast(sp.csr_matrix, X)
        for i in range(X_csr.shape[0]):
            start, end = X_csr.indptr[i], X_csr.indptr[i + 1]
            X_csr.data[start:end] = (
                X_csr.data[start:end] / self.weights_[X_csr.indices[start:end]]
            )
        return X_csr

    def fit_transform(
        self, X: sp.spmatrix | NDArray[np.float64], y=None
    ) -> sp.spmatrix:
        """
        Fit and transform in one step.

        Args:
            X: Input data.
            y: Ignored.

        Returns:
            Transformed data.
        """
        self.fit(X)
        return self.transform(X)


class Vectorizer:
    """
    Vectorize texts into a sparse, two-dimensional matrix.

    This class wraps sklearn vectorizers with a simplified API
    for authorship verification tasks.

    Supported vector space models:
        - 'tf': Simple relative term frequency
        - 'tf_scaled': TF normalized with MinMaxScaler
        - 'tf_std': TF normalized with StdDevScaler (for Burrows's Delta)
        - 'tf_idf': Traditional TF-IDF
        - 'bin': Binary model (presence/absence only)
    """

    def __init__(
        self,
        mfi: int = 100,
        ngram_type: str = "word",
        ngram_size: int = 1,
        vocabulary: list[str] | None = None,
        vector_space: str = "tf",
        lowercase: bool = True,
        min_df: float = 0.0,
        max_df: float = 1.0,
        ignore: list[str] | None = None,
    ):
        """
        Initialize the vectorizer.

        Args:
            mfi: Maximum number of features to extract. Defaults to 100.
            ngram_type: Type of n-grams ('word', 'char', 'char_wb').
            ngram_size: Size of n-grams. Defaults to 1.
            vocabulary: Predefined vocabulary to use.
            vector_space: Vector space model to use.
            lowercase: Whether to lowercase input. Defaults to True.
            min_df: Minimum document frequency for features.
            max_df: Maximum document frequency for features.
            ignore: List of features to ignore (e.g., stopwords).

        Raises:
            ValueError: If vector_space is not supported.
        """
        if ignore is None:
            ignore = []

        if vector_space not in ("tf", "tf_scaled", "tf_std", "tf_idf", "bin"):
            raise ValueError(f"Unsupported vector space model: {vector_space}")

        self.params = {
            "max_features": mfi,
            "max_df": max_df,
            "min_df": min_df,
            "preprocessor": None,
            "ngram_range": (ngram_size, ngram_size),
            "lowercase": lowercase,
            "vocabulary": vocabulary,
            "decode_error": "ignore",
            "stop_words": ignore,
        }

        if ngram_type == "word":
            self.params["tokenizer"] = identity
        elif ngram_type in ("char", "char_wb"):
            self.params["analyzer"] = ngram_type

        n = Normalizer(norm="l2", copy=False)

        if vector_space == "tf":
            self.params["use_idf"] = False
            v = TfidfVectorizer(**self.params)
            self.transformer = Pipeline([("s1", v), ("s2", n)])

        elif vector_space == "tf_std":
            self.params["use_idf"] = False
            v = TfidfVectorizer(**self.params)
            scaler = StdDevScaler()
            self.transformer = Pipeline([("s1", v), ("s2", scaler), ("s3", n)])

        elif vector_space == "tf_idf":
            self.params["use_idf"] = True
            v = TfidfVectorizer(**self.params)
            self.transformer = Pipeline([("s1", v), ("s2", n)])

        elif vector_space == "bin":
            self.params["binary"] = True
            v = CountVectorizer(**self.params)
            self.transformer = Pipeline([("s1", v), ("s2", n)])

    def fit(self, texts: Collection[str]) -> Vectorizer:
        """
        Fit the vectorizer to texts.

        Args:
            texts: List of text strings to fit on.

        Returns:
            self: The fitted vectorizer.
        """
        self.transformer.fit(texts)
        self.feature_names = self.transformer.named_steps["s1"].get_feature_names_out()
        return self

    def transform(self, texts: Collection[str]) -> NDArray[np.float64]:
        """
        Transform texts to feature vectors.

        Args:
            texts: List of text strings to transform.

        Returns:
            Dense array of shape (n_texts, n_features).
        """
        return self.transformer.transform(texts).toarray()

    def fit_transform(self, texts: Collection[str]) -> NDArray[np.float64]:
        """
        Fit and transform in one step.

        Args:
            texts: List of text strings.

        Returns:
            Dense array of shape (n_texts, n_features).
        """
        self.fit(texts)
        return self.transform(texts)
