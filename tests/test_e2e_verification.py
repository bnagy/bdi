#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end verification test for BDIVerifier.

This test uses the PAN 2014 du_essays corpus to verify that the BDIVerifier
produces consistent results with a seeded RNG.
"""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from bdi import BDIVerifier  # type: ignore

# Load gold results from numpy file
GOLD_SCORES = np.load("tests/gold_e2e_results.npy")


def load_and_prepare_data(data_dir: str):
    """Load PAN dataset and prepare for verification."""
    from bdi.utilities import load_pan_dataset

    train_data, test_data = load_pan_dataset(data_dir)
    train_labels, train_documents = zip(*train_data)
    test_labels, test_documents = zip(*test_data)

    # Vectorize using char 2-4-grams
    vectorizer = make_pipeline(
        TfidfVectorizer(
            sublinear_tf=True,
            use_idf=False,
            norm="l2",
            analyzer="char",
            ngram_range=(2, 4),
            max_features=10000,
        ),
        FunctionTransformer(lambda x: x.todense(), accept_sparse=True),
    )

    train_X = vectorizer.fit_transform(train_documents)
    test_X = vectorizer.transform(test_documents)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels + test_labels)
    train_y = np.array(label_encoder.transform(train_labels))
    test_y = np.array(label_encoder.transform(test_labels))

    return train_X, train_y, test_X, test_y


class TestE2EVerification:
    """End-to-end verification tests for BDIVerifier."""

    @pytest.fixture
    def prepared_data(self):
        """Load and prepare the PAN 2014 du_essays dataset."""
        data_dir = "/Users/ben/code/ruzicka/data/2014/du_essays/train"
        return load_and_prepare_data(data_dir)

    def test_new_verifier_matches_gold_results(self, prepared_data):
        """Test that BDIVerifier produces gold standard results.

        These gold results were generated using the same parameters
        (minmax metric, ranked method, 100 bootstrap iterations,
        random_state=1066, rnd_prop=0.35, nb_imposters=30).
        """
        train_X, train_y, test_X, test_y = prepared_data

        # Run BDIVerifier with same parameters as gold results
        verifier = BDIVerifier(
            metric="minmax",
            method="ranked",
            nb_bootstrap_iter=100,
            random_state=1066,
            rnd_prop=0.35,
        )
        verifier.fit(train_X, train_y)
        scores = verifier.predict_proba(test_X, test_y, nb_imposters=30)

        # Verify against gold standard
        assert np.allclose(
            scores, GOLD_SCORES, rtol=1e-10, atol=1e-10
        ), f"Scores don't match gold standard! Max diff: {np.max(np.abs(scores - GOLD_SCORES))}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
