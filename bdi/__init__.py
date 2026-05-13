#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BDIVerifier: Authorship Verification in the General Imposters Framework.

This package provides a clean implementation of the Bootstrap Distance Imposters
(BDI) verification algorithm for authorship attribution.

Example:
    >>> from bdi import BDIVerifier, Vectorizer
    >>> verifier = BDIVerifier(metric="manhattan", method="ranked")
    >>> verifier.fit(X_train, y_train)
    >>> scores = verifier.predict_proba(X_test, y_test)
"""

import logging

logger = logging.getLogger("bdi")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(name)s:%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False

__version__ = "1.0.0"

from .verifier import BDIVerifier
from .vectorizer import Vectorizer
from .score_shifting import ScoreShifter
from .evaluation import accuracy, auc, c_at_1, pan_metrics
from .metrics import manhattan, euclidean, minmax, common_ngrams, cosine, nini

__all__ = [
    "BDIVerifier",
    "Vectorizer",
    "ScoreShifter",
    "accuracy",
    "auc",
    "c_at_1",
    "pan_metrics",
    "manhattan",
    "euclidean",
    "minmax",
    "common_ngrams",
    "cosine",
    "nini",
]
