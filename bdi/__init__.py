#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BDIVerifier: Authorship Verification in the General Imposters Framework.

This package provides a clean implementation of the Bootstrap Distance Imposters
(BDI) verification algorithm for authorship attribution, plus the Eder
Bootstrap Consensus Tree (BCT) visualization.

Example:
    >>> from bdi import BDIVerifier, Vectorizer
    >>> verifier = BDIVerifier(metric="manhattan", method="ranked")
    >>> verifier.fit(X_train, y_train)
    >>> scores = verifier.predict_proba(X_test, y_test)

    >>> from bdi import eder_bct, plot_bct
    >>> graph_trim = eder_bct(X, y, n=1000)
    >>> fig, ax = plot_bct(graph_trim, work_names, y)
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

# Import submodules - noqa: E402
from .verifier import BDIVerifier  # noqa: E402
from .vectorizer import Vectorizer  # noqa: E402
from .score_shifting import ScoreShifter  # noqa: E402
from .evaluation import accuracy, auc, c_at_1, pan_metrics  # noqa: E402
from .metrics import (  # noqa: E402
    manhattan,
    euclidean,
    minmax,
    common_ngrams,
    cosine,
    nini,
)
from .bct import eder_bct, plot_bct  # noqa: E402

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
    "eder_bct",
    "plot_bct",
]
