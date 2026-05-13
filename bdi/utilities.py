#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for authorship verification datasets.

This module provides functions for loading PAN-style datasets,
splitting data, and other common operations.
"""

import codecs
import glob
import os
from typing import Collection, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .vectorizer import Vectorizer


def load_pan_dataset(
    directory: str, ext: str = "txt", encoding: str = "utf8"
) -> Tuple[list, list]:
    """
    Load a PAN-style dataset from a directory.

    The directory should contain subdirectories for each problem/author.
    Files named 'unknown' are treated as test data; all others as training data.

    Args:
        directory: Path to the dataset directory.
        ext: File extension to load. Defaults to "txt".
        encoding: File encoding. Defaults to "utf8".

    Returns:
        Tuple of (train_data, test_data) where each is a list of
        (author, document) tuples.

    References:
        http://www.uni-weimar.de/medien/webis/events/pan-15/pan15-web/author-identification.html
    """
    train_data, test_data = [], []

    for author in sorted(os.listdir(directory)):
        path = os.sep.join((directory, author))
        if os.path.isdir(path):
            for filepath in sorted(glob.glob(path + "/*." + ext)):
                text = codecs.open(filepath, mode="r", encoding=encoding).read()
                name = os.path.splitext(os.path.basename(filepath))[0]
                if name == "unknown":
                    test_data.append((author, text))
                else:
                    train_data.append((author, text))
    return train_data, test_data


def get_vocab_size(
    corpus_dir: str,
    ngram_type: str,
    ngram_size: int,
    min_df: float = 0.0,
    phase: str = "train",
) -> int:
    """
    Get the maximum vocabulary size for given vectorizer settings.

    Args:
        corpus_dir: Path to the corpus directory.
        ngram_type: Type of n-grams ('word', 'char', 'char_wb').
        ngram_size: Size of n-grams.
        min_df: Minimum document frequency. Defaults to 0.0.
        phase: 'train' or 'test' phase. Defaults to "train".

    Returns:
        Maximum number of vocabulary items available.
    """
    train_data, _ = load_pan_dataset(corpus_dir + "/" + phase)
    train_labels, train_documents = zip(*train_data)

    vectorizer = Vectorizer(
        mfi=2147483647, ngram_type=ngram_type, ngram_size=ngram_size, min_df=min_df
    )
    vectorizer.fit(train_documents)

    return len(vectorizer.feature_names)


def load_ground_truth(filepath: str, labels: Collection[str]) -> list:
    """
    Load ground truth labels from a PAN truth file.

    Args:
        filepath: Path to the truth.txt file.
        labels: Author labels to order the results.

    Returns:
        List of ground truth scores (0.0 or 1.0) in the order specified by labels.

    References:
        http://www.uni-weimar.de/medien/webis/events/pan-14/pan14-web/author-identification.html
    """
    ground_truth = {}

    for line in open(filepath).readlines():
        problem_id, outcome = line.strip().split()
        if outcome == "Y":
            outcome = 1.0
        elif outcome == "N":
            outcome = 0.0
        ground_truth[problem_id] = outcome

    return [ground_truth[label] for label in labels]


def train_dev_split(
    train_X: np.ndarray, train_y: list, random_state: int = 1027
) -> Tuple[np.ndarray, list, np.ndarray, list, list]:
    """
    Create a 50-50 train/dev split with balanced verification problems.

    For each document, creates both a same-author and different-author
    verification problem.

    Args:
        train_X: Feature matrix of shape (n_samples, n_features).
        train_y: Author labels for each document.
        random_state: Random seed. Defaults to 1027.

    Returns:
        Tuple of (X_dev, y_dev, X_test, y_test, test_gt_scores) where
        test_gt_scores indicates whether each test problem is same-author (1.0)
        or different-author (0.0).
    """
    X_dev, X_test, y_dev, y_test = train_test_split(
        train_X, train_y, test_size=0.5, random_state=random_state, stratify=train_y
    )
    test_gt_scores: list = []

    np.random.seed(random_state)
    author_options = set(train_y)
    rnd_idxs = np.random.choice(len(y_test), int(len(y_test) / 2))

    for idx, y in enumerate(y_test):
        if idx in rnd_idxs:
            real_author = y_test[idx]
            other_authors = [a for a in author_options if a != real_author]
            fake_author = np.random.choice(other_authors, 1)[0]
            y_test[idx] = fake_author
            test_gt_scores.append(0.0)
        else:
            test_gt_scores.append(1.0)

    return np.asarray(X_dev), y_dev, np.asarray(X_test), y_test, test_gt_scores


def binarize(scores: Collection[float]) -> list:
    """
    Convert scores to PAN-style binary labels.

    Args:
        scores: Scores in [0, 1].

    Returns:
        List of 'Y' (score > 0.5), 'N' (score < 0.5), or 'X' (score == 0.5).
    """
    scs = []
    for sc in scores:
        if sc == 0.5:
            scs.append("X")
        elif sc < 0.5:
            scs.append("N")
        else:
            scs.append("Y")
    return scs


def make_up_lies(X: np.ndarray, y: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create negative examples by assigning wrong labels.

    This is useful for fitting score shifters that need true negatives
    to calculate AUC.

    Args:
        X: Feature matrix.
        y: Author labels.

    Returns:
        Tuple of (X_combined, y_combined, ground_truth) where ground_truth
        is 1.0 for original examples and 0.0 for the "lies".
    """
    lies_labels = []
    n_labels = max(y) + 1
    for lab in y:
        while True:
            r = np.random.randint(n_labels)
            if r != lab:
                lies_labels.append(r)
                break
    ret_X = np.concatenate([X, X.copy()])
    ret_y = np.concatenate([y, lies_labels])
    ground_truth = np.concatenate([[1.0] * len(X), [0.0] * len(X)])
    return (ret_X, ret_y, ground_truth)
