#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for utilities in bdi.utilities.

These tests verify the correctness of utility functions.
"""

import numpy as np
import tempfile
import os

from bdi.utilities import (
    binarize,
    make_up_lies,
    load_ground_truth,
    train_dev_split,
    load_pan_dataset,
)


class TestBinarize:
    """Tests for binarize function."""

    def test_binarize_below_threshold(self):
        """Scores below 0.5 should become 'N'."""
        scores = [0.1, 0.3, 0.49]
        result = binarize(scores)
        assert result == ["N", "N", "N"]

    def test_binarize_above_threshold(self):
        """Scores above 0.5 should become 'Y'."""
        scores = [0.51, 0.7, 0.99]
        result = binarize(scores)
        assert result == ["Y", "Y", "Y"]

    def test_binarize_at_threshold(self):
        """Scores at 0.5 should become 'X'."""
        scores = [0.5, 0.5, 0.5]
        result = binarize(scores)
        assert result == ["X", "X", "X"]

    def test_binarize_mixed(self):
        """Test mixed scores."""
        scores = [0.1, 0.5, 0.9]
        result = binarize(scores)
        assert result == ["N", "X", "Y"]


class TestMakeUpLies:
    """Tests for make_up_lies function."""

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = [0, 1, 2]

        ret_X, ret_y, ground_truth = make_up_lies(X, y)

        assert ret_X.shape[0] == 6  # Doubled
        assert len(ret_y) == 6
        assert len(ground_truth) == 6

    def test_ground_truth_values(self):
        """Test that ground truth has correct values."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = [0, 1]

        ret_X, ret_y, ground_truth = make_up_lies(X, y)

        # First half should be 1.0 (correct), second half 0.0 (lies)
        assert all(ground_truth[:2] == 1.0)
        assert all(ground_truth[2:] == 0.0)

    def test_lies_are_different(self):
        """Test that lies have different labels."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = [0, 1, 2]

        ret_X, ret_y, ground_truth = make_up_lies(X, y)

        # Lies should have different labels than original
        for i in range(3):
            assert ret_y[i] != ret_y[i + 3]


class TestLoadGroundTruth:
    """Tests for load_ground_truth function."""

    def test_load_ground_truth_basic(self):
        """Test basic ground truth loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("prob1 Y\n")
            f.write("prob2 N\n")
            f.write("prob3 Y\n")
            filepath = f.name

        try:
            labels = ["prob1", "prob2", "prob3"]
            result = load_ground_truth(filepath, labels)
            assert result == [1.0, 0.0, 1.0]
        finally:
            os.unlink(filepath)

    def test_load_ground_truth_all_yes(self):
        """Test with all Y outcomes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("prob1 Y\n")
            f.write("prob2 Y\n")
            filepath = f.name

        try:
            labels = ["prob1", "prob2"]
            result = load_ground_truth(filepath, labels)
            assert result == [1.0, 1.0]
        finally:
            os.unlink(filepath)


class TestTrainDevSplit:
    """Tests for train_dev_split function."""

    def test_train_dev_split_shapes(self):
        """Test that split produces correct shapes."""
        X = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]
        )
        y = [0, 1, 2, 0, 1, 2]

        X_dev, y_dev, X_test, y_test, gt = train_dev_split(X, y)

        assert X_dev.shape[0] == 3
        assert X_test.shape[0] == 3
        assert len(y_dev) == 3
        assert len(y_test) == 3
        assert len(gt) == 3

    def test_train_dev_split_ground_truth(self):
        """Test that ground truth has correct values."""
        X = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.3, 0.7], [0.8, 0.2]]
        )
        y = [0, 1, 2, 0, 1, 2]

        X_dev, y_dev, X_test, y_test, gt = train_dev_split(X, y)

        # Ground truth should be 0.0 or 1.0
        assert all(g in [0.0, 1.0] for g in gt)


class TestLoadPanDataset:
    """Tests for load_pan_dataset function."""

    def test_load_pan_dataset_basic(self):
        """Test basic PAN dataset loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create author directory
            author_dir = os.path.join(tmpdir, "author1")
            os.makedirs(author_dir)

            # Create training file
            with open(os.path.join(author_dir, "doc1.txt"), "w") as f:
                f.write("Some text content")

            # Create unknown file
            with open(os.path.join(author_dir, "unknown.txt"), "w") as f:
                f.write("Unknown text")

            train_data, test_data = load_pan_dataset(tmpdir)

            assert len(train_data) == 1
            assert len(test_data) == 1
            assert train_data[0][0] == "author1"
            assert test_data[0][0] == "author1"

    def test_load_pan_dataset_multiple_authors(self):
        """Test with multiple authors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                author_dir = os.path.join(tmpdir, f"author{i}")
                os.makedirs(author_dir)
                with open(os.path.join(author_dir, "doc1.txt"), "w") as f:
                    f.write(f"Text from author {i}")

            train_data, test_data = load_pan_dataset(tmpdir)

            assert len(train_data) == 2
            assert len(test_data) == 0


class TestGetVocabSize:
    """Tests for get_vocab_size function."""

    def test_get_vocab_size_basic(self):
        """Test basic vocabulary size calculation."""
        from bdi.utilities import get_vocab_size

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create train subdirectory
            train_dir = os.path.join(tmpdir, "train")
            os.makedirs(train_dir)

            # Create author directory
            author_dir = os.path.join(train_dir, "author1")
            os.makedirs(author_dir)

            # Create training file
            with open(os.path.join(author_dir, "doc1.txt"), "w") as f:
                f.write("hello world hello")

            vocab_size = get_vocab_size(tmpdir, ngram_type="word", ngram_size=1)
            assert vocab_size >= 1
