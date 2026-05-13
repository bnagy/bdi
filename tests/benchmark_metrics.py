#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark suite for distance metrics in bdi.metrics.

This module provides comprehensive benchmarks for all distance metrics,
testing various vector sizes and similarity levels. Results are saved
to a JSON file for baseline comparison.
"""

import json
import time
import numpy as np
from typing import Callable

from bdi.metrics import manhattan, euclidean, minmax, common_ngrams, cosine, nini

# Configuration for benchmarks
VECTOR_SIZES = [100, 500, 1000, 5000, 10000]
ITERATIONS = 100
WARMUP = 10

METRICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "minmax": minmax,
    "common_ngrams": common_ngrams,
    "cosine": cosine,
    "nini": nini,
}


def generate_test_vectors(
    size: int, similarity: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two vectors with specified similarity level.

    Args:
        size: Number of features in vectors.
        similarity: Target similarity (0.0 to 1.0).

    Returns:
        Tuple of two numpy arrays.
    """
    np.random.seed(42)
    x = np.random.rand(size)

    if similarity == 1.0:
        y = x.copy()
    elif similarity == 0.0:
        y = np.random.rand(size)
    else:
        # Mix of shared and random elements
        shared_count = int(size * similarity)
        y = x.copy()
        # Perturb some elements
        perturb_indices = np.random.choice(size, size - shared_count, replace=False)
        y[perturb_indices] = np.random.rand(size - shared_count)

    return x, y


def benchmark_metric(
    metric_func: Callable,
    x: np.ndarray,
    y: np.ndarray,
    iterations: int = ITERATIONS,
    warmup: int = WARMUP,
) -> dict:
    """
    Benchmark a single metric function.

    Args:
        metric_func: The distance function to benchmark.
        x: First vector.
        y: Second vector.
        iterations: Number of benchmark iterations.
        warmup: Number of warmup iterations.

    Returns:
        Dictionary with benchmark results.
    """
    # Warmup
    for _ in range(warmup):
        metric_func(x, y)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = metric_func(x, y)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "median_ms": np.median(times) * 1000,
    }


def run_benchmarks() -> dict:
    """
    Run all benchmarks and return results.

    Returns:
        Dictionary with all benchmark results.
    """
    results = {
        "config": {
            "vector_sizes": VECTOR_SIZES,
            "iterations": ITERATIONS,
            "warmup": WARMUP,
        },
        "metrics": {},
    }

    for metric_name, metric_func in METRICS.items():
        print(f"Benchmarking {metric_name}...")
        results["metrics"][metric_name] = {"sizes": {}}

        for size in VECTOR_SIZES:
            results["metrics"][metric_name]["sizes"][str(size)] = {}

            for similarity in [0.0, 0.25, 0.5, 0.75, 1.0]:
                x, y = generate_test_vectors(size, similarity)
                bench_result = benchmark_metric(metric_func, x, y)
                results["metrics"][metric_name]["sizes"][str(size)][
                    f"sim_{int(similarity*100)}"
                ] = bench_result

    return results


def main():
    """Run benchmarks and save results."""
    print("Running metric benchmarks...")
    results = run_benchmarks()

    # Save results
    output_path = "tests/benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark results saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (mean time in ms for 1000-dim vectors)")
    print("=" * 80)
    print(
        f"{'Metric':<15} {'Sim 0%':<12} {'Sim 25%':<12} {'Sim 50%':<12} {'Sim 75%':<12} {'Sim 100%':<12}"
    )
    print("-" * 80)

    for metric_name, metric_data in results["metrics"].items():
        size_1000 = metric_data["sizes"]["1000"]
        row = [metric_name]
        for sim in ["sim_0", "sim_25", "sim_50", "sim_75", "sim_100"]:
            mean_ms = size_1000[sim]["mean_ms"]
            row.append(f"{mean_ms:.4f}")
        print(
            f"{row[0]:<15} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12} {row[5]:<12}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
