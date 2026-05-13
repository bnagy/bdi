"""Pytest configuration for BDI tests."""

import os

# Disable numba JIT for proper coverage reporting
os.environ["NUMBA_DISABLE_JIT"] = "1"
