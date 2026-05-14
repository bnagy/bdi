#!/usr/bin/env python
"""Verify scipy version and percentileofscore behavior."""

import scipy
import numpy as np
import scipy.stats as sp

print(f"scipy version: {scipy.__version__}")
print(f"numpy version: {np.__version__}")

# Test percentileofscore with basic data
x = np.array([-0.5, -0.3, 0.0, 0.1, 0.3])
result = sp.stats.percentileofscore(x, 0)
print(f"percentileofscore test: {result} (expected: 60.0)")
