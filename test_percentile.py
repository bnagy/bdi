#!/usr/bin/env python
import numpy as np
import scipy.stats as sp

# Test percentileofscore with the exact data pattern
a = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
score = 0

left = np.count_nonzero(a < score)
right = np.count_nonzero(a <= score)
n = len(a)
plus1 = left < right
perct = (left + right + plus1) * (50.0 / n)

print(f"left: {left}, right: {right}, n: {n}")
print(f"plus1: {plus1}")
print(f"manual perct: {perct}")
print(f"result: {(100 - perct) / 100}")
