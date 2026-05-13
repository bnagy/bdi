# BDIVerifier

**Bootstrap Distance Imposters verification for authorship attribution.**

This package provides a clean implementation of the BDI (Bootstrap Distance Imposters) verification algorithm for authorship attribution tasks. BDI is an update to the General Imposters method that incorporates improvements from Potha and Stamatatos, along with a novel bootstrapping method that provides better interpretability.

## Installation

```bash
pip install bdi
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from bdi import BDIVerifier, Vectorizer

# Create training data
X_train = np.random.rand(100, 50)  # 100 documents, 50 features
y_train = np.array([0] * 50 + [1] * 50)  # Two authors

# Initialize and fit the verifier
verifier = BDIVerifier(metric="manhattan", method="ranked")
verifier.fit(X_train, y_train)

# Predict on test data
X_test = np.random.rand(10, 50)
y_test = np.array([0] * 5 + [1] * 5)
probas = verifier.predict_proba(X_test, y_test)
```

## API Reference

### BDIVerifier

The main verification class implementing the Bootstrap Distance Imposters algorithm.

```python
BDIVerifier(
    metric="manhattan",       # Distance metric: manhattan, euclidean, minmax, cng, cosine, nini
    method="ranked",          # Imposters selection: ranked, random, closest
    nb_bootstrap_iter=100,    # Number of bootstrap iterations
    random_state=None,        # Random seed for reproducibility
    rnd_prop=0.35,            # Proportion of imposters for random method
    balance=False,            # Balance class sizes
)
```

**Methods:**

- `fit(X, y)` - Fit the verifier on training data
- `predict_proba(X, y, nb_imposters=-1)` - Predict verification probabilities

### Vectorizer

Convert text documents to numerical feature vectors.

```python
Vectorizer(
    mfi=100,                  # Maximum features to extract
    ngram_type="word",        # Type: word, char, char_wb
    ngram_size=1,             # N-gram size
    vector_space="tf",        # Model: tf, tf_scaled, tf_std, tf_idf, bin
    lowercase=True,           # Lowercase input
    min_df=0.0,               # Minimum document frequency
    max_df=1.0,               # Maximum document frequency
)
```

### ScoreShifter

Optimize verification scores for PAN metrics (AUC × c@1).

```python
from bdi import ScoreShifter

shifter = ScoreShifter(grid_size=100)
shifter.fit(predicted_scores, ground_truth)
corrected = shifter.transform(new_scores)
```

### Evaluation Metrics

```python
from bdi import accuracy, auc, c_at_1, pan_metrics

acc = accuracy(predictions, ground_truth)
auc_score = auc(predictions, ground_truth)
c1 = c_at_1(predictions, ground_truth)
acc, auc_score, c1 = pan_metrics(predictions, ground_truth)
```

## Distance Metrics

| Metric | Description |
|--------|-------------|
| `manhattan` | L1 distance |
| `euclidean` | L2 distance |
| `minmax` | Ružička (1 - intersection over union) |
| `cng` | Common n-grams distance |
| `cosine` | Cosine distance |
| `nini` | Nini distance for binary vectors |

## Methods

| Method | Description |
|--------|-------------|
| `ranked` | Select imposters based on distance ranking |
| `random` | Randomly sample imposters |
| `closest` | Use closest imposters to test document |

## Algorithm

The Bootstrap Distance Imposters algorithm works by:

1. **Bootstrapping**: Random feature subsets are sampled at each iteration
2. **Distance Comparison**: Test document distances to candidate vs imposter documents are compared
3. **Score Calculation**: Verification scores are calculated based on distance differences

Unlike the original General Imposters method which outputs binarized "votes", BDI outputs a bootstrapped distribution of distance differences. This provides better interpretability - positive matches show distributions centred around positive values, while "none of the above" results centre around zero.

## References

- Koppel, M. and Winter, Y. (2014). Determining if Two Documents are by the Same Author. JASIST, 65(1): 178-187.
- Potha, C. and Stamatatos, E. (2017). Improved Imposters Approach for Authorship Verification.
- Kestemont, M. et al. (2015). Computational Authorship Verification Method Attributes New Work to Major 2nd Century African Author. JASIST.

## License

MIT License
