#!/usr/bin/env python
import sys

sys.path.insert(0, "/Users/ben/code/bdi")
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from bdi.utilities import load_pan_dataset
from bdi import BDIVerifier

data_dir = "/Users/ben/code/ruzicka/data/2014/du_essays/train"
train_data, test_data = load_pan_dataset(data_dir)
train_labels, train_documents = zip(*train_data)
test_labels, test_documents = zip(*test_data)

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

label_encoder = LabelEncoder()
label_encoder.fit(train_labels + test_labels)
train_y = np.array(label_encoder.transform(train_labels))
test_y = np.array(label_encoder.transform(test_labels))

verifier = BDIVerifier(
    metric="minmax",
    method="ranked",
    nb_bootstrap_iter=100,
    random_state=1066,
    rnd_prop=0.35,
)
verifier.fit(train_X, train_y)
scores = verifier.predict_proba(test_X, test_y, nb_imposters=30)

print(f"Scores: {scores}")
print(f"Length: {len(scores)}")
print("\nFormatted for gold_e2e_results.py:")
print("GOLD_SCORES = np.array([")
for i in range(0, len(scores), 10):
    row = scores[i : i + 10]
    print("    " + ", ".join(f"{s:.2f}" if s == int(s) else f"{s}" for s in row) + ",")
print("])")
