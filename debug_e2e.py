#!/usr/bin/env python
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from bdi.utilities import load_pan_dataset
from bdi import BDIVerifier

data_dir = "tests/data/du_essays/train"
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
        max_features=1000,
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
    nb_bootstrap_iter=10,
    random_state=1066,
    rnd_prop=0.35,
)
verifier.fit(train_X, train_y)
scores = verifier.predict_proba(test_X, test_y, nb_imposters=30)

GOLD_SCORES = np.load("tests/gold_e2e_results.npy")
print("scores[:10]:", scores[:10])
print("GOLD_SCORES[:10]:", GOLD_SCORES[:10])
print("Max diff:", np.max(np.abs(scores - GOLD_SCORES)))
print("Indices with diff > 0.01:", np.where(np.abs(scores - GOLD_SCORES) > 0.01)[0])
