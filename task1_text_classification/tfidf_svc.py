"""
Task 1: TF-IDF + LinearSVC Baseline
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


def build_tfidf_svc(max_features=50000, ngram_range=(1, 2), C=1.0):
    """Build a TF-IDF + LinearSVC pipeline."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )),
        ("svc", LinearSVC(C=C, max_iter=10000, random_state=42)),
    ])
    return pipeline


def train_and_predict(pipeline, train_texts, train_labels, test_texts):
    """Train the pipeline and return predictions."""
    pipeline.fit(train_texts, train_labels)
    preds = pipeline.predict(test_texts)
    return preds
