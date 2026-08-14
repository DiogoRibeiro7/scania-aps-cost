import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from scania_aps.costs import optimize_score_threshold
from scania_aps.metrics import evaluate_scores
from scania_aps.scoring import positive_class_scores


def test_probability_and_margin_models_are_supported() -> None:
    X, y = make_classification(n_samples=120, n_features=6, random_state=42)

    logistic = LogisticRegression().fit(X, y)
    logistic_scores = positive_class_scores(logistic, X)
    assert logistic_scores.kind == "probability"
    assert np.all((logistic_scores.values >= 0.0) & (logistic_scores.values <= 1.0))

    svm = LinearSVC().fit(X, y)
    svm_scores = positive_class_scores(svm, X)
    assert svm_scores.kind == "decision"
    threshold = optimize_score_threshold(y, svm_scores.values).threshold
    result = evaluate_scores(y, svm_scores.values, threshold, score_kind="decision")
    assert result.total_cost >= 0.0
    assert result.brier is None
