import numpy as np
import pytest

from scania_aps.costs import bayes_threshold, maintenance_cost, optimize_threshold


def test_maintenance_cost_matches_industrial_definition() -> None:
    y_true = np.array([1, 1, 0, 0, 0], dtype=np.int8)
    y_pred = np.array([1, 0, 1, 0, 0], dtype=np.int8)

    result = maintenance_cost(y_true, y_pred)

    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.total_cost == 510.0


def test_bayes_threshold_uses_cost_ratio() -> None:
    assert bayes_threshold() == pytest.approx(10.0 / 510.0)


def test_threshold_optimizer_prefers_catching_expensive_failure() -> None:
    y_true = np.array([1, 0, 0], dtype=np.int8)
    probabilities = np.array([0.20, 0.15, 0.10], dtype=np.float64)

    result = optimize_threshold(y_true, probabilities)
    pred = (probabilities >= result.threshold).astype(np.int8)

    assert maintenance_cost(y_true, pred).total_cost == 0.0
