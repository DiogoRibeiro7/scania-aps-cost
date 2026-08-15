from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scania_aps.models.logistic import LogisticConfig
from scania_aps.optimization import sample_logistic_configs, tune_logistic


@pytest.fixture
def development_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 240
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series(((X["a"] + rng.normal(scale=0.4, size=n)) > 0.8).astype(int))
    X.iloc[0, 0] = np.nan
    return X[:160], y[:160], X[160:], y[160:]


def test_sampling_returns_the_requested_number_of_configs() -> None:
    assert len(sample_logistic_configs(5)) == 5


def test_sampling_is_reproducible_for_a_fixed_seed() -> None:
    assert sample_logistic_configs(8, random_state=7) == sample_logistic_configs(8, random_state=7)


def test_different_seeds_explore_different_configurations() -> None:
    a = sample_logistic_configs(12, random_state=1)
    b = sample_logistic_configs(12, random_state=2)
    assert a != b


def test_sampled_configs_are_valid() -> None:
    for config in sample_logistic_configs(20, random_state=3):
        assert isinstance(config, LogisticConfig)
        assert config.C > 0
        if config.penalty == "elasticnet":
            assert config.l1_ratio is not None
            assert 0.0 <= config.l1_ratio <= 1.0


def test_tuning_returns_one_result_per_trial(
    development_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    X_fit, y_fit, X_tune, y_tune = development_data

    best, results = tune_logistic(X_fit, y_fit, X_tune, y_tune, n_trials=3)

    assert len(results) == 3
    assert best in [result.config for result in results]
    for result in results:
        assert result.tune_cost >= 0
        assert 0.0 <= result.threshold <= 1.0


def test_tuning_selects_the_lowest_cost_candidate(
    development_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Selection is on cost, with PR-AUC only breaking ties. Not on accuracy."""

    X_fit, y_fit, X_tune, y_tune = development_data

    best, results = tune_logistic(X_fit, y_fit, X_tune, y_tune, n_trials=4)

    cheapest = min(result.tune_cost for result in results)
    chosen = next(result for result in results if result.config == best)
    assert chosen.tune_cost == cheapest
