"""Hyperparameter search using the actual maintenance objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scania_aps._types import Estimator
from scania_aps.costs import optimize_threshold
from scania_aps.metrics import evaluate_probabilities
from scania_aps.models.logistic import LogisticConfig, build_logistic_pipeline


@dataclass(frozen=True)
class LogisticCandidateResult:
    """One candidate's hyperparameters and development performance."""

    config: LogisticConfig
    threshold: float
    tune_cost: float
    pr_auc: float


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_logistic_configs(n_trials: int, *, random_state: int = 42) -> list[LogisticConfig]:
    """Generate a reproducible search over regularization and class weighting."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    rng = np.random.default_rng(random_state)
    penalties = np.array(["l1", "l2", "elasticnet"], dtype=object)
    positive_weights: list[float | None] = [None, 5.0, 10.0, 25.0, 50.0]

    configs: list[LogisticConfig] = [
        # A near-unregularized reference point.
        LogisticConfig(penalty="l2", C=1e6, positive_class_weight=None),
    ]

    for _ in range(max(n_trials - 1, 0)):
        penalty = str(rng.choice(penalties))
        l1_ratio = float(rng.uniform(0.05, 0.95)) if penalty == "elasticnet" else None
        configs.append(
            LogisticConfig(
                penalty=penalty,  # type: ignore[arg-type]
                C=_log_uniform(rng, 1e-4, 1e2),
                l1_ratio=l1_ratio,
                positive_class_weight=positive_weights[int(rng.integers(len(positive_weights)))],
            )
        )

    return configs


def tune_logistic(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    *,
    n_trials: int = 24,
    random_state: int = 42,
) -> tuple[LogisticConfig, list[LogisticCandidateResult]]:
    """Select logistic hyperparameters by minimum empirical maintenance cost."""

    results: list[LogisticCandidateResult] = []

    for config in sample_logistic_configs(n_trials, random_state=random_state):
        model: Estimator = build_logistic_pipeline(config)
        model.fit(X_fit, y_fit)
        probabilities = model.predict_proba(X_tune)[:, 1]
        threshold_result = optimize_threshold(y_tune.to_numpy(), probabilities)
        evaluation = evaluate_probabilities(
            y_tune.to_numpy(),
            probabilities,
            threshold_result.threshold,
        )
        results.append(
            LogisticCandidateResult(
                config=config,
                threshold=threshold_result.threshold,
                tune_cost=evaluation.total_cost,
                pr_auc=evaluation.pr_auc,
            )
        )

    best = min(results, key=lambda result: (result.tune_cost, -result.pr_auc))
    return best.config, results
