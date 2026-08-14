"""End-to-end experiments on the official Scania train/test split."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from scania_aps.costs import bayes_threshold, optimize_threshold
from scania_aps.data import read_raw_csv
from scania_aps.metrics import Evaluation, evaluate_probabilities
from scania_aps.models.boosted import BoostedConfig, build_boosted_classifier
from scania_aps.models.logistic import build_logistic_pipeline
from scania_aps.optimization import tune_logistic
from scania_aps.split import development_split


def _write_evaluation(path: Path, evaluation: Evaluation, extra: dict[str, Any]) -> None:
    payload: dict[str, Any] = {"evaluation": evaluation.to_dict(), **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_logistic_experiment(
    train_csv: Path,
    test_csv: Path,
    artifacts_dir: Path,
    *,
    n_trials: int = 24,
) -> Evaluation:
    """Tune regularized logistic regression and evaluate once on the official test set."""

    train = read_raw_csv(train_csv)
    test = read_raw_csv(test_csv)
    split = development_split(train.X, train.y)

    best_config, candidates = tune_logistic(
        split.X_fit,
        split.y_fit,
        split.X_tune,
        split.y_tune,
        n_trials=n_trials,
    )

    # Refit after model selection, but keep the threshold subset untouched.
    X_refit = pd.concat([split.X_fit, split.X_tune], axis=0)
    y_refit = pd.concat([split.y_fit, split.y_tune], axis=0)
    model = build_logistic_pipeline(best_config)
    model.fit(X_refit, y_refit)

    threshold_probabilities = model.predict_proba(split.X_threshold)[:, 1]
    threshold_result = optimize_threshold(split.y_threshold.to_numpy(), threshold_probabilities)

    test_probabilities = model.predict_proba(test.X)[:, 1]
    evaluation = evaluate_probabilities(
        test.y.to_numpy(),
        test_probabilities,
        threshold_result.threshold,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "logistic_model.joblib")
    _write_evaluation(
        artifacts_dir / "logistic_results.json",
        evaluation,
        {
            "model": "regularized_logistic_regression",
            "best_config": asdict(best_config),
            "empirical_threshold": threshold_result.threshold,
            "bayes_threshold_if_calibrated": bayes_threshold(),
            "n_candidates": len(candidates),
        },
    )
    return evaluation


def run_boosted_experiment(
    train_csv: Path,
    test_csv: Path,
    artifacts_dir: Path,
    *,
    config: BoostedConfig | None = None,
) -> Evaluation:
    """Fit a nonlinear boosted-tree reference model and optimize its threshold."""

    train = read_raw_csv(train_csv)
    test = read_raw_csv(test_csv)
    split = development_split(train.X, train.y)
    selected_config = config or BoostedConfig(scale_pos_weight=20.0)

    model = build_boosted_classifier(selected_config)
    model.fit(split.X_fit, split.y_fit)

    # Use the tuning subset for a simple model sanity check and refit before thresholding.
    tune_probs = model.predict_proba(split.X_tune)[:, 1]
    tune_threshold = optimize_threshold(split.y_tune.to_numpy(), tune_probs)

    X_refit = pd.concat([split.X_fit, split.X_tune], axis=0)
    y_refit = pd.concat([split.y_fit, split.y_tune], axis=0)
    model = build_boosted_classifier(selected_config)
    model.fit(X_refit, y_refit)

    threshold_probs = model.predict_proba(split.X_threshold)[:, 1]
    threshold_result = optimize_threshold(split.y_threshold.to_numpy(), threshold_probs)
    test_probs = model.predict_proba(test.X)[:, 1]
    evaluation = evaluate_probabilities(test.y.to_numpy(), test_probs, threshold_result.threshold)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "boosted_model.joblib")
    _write_evaluation(
        artifacts_dir / "boosted_results.json",
        evaluation,
        {
            "model": "xgboost",
            "config": asdict(selected_config),
            "pre_refit_tune_cost": tune_threshold.cost.total_cost,
            "empirical_threshold": threshold_result.threshold,
        },
    )
    return evaluation
