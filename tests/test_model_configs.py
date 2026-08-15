"""Guard rails on the model configuration dataclasses.

Every ``validate`` branch here rejects a hyperparameter that would otherwise
reach an estimator and either crash deep inside a third-party library or, worse,
train something quietly meaningless. None of these branches were covered before,
so a broken guard would have gone unnoticed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from scania_aps.models.boosted import BoostedConfig
from scania_aps.models.lightgbm import LightGBMConfig
from scania_aps.models.logistic import LogisticConfig
from scania_aps.models.svm import LinearSVMConfig
from scania_aps.models.trees import TreeEnsembleConfig

CONFIGS: dict[str, Any] = {
    "logistic": LogisticConfig(),
    "linear_svm": LinearSVMConfig(),
    "random_forest": TreeEnsembleConfig(kind="random_forest"),
    "xgboost": BoostedConfig(),
    "lightgbm": LightGBMConfig(),
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_default_configurations_are_valid(name: str) -> None:
    """The shipped defaults must pass their own validation."""

    CONFIGS[name].validate()


@pytest.mark.parametrize(
    ("config", "field", "value", "message"),
    [
        # Logistic
        (LogisticConfig(), "C", 0.0, "C must be positive"),
        (LogisticConfig(), "C", -1.0, "C must be positive"),
        (LogisticConfig(penalty="elasticnet", l1_ratio=0.5), "l1_ratio", 1.5, "l1_ratio"),
        (LogisticConfig(penalty="elasticnet", l1_ratio=0.5), "l1_ratio", -0.1, "l1_ratio"),
        (LogisticConfig(penalty="l2"), "l1_ratio", 0.5, "only valid with the elasticnet"),
        (LogisticConfig(), "positive_class_weight", 0.0, "positive_class_weight"),
        (LogisticConfig(), "max_iter", 0, "max_iter"),
        # Linear SVM
        (LinearSVMConfig(), "C", 0.0, "C must be positive"),
        (LinearSVMConfig(), "positive_class_weight", -5.0, "positive_class_weight"),
        (LinearSVMConfig(), "max_iter", -1, "max_iter"),
        # Tree ensembles
        (TreeEnsembleConfig(kind="random_forest"), "kind", "gradient_forest", "Unsupported"),
        (TreeEnsembleConfig(kind="random_forest"), "n_estimators", 0, "n_estimators"),
        (TreeEnsembleConfig(kind="extra_trees"), "max_depth", 0, "max_depth"),
        (TreeEnsembleConfig(kind="random_forest"), "min_samples_leaf", 0, "min_samples_leaf"),
        (TreeEnsembleConfig(kind="random_forest"), "max_features", 1.5, "max_features"),
        (TreeEnsembleConfig(kind="random_forest"), "max_features", 0.0, "max_features"),
        (TreeEnsembleConfig(kind="random_forest"), "positive_class_weight", 0.0, "positive"),
        # XGBoost
        (BoostedConfig(), "n_estimators", 0, "n_estimators"),
        (BoostedConfig(), "learning_rate", 0.0, "learning_rate"),
        (BoostedConfig(), "learning_rate", 1.5, "learning_rate"),
        (BoostedConfig(), "max_depth", 0, "max_depth"),
        (BoostedConfig(), "reg_alpha", -1.0, "Regularization"),
        (BoostedConfig(), "reg_lambda", -1.0, "Regularization"),
        (BoostedConfig(), "min_child_weight", -1.0, "Regularization"),
        (BoostedConfig(), "subsample", 1.5, "subsample"),
        (BoostedConfig(), "colsample_bytree", 0.0, "colsample_bytree"),
        (BoostedConfig(), "scale_pos_weight", 0.0, "scale_pos_weight"),
        # LightGBM
        (LightGBMConfig(), "n_estimators", 0, "positive"),
        (LightGBMConfig(), "num_leaves", 0, "positive"),
        (LightGBMConfig(), "learning_rate", 1.5, "learning_rate"),
        (LightGBMConfig(), "subsample", 0.0, "subsample"),
        (LightGBMConfig(), "colsample_bytree", 1.5, "colsample"),
        (LightGBMConfig(), "reg_alpha", -1.0, "non-negative"),
        (LightGBMConfig(), "positive_class_weight", 0.0, "positive_class_weight"),
    ],
    ids=lambda v: str(v)[:24],
)
def test_invalid_hyperparameters_are_rejected(
    config: Any, field: str, value: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(config, **{field: value}).validate()


def test_elasticnet_requires_an_l1_ratio() -> None:
    """The penalty and its mixing parameter have to agree, in both directions."""

    LogisticConfig(penalty="elasticnet", l1_ratio=0.5).validate()
    LogisticConfig(penalty="l1", l1_ratio=None).validate()

    with pytest.raises(ValueError, match="l1_ratio"):
        LogisticConfig(penalty="elasticnet", l1_ratio=None).validate()


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_l1_ratio_boundaries_are_accepted(boundary: float) -> None:
    """0 and 1 are pure L2 and pure L1; both are legitimate elastic-net corners."""

    LogisticConfig(penalty="elasticnet", l1_ratio=boundary).validate()


@pytest.mark.parametrize("kind", ["random_forest", "extra_trees"])
def test_both_tree_kinds_are_supported(kind: str) -> None:
    TreeEnsembleConfig(kind=kind).validate()  # type: ignore[arg-type]


def test_string_max_features_bypasses_the_float_range_check() -> None:
    """"sqrt" and "log2" are valid; only float values carry a range constraint."""

    for value in ("sqrt", "log2"):
        TreeEnsembleConfig(kind="random_forest", max_features=value).validate()
