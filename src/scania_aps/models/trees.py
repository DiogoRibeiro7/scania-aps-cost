"""Bagged tree ensembles for nonlinear APS failure prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TreeEnsemble = Literal["random_forest", "extra_trees"]


@dataclass(frozen=True)
class TreeEnsembleConfig:
    """Structural and stochastic regularization for tree ensembles."""

    kind: TreeEnsemble = "random_forest"
    n_estimators: int = 500
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str | float | None = "sqrt"
    positive_class_weight: float | None = None

    def validate(self) -> None:
        """Validate ensemble hyperparameters."""

        if self.kind not in {"random_forest", "extra_trees"}:
            raise ValueError("Unsupported tree ensemble kind.")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive when supplied.")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive.")
        if isinstance(self.max_features, float) and not 0.0 < self.max_features <= 1.0:
            raise ValueError("Float max_features must lie in (0, 1].")
        if self.positive_class_weight is not None and self.positive_class_weight <= 0:
            raise ValueError("positive_class_weight must be positive when supplied.")


def build_tree_pipeline(config: TreeEnsembleConfig) -> Pipeline:
    """Build Random Forest or Extra Trees with fit-only median imputation."""

    config.validate()
    class_weight = (
        None
        if config.positive_class_weight is None
        else {0: 1.0, 1: float(config.positive_class_weight)}
    )
    common = dict(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    estimator = (
        RandomForestClassifier(**common)
        if config.kind == "random_forest"
        else ExtraTreesClassifier(**common)
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", estimator),
        ]
    )
