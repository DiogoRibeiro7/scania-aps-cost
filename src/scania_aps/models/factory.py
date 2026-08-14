"""Unified model registry used by reproducible benchmark studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scania_aps.models.autoencoder import AutoencoderLogisticClassifier
from scania_aps.models.boosted import BoostedConfig, build_boosted_classifier
from scania_aps.models.lightgbm import LightGBMConfig, build_lightgbm_classifier
from scania_aps.models.logistic import LogisticConfig, build_logistic_pipeline
from scania_aps.models.mlp import TorchMLPClassifier
from scania_aps.models.svm import LinearSVMConfig, build_linear_svm_pipeline
from scania_aps.models.trees import TreeEnsembleConfig, build_tree_pipeline

ModelFamily = Literal[
    "logistic",
    "linear_svm",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "mlp",
    "autoencoder",
]


@dataclass(frozen=True)
class ModelCandidate:
    """Named model candidate and its serializable hyperparameters."""

    family: ModelFamily
    name: str
    parameters: dict[str, Any]


def build_candidate(candidate: ModelCandidate) -> Any:
    """Construct a fresh estimator from a model candidate."""

    p = dict(candidate.parameters)
    if candidate.family == "logistic":
        return build_logistic_pipeline(LogisticConfig(**p))
    if candidate.family == "linear_svm":
        return build_linear_svm_pipeline(LinearSVMConfig(**p))
    if candidate.family in {"random_forest", "extra_trees"}:
        return build_tree_pipeline(TreeEnsembleConfig(kind=candidate.family, **p))
    if candidate.family == "xgboost":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("model", build_boosted_classifier(BoostedConfig(**p))),
            ]
        )
    if candidate.family == "lightgbm":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("model", build_lightgbm_classifier(LightGBMConfig(**p))),
            ]
        )
    if candidate.family == "mlp":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("model", TorchMLPClassifier(**p)),
            ]
        )
    if candidate.family == "autoencoder":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("model", AutoencoderLogisticClassifier(**p)),
            ]
        )
    raise ValueError(f"Unsupported model family: {candidate.family}")
