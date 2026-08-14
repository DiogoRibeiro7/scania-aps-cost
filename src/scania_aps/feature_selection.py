"""Feature-selection and interpretation utilities for anonymized APS variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance

from scania_aps._types import FittedEstimator, ShapValues


@dataclass(frozen=True)
class RankedFeature:
    """Feature name and an importance/selection score."""

    feature: str
    score: float


def mutual_information_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    k: int | Literal["all"] = "all",
    random_state: int = 42,
) -> list[RankedFeature]:
    """Rank features by univariate mutual information with the failure label."""

    if k != "all" and (not isinstance(k, int) or k <= 0):
        raise ValueError("k must be a positive integer or 'all'.")
    numeric = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    selector = SelectKBest(
        score_func=lambda a, b: mutual_info_classif(a, b, random_state=random_state),
        k=k,
    )
    selector.fit(numeric, y)
    scores = np.asarray(selector.scores_, dtype=np.float64)
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
    return [RankedFeature(str(X.columns[i]), float(scores[i])) for i in order]


def permutation_ranking(
    model: FittedEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str = "average_precision",
    n_repeats: int = 8,
    random_state: int = 42,
) -> list[RankedFeature]:
    """Rank raw input features using held-out permutation importance."""

    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive.")
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    means: NDArray[np.float64] = np.asarray(result.importances_mean, dtype=np.float64)
    order = np.argsort(-means)
    return [RankedFeature(str(X.columns[i]), float(means[i])) for i in order]


def l1_nonzero_features(model: FittedEstimator) -> list[RankedFeature]:
    """Extract non-zero coefficients from a fitted L1/Elastic-Net pipeline.

    Missingness indicators created by the imputer are named ``missing::<raw>``
    when their source columns can be recovered.
    """

    if not hasattr(model, "named_steps"):
        raise TypeError("Expected a fitted sklearn-style pipeline.")
    estimator = model.named_steps.get("model")
    imputer = model.named_steps.get("imputer")
    if estimator is None or imputer is None or not hasattr(estimator, "coef_"):
        raise TypeError("Pipeline must contain fitted 'imputer' and linear 'model' steps.")

    coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    raw_names = [str(name) for name in getattr(model, "feature_names_in_", [])]
    names = list(raw_names)
    indicator = getattr(imputer, "indicator_", None)
    if indicator is not None and raw_names:
        for index in indicator.features_:
            names.append(f"missing::{raw_names[int(index)]}")
    if len(names) != len(coefficients):
        names = [f"feature_{i}" for i in range(len(coefficients))]

    ranked = [
        RankedFeature(name, float(value))
        for name, value in zip(names, coefficients, strict=True)
        if value != 0.0
    ]
    return sorted(ranked, key=lambda item: abs(item.score), reverse=True)


def tree_importance_ranking(model: FittedEstimator) -> list[RankedFeature]:
    """Extract impurity-based feature importances from a fitted tree pipeline."""

    if not hasattr(model, "named_steps"):
        raise TypeError("Expected a fitted sklearn-style pipeline.")
    estimator = model.named_steps.get("model")
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        raise TypeError("Pipeline does not expose tree feature importances.")
    importances = np.asarray(estimator.feature_importances_, dtype=np.float64)
    names = [str(name) for name in getattr(model, "feature_names_in_", [])]
    if len(names) != len(importances):
        # add_indicator changes dimensionality; fall back to stable synthetic names.
        names = [f"transformed_feature_{i}" for i in range(len(importances))]
    order = np.argsort(-importances)
    return [RankedFeature(names[i], float(importances[i])) for i in order]


def shap_values(model: FittedEstimator, X: pd.DataFrame, *, max_rows: int = 2000) -> ShapValues:
    """Compute optional SHAP values on a bounded held-out sample."""

    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    try:
        import shap
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional 'explain' dependency group to use SHAP.") from exc

    sample = X.iloc[:max_rows]
    explainer = shap.Explainer(model, sample)
    return explainer(sample)
