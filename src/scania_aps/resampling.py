"""Leakage-safe imbalance strategies implemented inside training pipelines."""

from __future__ import annotations

from typing import Literal

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from scania_aps._types import Estimator

ResamplingStrategy = Literal["none", "undersample", "smote"]


def build_resampled_pipeline(
    estimator: Estimator,
    *,
    strategy: ResamplingStrategy,
    scale: bool = True,
    random_state: int = 42,
) -> Estimator:
    """Wrap an estimator with imputation, optional scaling and fit-only resampling.

    ``imbalanced-learn`` pipelines call the sampler only during ``fit``. This
    prevents synthetic or discarded observations from leaking into validation,
    calibration, threshold-selection or test data.
    """

    if strategy not in {"none", "undersample", "smote"}:
        raise ValueError("Unsupported resampling strategy.")

    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional 'imbalance' dependency group.") from exc

    steps: list[tuple[str, Estimator]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    if strategy == "undersample":
        steps.append(("sampler", RandomUnderSampler(random_state=random_state)))
    elif strategy == "smote":
        steps.append(("sampler", SMOTE(random_state=random_state, k_neighbors=5)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)
