"""Utilities for obtaining comparable positive-class model scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

ScoreKind = Literal["probability", "decision"]


@dataclass(frozen=True)
class ModelScores:
    """Positive-class scores together with their semantic type."""

    values: NDArray[np.float64]
    kind: ScoreKind


def positive_class_scores(model: Any, X: Any) -> ModelScores:
    """Return positive-class probabilities or decision scores.

    Probability-producing models are preferred because calibration diagnostics
    are meaningful for them. Margin-based estimators such as ``LinearSVC`` are
    still usable through ``decision_function`` and receive their own optimal
    operating threshold.
    """

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("Binary predict_proba output with two columns is required.")
        return ModelScores(values=probabilities[:, 1], kind="probability")

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1)
        if not np.isfinite(scores).all():
            raise ValueError("Decision scores must be finite.")
        return ModelScores(values=scores, kind="decision")

    raise TypeError("Model must implement predict_proba or decision_function.")
