"""Probability calibration isolated from model fitting and threshold selection."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from scania_aps._types import FittedEstimator

CalibrationMethod = Literal["sigmoid", "isotonic"]


def calibrate_prefit_model(
    fitted_model: FittedEstimator,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    *,
    method: CalibrationMethod,
) -> FittedEstimator:
    """Calibrate an already-fitted model on a dedicated calibration subset.

    Scikit-learn's ``FrozenEstimator`` explicitly prevents refitting the base
    model. The returned estimator exposes calibrated ``predict_proba``.
    """

    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be 'sigmoid' or 'isotonic'.")

    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError as exc:  # pragma: no cover - for older sklearn only
        raise RuntimeError("Probability calibration requires scikit-learn >= 1.6.") from exc

    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    calibrated.fit(X_calibration, y_calibration)
    return calibrated
