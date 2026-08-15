import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from scania_aps.calibration import calibrate_prefit_model


@pytest.fixture
def fitted_model_and_calibration_set() -> tuple[LogisticRegression, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=200) > 0).astype(int))
    model = LogisticRegression(max_iter=500).fit(X[:120], y[:120])
    return model, X[120:], y[120:]


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrated_model_returns_two_column_probabilities(
    method: str,
    fitted_model_and_calibration_set: tuple[LogisticRegression, pd.DataFrame, pd.Series],
) -> None:
    model, X_cal, y_cal = fitted_model_and_calibration_set
    calibrated = calibrate_prefit_model(model, X_cal, y_cal, method=method)

    probabilities = calibrated.predict_proba(X_cal)
    assert probabilities.shape == (len(X_cal), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_calibration_does_not_refit_the_base_model(
    fitted_model_and_calibration_set: tuple[LogisticRegression, pd.DataFrame, pd.Series],
) -> None:
    """FrozenEstimator must keep the base model's parameters untouched.

    If the base model were refit on the calibration subset, the separation
    between fitting and calibration would be lost.
    """

    model, X_cal, y_cal = fitted_model_and_calibration_set
    before = model.coef_.copy()

    calibrate_prefit_model(model, X_cal, y_cal, method="sigmoid")

    np.testing.assert_array_equal(model.coef_, before)


def test_unknown_method_is_rejected(
    fitted_model_and_calibration_set: tuple[LogisticRegression, pd.DataFrame, pd.Series],
) -> None:
    model, X_cal, y_cal = fitted_model_and_calibration_set
    with pytest.raises(ValueError, match="sigmoid"):
        calibrate_prefit_model(model, X_cal, y_cal, method="platt")  # type: ignore[arg-type]
