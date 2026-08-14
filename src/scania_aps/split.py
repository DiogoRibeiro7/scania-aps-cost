"""Leakage-safe development splits for model selection and calibration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DevelopmentSplit:
    """Three-way stratified split kept for backward-compatible experiments."""

    X_fit: pd.DataFrame
    y_fit: pd.Series
    X_tune: pd.DataFrame
    y_tune: pd.Series
    X_threshold: pd.DataFrame
    y_threshold: pd.Series


@dataclass(frozen=True)
class ResearchSplit:
    """Four-way split separating tuning, calibration and threshold selection."""

    X_fit: pd.DataFrame
    y_fit: pd.Series
    X_tune: pd.DataFrame
    y_tune: pd.Series
    X_calibration: pd.DataFrame
    y_calibration: pd.Series
    X_threshold: pd.DataFrame
    y_threshold: pd.Series


def _validate_split_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if y.nunique(dropna=False) != 2:
        raise ValueError("The target must contain exactly two classes.")


def development_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fit_fraction: float = 0.70,
    tune_fraction: float = 0.15,
    random_state: int = 42,
) -> DevelopmentSplit:
    """Create fit/tune/threshold subsets without touching the official test set."""

    _validate_split_inputs(X, y)
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in (0, 1).")
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError("tune_fraction must lie in (0, 1).")
    if fit_fraction + tune_fraction >= 1.0:
        raise ValueError("fit_fraction + tune_fraction must be less than 1.")

    remainder_fraction = 1.0 - fit_fraction
    X_fit, X_remainder, y_fit, y_remainder = train_test_split(
        X, y, test_size=remainder_fraction, stratify=y, random_state=random_state
    )

    relative_tune = tune_fraction / remainder_fraction
    X_tune, X_threshold, y_tune, y_threshold = train_test_split(
        X_remainder,
        y_remainder,
        train_size=relative_tune,
        stratify=y_remainder,
        random_state=random_state + 1,
    )

    return DevelopmentSplit(
        X_fit=X_fit,
        y_fit=y_fit,
        X_tune=X_tune,
        y_tune=y_tune,
        X_threshold=X_threshold,
        y_threshold=y_threshold,
    )


def research_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fit_fraction: float = 0.65,
    tune_fraction: float = 0.15,
    calibration_fraction: float = 0.10,
    random_state: int = 42,
) -> ResearchSplit:
    """Create distinct fit, tune, calibration and threshold subsets.

    The official test set remains untouched. This separation lets calibration
    methods be compared without reusing the same observations to choose the
    deployment threshold.
    """

    _validate_split_inputs(X, y)
    fractions = (fit_fraction, tune_fraction, calibration_fraction)
    if any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("All split fractions must lie in (0, 1).")
    if sum(fractions) >= 1.0:
        raise ValueError("fit + tune + calibration fractions must sum to less than 1.")

    remainder = 1.0 - fit_fraction
    X_fit, X_rest, y_fit, y_rest = train_test_split(
        X, y, test_size=remainder, stratify=y, random_state=random_state
    )

    tune_relative = tune_fraction / remainder
    X_tune, X_after_tune, y_tune, y_after_tune = train_test_split(
        X_rest,
        y_rest,
        train_size=tune_relative,
        stratify=y_rest,
        random_state=random_state + 1,
    )

    after_tune_fraction = remainder - tune_fraction
    calibration_relative = calibration_fraction / after_tune_fraction
    X_cal, X_threshold, y_cal, y_threshold = train_test_split(
        X_after_tune,
        y_after_tune,
        train_size=calibration_relative,
        stratify=y_after_tune,
        random_state=random_state + 2,
    )

    return ResearchSplit(
        X_fit=X_fit,
        y_fit=y_fit,
        X_tune=X_tune,
        y_tune=y_tune,
        X_calibration=X_cal,
        y_calibration=y_cal,
        X_threshold=X_threshold,
        y_threshold=y_threshold,
    )
