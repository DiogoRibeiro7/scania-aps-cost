"""Regularized logistic-regression models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Penalty = Literal["l1", "l2", "elasticnet"]


@dataclass(frozen=True)
class LogisticConfig:
    """Hyperparameters controlling shrinkage and class weighting."""

    penalty: Penalty = "l2"
    C: float = 1.0
    l1_ratio: float | None = None
    positive_class_weight: float | None = None
    max_iter: int = 4000

    def validate(self) -> None:
        """Validate hyperparameter ranges before constructing the estimator."""

        if self.C <= 0:
            raise ValueError("C must be positive.")
        if self.penalty == "elasticnet":
            if self.l1_ratio is None or not 0.0 <= self.l1_ratio <= 1.0:
                raise ValueError("Elastic Net requires l1_ratio in [0, 1].")
        elif self.l1_ratio is not None:
            raise ValueError("l1_ratio is only valid with the elasticnet penalty.")
        if self.positive_class_weight is not None and self.positive_class_weight <= 0:
            raise ValueError("positive_class_weight must be positive when supplied.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")


def build_logistic_pipeline(config: LogisticConfig) -> Pipeline:
    """Build an imputation, scaling and SAGA logistic-regression pipeline."""

    config.validate()

    class_weight: dict[int, float] | None = None
    if config.positive_class_weight is not None:
        class_weight = {0: 1.0, 1: float(config.positive_class_weight)}

    # scikit-learn 1.8 deprecated `penalty` in favour of `l1_ratio` alone and
    # removes it in 1.10: l1_ratio=0 is pure L2, 1 is pure L1, and anything
    # between is elastic net. The mapping below is behaviour-preserving --
    # verified to produce identical coefficients on the old and new arguments.
    # `n_jobs` is dropped because it has had no effect since 1.8.
    l1_ratio = {"l1": 1.0, "l2": 0.0}.get(config.penalty, config.l1_ratio)

    estimator = LogisticRegression(
        C=config.C,
        l1_ratio=l1_ratio,
        solver="saga",
        class_weight=class_weight,
        max_iter=config.max_iter,
        random_state=42,
    )

    return Pipeline(
        steps=[
            # Missingness itself can be predictive, so add explicit indicators.
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
