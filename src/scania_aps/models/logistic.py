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


#: Solver chosen per penalty family.
#:
#: ``saga`` supports all three penalties, which makes it tempting as a single
#: uniform choice, but on this data (42,000 rows x 340 columns after
#: missingness indicators) it does not converge within any practical iteration
#: budget and is 30-170x slower than a solver that does. A capped ``saga`` fit
#: is not the model that was asked for: measured on an 8,000-row sample, it
#: zeroed 121 coefficients under an L1 penalty where the converged
#: ``liblinear`` fit zeroed 185.
#:
#: So each penalty gets the fastest solver that actually reaches an optimum.
#: Elastic net has no such option -- ``saga`` is the only solver that supports
#: it -- and those fits remain non-converged. See ``docs/methodology.md``.
_SOLVERS: dict[str, str] = {
    "l2": "lbfgs",  # converges in ~100 iterations
    "l1": "liblinear",  # converges in ~30 iterations
    "elasticnet": "saga",  # the only option; does not converge on this data
}


def build_logistic_pipeline(config: LogisticConfig) -> Pipeline:
    """Build an imputation, scaling and logistic-regression pipeline.

    The solver is chosen from the penalty family rather than fixed, because the
    one solver that covers all three penalties fails to converge here.
    """

    config.validate()

    class_weight: dict[int, float] | None = None
    if config.positive_class_weight is not None:
        class_weight = {0: 1.0, 1: float(config.positive_class_weight)}

    # scikit-learn 1.8 deprecated `penalty` in favour of `l1_ratio` alone and
    # removes it in 1.10: l1_ratio=0 is pure L2, 1 is pure L1, and anything
    # between is elastic net.
    l1_ratio = {"l1": 1.0, "l2": 0.0}.get(config.penalty, config.l1_ratio)

    estimator = LogisticRegression(
        C=config.C,
        l1_ratio=l1_ratio,
        solver=_SOLVERS[config.penalty],
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
