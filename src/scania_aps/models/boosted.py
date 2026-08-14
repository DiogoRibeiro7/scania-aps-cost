"""Optional XGBoost model for nonlinear interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BoostedConfig:
    """Regularization and optimization settings for gradient boosting."""

    n_estimators: int = 600
    learning_rate: float = 0.04
    max_depth: int = 6
    min_child_weight: float = 3.0
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0

    def validate(self) -> None:
        """Validate the search space before estimator construction."""

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1].")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if self.min_child_weight < 0 or self.reg_alpha < 0 or self.reg_lambda < 0:
            raise ValueError("Regularization parameters must be non-negative.")
        if not 0.0 < self.subsample <= 1.0 or not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("subsample and colsample_bytree must lie in (0, 1].")
        if self.scale_pos_weight <= 0:
            raise ValueError("scale_pos_weight must be positive.")


def build_boosted_classifier(config: BoostedConfig) -> Any:
    """Construct an XGBoost classifier.

    XGBoost is an optional dependency so importing the base package does not
    require it.
    """

    config.validate()

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install the optional 'boost' dependency group to use XGBoost.") from exc

    return XGBClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        scale_pos_weight=config.scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
