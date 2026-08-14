"""Optional LightGBM model exposing leaf-wise regularization controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LightGBMConfig:
    """Optimization, complexity and imbalance settings for LightGBM."""

    n_estimators: int = 700
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    positive_class_weight: float = 1.0

    def validate(self) -> None:
        """Validate LightGBM parameters."""

        if self.n_estimators <= 0 or self.num_leaves <= 1 or self.min_child_samples <= 0:
            raise ValueError("Estimator, leaf and child counts must be positive.")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1].")
        if not 0.0 < self.subsample <= 1.0 or not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("subsample and colsample_bytree must lie in (0, 1].")
        if self.reg_alpha < 0 or self.reg_lambda < 0:
            raise ValueError("Regularization parameters must be non-negative.")
        if self.positive_class_weight <= 0:
            raise ValueError("positive_class_weight must be positive.")


def build_lightgbm_classifier(config: LightGBMConfig) -> Any:
    """Construct LightGBM while keeping it an optional dependency."""

    config.validate()
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the optional 'boost' dependency group to use LightGBM.") from exc

    return LGBMClassifier(
        objective="binary",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        scale_pos_weight=config.positive_class_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
