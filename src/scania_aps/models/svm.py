"""Linear support-vector-machine baseline."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class LinearSVMConfig:
    """Margin regularization and class-weight settings."""

    C: float = 1.0
    positive_class_weight: float | None = None
    max_iter: int = 10_000

    def validate(self) -> None:
        """Validate configuration values."""

        if self.C <= 0:
            raise ValueError("C must be positive.")
        if self.positive_class_weight is not None and self.positive_class_weight <= 0:
            raise ValueError("positive_class_weight must be positive when supplied.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")


def build_linear_svm_pipeline(config: LinearSVMConfig) -> Pipeline:
    """Build a scaled linear SVM with median imputation and missing indicators."""

    config.validate()
    class_weight = (
        None
        if config.positive_class_weight is None
        else {0: 1.0, 1: float(config.positive_class_weight)}
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LinearSVC(
                    C=config.C,
                    class_weight=class_weight,
                    dual="auto",
                    max_iter=config.max_iter,
                    random_state=42,
                ),
            ),
        ]
    )
