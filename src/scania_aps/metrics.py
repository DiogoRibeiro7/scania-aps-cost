"""Evaluation metrics centered on operational cost."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scania_aps.costs import maintenance_cost

ScoreKind = Literal["probability", "decision"]


@dataclass(frozen=True)
class Evaluation:
    """Model evaluation at a fixed operating threshold."""

    threshold: float
    total_cost: float
    cost_per_observation: float
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    pr_auc: float
    roc_auc: float
    always_negative_cost: float
    always_positive_cost: float
    saving_vs_always_negative: float
    saving_vs_always_positive: float
    score_kind: ScoreKind = "probability"
    brier: float | None = None
    log_loss: float | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        """Convert to a JSON-serializable mapping."""

        return asdict(self)


def evaluate_scores(
    y_true: NDArray[np.integer],
    scores: NDArray[np.floating],
    threshold: float,
    *,
    score_kind: ScoreKind,
) -> Evaluation:
    """Evaluate arbitrary binary-classification scores at a fixed threshold.

    Ranking metrics accept either margins or probabilities. Calibration metrics
    are computed only when ``score_kind='probability'``.
    """

    truth = np.asarray(y_true, dtype=np.int8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)

    if len(truth) != len(values):
        raise ValueError("y_true and scores must have identical lengths.")
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite.")
    if score_kind == "probability" and np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Probability scores must lie in [0, 1].")

    pred = (values >= threshold).astype(np.int8)
    observed = maintenance_cost(truth, pred)
    always_negative = maintenance_cost(truth, np.zeros_like(truth)).total_cost
    always_positive = maintenance_cost(truth, np.ones_like(truth)).total_cost

    brier: float | None = None
    cross_entropy: float | None = None
    if score_kind == "probability":
        brier = float(brier_score_loss(truth, values))
        cross_entropy = float(log_loss(truth, values, labels=[0, 1]))

    return Evaluation(
        threshold=float(threshold),
        total_cost=observed.total_cost,
        cost_per_observation=observed.cost_per_observation,
        false_positives=observed.false_positives,
        false_negatives=observed.false_negatives,
        precision=float(precision_score(truth, pred, zero_division=0)),
        recall=float(recall_score(truth, pred, zero_division=0)),
        pr_auc=float(average_precision_score(truth, values)),
        roc_auc=float(roc_auc_score(truth, values)),
        always_negative_cost=float(always_negative),
        always_positive_cost=float(always_positive),
        saving_vs_always_negative=float(always_negative - observed.total_cost),
        saving_vs_always_positive=float(always_positive - observed.total_cost),
        score_kind=score_kind,
        brier=brier,
        log_loss=cross_entropy,
    )


def evaluate_probabilities(
    y_true: NDArray[np.integer],
    probabilities: NDArray[np.floating],
    threshold: float,
) -> Evaluation:
    """Evaluate probabilities with the industrial cost and calibration metrics."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1].")
    return evaluate_scores(y_true, probabilities, threshold, score_kind="probability")
