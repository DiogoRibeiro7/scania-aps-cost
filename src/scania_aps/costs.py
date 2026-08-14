"""Industrial misclassification cost and threshold optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FALSE_POSITIVE_COST: float = 10.0
FALSE_NEGATIVE_COST: float = 500.0


@dataclass(frozen=True)
class CostResult:
    """Cost decomposition for a binary decision rule."""

    false_positives: int
    false_negatives: int
    total_cost: float
    cost_per_observation: float


@dataclass(frozen=True)
class ThresholdResult:
    """Optimal score threshold and its resulting development cost."""

    threshold: float
    cost: CostResult


def _as_binary_vector(values: NDArray[np.integer] | NDArray[np.bool_]) -> NDArray[np.int8]:
    array = np.asarray(values).reshape(-1)
    if not np.isin(array, [0, 1]).all():
        raise ValueError("Binary values must contain only 0 and 1.")
    return array.astype(np.int8, copy=False)


def maintenance_cost(
    y_true: NDArray[np.integer] | NDArray[np.bool_],
    y_pred: NDArray[np.integer] | NDArray[np.bool_],
    *,
    false_positive_cost: float = FALSE_POSITIVE_COST,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
) -> CostResult:
    """Compute the Scania challenge cost."""

    truth = _as_binary_vector(y_true)
    pred = _as_binary_vector(y_pred)

    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("Misclassification costs must be non-negative.")

    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    total = false_positive_cost * fp + false_negative_cost * fn

    return CostResult(
        false_positives=fp,
        false_negatives=fn,
        total_cost=float(total),
        cost_per_observation=float(total / max(len(truth), 1)),
    )


def bayes_threshold(
    *,
    false_positive_cost: float = FALSE_POSITIVE_COST,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
) -> float:
    """Return the Bayes threshold for calibrated probabilities.

    Predicting failure is optimal when

    ``C_FP * P(Y=0|x) < C_FN * P(Y=1|x)``.
    """

    denominator = false_positive_cost + false_negative_cost
    if denominator <= 0:
        raise ValueError("The sum of misclassification costs must be positive.")
    return float(false_positive_cost / denominator)


def optimize_score_threshold(
    y_true: NDArray[np.integer] | NDArray[np.bool_],
    scores: NDArray[np.floating],
    *,
    false_positive_cost: float = FALSE_POSITIVE_COST,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
) -> ThresholdResult:
    """Find the exact empirical cost-minimizing threshold for arbitrary scores.

    The implementation sorts scores once, groups ties, and evaluates every
    distinct decision boundary in ``O(n log n)`` time. Higher scores are
    assumed to indicate stronger evidence for the positive class.
    """

    truth = _as_binary_vector(y_true)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)

    if len(truth) != len(values):
        raise ValueError("y_true and scores must have identical lengths.")
    if len(values) == 0:
        raise ValueError("At least one score is required.")
    if not np.isfinite(values).all():
        raise ValueError("Scores must be finite.")
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("Misclassification costs must be non-negative.")

    order = np.argsort(-values, kind="mergesort")
    score_sorted = values[order]
    y_sorted = truth[order]

    cumulative_tp = np.cumsum(y_sorted)
    cumulative_fp = np.cumsum(1 - y_sorted)
    total_positives = int(np.sum(y_sorted))

    best_fp = 0
    best_fn = total_positives
    best_total = false_negative_cost * best_fn
    best_threshold = float(np.nextafter(score_sorted[0], np.inf))

    boundaries = np.flatnonzero(np.r_[score_sorted[:-1] != score_sorted[1:], True])

    for index in boundaries:
        fp = int(cumulative_fp[index])
        tp = int(cumulative_tp[index])
        fn = total_positives - tp
        total = false_positive_cost * fp + false_negative_cost * fn

        if index == len(score_sorted) - 1:
            threshold = float(np.nextafter(score_sorted[index], -np.inf))
        else:
            threshold = float((score_sorted[index] + score_sorted[index + 1]) / 2.0)

        if total < best_total:
            best_fp = fp
            best_fn = fn
            best_total = total
            best_threshold = threshold

    return ThresholdResult(
        threshold=best_threshold,
        cost=CostResult(
            false_positives=best_fp,
            false_negatives=best_fn,
            total_cost=float(best_total),
            cost_per_observation=float(best_total / len(truth)),
        ),
    )


def optimize_threshold(
    y_true: NDArray[np.integer] | NDArray[np.bool_],
    probabilities: NDArray[np.floating],
    *,
    false_positive_cost: float = FALSE_POSITIVE_COST,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
) -> ThresholdResult:
    """Find the exact empirical cost-minimizing probability threshold."""

    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not np.isfinite(probs).all() or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("Probabilities must be finite and lie in [0, 1].")
    return optimize_score_threshold(
        y_true,
        probs,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
