"""Candidate generation for model-family optimization experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np

from scania_aps.models.factory import ModelCandidate, ModelFamily

SearchProfile = Literal["quick", "full"]


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _optional_weight(rng: np.random.Generator, options: Sequence[float | None]) -> float | None:
    """Draw from a sequence that mixes ``None`` with class-weight values.

    ``Generator.choice`` already coerces such a sequence to an object array
    internally. Doing it explicitly leaves the draw bit-for-bit identical while
    giving the call a static type that survives ``--strict``.
    """

    return cast("float | None", rng.choice(np.asarray(options, dtype=object)))


def candidates_for_family(
    family: ModelFamily,
    *,
    profile: SearchProfile = "quick",
    random_state: int = 42,
) -> list[ModelCandidate]:
    """Generate reproducible candidates exposing each family's key controls."""

    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'.")
    rng = np.random.default_rng(random_state)
    n = 3 if profile == "quick" else 12
    result: list[ModelCandidate] = []

    if family == "logistic":
        base = [
            ("l2", 1.0, None, None),
            ("l1", 0.1, None, 20.0),
            ("elasticnet", 0.1, 0.5, 20.0),
        ]
        for index, (penalty, C, l1_ratio, weight) in enumerate(base):
            result.append(
                ModelCandidate(
                    family=family,
                    name=f"logistic_{index}",
                    parameters={
                        "penalty": penalty,
                        "C": C,
                        "l1_ratio": l1_ratio,
                        "positive_class_weight": weight,
                    },
                )
            )
        while len(result) < n:
            penalty = str(rng.choice(["l1", "l2", "elasticnet"]))
            result.append(
                ModelCandidate(
                    family=family,
                    name=f"logistic_{len(result)}",
                    parameters={
                        "penalty": penalty,
                        "C": _log_uniform(rng, 1e-4, 1e2),
                        "l1_ratio": float(rng.uniform(0.05, 0.95))
                        if penalty == "elasticnet"
                        else None,
                        "positive_class_weight": _optional_weight(
                            rng, [None, 5.0, 10.0, 20.0, 50.0]
                        ),
                    },
                )
            )
        return result

    if family == "linear_svm":
        grid = [(0.01, None), (0.1, 20.0), (1.0, 50.0)]
        for C, weight in grid:
            result.append(
                ModelCandidate(
                    family=family,
                    name=f"svm_C{C:g}_w{weight}",
                    parameters={"C": C, "positive_class_weight": weight},
                )
            )
        while len(result) < n:
            result.append(
                ModelCandidate(
                    family=family,
                    name=f"svm_{len(result)}",
                    parameters={
                        "C": _log_uniform(rng, 1e-4, 1e2),
                        "positive_class_weight": _optional_weight(
                            rng, [None, 5.0, 10.0, 20.0, 50.0]
                        ),
                    },
                )
            )
        return result

    if family in {"random_forest", "extra_trees"}:
        defaults: list[dict[str, Any]] = [
            {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 500, "max_depth": 18, "min_samples_leaf": 3, "max_features": 0.5},
            {
                "n_estimators": 600,
                "max_depth": 10,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "positive_class_weight": 20.0,
            },
        ]
        for index, params in enumerate(defaults):
            result.append(
                ModelCandidate(family=family, name=f"{family}_{index}", parameters=params)
            )
        while len(result) < n:
            depth_choice = rng.choice([0, 8, 12, 18, 24])
            result.append(
                ModelCandidate(
                    family=family,
                    name=f"{family}_{len(result)}",
                    parameters={
                        "n_estimators": int(rng.integers(300, 900)),
                        "max_depth": None if depth_choice == 0 else int(depth_choice),
                        "min_samples_leaf": int(rng.choice([1, 2, 5, 10, 20])),
                        "max_features": ["sqrt", "log2", 0.5, 0.8][int(rng.integers(4))],
                        "positive_class_weight": _optional_weight(rng, [None, 10.0, 20.0, 50.0]),
                    },
                )
            )
        return result

    if family == "xgboost":
        defaults = [
            {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 5,
                "min_child_weight": 3.0,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "scale_pos_weight": 20.0,
            },
            {
                "n_estimators": 800,
                "learning_rate": 0.03,
                "max_depth": 7,
                "min_child_weight": 5.0,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
                "scale_pos_weight": 20.0,
            },
            {
                "n_estimators": 350,
                "learning_rate": 0.08,
                "max_depth": 3,
                "min_child_weight": 10.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 1.0,
                "reg_lambda": 10.0,
                "scale_pos_weight": 50.0,
            },
        ]
        for i, params in enumerate(defaults):
            result.append(ModelCandidate(family=family, name=f"xgboost_{i}", parameters=params))
        while len(result) < n:
            result.append(
                ModelCandidate(
                    family=family,
                    name=f"xgboost_{len(result)}",
                    parameters={
                        "n_estimators": int(rng.integers(300, 1000)),
                        "learning_rate": _log_uniform(rng, 0.01, 0.15),
                        "max_depth": int(rng.integers(3, 9)),
                        "min_child_weight": float(rng.choice([1, 3, 5, 10, 20])),
                        "subsample": float(rng.uniform(0.65, 1.0)),
                        "colsample_bytree": float(rng.uniform(0.65, 1.0)),
                        "reg_alpha": _log_uniform(rng, 1e-4, 5.0),
                        "reg_lambda": _log_uniform(rng, 0.1, 20.0),
                        "scale_pos_weight": float(rng.choice([1, 10, 20, 50])),
                    },
                )
            )
        return result

    if family == "lightgbm":
        defaults = [
            {
                "n_estimators": 600,
                "learning_rate": 0.04,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "positive_class_weight": 20.0,
            },
            {
                "n_estimators": 850,
                "learning_rate": 0.025,
                "num_leaves": 63,
                "min_child_samples": 40,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.5,
                "reg_lambda": 5.0,
                "positive_class_weight": 20.0,
            },
            {
                "n_estimators": 450,
                "learning_rate": 0.06,
                "num_leaves": 15,
                "min_child_samples": 80,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 1.0,
                "reg_lambda": 10.0,
                "positive_class_weight": 50.0,
            },
        ]
        for i, params in enumerate(defaults):
            result.append(ModelCandidate(family=family, name=f"lightgbm_{i}", parameters=params))
        while len(result) < n:
            result.append(
                ModelCandidate(
                    family=family,
                    name=f"lightgbm_{len(result)}",
                    parameters={
                        "n_estimators": int(rng.integers(300, 1000)),
                        "learning_rate": _log_uniform(rng, 0.01, 0.15),
                        "num_leaves": int(rng.choice([15, 31, 63, 127])),
                        "min_child_samples": int(rng.choice([10, 20, 40, 80, 120])),
                        "subsample": float(rng.uniform(0.65, 1.0)),
                        "colsample_bytree": float(rng.uniform(0.65, 1.0)),
                        "reg_alpha": _log_uniform(rng, 1e-4, 5.0),
                        "reg_lambda": _log_uniform(rng, 0.1, 20.0),
                        "positive_class_weight": float(rng.choice([1, 10, 20, 50])),
                    },
                )
            )
        return result

    if family == "mlp":
        variants = [
            {
                "optimizer": "sgd",
                "learning_rate": 0.02,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "batch_norm": False,
                "scheduler": "cosine",
                "loss": "bce",
                "max_epochs": 80,
            },
            {
                "optimizer": "adam",
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "dropout": 0.2,
                "batch_norm": True,
                "scheduler": "plateau",
                "loss": "bce",
                "max_epochs": 80,
            },
            {
                "optimizer": "adamw",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "dropout": 0.3,
                "batch_norm": True,
                "scheduler": "plateau",
                "loss": "focal",
                "positive_class_weight": 10.0,
                "max_epochs": 80,
            },
        ]
        for i, params in enumerate(variants):
            result.append(ModelCandidate(family=family, name=f"mlp_{i}", parameters=params))
        while len(result) < n:
            result.append(
                ModelCandidate(
                    family=family,
                    name=f"mlp_{len(result)}",
                    parameters={
                        "hidden_dims": [(128, 64), (256, 128), (256, 128, 64)][
                            int(rng.integers(3))
                        ],
                        "optimizer": str(rng.choice(["sgd", "adam", "adamw"])),
                        "learning_rate": _log_uniform(rng, 1e-4, 3e-2),
                        "weight_decay": _log_uniform(rng, 1e-7, 1e-2),
                        "dropout": float(rng.uniform(0.0, 0.5)),
                        "batch_norm": bool(rng.integers(0, 2)),
                        "batch_size": int(rng.choice([128, 256, 512, 1024])),
                        "scheduler": str(rng.choice(["none", "cosine", "plateau"])),
                        "loss": str(rng.choice(["bce", "focal"])),
                        "positive_class_weight": _optional_weight(rng, [None, 5.0, 10.0, 20.0]),
                        "max_epochs": 100,
                    },
                )
            )
        return result

    if family == "autoencoder":
        variants = [
            {
                "latent_dim": 16,
                "hidden_dim": 96,
                "weight_decay": 1e-5,
                "classifier_C": 0.1,
                "max_epochs": 40,
            },
            {
                "latent_dim": 32,
                "hidden_dim": 128,
                "weight_decay": 1e-5,
                "classifier_C": 1.0,
                "positive_class_weight": 20.0,
                "max_epochs": 40,
            },
            {
                "latent_dim": 64,
                "hidden_dim": 192,
                "weight_decay": 1e-4,
                "classifier_C": 0.1,
                "positive_class_weight": 20.0,
                "max_epochs": 40,
            },
        ]
        for i, params in enumerate(variants):
            result.append(ModelCandidate(family=family, name=f"autoencoder_{i}", parameters=params))
        while len(result) < n:
            result.append(
                ModelCandidate(
                    family=family,
                    name=f"autoencoder_{len(result)}",
                    parameters={
                        "latent_dim": int(rng.choice([8, 16, 32, 64])),
                        "hidden_dim": int(rng.choice([64, 128, 256])),
                        "learning_rate": _log_uniform(rng, 1e-4, 3e-3),
                        "weight_decay": _log_uniform(rng, 1e-7, 1e-3),
                        "classifier_C": _log_uniform(rng, 1e-3, 10.0),
                        "positive_class_weight": _optional_weight(rng, [None, 10.0, 20.0, 50.0]),
                        "max_epochs": 50,
                    },
                )
            )
        return result

    raise ValueError(f"Unsupported model family: {family}")
