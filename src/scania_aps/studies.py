"""Runnable research studies on the real Scania APS data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

from scania_aps.calibration import CalibrationMethod, calibrate_prefit_model
from scania_aps.costs import optimize_score_threshold
from scania_aps.data import read_raw_csv
from scania_aps.metrics import Evaluation, evaluate_scores
from scania_aps.model_search import SearchProfile, candidates_for_family
from scania_aps.models.factory import ModelCandidate, ModelFamily, build_candidate
from scania_aps.models.mlp import TorchMLPClassifier
from scania_aps.resampling import build_resampled_pipeline
from scania_aps.scoring import ModelScores, positive_class_scores
from scania_aps.split import ResearchSplit, research_split

CalibrationChoice = Literal["none", "sigmoid", "isotonic"]


@dataclass(frozen=True)
class CandidatePerformance:
    """Development performance used to select a model candidate."""

    name: str
    family: str
    parameters: dict[str, Any]
    threshold: float
    tune_cost: float
    pr_auc: float
    roc_auc: float
    score_kind: str


@dataclass(frozen=True)
class StudyResult:
    """Final evaluation and selected development configuration."""

    family: str
    candidate: ModelCandidate
    calibration: CalibrationChoice
    evaluation: Evaluation


def _evaluate_scores(y: pd.Series, scores: ModelScores) -> tuple[float, Evaluation]:
    threshold = optimize_score_threshold(y.to_numpy(), scores.values).threshold
    evaluation = evaluate_scores(y.to_numpy(), scores.values, threshold, score_kind=scores.kind)
    return threshold, evaluation


def _select_candidate(
    split: ResearchSplit,
    candidates: Iterable[ModelCandidate],
) -> tuple[ModelCandidate, list[CandidatePerformance]]:
    performances: list[CandidatePerformance] = []
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("At least one model candidate is required.")

    for candidate in candidate_list:
        model = build_candidate(candidate)
        model.fit(split.X_fit, split.y_fit)
        scores = positive_class_scores(model, split.X_tune)
        threshold, evaluation = _evaluate_scores(split.y_tune, scores)
        performances.append(
            CandidatePerformance(
                name=candidate.name,
                family=candidate.family,
                parameters=candidate.parameters,
                threshold=threshold,
                tune_cost=evaluation.total_cost,
                pr_auc=evaluation.pr_auc,
                roc_auc=evaluation.roc_auc,
                score_kind=scores.kind,
            )
        )

    best_performance = min(performances, key=lambda item: (item.tune_cost, -item.pr_auc))
    best_candidate = next(item for item in candidate_list if item.name == best_performance.name)
    return best_candidate, performances


def _fit_selected(
    split: ResearchSplit,
    candidate: ModelCandidate,
    *,
    calibration: CalibrationChoice,
) -> Any:
    X_refit = pd.concat([split.X_fit, split.X_tune], axis=0)
    y_refit = pd.concat([split.y_fit, split.y_tune], axis=0)
    model = build_candidate(candidate)
    model.fit(X_refit, y_refit)
    if calibration == "none":
        return model
    return calibrate_prefit_model(
        model,
        split.X_calibration,
        split.y_calibration,
        method=calibration,
    )


def _final_evaluation(model: Any, split: ResearchSplit, X_test: pd.DataFrame, y_test: pd.Series) -> Evaluation:
    threshold_scores = positive_class_scores(model, split.X_threshold)
    threshold = optimize_score_threshold(split.y_threshold.to_numpy(), threshold_scores.values).threshold
    test_scores = positive_class_scores(model, X_test)
    if test_scores.kind != threshold_scores.kind:
        raise RuntimeError("Score type changed between threshold and test prediction.")
    return evaluate_scores(y_test.to_numpy(), test_scores.values, threshold, score_kind=test_scores.kind)


def run_model_family_study(
    train_csv: Path,
    test_csv: Path,
    artifacts_dir: Path,
    *,
    families: Iterable[ModelFamily] = (
        "logistic",
        "linear_svm",
        "random_forest",
        "extra_trees",
        "xgboost",
        "lightgbm",
        "mlp",
        "autoencoder",
    ),
    profile: SearchProfile = "quick",
    calibration: CalibrationChoice = "none",
) -> pd.DataFrame:
    """Tune each model family on development cost and compare on official test data."""

    train = read_raw_csv(train_csv)
    test = read_raw_csv(test_csv)
    split = research_split(train.X, train.y)
    output = artifacts_dir / "model_study"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for family in families:
        candidates = candidates_for_family(family, profile=profile)
        best, trace = _select_candidate(split, candidates)
        model = _fit_selected(split, best, calibration=calibration)
        evaluation = _final_evaluation(model, split, test.X, test.y)
        try:
            joblib.dump(model, output / f"{family}_model.joblib")
        except Exception:
            # PyTorch-backed models can be environment-sensitive when pickled;
            # the configuration and metrics are always persisted regardless.
            pass
        (output / f"{family}_selection.json").write_text(
            json.dumps([asdict(item) for item in trace], indent=2, default=str), encoding="utf-8"
        )
        row: dict[str, Any] = {
            "family": family,
            "candidate": best.name,
            "calibration": calibration,
            **best.parameters,
            **evaluation.to_dict(),
        }
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["total_cost", "pr_auc"], ascending=[True, False])
    frame.to_csv(output / "comparison.csv", index=False)
    return frame


def run_calibration_study(
    train_csv: Path,
    test_csv: Path,
    artifacts_dir: Path,
    *,
    family: ModelFamily = "logistic",
    profile: SearchProfile = "quick",
) -> pd.DataFrame:
    """Compare uncalibrated, Platt/sigmoid and isotonic probabilities."""

    train = read_raw_csv(train_csv)
    test = read_raw_csv(test_csv)
    split = research_split(train.X, train.y)
    best, trace = _select_candidate(split, candidates_for_family(family, profile=profile))
    rows: list[dict[str, Any]] = []
    for method in ("none", "sigmoid", "isotonic"):
        model = _fit_selected(split, best, calibration=method)
        evaluation = _final_evaluation(model, split, test.X, test.y)
        rows.append({"family": family, "calibration": method, **evaluation.to_dict()})

    output = artifacts_dir / "calibration_study"
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(
        json.dumps([asdict(item) for item in trace], indent=2, default=str), encoding="utf-8"
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "comparison.csv", index=False)
    return frame


def _logistic_base(*, class_weight: dict[int, float] | None = None) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=0.1,
        class_weight=class_weight,
        solver="saga",
        max_iter=4000,
        random_state=42,
        n_jobs=-1,
    )


def run_imbalance_study(train_csv: Path, test_csv: Path, artifacts_dir: Path) -> pd.DataFrame:
    """Compare weighting, under-sampling, SMOTE and focal loss under the same cost metric."""

    train = read_raw_csv(train_csv)
    test = read_raw_csv(test_csv)
    split = research_split(train.X, train.y)
    X_refit = pd.concat([split.X_fit, split.X_tune, split.X_calibration], axis=0)
    y_refit = pd.concat([split.y_fit, split.y_tune, split.y_calibration], axis=0)

    strategies: list[tuple[str, Any]] = [
        ("none", build_resampled_pipeline(_logistic_base(), strategy="none")),
        (
            "class_weight",
            build_resampled_pipeline(_logistic_base(class_weight={0: 1.0, 1: 20.0}), strategy="none"),
        ),
        ("undersample", build_resampled_pipeline(_logistic_base(), strategy="undersample")),
        ("smote", build_resampled_pipeline(_logistic_base(), strategy="smote")),
        (
            "focal_loss_mlp",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        TorchMLPClassifier(
                            loss="focal",
                            focal_gamma=2.0,
                            optimizer="adamw",
                            weight_decay=1e-4,
                            dropout=0.3,
                            positive_class_weight=10.0,
                            max_epochs=80,
                        ),
                    ),
                ]
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for name, model in strategies:
        model.fit(X_refit, y_refit)
        evaluation = _final_evaluation(model, split, test.X, test.y)
        rows.append({"strategy": name, **evaluation.to_dict()})

    output = artifacts_dir / "imbalance_study"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("total_cost")
    frame.to_csv(output / "comparison.csv", index=False)
    return frame


def _mi_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return mutual_info_classif(X, y, random_state=42)


def run_feature_selection_study(
    train_csv: Path,
    test_csv: Path,
    artifacts_dir: Path,
) -> pd.DataFrame:
    """Compare L1, mutual information, RFE and tree-based feature selection."""

    train = read_raw_csv(train_csv)
    test = read_raw_csv(test_csv)
    split = research_split(train.X, train.y)
    X_refit = pd.concat([split.X_fit, split.X_tune, split.X_calibration], axis=0)
    y_refit = pd.concat([split.y_fit, split.y_tune, split.y_calibration], axis=0)

    linear = LogisticRegression(
        penalty="l2", C=0.1, class_weight={0: 1.0, 1: 20.0}, solver="saga", max_iter=4000,
        random_state=42, n_jobs=-1,
    )
    selectors: list[tuple[str, Any]] = [
        ("all_features", "passthrough"),
        (
            "l1_select_from_model",
            SelectFromModel(
                LogisticRegression(
                    penalty="l1", C=0.05, solver="saga", max_iter=4000, random_state=42, n_jobs=-1
                ),
                threshold="median",
            ),
        ),
        ("mutual_information_50", SelectKBest(score_func=_mi_score, k=50)),
        (
            "rfe_50",
            RFE(
                LogisticRegression(penalty="l2", C=0.1, solver="liblinear", max_iter=2000),
                n_features_to_select=50,
                step=0.2,
            ),
        ),
        (
            "extra_trees_select",
            SelectFromModel(
                ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1), threshold="median"
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    for name, selector in selectors:
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("selector", selector),
                ("model", linear),
            ]
        )
        pipeline.fit(X_refit, y_refit)
        evaluation = _final_evaluation(pipeline, split, test.X, test.y)
        selected_count: int | None = None
        if selector != "passthrough" and hasattr(pipeline.named_steps["selector"], "get_support"):
            selected_count = int(np.sum(pipeline.named_steps["selector"].get_support()))
        rows.append({"selection": name, "n_selected": selected_count, **evaluation.to_dict()})

    output = artifacts_dir / "feature_selection_study"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("total_cost")
    frame.to_csv(output / "comparison.csv", index=False)
    return frame


def run_xgboost_ablation(train_csv: Path, test_csv: Path, artifacts_dir: Path) -> pd.DataFrame:
    """Ablate major XGBoost regularization and decision components."""

    train = read_raw_csv(train_csv)
    test = read_raw_csv(test_csv)
    split = research_split(train.X, train.y)
    X_refit = pd.concat([split.X_fit, split.X_tune, split.X_calibration], axis=0)
    y_refit = pd.concat([split.y_fit, split.y_tune, split.y_calibration], axis=0)

    full = {
        "n_estimators": 700,
        "learning_rate": 0.04,
        "max_depth": 5,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "scale_pos_weight": 20.0,
    }
    variants = [
        ("full", full, True),
        ("no_l1_l2", {**full, "reg_alpha": 0.0, "reg_lambda": 0.0}, True),
        ("no_subsampling", {**full, "subsample": 1.0, "colsample_bytree": 1.0}, True),
        ("no_class_weight", {**full, "scale_pos_weight": 1.0}, True),
        ("threshold_0_5", full, False),
    ]

    rows: list[dict[str, Any]] = []
    for name, params, optimize_decision in variants:
        candidate = ModelCandidate("xgboost", name, params)
        model = build_candidate(candidate)
        model.fit(X_refit, y_refit)
        test_scores = positive_class_scores(model, test.X)
        if optimize_decision:
            threshold_scores = positive_class_scores(model, split.X_threshold)
            threshold = optimize_score_threshold(split.y_threshold.to_numpy(), threshold_scores.values).threshold
        else:
            if test_scores.kind != "probability":
                raise RuntimeError("0.5 ablation requires probability scores.")
            threshold = 0.5
        evaluation = evaluate_scores(test.y.to_numpy(), test_scores.values, threshold, score_kind=test_scores.kind)
        rows.append({"ablation": name, **evaluation.to_dict()})

    output = artifacts_dir / "ablation_study"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "xgboost_ablation.csv", index=False)
    return frame
