"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from scania_aps.data import TEST_FILENAME, TRAIN_FILENAME, download_dataset
from scania_aps.experiment import run_boosted_experiment, run_logistic_experiment
from scania_aps.models.factory import ModelFamily
from scania_aps.studies import (
    run_calibration_study,
    run_feature_selection_study,
    run_imbalance_study,
    run_model_family_study,
    run_xgboost_ablation,
)

ALL_FAMILIES: tuple[ModelFamily, ...] = (
    "logistic",
    "linear_svm",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "mlp",
    "autoencoder",
)


def _paths(root: Path) -> tuple[Path, Path, Path]:
    raw = root / "data" / "raw"
    return raw / TRAIN_FILENAME, raw / TEST_FILENAME, root / "artifacts"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Scania APS cost-sensitive ML")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="Download the official UCI dataset")

    logistic = subparsers.add_parser("train-logistic", help="Tune regularized logistic regression")
    logistic.add_argument("--trials", type=int, default=24)
    subparsers.add_parser("train-boosted", help="Train the XGBoost reference model")

    study = subparsers.add_parser("run-study", help="Compare all implemented model families")
    study.add_argument("--profile", choices=["quick", "full"], default="quick")
    study.add_argument("--calibration", choices=["none", "sigmoid", "isotonic"], default="none")
    study.add_argument(
        "--models",
        nargs="+",
        choices=list(ALL_FAMILIES),
        default=list(ALL_FAMILIES),
        help="Model families to run",
    )

    calibration = subparsers.add_parser("study-calibration", help="Platt vs isotonic calibration")
    calibration.add_argument("--model", choices=list(ALL_FAMILIES), default="logistic")
    calibration.add_argument("--profile", choices=["quick", "full"], default="quick")

    subparsers.add_parser("study-imbalance", help="Weighting vs sampling vs focal loss")
    subparsers.add_parser("study-features", help="Compare feature-selection methods")
    subparsers.add_parser("study-ablation", help="Run XGBoost regularization/decision ablations")
    return parser


def _print_table(frame: object) -> None:
    if hasattr(frame, "to_string"):
        print(frame.to_string(index=False))
    else:
        print(frame)


def main() -> None:
    """Run a selected repository workflow."""

    args = build_parser().parse_args()
    root: Path = args.root.resolve()
    train_path, test_path, artifacts = _paths(root)

    if args.command == "download":
        train, test = download_dataset(root / "data" / "raw")
        print(f"training data: {train}")
        print(f"test data: {test}")
        return

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Dataset not found. Run `scania-aps download` first.")

    if args.command == "train-logistic":
        result = run_logistic_experiment(train_path, test_path, artifacts, n_trials=args.trials)
        print(f"cost={result.total_cost:.0f}")
        print(f"false negatives={result.false_negatives}")
        print(f"false positives={result.false_positives}")
        print(f"PR-AUC={result.pr_auc:.4f}")
        print(f"threshold={result.threshold:.6f}")
        return
    if args.command == "train-boosted":
        result = run_boosted_experiment(train_path, test_path, artifacts)
        print(f"cost={result.total_cost:.0f}")
        return
    if args.command == "run-study":
        frame = run_model_family_study(
            train_path,
            test_path,
            artifacts,
            families=args.models,
            profile=args.profile,
            calibration=args.calibration,
        )
    elif args.command == "study-calibration":
        frame = run_calibration_study(
            train_path, test_path, artifacts, family=args.model, profile=args.profile
        )
    elif args.command == "study-imbalance":
        frame = run_imbalance_study(train_path, test_path, artifacts)
    elif args.command == "study-features":
        frame = run_feature_selection_study(train_path, test_path, artifacts)
    elif args.command == "study-ablation":
        frame = run_xgboost_ablation(train_path, test_path, artifacts)
    else:  # pragma: no cover
        raise RuntimeError(f"Unknown command: {args.command}")
    _print_table(frame)


if __name__ == "__main__":  # pragma: no cover
    main()
