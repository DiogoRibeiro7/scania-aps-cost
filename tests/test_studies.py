"""End-to-end tests for the study runners, focused on leakage safety.

CONTRIBUTING.md states the protocol these tests exist to defend: the official
test set is never used for a modelling decision, and the fit / tune /
calibration / threshold roles stay separate. Those properties are invisible to
a test that only checks a study produced a number, so the tests below spy on
what the study actually fits and thresholds on.
"""

from __future__ import annotations

import json
from collections.abc import Sized
from pathlib import Path

import pandas as pd
import pytest

from scania_aps import studies
from scania_aps._types import Estimator, FeatureMatrix
from scania_aps.costs import ThresholdResult
from scania_aps.data import read_raw_csv
from scania_aps.models.factory import ModelCandidate
from scania_aps.studies import (
    run_calibration_study,
    run_model_family_study,
)


@pytest.fixture
def fit_spy(monkeypatch: pytest.MonkeyPatch) -> list[set[float]]:
    """Record the ``row_id`` fingerprints of every design matrix passed to fit."""

    seen: list[set[float]] = []
    real_build = studies.build_candidate

    def spy_build(candidate: ModelCandidate) -> Estimator:
        estimator = real_build(candidate)
        real_fit = estimator.fit

        def fit(X: FeatureMatrix, y: object = None, **kwargs: object) -> Estimator:
            if isinstance(X, pd.DataFrame) and "row_id" in X.columns:
                seen.append(set(X["row_id"].astype(float)))
            return real_fit(X, y, **kwargs)

        estimator.fit = fit
        return estimator

    monkeypatch.setattr(studies, "build_candidate", spy_build)
    return seen


@pytest.fixture
def threshold_spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record how many observations each threshold optimization saw."""

    seen: list[int] = []
    real = studies.optimize_score_threshold

    def spy(y: Sized, scores: object, **kwargs: object) -> ThresholdResult:
        seen.append(len(y))
        return real(y, scores, **kwargs)

    monkeypatch.setattr(studies, "optimize_score_threshold", spy)
    return seen


def _row_ids(csv: Path) -> set[float]:
    return set(read_raw_csv(csv).X["row_id"].astype(float))


def test_model_family_study_produces_a_ranked_comparison(
    train_test_csvs: tuple[Path, Path], tmp_path: Path
) -> None:
    train_csv, test_csv = train_test_csvs
    artifacts = tmp_path / "artifacts"

    frame = run_model_family_study(
        train_csv, test_csv, artifacts, families=["logistic"], profile="quick"
    )

    assert len(frame) == 1
    for column in ("family", "candidate", "total_cost", "false_negatives", "false_positives"):
        assert column in frame.columns
    assert (artifacts / "model_study" / "comparison.csv").exists()

    trace = json.loads((artifacts / "model_study" / "logistic_selection.json").read_text())
    assert len(trace) == 3, "the quick profile evaluates three candidates"
    assert all("tune_cost" in entry for entry in trace)


def test_study_never_fits_on_the_official_test_set(
    train_test_csvs: tuple[Path, Path], tmp_path: Path, fit_spy: list[set[float]]
) -> None:
    """The headline invariant: no test-set row ever reaches an estimator's fit."""

    train_csv, test_csv = train_test_csvs
    test_ids = _row_ids(test_csv)

    run_model_family_study(
        train_csv, test_csv, tmp_path / "a", families=["logistic"], profile="quick"
    )

    assert fit_spy, "no fit calls were captured, so the assertion below proves nothing"
    leaked = set().union(*fit_spy) & test_ids
    assert not leaked, f"{len(leaked)} official test rows were used for fitting"


def test_candidate_selection_never_fits_on_held_out_development_roles(
    train_test_csvs: tuple[Path, Path], tmp_path: Path, fit_spy: list[set[float]]
) -> None:
    """Calibration and threshold rows must not be used to estimate parameters.

    Candidate selection fits on `fit`; the winner is refit on `fit + tune`.
    Neither stage may touch the calibration or threshold subsets.
    """

    train_csv, test_csv = train_test_csvs
    dataset = read_raw_csv(train_csv)
    split = studies.research_split(dataset.X, dataset.y)
    reserved = set(split.X_calibration["row_id"].astype(float)) | set(
        split.X_threshold["row_id"].astype(float)
    )

    run_model_family_study(
        train_csv, test_csv, tmp_path / "a", families=["logistic"], profile="quick"
    )

    for call in fit_spy:
        assert not (call & reserved), "a fit used calibration or threshold observations"


def test_threshold_is_never_chosen_on_the_official_test_set(
    train_test_csvs: tuple[Path, Path], tmp_path: Path, threshold_spy: list[int]
) -> None:
    train_csv, test_csv = train_test_csvs
    n_test = len(read_raw_csv(test_csv).y)
    dataset = read_raw_csv(train_csv)
    split = studies.research_split(dataset.X, dataset.y)

    run_model_family_study(
        train_csv, test_csv, tmp_path / "a", families=["logistic"], profile="quick"
    )

    assert threshold_spy, "no threshold optimization was captured"
    assert n_test not in threshold_spy, "a threshold was optimized on the official test set"
    assert len(split.y_threshold) in threshold_spy, "the threshold subset was never used"


def test_calibration_study_compares_all_three_methods(
    train_test_csvs: tuple[Path, Path], tmp_path: Path
) -> None:
    train_csv, test_csv = train_test_csvs

    frame = run_calibration_study(
        train_csv, test_csv, tmp_path / "a", family="logistic", profile="quick"
    )

    assert list(frame["calibration"]) == ["none", "sigmoid", "isotonic"]
    assert frame["total_cost"].notna().all()


def test_reported_cost_matches_the_documented_objective(
    train_test_csvs: tuple[Path, Path], tmp_path: Path
) -> None:
    """total_cost must equal 10*FP + 500*FN, not some other weighting."""

    train_csv, test_csv = train_test_csvs

    frame = run_model_family_study(
        train_csv, test_csv, tmp_path / "a", families=["logistic"], profile="quick"
    )

    row = frame.iloc[0]
    expected = 10 * row["false_positives"] + 500 * row["false_negatives"]
    assert row["total_cost"] == pytest.approx(expected)


def test_empty_family_list_produces_an_empty_comparison(
    train_test_csvs: tuple[Path, Path], tmp_path: Path
) -> None:
    train_csv, test_csv = train_test_csvs
    with pytest.raises((ValueError, KeyError)):
        run_model_family_study(train_csv, test_csv, tmp_path / "a", families=[], profile="quick")


@pytest.mark.slow
def test_feature_selection_study_runs(train_test_csvs: tuple[Path, Path], tmp_path: Path) -> None:
    train_csv, test_csv = train_test_csvs
    frame = studies.run_feature_selection_study(train_csv, test_csv, tmp_path / "a")
    assert not frame.empty
    assert "total_cost" in frame.columns


@pytest.mark.slow
def test_imbalance_study_runs(train_test_csvs: tuple[Path, Path], tmp_path: Path) -> None:
    # The study compares sampling against focal loss, so it needs both groups.
    pytest.importorskip("imblearn", reason="requires the optional 'imbalance' group")
    pytest.importorskip("torch", reason="the focal-loss arm requires the 'neural' group")
    train_csv, test_csv = train_test_csvs
    frame = studies.run_imbalance_study(train_csv, test_csv, tmp_path / "a")
    assert not frame.empty


@pytest.mark.slow
def test_xgboost_ablation_runs(train_test_csvs: tuple[Path, Path], tmp_path: Path) -> None:
    pytest.importorskip("xgboost", reason="requires the optional 'boost' group")
    train_csv, test_csv = train_test_csvs
    frame = studies.run_xgboost_ablation(train_csv, test_csv, tmp_path / "a")
    assert not frame.empty
    assert "ablation" in frame.columns
