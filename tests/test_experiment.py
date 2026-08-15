from __future__ import annotations

import json
from pathlib import Path

import pytest

from scania_aps.experiment import run_boosted_experiment, run_logistic_experiment


def test_logistic_experiment_evaluates_once_and_persists_its_result(
    train_test_csvs: tuple[Path, Path], tmp_path: Path
) -> None:
    train_csv, test_csv = train_test_csvs
    artifacts = tmp_path / "artifacts"

    evaluation = run_logistic_experiment(train_csv, test_csv, artifacts, n_trials=3)

    assert evaluation.total_cost >= 0
    assert evaluation.false_negatives >= 0
    assert evaluation.false_positives >= 0
    assert 0.0 <= evaluation.pr_auc <= 1.0

    written = list(artifacts.rglob("*.json"))
    assert written, "the experiment should persist a machine-readable result"
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["evaluation"]["total_cost"] == pytest.approx(evaluation.total_cost)
    assert "best_config" in payload, "the winning hyperparameters must be recoverable"


def test_logistic_experiment_reports_the_documented_cost(
    train_test_csvs: tuple[Path, Path], tmp_path: Path
) -> None:
    train_csv, test_csv = train_test_csvs

    evaluation = run_logistic_experiment(train_csv, test_csv, tmp_path / "a", n_trials=2)

    expected = 10 * evaluation.false_positives + 500 * evaluation.false_negatives
    assert evaluation.total_cost == pytest.approx(expected)


@pytest.mark.slow
def test_boosted_experiment_runs(train_test_csvs: tuple[Path, Path], tmp_path: Path) -> None:
    pytest.importorskip("xgboost", reason="requires the optional 'boost' group")
    train_csv, test_csv = train_test_csvs

    evaluation = run_boosted_experiment(train_csv, test_csv, tmp_path / "a")

    assert evaluation.total_cost >= 0
    assert 0.0 <= evaluation.pr_auc <= 1.0
