from __future__ import annotations

from pathlib import Path

import pytest

from scania_aps.cli import ALL_FAMILIES, _paths, build_parser, main


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-everything"])


@pytest.mark.parametrize(
    "argv",
    [
        ["download"],
        ["train-logistic"],
        ["train-boosted"],
        ["run-study"],
        ["study-calibration"],
        ["study-imbalance"],
        ["study-features"],
        ["study-ablation"],
    ],
    ids=lambda a: a[0],
)
def test_every_subcommand_parses(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]


def test_defaults_match_the_documented_behaviour() -> None:
    args = build_parser().parse_args(["run-study"])

    assert args.profile == "quick"
    assert args.calibration == "none"
    assert list(args.models) == list(ALL_FAMILIES)


def test_model_selection_accepts_a_subset() -> None:
    args = build_parser().parse_args(["run-study", "--models", "logistic", "xgboost"])
    assert args.models == ["logistic", "xgboost"]


def test_unknown_model_family_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-study", "--models", "transformer"])


@pytest.mark.parametrize("flag", ["--profile", "--calibration"])
def test_invalid_choices_are_rejected(flag: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-study", flag, "nonsense"])


def test_trials_is_parsed_as_an_integer() -> None:
    assert build_parser().parse_args(["train-logistic", "--trials", "7"]).trials == 7


def test_paths_are_derived_from_the_root() -> None:
    train, test, artifacts = _paths(Path("/repo"))

    assert train.parent == Path("/repo/data/raw")
    assert test.parent == Path("/repo/data/raw")
    assert artifacts == Path("/repo/artifacts")
    assert train != test


def test_missing_dataset_produces_an_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A study invoked before `download` should say so, not fail deep in pandas."""

    monkeypatch.setattr("sys.argv", ["scania-aps", "--root", str(tmp_path), "run-study"])

    with pytest.raises(FileNotFoundError, match="download"):
        main()


def test_download_command_is_dispatched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`download` must run before the dataset exists, unlike every other command."""

    called: list[Path] = []

    def fake_download(raw_dir: Path, **_: object) -> tuple[Path, Path]:
        called.append(raw_dir)
        return raw_dir / "train.csv", raw_dir / "test.csv"

    monkeypatch.setattr("scania_aps.cli.download_dataset", fake_download)
    monkeypatch.setattr("sys.argv", ["scania-aps", "--root", str(tmp_path), "download"])

    main()

    assert called == [tmp_path / "data" / "raw"]


@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("run-study", "run_model_family_study"),
        ("study-calibration", "run_calibration_study"),
        ("study-imbalance", "run_imbalance_study"),
        ("study-features", "run_feature_selection_study"),
        ("study-ablation", "run_xgboost_ablation"),
        ("train-logistic", "run_logistic_experiment"),
        ("train-boosted", "run_boosted_experiment"),
    ],
)
def test_each_command_dispatches_to_its_runner(
    command: str,
    target: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every subcommand must reach the right runner, with the right paths."""

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for name in ("aps_failure_training_set.csv", "aps_failure_test_set.csv"):
        (raw / name).write_text("preamble\nclass,aa_000\nneg,1\npos,2\n", encoding="utf-8")

    calls: list[tuple[Path, Path, Path]] = []

    class _Result:
        total_cost = 1.0
        false_negatives = 0
        false_positives = 0
        pr_auc = 0.5
        threshold = 0.02

        def to_string(self, **_: object) -> str:
            return "table"

    def fake_runner(train: Path, test: Path, artifacts: Path, **_: object) -> _Result:
        calls.append((train, test, artifacts))
        return _Result()

    monkeypatch.setattr(f"scania_aps.cli.{target}", fake_runner)
    monkeypatch.setattr("sys.argv", ["scania-aps", "--root", str(tmp_path), command])

    main()

    assert len(calls) == 1
    train, test, artifacts = calls[0]
    assert train.parent == raw
    assert artifacts == tmp_path / "artifacts"
    assert capsys.readouterr().out.strip(), "the command produced no output"
