from pathlib import Path

import pandas as pd
import pytest
from dataexcept import DataLoadingError, FileReadError

from scania_aps.data import read_raw_csv


def test_parser_finds_header_and_converts_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "mini.csv"
    path.write_text(
        "metadata line\nanother metadata line\nclass,aa_000,ab_000\nneg,1,na\npos,2,3\n",
        encoding="utf-8",
    )

    dataset = read_raw_csv(path)

    assert dataset.X.shape == (2, 2)
    assert dataset.y.tolist() == [0, 1]
    assert dataset.X["ab_000"].isna().sum() == 1


def test_header_is_located_regardless_of_preamble_length(tmp_path: Path) -> None:
    """The official files vary in preamble length, so it must not be hard-coded."""

    path = tmp_path / "long_preamble.csv"
    preamble = "\n".join(f"comment line {i}" for i in range(37))
    path.write_text(
        f"{preamble}\nclass,aa_000\nneg,1\npos,2\n",
        encoding="utf-8",
    )

    assert read_raw_csv(path).y.tolist() == [0, 1]


def test_quoted_header_is_recognised(tmp_path: Path) -> None:
    path = tmp_path / "quoted.csv"
    path.write_text('preamble\n"class","aa_000"\nneg,1\npos,2\n', encoding="utf-8")

    dataset = read_raw_csv(path)

    assert dataset.y.tolist() == [0, 1]
    assert list(dataset.X.columns) == ["aa_000"]


def test_labels_are_case_and_whitespace_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "messy_labels.csv"
    path.write_text("preamble\nclass,aa_000\n NEG ,1\nPos,2\n", encoding="utf-8")

    assert read_raw_csv(path).y.tolist() == [0, 1]


def test_target_is_a_compact_integer_type(tmp_path: Path) -> None:
    """60,000 rows of int8 rather than object matters for memory and for sklearn."""

    path = tmp_path / "dtype.csv"
    path.write_text("preamble\nclass,aa_000\nneg,1\npos,2\n", encoding="utf-8")

    assert read_raw_csv(path).y.dtype == "int8"


def test_non_numeric_feature_values_become_missing(tmp_path: Path) -> None:
    path = tmp_path / "junk.csv"
    path.write_text("preamble\nclass,aa_000\nneg,not_a_number\npos,2\n", encoding="utf-8")

    assert read_raw_csv(path).X["aa_000"].isna().sum() == 1


def test_missing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_raw_csv(tmp_path / "absent.csv")


def test_csv_read_failure_keeps_its_path_and_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "broken.csv"
    path.write_text("class,aa_000\nneg,1\n", encoding="utf-8")
    cause = pd.errors.ParserError("malformed CSV")

    def fail_read(*args: object, **kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(pd, "read_csv", fail_read)
    with pytest.raises(DataLoadingError) as caught:
        read_raw_csv(path)

    assert caught.value.source == str(path)
    assert caught.value.original is cause
    assert caught.value.__cause__ is cause


def test_csv_header_read_failure_keeps_its_path_and_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unreadable.csv"
    path.touch()
    cause = PermissionError("permission denied")

    def fail_open(self: Path, *args: object, **kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(FileReadError) as caught:
        read_raw_csv(path)

    assert caught.value.path == str(path)
    assert caught.value.original is cause
    assert caught.value.__cause__ is cause


def test_a_file_without_a_header_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "headerless.csv"
    path.write_text("just,some,columns\n1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="header"):
        read_raw_csv(path)


def test_unexpected_labels_are_rejected(tmp_path: Path) -> None:
    """A silently mismapped label would corrupt every cost this repository reports."""

    path = tmp_path / "bad_labels.csv"
    path.write_text("preamble\nclass,aa_000\nneg,1\nmaybe,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="maybe"):
        read_raw_csv(path)


def test_a_file_with_no_feature_columns_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "target_only.csv"
    path.write_text("preamble\nclass\nneg\npos\n", encoding="utf-8")

    with pytest.raises(ValueError, match="feature"):
        read_raw_csv(path)
