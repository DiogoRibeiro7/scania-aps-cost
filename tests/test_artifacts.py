"""Artifact writes retain their failed path and underlying filesystem error."""

from pathlib import Path

import pandas as pd
import pytest
from dataexcept import FileWriteError

from scania_aps.artifacts import create_artifact_dir, dump_model, write_csv, write_text


def test_text_write_failure_preserves_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "results.json"
    cause = PermissionError("read-only directory")

    def fail_write(self: Path, *args: object, **kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(FileWriteError) as caught:
        write_text(path, "{}")

    assert caught.value.path == str(path)
    assert caught.value.original is cause
    assert caught.value.__cause__ is cause


def test_csv_write_failure_preserves_cause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "comparison.csv"
    frame = pd.DataFrame({"cost": [10]})
    cause = OSError("disk full")

    def fail_csv(*args: object, **kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(frame, "to_csv", fail_csv)
    with pytest.raises(FileWriteError) as caught:
        write_csv(frame, path)

    assert caught.value.path == str(path)
    assert caught.value.__cause__ is cause


def test_model_write_failure_preserves_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.joblib"
    cause = OSError("disk full")

    def fail_dump(*args: object, **kwargs: object) -> None:
        raise cause

    monkeypatch.setattr("scania_aps.artifacts.joblib.dump", fail_dump)
    with pytest.raises(FileWriteError) as caught:
        dump_model(object(), path)

    assert caught.value.path == str(path)
    assert caught.value.__cause__ is cause


def test_directory_creation_failure_preserves_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifacts"
    cause = PermissionError("read-only parent")

    def fail_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    with pytest.raises(FileWriteError) as caught:
        create_artifact_dir(path)

    assert caught.value.path == str(path)
    assert caught.value.__cause__ is cause
