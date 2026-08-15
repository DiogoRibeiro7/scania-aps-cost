"""Tests for the runtime dataset download.

SECURITY.md puts this code path in scope, so it is exercised here with a local
fake archive rather than left to a live network call.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from scania_aps.data import TEST_FILENAME, TRAIN_FILENAME, download_dataset


def _archive_bytes(names: tuple[str, ...] = (TRAIN_FILENAME, TEST_FILENAME)) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, "preamble\nclass,aa_000\nneg,1\npos,2\n")
    return buffer.getvalue()


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@pytest.fixture
def fake_archive(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve a local zip instead of contacting the UCI archive."""

    requested: list[str] = []

    def fake_urlopen(url: str, **_: object) -> _FakeResponse:
        requested.append(url)
        return _FakeResponse(_archive_bytes())

    monkeypatch.setattr("scania_aps.data.urlopen", fake_urlopen)
    return requested


def test_download_extracts_both_official_files(tmp_path: Path, fake_archive: list[str]) -> None:
    train, test = download_dataset(tmp_path)

    assert train.exists() and test.exists()
    assert train.name == TRAIN_FILENAME
    assert test.name == TEST_FILENAME
    assert len(fake_archive) == 1
    assert fake_archive[0].startswith("https://")


def test_the_zip_is_not_left_behind(tmp_path: Path, fake_archive: list[str]) -> None:
    download_dataset(tmp_path)
    assert not list(tmp_path.glob("*.zip"))


def test_existing_files_short_circuit_the_download(tmp_path: Path, fake_archive: list[str]) -> None:
    download_dataset(tmp_path)
    assert len(fake_archive) == 1

    download_dataset(tmp_path)

    assert len(fake_archive) == 1, "a second call re-downloaded already-present data"


def test_overwrite_forces_a_fresh_download(tmp_path: Path, fake_archive: list[str]) -> None:
    download_dataset(tmp_path)
    download_dataset(tmp_path, overwrite=True)

    assert len(fake_archive) == 2


def test_an_archive_missing_the_expected_files_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(url: str, **_: object) -> _FakeResponse:
        return _FakeResponse(_archive_bytes(names=("something_else.csv",)))

    monkeypatch.setattr("scania_aps.data.urlopen", fake_urlopen)

    with pytest.raises(FileNotFoundError, match="expected"):
        download_dataset(tmp_path)


def test_raw_dir_must_be_a_path(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Path"):
        download_dataset(str(tmp_path))  # type: ignore[arg-type]


def test_the_target_directory_is_created(tmp_path: Path, fake_archive: list[str]) -> None:
    nested = tmp_path / "data" / "raw"
    assert not nested.exists()

    download_dataset(nested)

    assert nested.is_dir()
