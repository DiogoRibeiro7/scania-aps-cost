"""Persist experiment results with path-aware operational errors."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from dataexcept import FileWriteError, wrapping


def create_artifact_dir(path: Path) -> None:
    """Create an artifact directory, reporting filesystem failures with its path."""

    with wrapping(OSError, FileWriteError, path=str(path)):
        path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    """Save UTF-8 JSON text without hiding serialization errors."""

    with wrapping((OSError, UnicodeError), FileWriteError, path=str(path)):
        path.write_text(content, encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Save a study comparison as CSV."""

    with wrapping(OSError, FileWriteError, path=str(path)):
        frame.to_csv(path, index=False)


def dump_model(model: object, path: Path) -> None:
    """Persist a fitted estimator; keep model serialization errors distinct."""

    with wrapping(OSError, FileWriteError, path=str(path)):
        joblib.dump(model, path)
