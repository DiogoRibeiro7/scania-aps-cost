"""Dataset acquisition and parsing utilities.

The raw UCI files contain a descriptive preamble before the CSV header.  The
parser locates the header dynamically rather than assuming a fixed number of
metadata lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

UCI_ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip"
)
TRAIN_FILENAME = "aps_failure_training_set.csv"
TEST_FILENAME = "aps_failure_test_set.csv"


@dataclass(frozen=True)
class ScaniaDataset:
    """Container for features and binary targets."""

    X: pd.DataFrame
    y: pd.Series


def download_dataset(raw_dir: Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Download and extract the official UCI archive.

    Parameters
    ----------
    raw_dir:
        Directory in which the raw CSV files will be stored.
    overwrite:
        Download again even when both expected CSV files already exist.

    Returns
    -------
    tuple[Path, Path]
        Paths to the training and test CSV files.
    """

    if not isinstance(raw_dir, Path):
        raise TypeError("raw_dir must be a pathlib.Path")

    raw_dir.mkdir(parents=True, exist_ok=True)
    train_path = raw_dir / TRAIN_FILENAME
    test_path = raw_dir / TEST_FILENAME

    if not overwrite and train_path.exists() and test_path.exists():
        return train_path, test_path

    archive_path = raw_dir / "scania_aps.zip"
    with urlopen(UCI_ARCHIVE_URL, timeout=120) as response:  # noqa: S310
        archive_path.write_bytes(response.read())

    with ZipFile(archive_path) as archive:
        archive.extractall(raw_dir)

    archive_path.unlink(missing_ok=True)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("The UCI archive did not contain the expected Scania CSV files.")

    return train_path, test_path


def _header_row(path: Path) -> int:
    """Return the zero-based row containing the CSV header."""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            stripped = line.strip().lower()
            if stripped.startswith("class,") or stripped.startswith('"class",'):
                return line_number
    raise ValueError(f"Could not locate the CSV header in {path}")


def read_raw_csv(path: Path) -> ScaniaDataset:
    """Read one original Scania APS CSV file.

    The target is converted from ``pos`` / ``neg`` to integer 1 / 0. All
    feature columns are coerced to floating point; ``na`` becomes ``NaN``.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    header_row = _header_row(path)
    frame = pd.read_csv(path, skiprows=header_row, na_values=["na", "NA"])

    if "class" not in frame.columns:
        raise ValueError("Expected a 'class' target column.")

    raw_target = frame.pop("class").astype(str).str.strip().str.lower()
    invalid = sorted(set(raw_target.unique()) - {"pos", "neg"})
    if invalid:
        raise ValueError(f"Unexpected target labels: {invalid}")

    target = raw_target.map({"neg": 0, "pos": 1}).astype("int8")
    features = frame.apply(pd.to_numeric, errors="coerce")

    if features.shape[1] == 0:
        raise ValueError("No feature columns were parsed.")
    if len(features) != len(target):
        raise ValueError("Feature/target row counts are inconsistent.")

    return ScaniaDataset(X=features, y=target)
