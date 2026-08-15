"""Shared fixtures.

The real UCI archive is 60,000 rows fetched over the network, so it cannot be a
test dependency. These fixtures write files with the same *shape* as the
official CSVs: a descriptive preamble ahead of the header, a ``pos``/``neg``
class column, numeric features and ``na`` missing markers.

Every row carries a unique ``row_id`` value. It is an ordinary feature as far as
the parser is concerned, but it lets a test fingerprint exactly which rows an
estimator was fitted on, which is how the leakage checks in
``test_studies.py`` work.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

PREAMBLE = [
    "The dataset consists of data collected from heavy Scania trucks in everyday usage.",
    "The datasets' positive class consists of component failures for the APS system.",
    "",
    "Synthetic stand-in used by the test suite. Not the official data.",
    "",
]


def write_aps_csv(
    path: Path,
    *,
    n_rows: int,
    n_features: int = 6,
    positive_rate: float = 0.3,
    na_fraction: float = 0.08,
    id_offset: int = 0,
    seed: int = 0,
) -> Path:
    """Write one APS-shaped CSV and return its path.

    ``id_offset`` shifts the ``row_id`` fingerprints so a train and a test file
    can be generated with disjoint identifiers.
    """

    rng = np.random.default_rng(seed)
    names = [f"a{i:02d}_000" for i in range(n_features)]

    n_positive = max(2, int(round(n_rows * positive_rate)))
    labels = np.array(["neg"] * n_rows, dtype=object)
    labels[rng.choice(n_rows, size=n_positive, replace=False)] = "pos"

    # Give the positive class a real signal so fitted models are not degenerate.
    values = rng.normal(size=(n_rows, n_features)) * 10.0
    values[labels == "pos"] += 6.0

    mask = rng.random((n_rows, n_features)) < na_fraction
    lines = [*PREAMBLE, "class,row_id," + ",".join(names)]
    for i in range(n_rows):
        cells = ["na" if mask[i, j] else f"{values[i, j]:.6f}" for j in range(n_features)]
        lines.append(f"{labels[i]},{id_offset + i}," + ",".join(cells))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def aps_csv_factory(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory writing APS-shaped CSVs into a temporary directory."""

    def factory(name: str, **kwargs: object) -> Path:
        return write_aps_csv(tmp_path / name, **kwargs)  # type: ignore[arg-type]

    return factory


@pytest.fixture
def train_test_csvs(aps_csv_factory: Callable[..., Path]) -> tuple[Path, Path]:
    """A train/test pair whose ``row_id`` fingerprints do not overlap."""

    train = aps_csv_factory("aps_failure_training_set.csv", n_rows=220, seed=1, id_offset=0)
    test = aps_csv_factory("aps_failure_test_set.csv", n_rows=80, seed=2, id_offset=100_000)
    return train, test
