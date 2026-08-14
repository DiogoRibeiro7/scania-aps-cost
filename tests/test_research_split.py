import numpy as np
import pandas as pd

from scania_aps.split import research_split


def test_research_split_is_disjoint_and_complete() -> None:
    X = pd.DataFrame({"x": np.arange(400), "z": np.arange(400) ** 2})
    y = pd.Series(([0] * 360) + ([1] * 40), dtype="int8")

    result = research_split(X, y)
    parts = [
        result.X_fit,
        result.X_tune,
        result.X_calibration,
        result.X_threshold,
    ]
    all_indices = set().union(*(set(part.index) for part in parts))

    assert len(all_indices) == len(X)
    for i, left in enumerate(parts):
        for right in parts[i + 1 :]:
            assert set(left.index).isdisjoint(right.index)
