import numpy as np
import pandas as pd

from scania_aps.split import development_split


def test_development_split_is_disjoint_and_complete() -> None:
    X = pd.DataFrame({"x": np.arange(200), "z": np.arange(200) ** 2})
    y = pd.Series(([0] * 180) + ([1] * 20), dtype="int8")

    result = development_split(X, y)
    indices = set(result.X_fit.index) | set(result.X_tune.index) | set(result.X_threshold.index)

    assert len(indices) == len(X)
    assert set(result.X_fit.index).isdisjoint(result.X_tune.index)
    assert set(result.X_fit.index).isdisjoint(result.X_threshold.index)
    assert set(result.X_tune.index).isdisjoint(result.X_threshold.index)
