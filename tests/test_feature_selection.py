import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scania_aps.feature_selection import (
    RankedFeature,
    l1_nonzero_features,
    mutual_information_ranking,
    permutation_ranking,
    tree_importance_ranking,
)


@pytest.fixture
def signal_data() -> tuple[pd.DataFrame, pd.Series]:
    """Two informative columns and three pure noise columns."""

    rng = np.random.default_rng(0)
    n = 300
    informative = rng.normal(size=(n, 2))
    noise = rng.normal(size=(n, 3))
    y = pd.Series(((informative.sum(axis=1) + rng.normal(scale=0.3, size=n)) > 0).astype(int))
    X = pd.DataFrame(
        np.hstack([informative, noise]),
        columns=["signal_a", "signal_b", "noise_a", "noise_b", "noise_c"],
    )
    return X, y


def _l1_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(penalty="l1", solver="liblinear", C=0.5)),
        ]
    )
    return pipeline.fit(X, y)


def test_mutual_information_ranks_signal_above_noise(
    signal_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = signal_data
    ranking = mutual_information_ranking(X, y)

    assert {r.feature for r in ranking[:2]} == {"signal_a", "signal_b"}
    assert all(isinstance(r, RankedFeature) for r in ranking)
    assert ranking == sorted(ranking, key=lambda r: r.score, reverse=True)


def test_permutation_ranking_orders_by_importance(
    signal_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = signal_data
    model = _l1_pipeline(X, y)

    ranking = permutation_ranking(model, X, y, n_repeats=3)

    assert len(ranking) == X.shape[1]
    assert ranking == sorted(ranking, key=lambda r: r.score, reverse=True)
    assert {r.feature for r in ranking[:2]} == {"signal_a", "signal_b"}


def test_permutation_ranking_rejects_non_positive_repeats(
    signal_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = signal_data
    with pytest.raises(ValueError, match="n_repeats"):
        permutation_ranking(_l1_pipeline(X, y), X, y, n_repeats=0)


def test_l1_selection_returns_only_non_zero_coefficients(
    signal_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = signal_data
    ranking = l1_nonzero_features(_l1_pipeline(X, y))

    assert ranking, "L1 should retain at least one feature on separable data"
    assert all(r.score != 0.0 for r in ranking)
    assert ranking == sorted(ranking, key=lambda r: abs(r.score), reverse=True)


def test_l1_selection_names_missingness_indicators(
    signal_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """add_indicator appends columns; they must be labelled, not left anonymous."""

    X, y = signal_data
    X = X.copy()
    X.loc[X.index[:40], "noise_a"] = np.nan

    names = [r.feature for r in l1_nonzero_features(_l1_pipeline(X, y))]

    assert not any(name.startswith("feature_") for name in names), (
        "fell back to synthetic names, so indicator naming broke"
    )


def test_tree_importance_ranking(signal_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = signal_data
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=25, random_state=0)),
        ]
    ).fit(X, y)

    ranking = tree_importance_ranking(pipeline)

    assert len(ranking) == X.shape[1]
    assert ranking == sorted(ranking, key=lambda r: r.score, reverse=True)


@pytest.mark.parametrize(
    "extractor", [l1_nonzero_features, tree_importance_ranking], ids=["l1", "tree"]
)
def test_extractors_reject_bare_estimators(extractor: object) -> None:
    with pytest.raises(TypeError):
        extractor(LogisticRegression())  # type: ignore[operator]


def test_tree_extractor_rejects_a_pipeline_without_importances(
    signal_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = signal_data
    with pytest.raises(TypeError, match="importances"):
        tree_importance_ranking(_l1_pipeline(X, y))
