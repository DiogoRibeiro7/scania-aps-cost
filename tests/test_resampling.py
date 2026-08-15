import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from scania_aps.resampling import build_resampled_pipeline

pytest.importorskip("imblearn", reason="requires the optional 'imbalance' dependency group")


@pytest.fixture
def imbalanced_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=list("abcde"))
    y = pd.Series(np.r_[np.zeros(270, dtype=int), np.ones(30, dtype=int)])
    X.loc[y == 1] += 2.0
    X.iloc[0, 0] = np.nan  # the pipeline must impute before sampling
    return X, y


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="strategy"):
        build_resampled_pipeline(LogisticRegression(), strategy="oversample")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("strategy", "expects_sampler"),
    [("none", False), ("undersample", True), ("smote", True)],
)
def test_sampler_present_only_for_resampling_strategies(
    strategy: str, expects_sampler: bool
) -> None:
    pipeline = build_resampled_pipeline(LogisticRegression(), strategy=strategy)  # type: ignore[arg-type]
    assert ("sampler" in pipeline.named_steps) is expects_sampler
    assert "imputer" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_scaling_can_be_disabled() -> None:
    assert "scaler" in build_resampled_pipeline(LogisticRegression(), strategy="none").named_steps
    assert (
        "scaler"
        not in build_resampled_pipeline(
            LogisticRegression(), strategy="none", scale=False
        ).named_steps
    )


@pytest.mark.parametrize("strategy", ["none", "undersample", "smote"])
def test_resampling_never_changes_the_prediction_row_count(
    strategy: str, imbalanced_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Sampling happens inside fit only.

    imbalanced-learn pipelines skip the sampler at predict time. If that ever
    stopped holding, evaluation sets would be silently resampled and every
    reported cost would be wrong.
    """

    X, y = imbalanced_data
    pipeline = build_resampled_pipeline(LogisticRegression(max_iter=500), strategy=strategy)  # type: ignore[arg-type]
    pipeline.fit(X, y)

    assert len(pipeline.predict(X)) == len(X)
    assert pipeline.predict_proba(X).shape == (len(X), 2)
