import pytest

from scania_aps.models.logistic import LogisticConfig, build_logistic_pipeline


def test_elastic_net_requires_valid_mixing_parameter() -> None:
    with pytest.raises(ValueError):
        build_logistic_pipeline(LogisticConfig(penalty="elasticnet", C=1.0, l1_ratio=None))


def test_l2_pipeline_builds() -> None:
    pipeline = build_logistic_pipeline(LogisticConfig(penalty="l2", C=0.1))
    assert pipeline.named_steps["model"].C == 0.1


def test_penalty_maps_onto_the_l1_ratio_api() -> None:
    """scikit-learn removes `penalty` in 1.10; the family must ride on l1_ratio.

    l1_ratio 0 is pure L2, 1 is pure L1, and anything between is elastic net.
    """

    cases = {"l2": 0.0, "l1": 1.0}
    for penalty, expected in cases.items():
        pipeline = build_logistic_pipeline(LogisticConfig(penalty=penalty))  # type: ignore[arg-type]
        model = pipeline.named_steps["model"]
        assert model.l1_ratio == expected
        assert getattr(model, "penalty", None) in (None, "deprecated"), (
            "the deprecated penalty argument must not be passed"
        )

    elastic = build_logistic_pipeline(
        LogisticConfig(penalty="elasticnet", l1_ratio=0.3)
    ).named_steps["model"]
    assert elastic.l1_ratio == 0.3


def test_fitting_emits_no_deprecation_warnings() -> None:
    """A published notebook should not be full of library deprecation noise."""

    import warnings

    import numpy as np
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300, n_features=12, n_informative=5, weights=[0.85, 0.15], random_state=0
    )
    X = X.astype(np.float64)

    for penalty, ratio in (("l2", None), ("l1", None), ("elasticnet", 0.5)):
        config = LogisticConfig(penalty=penalty, C=0.5, l1_ratio=ratio, max_iter=200)  # type: ignore[arg-type]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_logistic_pipeline(config).fit(X, y)

        offenders = [
            f"{w.category.__name__}: {w.message}"
            for w in caught
            if issubclass(w.category, FutureWarning | DeprecationWarning)
            or "Inconsistent values" in str(w.message)
        ]
        assert not offenders, f"{penalty}: {offenders}"


def test_solver_is_chosen_per_penalty() -> None:
    """saga covers all three penalties but does not converge on this data.

    Each penalty gets the fastest solver that reaches an optimum. Elastic net
    has no alternative, so it keeps saga.
    """

    expected = {"l2": "lbfgs", "l1": "liblinear", "elasticnet": "saga"}
    for penalty, solver in expected.items():
        ratio = 0.5 if penalty == "elasticnet" else None
        config = LogisticConfig(penalty=penalty, l1_ratio=ratio)  # type: ignore[arg-type]
        assert build_logistic_pipeline(config).named_steps["model"].solver == solver


def test_pure_penalties_converge() -> None:
    """A capped fit is not the model that was requested, so it must converge."""

    import warnings

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.exceptions import ConvergenceWarning

    X, y = make_classification(
        n_samples=1500, n_features=60, n_informative=8, weights=[0.9, 0.1], random_state=0
    )

    for penalty in ("l1", "l2"):
        config = LogisticConfig(penalty=penalty, C=1.0, max_iter=4000)  # type: ignore[arg-type]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = build_logistic_pipeline(config).fit(X, y)

        stalled = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        assert not stalled, f"{penalty} did not converge: {[str(w.message) for w in stalled]}"

        model = pipeline.named_steps["model"]
        assert int(np.max(model.n_iter_)) < config.max_iter
