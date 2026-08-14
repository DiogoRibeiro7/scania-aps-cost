import pytest

from scania_aps.models.logistic import LogisticConfig, build_logistic_pipeline


def test_elastic_net_requires_valid_mixing_parameter() -> None:
    with pytest.raises(ValueError):
        build_logistic_pipeline(LogisticConfig(penalty="elasticnet", C=1.0, l1_ratio=None))


def test_l2_pipeline_builds() -> None:
    pipeline = build_logistic_pipeline(LogisticConfig(penalty="l2", C=0.1))
    assert pipeline.named_steps["model"].C == 0.1
