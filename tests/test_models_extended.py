import numpy as np
from sklearn.datasets import make_classification

from scania_aps.model_search import candidates_for_family
from scania_aps.models.autoencoder import AutoencoderLogisticClassifier
from scania_aps.models.factory import build_candidate
from scania_aps.models.mlp import TorchMLPClassifier
from scania_aps.models.svm import LinearSVMConfig, build_linear_svm_pipeline
from scania_aps.models.trees import TreeEnsembleConfig, build_tree_pipeline


def test_svm_and_tree_builders_expose_expected_estimators() -> None:
    svm = build_linear_svm_pipeline(LinearSVMConfig(C=0.25))
    forest = build_tree_pipeline(TreeEnsembleConfig(kind="random_forest", n_estimators=10))
    extra = build_tree_pipeline(TreeEnsembleConfig(kind="extra_trees", n_estimators=10))

    assert svm.named_steps["model"].C == 0.25
    assert forest.named_steps["model"].n_estimators == 10
    assert extra.named_steps["model"].n_estimators == 10


def test_all_model_families_have_quick_candidates() -> None:
    families = [
        "logistic",
        "linear_svm",
        "random_forest",
        "extra_trees",
        "xgboost",
        "lightgbm",
        "mlp",
        "autoencoder",
    ]
    for family in families:
        candidates = candidates_for_family(family, profile="quick")
        assert len(candidates) == 3
        model = build_candidate(candidates[0])
        assert model is not None


def test_torch_models_fit_small_dataset() -> None:
    X, y = make_classification(
        n_samples=160,
        n_features=12,
        n_informative=6,
        weights=[0.8, 0.2],
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    mlp = TorchMLPClassifier(
        hidden_dims=(16,),
        batch_norm=False,
        dropout=0.1,
        max_epochs=3,
        patience=2,
        batch_size=64,
        scheduler="none",
    ).fit(X, y)
    assert mlp.predict_proba(X[:5]).shape == (5, 2)

    autoencoder = AutoencoderLogisticClassifier(
        latent_dim=4,
        hidden_dim=8,
        max_epochs=2,
        batch_size=64,
    ).fit(X, y)
    assert autoencoder.predict_proba(X[:5]).shape == (5, 2)
