"""Model builders and sklearn-compatible estimators."""

from scania_aps.models.autoencoder import AutoencoderLogisticClassifier
from scania_aps.models.factory import ModelCandidate, ModelFamily, build_candidate
from scania_aps.models.logistic import LogisticConfig, build_logistic_pipeline
from scania_aps.models.mlp import TorchMLPClassifier
from scania_aps.models.svm import LinearSVMConfig, build_linear_svm_pipeline
from scania_aps.models.trees import TreeEnsembleConfig, build_tree_pipeline

__all__ = [
    "AutoencoderLogisticClassifier",
    "LinearSVMConfig",
    "LogisticConfig",
    "ModelCandidate",
    "ModelFamily",
    "TorchMLPClassifier",
    "TreeEnsembleConfig",
    "build_candidate",
    "build_linear_svm_pipeline",
    "build_logistic_pipeline",
    "build_tree_pipeline",
]
