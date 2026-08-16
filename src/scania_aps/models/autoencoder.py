"""Autoencoder representation learning followed by a sparse linear classifier."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class AutoencoderLogisticClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Learn a nonlinear latent representation, then classify in latent space."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 512,
        max_epochs: int = 50,
        classifier_C: float = 1.0,
        positive_class_weight: float | None = None,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.classifier_C = classifier_C
        self.positive_class_weight = positive_class_weight
        self.random_state = random_state
        self.device = device

    def fit(self, X: NDArray[np.floating], y: NDArray[np.integer]) -> AutoencoderLogisticClassifier:
        """Pretrain the autoencoder and fit logistic regression on its latent codes."""

        X_checked, y_checked = check_X_y(X, y, dtype=np.float32, ensure_all_finite=True)
        if self.latent_dim <= 0 or self.hidden_dim <= 0 or self.max_epochs <= 0:
            raise ValueError("Network dimensions and max_epochs must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.classifier_C <= 0:
            raise ValueError("Invalid optimization or classifier regularization value.")

        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Install the optional 'neural' dependency group to use PyTorch."
            ) from exc

        torch.manual_seed(self.random_state)
        if self.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device)
        self.device_ = str(device)

        input_dim = int(X_checked.shape[1])
        encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, input_dim),
        )
        autoencoder = nn.Sequential(encoder, decoder).to(device)
        optimizer = torch.optim.AdamW(
            autoencoder.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(np.asarray(X_checked, dtype=np.float32))),
            batch_size=self.batch_size,
            shuffle=True,
        )
        history: list[float] = []
        for _ in range(self.max_epochs):
            losses: list[float] = []
            autoencoder.train()
            for (features,) in loader:
                features = features.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstructed = autoencoder(features)
                loss = loss_fn(reconstructed, features)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            history.append(float(np.mean(losses)))

        encoder = encoder.to(device)
        encoder.eval()
        with torch.no_grad():
            latent = encoder(torch.from_numpy(X_checked).to(device)).cpu().numpy()

        class_weight = (
            None
            if self.positive_class_weight is None
            else {0: 1.0, 1: float(self.positive_class_weight)}
        )
        classifier = LogisticRegression(
            C=self.classifier_C,
            l1_ratio=0.0,  # pure L2; `penalty` is removed in scikit-learn 1.10
            class_weight=class_weight,
            solver="lbfgs",
            max_iter=3000,
            random_state=self.random_state,
        ).fit(latent, y_checked)

        self.encoder_ = encoder
        self.classifier_ = classifier
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_in_ = input_dim
        self.reconstruction_history_ = history
        return self

    def _transform(self, X: NDArray[np.floating]) -> NDArray[np.float32]:
        check_is_fitted(self, attributes=["encoder_", "classifier_", "n_features_in_"])
        X_checked = check_array(X, dtype=np.float32, ensure_all_finite=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training data.")
        import torch

        device = torch.device(self.device_)
        self.encoder_.eval()
        with torch.no_grad():
            latent = self.encoder_(torch.from_numpy(X_checked).to(device)).cpu().numpy()
        return np.asarray(latent, dtype=np.float32)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return probabilities from the latent-space classifier."""

        return np.asarray(self.classifier_.predict_proba(self._transform(X)), dtype=np.float64)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.int8]:
        """Return binary predictions at 0.5 probability."""

        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)
