"""PyTorch multilayer perceptron with explicit optimization experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from scania_aps._types import Tensor

OptimizerName = Literal["sgd", "adam", "adamw"]
SchedulerName = Literal["none", "cosine", "plateau"]
LossName = Literal["bce", "focal"]


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Small sklearn-compatible PyTorch classifier for tabular experiments.

    The estimator intentionally exposes optimizer, weight decay, dropout,
    batch normalization, batch size, gradient clipping and early stopping so
    their effect on the real maintenance objective can be studied directly.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        batch_norm: bool = True,
        optimizer: OptimizerName = "adamw",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 100,
        validation_fraction: float = 0.12,
        patience: int = 12,
        gradient_clip: float | None = 5.0,
        scheduler: SchedulerName = "plateau",
        loss: LossName = "bce",
        focal_gamma: float = 2.0,
        positive_class_weight: float | None = None,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.gradient_clip = gradient_clip
        self.scheduler = scheduler
        self.loss = loss
        self.focal_gamma = focal_gamma
        self.positive_class_weight = positive_class_weight
        self.random_state = random_state
        self.device = device

    def _validate_hyperparameters(self) -> None:
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise ValueError("Unsupported optimizer.")
        if self.scheduler not in {"none", "cosine", "plateau"}:
            raise ValueError("Unsupported scheduler.")
        if self.loss not in {"bce", "focal"}:
            raise ValueError("Unsupported loss.")
        if self.learning_rate <= 0 or self.batch_size <= 0 or self.max_epochs <= 0:
            raise ValueError("learning_rate, batch_size and max_epochs must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie in (0, 0.5).")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative.")
        if self.positive_class_weight is not None and self.positive_class_weight <= 0:
            raise ValueError("positive_class_weight must be positive when supplied.")

    def fit(self, X: NDArray[np.floating], y: NDArray[np.integer]) -> TorchMLPClassifier:
        """Fit the network using an internal stratified validation split."""

        self._validate_hyperparameters()
        X_checked, y_checked = check_X_y(X, y, dtype=np.float32, ensure_all_finite=True)
        y_array = np.asarray(y_checked, dtype=np.int64)
        if not np.isin(y_array, [0, 1]).all():
            raise ValueError("TorchMLPClassifier requires binary labels 0/1.")

        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Install the optional 'neural' dependency group to use PyTorch."
            ) from exc

        from sklearn.model_selection import train_test_split

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X_train, X_val, y_train, y_val = train_test_split(
            X_checked,
            y_array,
            test_size=self.validation_fraction,
            stratify=y_array,
            random_state=self.random_state,
        )

        if self.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device)
        self.device_ = str(device)

        layers: list[nn.Module] = []
        in_features = X_checked.shape[1]
        for width in self.hidden_dims:
            layers.append(nn.Linear(in_features, width))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_features = width
        layers.append(nn.Linear(in_features, 1))
        model = nn.Sequential(*layers).to(device)

        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

        scheduler: object | None
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        elif self.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=4, factor=0.5
            )
        else:
            scheduler = None

        pos_weight = None
        if self.positive_class_weight is not None:
            pos_weight = torch.tensor(
                [self.positive_class_weight], dtype=torch.float32, device=device
            )
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        def loss_fn(logits: Tensor, targets: Tensor) -> Tensor:
            raw = bce(logits, targets)
            if self.loss == "bce":
                return raw.mean()
            probs = torch.sigmoid(logits)
            pt = torch.where(targets > 0.5, probs, 1.0 - probs)
            return (((1.0 - pt) ** self.focal_gamma) * raw).mean()

        train_ds = TensorDataset(
            torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_train, dtype=np.float32)).reshape(-1, 1),
        )
        loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        X_val_t = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(device)
        y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).reshape(-1, 1).to(device)

        best_state: dict[str, Tensor] | None = None
        best_val = float("inf")
        stale_epochs = 0
        history: list[dict[str, float]] = []

        for epoch in range(self.max_epochs):
            model.train()
            epoch_losses: list[float] = []
            for features, target in loader:
                features = features.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss_value = loss_fn(logits, target)
                loss_value.backward()
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                optimizer.step()
                epoch_losses.append(float(loss_value.detach().cpu().item()))

            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(X_val_t), y_val_t).detach().cpu().item())
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

            if scheduler is not None:
                if self.scheduler == "plateau":
                    scheduler.step(val_loss)  # type: ignore[union-attr]
                else:
                    scheduler.step()  # type: ignore[union-attr]

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_in_ = int(X_checked.shape[1])
        self.history_ = history
        self.best_validation_loss_ = best_val
        return self

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return two-column class probabilities."""

        check_is_fitted(self, attributes=["model_", "classes_", "n_features_in_"])
        X_checked = check_array(X, dtype=np.float32, ensure_all_finite=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training data.")
        import torch

        device = torch.device(self.device_)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X_checked).to(device)).reshape(-1)
            positive = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.int8]:
        """Predict using the conventional 0.5 probability threshold."""

        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)
