"""Shared type aliases for objects scikit-learn does not type statically.

This project deliberately accepts *any* estimator that follows the scikit-learn
API: plain scikit-learn estimators, ``Pipeline`` and ``imblearn.pipeline.Pipeline``
objects, XGBoost and LightGBM wrappers, ``CalibratedClassifierCV``, and the
PyTorch estimators defined in :mod:`scania_aps.models`. No common base class
covers that set, and several helpers branch on capability at runtime
(``predict_proba`` versus ``decision_function``, presence of ``named_steps``)
rather than on a declared interface.

The aliases below make that intent explicit. ``Estimator`` marks a value as
duck-typed on purpose, which is different from an ``Any`` that nobody thought
about.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Estimator: TypeAlias = Any
"""An unfitted or fitted object following the scikit-learn estimator API."""

FittedEstimator: TypeAlias = Any
"""A fitted estimator, i.e. one whose ``fit`` has already been called."""

FeatureMatrix: TypeAlias = "pd.DataFrame | NDArray[np.float64] | NDArray[np.float32]"
"""A design matrix, either a labelled frame or a dense float array."""

Tensor: TypeAlias = Any
"""A PyTorch tensor.

``torch`` is an optional dependency imported inside the functions that need
it, so it cannot be named in a module-level annotation. This alias marks the
value as a tensor for readers without forcing the import.
"""

ShapValues: TypeAlias = Any
"""SHAP's own return type, which varies by explainer and is not statically typed."""
