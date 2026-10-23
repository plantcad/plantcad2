import copy

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from exp586_evals.full_probe import _validate_roundtrip


def test_probe_roundtrip_requires_identical_fitted_parameters() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(64, 12)).astype(np.float32)
    target = rng.normal(size=64).astype(np.float32)
    estimator = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(solver="lsqr"))]
    ).fit(features, target)
    restored = copy.deepcopy(estimator)
    expected = estimator.predict(features)

    _validate_roundtrip(estimator, restored, features, expected)

    restored.named_steps["ridge"].coef_[0] += np.float32(1e-3)
    with pytest.raises(AssertionError):
        _validate_roundtrip(estimator, restored, features, expected)
