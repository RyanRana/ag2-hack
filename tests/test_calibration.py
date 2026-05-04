"""Tests for temperature scaling calibration."""

from __future__ import annotations

import numpy as np
import pytest

from pesto.calibration import (
    apply_temperature,
    learn_temperature,
    load_temperature,
    save_temperature,
)


def _make_overconfident_logits(n: int = 200, c: int = 7, seed: int = 42):
    """Simulate overconfident model: logits are scaled up so softmax is peaky."""
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, c, size=n)
    # Ground-truth logits: correct class gets +2, rest ~0
    logits = rng.randn(n, c) * 0.3
    logits[np.arange(n), labels] += 2.0
    # Scale up to make overconfident (T_true ~ 3.0 would fix this)
    logits *= 3.0
    return logits, labels


def test_learn_temperature_reduces_overconfidence():
    logits, labels = _make_overconfident_logits()
    T = learn_temperature(logits, labels)
    # Overconfident logits → T should be > 1.0 (softens the distribution)
    assert T > 1.0, f"Expected T > 1.0 for overconfident logits, got {T}"


def test_learn_temperature_identity_on_calibrated():
    """Well-calibrated logits should yield T ≈ 1.0."""
    rng = np.random.RandomState(0)
    n, c = 500, 7
    labels = rng.randint(0, c, size=n)
    logits = rng.randn(n, c) * 0.5
    logits[np.arange(n), labels] += 1.5
    T = learn_temperature(logits, labels)
    assert 0.5 < T < 2.0, f"Expected T near 1.0, got {T}"


def test_apply_temperature():
    logits = np.array([2.0, 1.0, 0.5])
    scaled = apply_temperature(logits, 2.0)
    np.testing.assert_allclose(scaled, [1.0, 0.5, 0.25])


def test_apply_temperature_rejects_non_positive():
    with pytest.raises(ValueError):
        apply_temperature(np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        apply_temperature(np.array([1.0]), -1.0)


def test_save_load_roundtrip(tmp_path):
    save_temperature("test_agent", 2.5, directory=tmp_path)
    loaded = load_temperature("test_agent", directory=tmp_path)
    assert loaded == 2.5


def test_load_missing_returns_default(tmp_path):
    T = load_temperature("nonexistent_agent", directory=tmp_path)
    assert T == 1.0
