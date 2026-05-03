"""Temperature scaling — post-hoc calibration for ML agents.

Each ML classifier (DiseaseClassifier, HealthClassifier) produces logits
that are typically overconfident.  Temperature scaling learns a single
scalar T > 0 per model such that ``softmax(logits / T)`` minimises
negative-log-likelihood on a held-out calibration set.

Usage:
    # Offline (one-time):
    T = learn_temperature(logits_array, labels_array)
    save_temperature("disease_classifier", T)

    # At inference (inside emit_constraint):
    T = load_temperature("disease_classifier")   # defaults to 1.0
    calibrated_logits = logits / T
    probs = softmax(calibrated_logits)

Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Default directory for persisted temperature parameters.
_CALIBRATION_DIR = Path(__file__).parent.parent / "data" / "calibration"


def learn_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    lr: float = 0.01,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> float:
    """Learn optimal temperature T via gradient descent on NLL.

    Parameters
    ----------
    logits : (N, C) array of raw model logits.
    labels : (N,) array of integer ground-truth class indices.
    lr     : Learning rate for gradient descent.
    max_iter : Maximum optimisation steps.
    tol    : Convergence tolerance on T change.

    Returns
    -------
    Optimal temperature T > 0.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    N, C = logits.shape
    assert labels.shape == (N,), f"labels shape {labels.shape} != ({N},)"

    T = 1.0  # initialise at identity
    for _ in range(max_iter):
        scaled = logits / T
        # Numerically stable log-softmax
        max_s = scaled.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(scaled - max_s), axis=1, keepdims=True)) + max_s
        log_probs = scaled - log_sum_exp  # (N, C)

        # NLL = -mean( log_probs[i, labels[i]] )
        nll = -np.mean(log_probs[np.arange(N), labels])

        # Gradient of NLL w.r.t. T
        probs = np.exp(log_probs)  # (N, C)
        # d(NLL)/dT = (1/T^2) * mean( logits[i,labels[i]] - sum_c probs[i,c]*logits[i,c] )
        correct_logits = logits[np.arange(N), labels]  # (N,)
        expected_logits = np.sum(probs * logits, axis=1)  # (N,)
        grad = np.mean(expected_logits - correct_logits) / (T * T)

        T_new = T - lr * grad
        T_new = max(T_new, 0.01)  # keep T positive

        if abs(T_new - T) < tol:
            T = T_new
            break
        T = T_new

    return float(T)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by learned temperature: logits / T."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return logits / temperature


def save_temperature(agent_name: str, temperature: float, directory: Path | None = None) -> Path:
    """Persist a learned temperature scalar to disk."""
    d = directory or _CALIBRATION_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{agent_name}_temperature.json"
    path.write_text(json.dumps({"agent": agent_name, "temperature": temperature}))
    return path


def load_temperature(agent_name: str, directory: Path | None = None) -> float:
    """Load a persisted temperature scalar. Returns 1.0 if not found."""
    d = directory or _CALIBRATION_DIR
    path = d / f"{agent_name}_temperature.json"
    if not path.exists():
        return 1.0
    data = json.loads(path.read_text())
    return float(data["temperature"])
