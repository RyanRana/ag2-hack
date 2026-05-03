"""Cross-examination — pairwise per-plant disagreement metrics.

Enhanced with weighted paradigm pairs:
  - ML-vs-ML disagreement:         weight 1.0
  - ML-vs-biophysics disagreement: weight 1.5 (cross-paradigm)
  - ML-vs-anomaly disagreement:    weight 2.0 (anomaly flags are rare)
"""

from __future__ import annotations

import numpy as np

from pulse.latent import CONDITION_LABELS
from pulse.messages import ConstraintMessage, CrossExamMessage


# Agent name → paradigm for weighted cross-examination
_AGENT_PARADIGM: dict[str, str] = {
    "disease_classifier": "ml",
    "health_classifier": "ml",
    "weed_detector": "ml",
    "segmentation": "cv",
    "growth_stage": "ml",
    "anomaly_detector": "anomaly",
    "water_balance": "biophysics",
    "weather_prior": "physics",
    "pesticide_fate": "physics",
    "ecological_dynamics": "biology",
    "vlm_reasoner": "ml",
    "skeptic": "ml",
}

# Weight multiplier for paradigm pair disagreements
_PARADIGM_PAIR_WEIGHTS: dict[tuple[str, str], float] = {
    ("ml", "ml"): 1.0,
    ("ml", "cv"): 1.2,
    ("ml", "biophysics"): 1.5,
    ("ml", "physics"): 1.5,
    ("ml", "anomaly"): 2.0,
    ("cv", "biophysics"): 1.3,
    ("cv", "anomaly"): 1.8,
    ("biophysics", "physics"): 1.0,
    ("anomaly", "biophysics"): 1.8,
}


def _pair_weight(agent_a: str, agent_b: str) -> float:
    """Get the disagreement weight for a pair of agents."""
    pa = _AGENT_PARADIGM.get(agent_a, "ml")
    pb = _AGENT_PARADIGM.get(agent_b, "ml")
    # Check both orderings
    w = _PARADIGM_PAIR_WEIGHTS.get((pa, pb))
    if w is not None:
        return w
    w = _PARADIGM_PAIR_WEIGHTS.get((pb, pa))
    if w is not None:
        return w
    return 1.0  # default for unknown pairs


def cross_examine(constraints: dict[str, ConstraintMessage]) -> list[CrossExamMessage]:
    """Compute pairwise per-plant disagreement matrices.

    Returns one CrossExamMessage per (evaluator, target) pair, where
    ``per_plant_disagreement`` is the KL divergence WEIGHTED by paradigm
    pair type, and ``per_plant_diagnostic`` records which condition labels
    the disagreement concentrates on.
    """
    msgs: list[CrossExamMessage] = []
    agents = list(constraints.keys())
    for i, ai in enumerate(agents):
        for j, aj in enumerate(agents):
            if i == j:
                continue
            ci = constraints[ai]
            cj = constraints[aj]
            weight = _pair_weight(ai, aj)
            shared = set(ci.per_plant_log_likelihoods) & set(cj.per_plant_log_likelihoods)
            per_plant_dis: dict[int, float] = {}
            per_plant_diag: dict[int, dict[str, float]] = {}
            for pid in shared:
                ll_i = np.asarray(ci.per_plant_log_likelihoods[pid])
                ll_j = np.asarray(cj.per_plant_log_likelihoods[pid])
                pi = _softmax(ll_i)
                pj = _softmax(ll_j)
                kl = float(np.sum(pi * (np.log(pi + 1e-12) - np.log(pj + 1e-12))))
                per_plant_dis[pid] = kl * weight
                axis_diag: dict[str, float] = {}
                for k, label in enumerate(CONDITION_LABELS):
                    axis_diag[label] = float(abs(pi[k] - pj[k]))
                per_plant_diag[pid] = axis_diag
            msgs.append(CrossExamMessage(
                evaluator=ai,
                target=aj,
                per_plant_disagreement=per_plant_dis,
                per_plant_diagnostic=per_plant_diag,
            ))
    return msgs


def max_disagreement_per_plant(
    cross_exam_msgs: list[CrossExamMessage],
) -> dict[int, float]:
    """Across all evaluator/target pairs, max disagreement per plant_id."""
    out: dict[int, float] = {}
    for m in cross_exam_msgs:
        for pid, kl in m.per_plant_disagreement.items():
            out[pid] = max(out.get(pid, 0.0), kl)
    return out


def disputed_plants(
    cross_exam_msgs: list[CrossExamMessage], threshold: float = 1.5
) -> list[int]:
    """List plant_ids with max KL disagreement above threshold."""
    maxes = max_disagreement_per_plant(cross_exam_msgs)
    return sorted(pid for pid, kl in maxes.items() if kl > threshold)


def dominant_axes_for_plant(
    cross_exam_msgs: list[CrossExamMessage], plant_id: int
) -> dict[str, float]:
    """Sum the per-axis disagreement magnitudes across all pairs for a plant."""
    totals: dict[str, float] = {label: 0.0 for label in CONDITION_LABELS}
    for m in cross_exam_msgs:
        diag = m.per_plant_diagnostic.get(plant_id, {})
        for label, v in diag.items():
            totals[label] = totals.get(label, 0.0) + v
    return totals


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
