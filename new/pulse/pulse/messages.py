"""Inter-agent message protocol — the firewall.

Every payload exchanged between agents in the inference loop is one of the
dataclasses below. Free-form prose fields (``text``, ``message``, ``prose``,
``explanation``, ``commentary``, ``description``, ``narrative``,
``reasoning``) are FORBIDDEN. The architectural firewall is enforced by
``tests/test_protocol_firewall.py``.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConstraintMessage:
    """A model agent's per-plant likelihood contribution. NO PROSE."""

    sender: str
    timestamp: float
    iteration: int
    # Map plant_id -> log-likelihood vector over CONDITION_LABELS
    per_plant_log_likelihoods: dict[int, np.ndarray]
    # Per-plant residual / out-of-distribution score for skeptic consumption
    per_plant_residual: dict[int, float]
    # Agent self-confidence per plant
    per_plant_confidence: dict[int, float]
    # Which condition labels this agent is calibrated to discriminate
    labels_discriminated: list[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class CrossExamMessage:
    """Pairwise disagreement metric between agents per plant."""

    evaluator: str
    target: str
    per_plant_disagreement: dict[int, float]  # KL-divergence
    per_plant_diagnostic: dict[int, dict[str, float]]


@dataclass
class HypothesisMessage:
    """Skeptic's alternative explanation for an asymmetry pattern."""

    sender: str
    timestamp: float
    plant_id: int
    hypothesis_id: str
    log_posterior: float
    evidence_axes: dict[str, float]


@dataclass
class ActionMessage:
    """Controller's per-plant intervention recommendation."""

    sender: str
    timestamp: float
    plant_id: int
    action_type: str
    action_params: dict
    expected_information_gain: float
    expected_utility: float = 0.0
    physics_hazard_score: float = 0.0
    target_hypothesis: str | None = None


@dataclass
class InterventionAssessmentMessage:
    """Physics agent's assessment of a candidate intervention. NO PROSE.

    Emitted per (plant_id, action_type) candidate.
    """

    sender: str
    timestamp: float
    plant_id: int
    action_type: str
    action_params: dict
    off_target_deposition: dict[int, float]
    soil_half_life_days: float
    time_to_offsite_hours: float | None
    hazard_score: float
    hazard_breakdown: dict[str, float]


@dataclass
class TrajectoryMessage:
    """Ecology agent's 30-day population forecast under a candidate
    intervention. NO PROSE."""

    sender: str
    timestamp: float
    plant_id: int
    action_type: str
    action_params: dict
    days: list[float]
    pest_trajectory: list[float]
    predator_trajectory: list[float]
    parasitoid_trajectory: list[float]
    ecological_cost_score: float
    cost_breakdown: dict[str, float]
