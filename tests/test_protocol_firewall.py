"""The firewall test. If this fails, Pesto has degraded into a chatbot."""

import dataclasses

import numpy as np

from pesto.messages import (
    ActionMessage,
    ConstraintMessage,
    CrossExamMessage,
    HypothesisMessage,
    InterventionAssessmentMessage,
    TrajectoryMessage,
)


FORBIDDEN_FIELDS = {
    "text",
    "message",
    "prose",
    "explanation",
    "commentary",
    "description",
    "narrative",
    "reasoning",
}


def test_no_prose_fields_in_messages():
    """Architecture firewall — no prose fields anywhere in the protocol."""
    for cls in [
        ConstraintMessage,
        CrossExamMessage,
        ActionMessage,
        HypothesisMessage,
        InterventionAssessmentMessage,
        TrajectoryMessage,
    ]:
        field_names = {f.name for f in dataclasses.fields(cls)}
        violations = field_names & FORBIDDEN_FIELDS
        assert not violations, (
            f"{cls.__name__} contains forbidden prose fields: {violations}"
        )


def test_action_params_no_strings():
    """Action parameters must be numerical."""
    msg = ActionMessage(
        sender="controller",
        timestamp=0.0,
        plant_id=0,
        action_type="laser_zap",
        action_params={"intensity_pct": 70, "duration_ms": 200},
        expected_information_gain=0.5,
    )
    for v in msg.action_params.values():
        assert isinstance(v, (int, float, np.floating, np.integer)), (
            f"Action param contains non-numeric: {v}"
        )


def test_intervention_assessment_breakdown_no_strings():
    """Hazard breakdown is numerical-only."""
    msg = InterventionAssessmentMessage(
        sender="pesticide_fate",
        timestamp=0.0,
        plant_id=0,
        action_type="targeted_spray",
        action_params={"volume_ml": 5.0, "concentration_g_l": 360.0},
        off_target_deposition={1: 0.18, 2: 0.04},
        soil_half_life_days=23.0,
        time_to_offsite_hours=4.2,
        hazard_score=0.71,
        hazard_breakdown={
            "drift_max_ppm": 0.18,
            "neighbors_at_risk": 3.0,
            "ditch_arrival_hours": 4.2,
        },
    )
    for v in msg.hazard_breakdown.values():
        assert isinstance(v, (int, float, np.floating, np.integer)), (
            f"hazard_breakdown contains non-numeric: {v}"
        )


def test_trajectory_message_no_strings():
    """Ecological cost breakdown is numerical-only; trajectories are float lists."""
    days = list(range(0, 31))
    msg = TrajectoryMessage(
        sender="ecological_dynamics",
        timestamp=0.0,
        plant_id=0,
        action_type="targeted_spray",
        action_params={"volume_ml": 5.0},
        days=[float(d) for d in days],
        pest_trajectory=[1.0] * 31,
        predator_trajectory=[1.0] * 31,
        parasitoid_trajectory=[1.0] * 31,
        ecological_cost_score=0.42,
        cost_breakdown={
            "predator_drop_pct_d14": 0.47,
            "pest_rebound_factor_d30": 2.3,
            "parasitoid_drop_pct_d14": 0.61,
        },
    )
    for v in msg.cost_breakdown.values():
        assert isinstance(v, (int, float, np.floating, np.integer)), (
            f"cost_breakdown contains non-numeric: {v}"
        )
    for traj in [
        msg.pest_trajectory,
        msg.predator_trajectory,
        msg.parasitoid_trajectory,
    ]:
        for v in traj:
            assert isinstance(v, (int, float, np.floating, np.integer))
