"""Ecology agent tests — Lotka-Volterra dynamics + pesticide toxicity."""

from __future__ import annotations

from pesto.agents.ecological_dynamics import EcologicalDynamicsAgent
from pesto.biology.lotka_volterra import simulate_population_trajectory


def test_no_intervention_stable_populations():
    """Without pesticide, predator population stays well above zero."""
    traj = simulate_population_trajectory(
        initial_pest=100.0,
        initial_predator=20.0,
        initial_parasitoid=15.0,
        chemistry_id=None,
        initial_concentration_ppm=0.0,
        decay_k_per_day=0.0,
        horizon_days=30,
    )
    assert min(traj["predator"]) > 5.0


def test_chlorpyrifos_devastates_predators():
    """Broad-spectrum insecticide causes massive predator drop by day 14."""
    traj = simulate_population_trajectory(
        initial_pest=100.0,
        initial_predator=20.0,
        initial_parasitoid=15.0,
        chemistry_id="chlorpyrifos",
        initial_concentration_ppm=1.0,
        decay_k_per_day=0.023,
        horizon_days=30,
    )
    assert traj["predator"][14] < traj["predator"][0] * 0.5


def test_glyphosate_minimal_predator_impact():
    """Herbicide has minimal direct effect on predators."""
    traj = simulate_population_trajectory(
        initial_pest=100.0,
        initial_predator=20.0,
        initial_parasitoid=15.0,
        chemistry_id="glyphosate",
        initial_concentration_ppm=1.0,
        decay_k_per_day=0.030,
        horizon_days=30,
    )
    assert traj["predator"][14] > traj["predator"][0] * 0.7


def test_agent_returns_trajectory_message():
    """Agent emits a valid TrajectoryMessage with low eco cost for laser zap."""
    agent = EcologicalDynamicsAgent()
    payload = {
        "plant_id": 0,
        "action_type": "laser_zap",
        "action_params": {},
        "chemistry": None,
        "initial_populations": {"pest": 100.0, "predator": 20.0, "parasitoid": 15.0},
    }
    msg = agent.assess_intervention(payload)
    assert len(msg.days) == 31
    assert 0.0 <= msg.ecological_cost_score <= 1.0
    assert msg.ecological_cost_score < 0.2  # laser zap = no chemicals


def test_agent_chlorpyrifos_high_eco_cost():
    """Agent flags chlorpyrifos with eco_cost > 0.7 per §13 deliverable."""
    agent = EcologicalDynamicsAgent()
    payload = {
        "plant_id": 0,
        "action_type": "targeted_spray",
        "action_params": {"volume_ml": 5.0, "concentration_g_l": 360.0},
        "chemistry": "chlorpyrifos",
        "initial_populations": {"pest": 100.0, "predator": 20.0, "parasitoid": 15.0},
    }
    msg = agent.assess_intervention(payload)
    assert msg.ecological_cost_score > 0.7


def test_agent_laser_zap_low_eco_cost():
    """Laser zap < 0.1 per §13 deliverable."""
    agent = EcologicalDynamicsAgent()
    payload = {
        "plant_id": 0,
        "action_type": "laser_zap",
        "action_params": {},
        "chemistry": None,
        "initial_populations": {"pest": 100.0, "predator": 20.0, "parasitoid": 15.0},
    }
    msg = agent.assess_intervention(payload)
    assert msg.ecological_cost_score < 0.1
