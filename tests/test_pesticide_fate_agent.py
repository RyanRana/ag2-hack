"""PesticideFateAgent unit tests — physics-only, no LLM, no models."""

from __future__ import annotations

import numpy as np

from pesto.agents.pesticide_fate import PesticideFateAgent
from pesto.physics.drift import (
    deposition_at_locations,
    gaussian_plume_concentration,
    soil_persistence_curve,
)


def test_no_chemicals_zero_hazard():
    """Laser zap is a non-chemical action — must produce zero hazard."""
    agent = PesticideFateAgent()
    field = {
        "plants": [
            {"plant_id": 0, "xy_m": [0.0, 0.0], "is_target": True},
            {"plant_id": 1, "xy_m": [0.5, 0.0], "is_target": False},
        ],
        "wind_dir_deg": 270,
        "wind_speed_m_s": 4.0,
    }
    a = agent.assess_intervention(
        plant_id=0,
        action_type="laser_zap",
        action_params={},
        field_state=field,
        chemistry_id="glyphosate",
    )
    assert a.hazard_score == 0.0
    assert a.off_target_deposition == {}


def test_spray_with_close_neighbor_high_hazard():
    """Spraying glyphosate next to a healthy plant 30cm downwind should flag hazard."""
    agent = PesticideFateAgent()
    field = {
        "plants": [
            {"plant_id": 0, "xy_m": [0.0, 0.0], "is_target": True},
            {"plant_id": 1, "xy_m": [0.3, 0.0], "is_target": False},
        ],
        "wind_dir_deg": 270,  # wind from west → flow east → +x is downwind
        "wind_speed_m_s": 2.0,
    }
    a = agent.assess_intervention(
        plant_id=0,
        action_type="targeted_spray",
        action_params={"volume_ml": 5.0, "concentration_g_l": 360.0,
                       "application_duration_s": 0.3},
        field_state=field,
        chemistry_id="glyphosate",
    )
    assert 1 in a.off_target_deposition
    assert a.off_target_deposition[1] > 0
    assert a.hazard_score > 0.0


def test_upwind_neighbor_zero_deposition():
    """A plant directly upwind of the source receives ~0 deposition."""
    agent = PesticideFateAgent()
    field = {
        "plants": [
            {"plant_id": 0, "xy_m": [0.0, 0.0], "is_target": True},
            {"plant_id": 1, "xy_m": [-1.0, 0.0], "is_target": False},
        ],
        "wind_dir_deg": 270,
        "wind_speed_m_s": 4.0,
    }
    a = agent.assess_intervention(
        plant_id=0,
        action_type="targeted_spray",
        action_params={"volume_ml": 5.0, "concentration_g_l": 360.0,
                       "application_duration_s": 0.3},
        field_state=field,
        chemistry_id="glyphosate",
    )
    assert a.off_target_deposition[1] < 1e-6


def test_atrazine_more_persistent_than_glyphosate():
    """Atrazine DT50=75 days >> glyphosate DT50=23 days → higher persist term."""
    agent = PesticideFateAgent()
    field = {
        "plants": [
            {"plant_id": 0, "xy_m": [0.0, 0.0], "is_target": True},
            {"plant_id": 1, "xy_m": [0.4, 0.0], "is_target": False},
        ],
        "wind_dir_deg": 270,
        "wind_speed_m_s": 2.0,
    }
    glyph = agent.assess_intervention(0, "targeted_spray",
                                      {"volume_ml": 5.0, "concentration_g_l": 360.0,
                                       "application_duration_s": 0.3},
                                      field, "glyphosate")
    atr = agent.assess_intervention(0, "targeted_spray",
                                    {"volume_ml": 5.0, "concentration_g_l": 360.0,
                                     "application_duration_s": 0.3},
                                    field, "atrazine")
    assert atr.soil_half_life_days > glyph.soil_half_life_days
    assert atr.hazard_score >= glyph.hazard_score


def test_gaussian_plume_upwind_zero():
    """Sanity: plume concentration upwind of source is exactly zero."""
    C = gaussian_plume_concentration(
        x_m=np.array([-2.0, -0.5]),
        y_m=np.array([0.0, 0.0]),
        z_m=np.array([1.0, 1.0]),
        Q_g_per_s=10.0,
        u_m_per_s=2.0,
    )
    np.testing.assert_array_equal(C, np.zeros_like(C))


def test_soil_persistence_halflife():
    """C(DT50) = C0/2."""
    days = np.array([0.0, 23.0, 46.0])
    out = soil_persistence_curve(initial_conc_g_kg=10.0, dt50_days=23.0, days=days)
    assert abs(out[0] - 10.0) < 1e-9
    assert abs(out[1] - 5.0) < 1e-3
    assert abs(out[2] - 2.5) < 1e-3
