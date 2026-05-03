"""Water balance agent tests — FAO-56 Penman-Monteith + van Genuchten."""

from __future__ import annotations

from pulse.agents.water_balance import WaterBalanceAgent
from pulse.biophysics.water_balance import (
    penman_monteith_et0,
    soil_water_potential,
    water_stress_index,
)
from pulse.latent import CONDITION_LABELS


# --- Math sanity ---------------------------------------------------------


def test_well_watered_loam_no_stress():
    wsi = water_stress_index(
        theta=0.32,
        soil_texture="loam",
        crop_type="tomato",
        T_C=22.0,
        RH_pct=60,
        u2_m_s=1.5,
        R_n_MJ_m2_d=14.0,
    )
    assert wsi["stress_index"] < 0.1, f"expected near-zero, got {wsi}"


def test_dry_loam_high_stress():
    """Loam well below field capacity in hot dry weather → high stress index.

    Note: with Carsel-Parrish van Genuchten parameters, sand has a very steep
    retention curve and *never* exhibits high matric tension at typical theta
    values. Loam (and loam-class soils) is the realistic scenario.
    """
    wsi = water_stress_index(
        theta=0.09,
        soil_texture="loam",
        crop_type="tomato",
        T_C=35.0,
        RH_pct=20,
        u2_m_s=4.0,
        R_n_MJ_m2_d=24.0,
    )
    assert wsi["stress_index"] > 0.6, f"expected high stress, got {wsi}"


def test_penman_monteith_increases_with_temperature():
    cool = penman_monteith_et0(15.0, 60.0, 2.0, 12.0)
    hot = penman_monteith_et0(35.0, 60.0, 2.0, 12.0)
    assert hot > cool


def test_van_genuchten_dry_soil_negative_potential():
    psi_dry = soil_water_potential(theta=0.09, soil_texture="loam")
    psi_wet = soil_water_potential(theta=0.32, soil_texture="loam")
    assert psi_dry < -100.0  # very negative
    assert psi_wet > -50.0   # near field capacity


# --- Agent end-to-end ----------------------------------------------------


def test_agent_suppresses_water_stress_in_well_watered_field():
    agent = WaterBalanceAgent()
    payload = {
        "latent": {"iteration": 0},
        "field_state": {
            "soil_moisture_m3_m3": 0.32,
            "soil_texture": "loam",
            "T_C": 22.0,
            "RH_pct": 60,
            "u2_m_s": 1.5,
            "R_n_MJ_m2_d": 14.0,
            "crop_type": "tomato",
            "plants": [{"plant_id": 0}, {"plant_id": 1}],
        },
    }
    c = agent.emit_constraint(payload)
    i_water = CONDITION_LABELS.index("water_stress")
    for ll in c.per_plant_log_likelihoods.values():
        assert ll[i_water] < 0
    assert c.metadata["stress_index"] < 0.1


def test_agent_elevates_water_stress_in_dry_field():
    agent = WaterBalanceAgent()
    payload = {
        "latent": {"iteration": 1},
        "field_state": {
            "soil_moisture_m3_m3": 0.09,
            "soil_texture": "loam",
            "T_C": 35.0,
            "RH_pct": 20,
            "u2_m_s": 4.0,
            "R_n_MJ_m2_d": 24.0,
            "crop_type": "tomato",
            "plants": [{"plant_id": 0}],
        },
    }
    c = agent.emit_constraint(payload)
    i_water = CONDITION_LABELS.index("water_stress")
    i_disease = CONDITION_LABELS.index("disease")
    ll = c.per_plant_log_likelihoods[0]
    assert ll[i_water] > 1.0  # strong elevation
    assert ll[i_disease] < 0  # disease suppressed
    assert c.metadata["stress_index"] > 0.6
