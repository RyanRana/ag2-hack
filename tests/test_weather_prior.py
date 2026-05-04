"""Tests for WeatherPriorAgent — uses synthetic weather data, no API calls."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from pesto.agents.weather_prior import (
    WeatherPriorAgent,
    compute_priors_from_field_state,
    compute_weather_priors,
)
from pesto.latent import CONDITION_LABELS, FieldLatentState, PlantInstance


def _make_latent(n_plants: int = 3) -> FieldLatentState:
    f = FieldLatentState(image_shape=(480, 640))
    for i in range(n_plants):
        f.plants.append(PlantInstance(
            plant_id=i,
            bbox=(i * 100, 0, i * 100 + 80, 80),
        ))
    return f


def _drought_weather() -> dict:
    """7 days with no rain, current temp 32C."""
    return {
        "daily": {
            "precipitation_sum": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "temperature_2m_max": [30, 31, 32, 33, 32, 31, 32],
            "relative_humidity_2m_mean": [30, 28, 25, 27, 26, 29, 28],
        },
        "current_weather": {"temperature": 32.0},
    }


def _humid_warm_weather() -> dict:
    """5 days of high humidity + warm temps → fungal risk."""
    return {
        "daily": {
            "precipitation_sum": [5.0, 3.0, 2.0, 4.0, 6.0],
            "temperature_2m_max": [26, 27, 25, 28, 27],
            "relative_humidity_2m_mean": [85, 88, 82, 90, 87],
        },
        "current_weather": {"temperature": 26.0},
    }


def _heavy_rain_weather() -> dict:
    """Heavy rain → nutrient leaching."""
    return {
        "daily": {
            "precipitation_sum": [20.0, 25.0, 18.0, 15.0, 12.0],
            "temperature_2m_max": [22, 21, 20, 23, 22],
            "relative_humidity_2m_mean": [75, 80, 78, 72, 70],
        },
        "current_weather": {"temperature": 22.0},
    }


def _cool_dry_weather() -> dict:
    """Cool and dry — low stress."""
    return {
        "daily": {
            "precipitation_sum": [1.0, 0.5, 2.0, 0.0, 1.0],
            "temperature_2m_max": [12, 11, 13, 10, 12],
            "relative_humidity_2m_mean": [45, 42, 48, 40, 43],
        },
        "current_weather": {"temperature": 12.0},
    }


def test_drought_elevates_water_stress():
    adj = compute_weather_priors(_drought_weather())
    assert adj["water_stress"] > 0.5
    assert adj["healthy_crop"] < 0


def test_humid_warm_elevates_disease():
    adj = compute_weather_priors(_humid_warm_weather())
    assert adj["disease"] > 0.5


def test_heavy_rain_elevates_nutrient_stress():
    adj = compute_weather_priors(_heavy_rain_weather())
    assert adj["nutrient_stress"] > 0.3


def test_cool_dry_suppresses_stress():
    adj = compute_weather_priors(_cool_dry_weather())
    assert adj["disease"] < 0
    assert adj["pest_damage"] < 0
    assert adj["healthy_crop"] > 0


def test_agent_emits_constraint_with_mock_weather():
    """Agent should emit valid ConstraintMessage with mocked weather."""
    agent = WeatherPriorAgent(latitude=36.7, longitude=-119.8)
    latent = _make_latent(3)

    with patch.object(agent, "_get_weather", return_value=_drought_weather()):
        msg = agent.emit_constraint(None, latent)

    assert msg.sender == "weather_prior"
    assert len(msg.per_plant_log_likelihoods) == 3
    for pid in range(3):
        ll = msg.per_plant_log_likelihoods[pid]
        assert len(ll) == len(CONDITION_LABELS)
        i_water = CONDITION_LABELS.index("water_stress")
        assert ll[i_water] > 0  # drought should elevate water stress


def test_agent_handles_api_failure():
    """Agent should still emit a valid ConstraintMessage when API fails."""
    agent = WeatherPriorAgent()
    latent = _make_latent(2)

    with patch.object(agent, "_get_weather", return_value=None):
        msg = agent.emit_constraint(None, latent)

    assert msg.sender == "weather_prior"
    assert len(msg.per_plant_log_likelihoods) == 2
    # All adjustments should be zero (no data)
    for pid in range(2):
        ll = msg.per_plant_log_likelihoods[pid]
        assert np.allclose(ll, 0.0)


def test_fallback_priors_from_field_state():
    field_state = {
        "T_C": 35.0,
        "RH_pct": 85.0,
        "soil_moisture_m3_m3": 0.10,
    }
    adj = compute_priors_from_field_state(field_state)
    assert adj["water_stress"] > 0
    assert adj["disease"] > 0


def test_protocol_firewall_compliance():
    """WeatherPriorAgent ConstraintMessage must have no prose fields."""
    import dataclasses
    from pesto.messages import ConstraintMessage

    forbidden = {"text", "message", "prose", "explanation", "commentary",
                 "description", "narrative", "reasoning"}
    field_names = {f.name for f in dataclasses.fields(ConstraintMessage)}
    assert not (field_names & forbidden)
