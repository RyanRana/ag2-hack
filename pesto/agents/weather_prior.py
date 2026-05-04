"""WeatherPriorAgent — contextual prior adjustment from weather data.

Paradigm: Physics (environmental context).
Runs BEFORE ML agents to shift the shared prior Psi based on current and
recent weather conditions:

  - 5+ days without rain + high temp  → P(water_stress) ↑
  - High humidity + warm temp         → P(fungal disease) ↑
  - Recent heavy rain                 → P(nutrient_stress) ↑ (leaching)
  - Cool + dry                        → all stress priors ↓

Data source: Open-Meteo free API (no key required). Falls back to
user-supplied field_state values if the API is unreachable.

Same ChannelAgent envelope, same ConstraintMessage protocol.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import requests

from pesto.agents.base import ChannelAgent
from pesto.latent import CONDITION_LABELS, FieldLatentState
from pesto.messages import ConstraintMessage


# Open-Meteo endpoint (free, no API key)
_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather(
    latitude: float,
    longitude: float,
    *,
    past_days: int = 7,
    forecast_days: int = 2,
    timeout: float = 5.0,
) -> dict[str, Any] | None:
    """Fetch current + recent weather from Open-Meteo.

    Returns None on any network failure (offline operation supported).
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean",
        "current_weather": "true",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
    }
    try:
        resp = requests.get(_OPEN_METEO_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


def compute_weather_priors(weather: dict[str, Any]) -> dict[str, float]:
    """Derive condition-prior adjustments from weather data.

    Returns a dict of CONDITION_LABEL -> log-likelihood adjustment.
    Positive = condition more likely given weather. Negative = less likely.
    """
    adjustments: dict[str, float] = {label: 0.0 for label in CONDITION_LABELS}

    daily = weather.get("daily", {})
    precip_sums = daily.get("precipitation_sum", [])
    temp_maxes = daily.get("temperature_2m_max", [])
    humidity_means = daily.get("relative_humidity_2m_mean", [])

    current = weather.get("current_weather", {})
    current_temp = current.get("temperature", 25.0)

    # --- Drought signal: consecutive days with <1mm rain ---
    if precip_sums:
        # Count recent dry days (last 7 days of available data)
        recent_precip = precip_sums[-7:] if len(precip_sums) >= 7 else precip_sums
        dry_days = sum(1 for p in recent_precip if p is not None and p < 1.0)

        if dry_days >= 5 and current_temp > 28:
            adjustments["water_stress"] = 1.5
            adjustments["healthy_crop"] = -0.5
        elif dry_days >= 3:
            adjustments["water_stress"] = 0.7

    # --- Fungal disease signal: high humidity + warm temps ---
    if humidity_means and temp_maxes:
        recent_humidity = [h for h in humidity_means[-5:] if h is not None]
        recent_temps = [t for t in temp_maxes[-5:] if t is not None]

        if recent_humidity and recent_temps:
            avg_humidity = sum(recent_humidity) / len(recent_humidity)
            avg_temp = sum(recent_temps) / len(recent_temps)

            if avg_humidity > 80 and avg_temp > 20:
                adjustments["disease"] = 1.2
                adjustments["healthy_crop"] = -0.3
            elif avg_humidity > 70 and avg_temp > 18:
                adjustments["disease"] = 0.5

    # --- Nutrient leaching signal: heavy recent rain ---
    if precip_sums:
        recent_precip = precip_sums[-5:] if len(precip_sums) >= 5 else precip_sums
        total_recent = sum(p for p in recent_precip if p is not None)
        if total_recent > 80:  # >80mm in 5 days = heavy
            adjustments["nutrient_stress"] = 0.8
        elif total_recent > 50:
            adjustments["nutrient_stress"] = 0.4

    # --- Cool + dry = low stress overall ---
    if current_temp < 15 and adjustments["water_stress"] == 0.0:
        adjustments["disease"] = min(adjustments["disease"], 0.0) - 0.3
        adjustments["pest_damage"] = -0.3
        adjustments["healthy_crop"] = 0.3

    return adjustments


def compute_priors_from_field_state(field_state: dict) -> dict[str, float]:
    """Fallback: derive weather priors from field_state values when API is down."""
    adjustments: dict[str, float] = {label: 0.0 for label in CONDITION_LABELS}

    temp = field_state.get("T_C", 25.0)
    rh = field_state.get("RH_pct", 60.0)
    soil_moisture = field_state.get("soil_moisture_m3_m3", 0.25)

    if rh > 80 and temp > 20:
        adjustments["disease"] = 1.0
    if soil_moisture < 0.15 and temp > 28:
        adjustments["water_stress"] = 1.2
        adjustments["healthy_crop"] = -0.4

    return adjustments


class WeatherPriorAgent(ChannelAgent):
    """Pre-inference context agent that adjusts priors from weather data."""

    def __init__(
        self,
        latitude: float = 36.7,  # default: California Central Valley
        longitude: float = -119.8,
    ) -> None:
        super().__init__(name="weather_prior")
        self.latitude = latitude
        self.longitude = longitude
        self._cached_weather: dict[str, Any] | None = None
        self._cache_timestamp: float = 0.0
        self._cache_ttl: float = 3600.0  # 1 hour

    def _get_weather(self) -> dict[str, Any] | None:
        """Fetch weather with 1-hour cache."""
        now = time.time()
        if self._cached_weather and (now - self._cache_timestamp) < self._cache_ttl:
            return self._cached_weather
        weather = fetch_weather(self.latitude, self.longitude)
        if weather:
            self._cached_weather = weather
            self._cache_timestamp = now
        return weather

    def emit_constraint(
        self, image_path: str | None, latent: FieldLatentState
    ) -> ConstraintMessage:
        weather = self._get_weather()

        if weather:
            adjustments = compute_weather_priors(weather)
            source = "open_meteo"
        else:
            adjustments = {label: 0.0 for label in CONDITION_LABELS}
            source = "none"

        # Build per-plant log-likelihoods from weather adjustments
        # (same adjustment applied uniformly — weather affects all plants equally)
        log_lik = np.array([adjustments.get(label, 0.0) for label in CONDITION_LABELS])

        per_ll: dict[int, np.ndarray] = {}
        per_resid: dict[int, float] = {}
        per_conf: dict[int, float] = {}

        # Confidence is low — weather is a weak prior, not a detector
        max_adj = max(abs(v) for v in adjustments.values()) if adjustments else 0.0
        confidence = float(min(0.3 + max_adj * 0.2, 0.6))

        for plant in latent.plants:
            per_ll[plant.plant_id] = log_lik.copy()
            per_resid[plant.plant_id] = 0.0
            per_conf[plant.plant_id] = confidence

        active_labels = [label for label, v in adjustments.items() if abs(v) > 0.01]
        if not active_labels:
            active_labels = ["healthy_crop"]

        return ConstraintMessage(
            sender=self.name,
            timestamp=time.time(),
            iteration=latent.iteration,
            per_plant_log_likelihoods=per_ll,
            per_plant_residual=per_resid,
            per_plant_confidence=per_conf,
            labels_discriminated=active_labels,
            metadata={
                "weather_source": source,
                "adjustments": {k: float(v) for k, v in adjustments.items() if abs(v) > 0.01},
            },
        )
