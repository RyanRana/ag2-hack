"""Lotka-Volterra-with-toxicity for crop-pest-predator-parasitoid triads.

References:
- Crowder & Northfield 2014 ('Predator diversity strengthens herbivore
  suppression') — predator-pest dynamics.
- EPA ECOTOX database — LC50 values.
- AERU University of Hertfordshire PPDB.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import odeint


# LC50 (concentration killing 50%) per chemical class. Values are
# illustrative — production should pull from EPA ECOTOX directly.
TOXICITY_DB: dict[str, dict[str, float]] = {
    "glyphosate": {
        "predator_LC50_ppm": 100.0,
        "parasitoid_LC50_ppm": 50.0,
        "predator_max_mortality_d": 0.05,
        "parasitoid_max_mortality_d": 0.10,
    },
    "2_4_d": {
        "predator_LC50_ppm": 80.0,
        "parasitoid_LC50_ppm": 30.0,
        "predator_max_mortality_d": 0.08,
        "parasitoid_max_mortality_d": 0.15,
    },
    "atrazine": {
        "predator_LC50_ppm": 60.0,
        "parasitoid_LC50_ppm": 25.0,
        "predator_max_mortality_d": 0.12,
        "parasitoid_max_mortality_d": 0.20,
    },
    "chlorpyrifos": {
        # Broad-spectrum organophosphate insecticide — devastating to non-targets.
        "predator_LC50_ppm": 0.05,
        "parasitoid_LC50_ppm": 0.02,
        "predator_max_mortality_d": 0.85,
        "parasitoid_max_mortality_d": 0.92,
    },
}

DEFAULT_PARAMS: dict[str, float] = {
    "r_pest": 0.30,
    "K_pest": 1000.0,
    "a_predation": 0.0008,
    "e_conversion": 0.10,
    "m_predator": 0.05,
    "m_parasitoid": 0.04,
    "a_parasitism": 0.0006,
}


def lotka_volterra_with_pesticide(
    state: np.ndarray,
    t: float,
    params: dict,
    pesticide_C0: float,
    pesticide_decay_k: float,
    toxicity: dict,
) -> np.ndarray:
    """ODE state vector: [P (pest), R (predator), Pa (parasitoid)]."""
    P, R, Pa = state
    P = max(P, 0.0)
    R = max(R, 0.0)
    Pa = max(Pa, 0.0)
    C_t = pesticide_C0 * math.exp(-pesticide_decay_k * t)

    pred_LC = max(toxicity["predator_LC50_ppm"], 1e-9)
    para_LC = max(toxicity["parasitoid_LC50_ppm"], 1e-9)
    mu_pred = (
        toxicity["predator_max_mortality_d"]
        * (C_t / pred_LC) / (1.0 + C_t / pred_LC)
    )
    mu_para = (
        toxicity["parasitoid_max_mortality_d"]
        * (C_t / para_LC) / (1.0 + C_t / para_LC)
    )
    is_insecticide = toxicity["predator_max_mortality_d"] > 0.5
    mu_pest_max = 0.7 if is_insecticide else 0.0
    mu_pest_t = mu_pest_max * (C_t / 0.1) / (1.0 + C_t / 0.1)

    dP = (
        params["r_pest"] * P * (1.0 - P / params["K_pest"])
        - params["a_predation"] * P * R
        - params["a_parasitism"] * P * Pa
        - mu_pest_t * P
    )
    dR = (
        params["e_conversion"] * params["a_predation"] * P * R
        - params["m_predator"] * R
        - mu_pred * R
    )
    dPa = (
        params["e_conversion"] * params["a_parasitism"] * P * Pa
        - params["m_parasitoid"] * Pa
        - mu_para * Pa
    )
    return np.array([dP, dR, dPa])


def simulate_population_trajectory(
    initial_pest: float,
    initial_predator: float,
    initial_parasitoid: float,
    chemistry_id: str | None,
    initial_concentration_ppm: float,
    decay_k_per_day: float,
    horizon_days: int = 30,
    params: dict | None = None,
) -> dict:
    """Simulate 30-day trajectory under a candidate intervention scenario."""
    params = params or DEFAULT_PARAMS
    if chemistry_id is None or initial_concentration_ppm <= 0:
        toxicity = {
            "predator_LC50_ppm": 1e9,
            "parasitoid_LC50_ppm": 1e9,
            "predator_max_mortality_d": 0.0,
            "parasitoid_max_mortality_d": 0.0,
        }
    else:
        toxicity = TOXICITY_DB[chemistry_id]
    t = np.linspace(0.0, float(horizon_days), horizon_days + 1)
    y0 = np.array([initial_pest, initial_predator, initial_parasitoid])
    sol = odeint(
        lotka_volterra_with_pesticide,
        y0,
        t,
        args=(params, initial_concentration_ppm, decay_k_per_day, toxicity),
        full_output=False,
    )
    # Cap populations to prevent overflow per §14.O.
    sol = np.clip(sol, 0.0, 10.0 * params["K_pest"])
    return {
        "days": t.tolist(),
        "pest": sol[:, 0].tolist(),
        "predator": sol[:, 1].tolist(),
        "parasitoid": sol[:, 2].tolist(),
    }
