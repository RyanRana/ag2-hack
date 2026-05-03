"""Soil-plant-atmosphere water balance.

FAO-56 Penman-Monteith reference evapotranspiration + van Genuchten soil
water retention. Single-step model for current state. Sources:
USDA-NRCS soil texture parameters; FAO-56 paper for ET0 and Kc.
"""

from __future__ import annotations

import numpy as np


# van Genuchten parameters by USDA soil texture class.
# theta_r, theta_s, alpha (1/cm), n
SOIL_TEXTURES: dict[str, dict[str, float]] = {
    "sand":       {"theta_r": 0.045, "theta_s": 0.430, "alpha": 0.145, "n": 2.68},
    "loamy_sand": {"theta_r": 0.057, "theta_s": 0.410, "alpha": 0.124, "n": 2.28},
    "sandy_loam": {"theta_r": 0.065, "theta_s": 0.410, "alpha": 0.075, "n": 1.89},
    "loam":       {"theta_r": 0.078, "theta_s": 0.430, "alpha": 0.036, "n": 1.56},
    "silt":       {"theta_r": 0.034, "theta_s": 0.460, "alpha": 0.016, "n": 1.37},
    "silt_loam":  {"theta_r": 0.067, "theta_s": 0.450, "alpha": 0.020, "n": 1.41},
    "clay_loam":  {"theta_r": 0.095, "theta_s": 0.410, "alpha": 0.019, "n": 1.31},
    "clay":       {"theta_r": 0.068, "theta_s": 0.380, "alpha": 0.008, "n": 1.09},
}

# FAO-56 mid-season crop coefficients.
CROP_KC: dict[str, float] = {
    "corn": 1.20, "soybean": 1.15, "wheat": 1.15, "tomato": 1.15,
    "lettuce": 1.00, "cotton": 1.20, "potato": 1.15, "default": 1.10,
}


def saturation_vapor_pressure(T_C: float) -> float:
    """Tetens formula. Returns kPa."""
    return 0.6108 * np.exp((17.27 * T_C) / (T_C + 237.3))


def penman_monteith_et0(
    T_C: float,
    RH_pct: float,
    u2_m_s: float,
    R_n_MJ_m2_d: float,
    elevation_m: float = 100.0,
    G_MJ_m2_d: float = 0.0,
) -> float:
    """FAO-56 reference evapotranspiration in mm/day."""
    # Bound inputs to physically reasonable ranges.
    T_C = float(np.clip(T_C, -10.0, 50.0))
    RH_pct = float(np.clip(RH_pct, 5.0, 95.0))
    u2_m_s = float(max(u2_m_s, 0.5))
    e_s = saturation_vapor_pressure(T_C)
    e_a = e_s * (RH_pct / 100.0)
    Delta = 4098.0 * e_s / ((T_C + 237.3) ** 2)
    P = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    gamma = 0.000665 * P
    num = (
        0.408 * Delta * (R_n_MJ_m2_d - G_MJ_m2_d)
        + gamma * (900.0 / (T_C + 273.0)) * u2_m_s * (e_s - e_a)
    )
    den = Delta + gamma * (1.0 + 0.34 * u2_m_s)
    return float(num / den)


def soil_water_potential(theta: float, soil_texture: str) -> float:
    """Inverse van Genuchten — returns plant-perspective psi in kPa (negative=drier)."""
    params = SOIL_TEXTURES[soil_texture]
    theta_r = params["theta_r"]
    theta_s = params["theta_s"]
    alpha_inv_cm = params["alpha"]
    n = params["n"]

    Se = (theta - theta_r) / (theta_s - theta_r)
    # Clamp Se to (0, 1) without losing dry-end resolution (§14.L).
    Se = float(np.clip(Se, 1e-9, 1.0 - 1e-9))
    if abs(n - 1.0) < 1e-6:
        # Degenerate case — fall back to PWP/sat per §14.L.
        if Se < 0.05:
            return -1500.0
        if Se > 0.95:
            return 0.0
    m = 1.0 - 1.0 / n
    psi_cm = (1.0 / alpha_inv_cm) * ((Se ** (-1.0 / m)) - 1.0) ** (1.0 / n)
    # 1 cm water column = 0.0981 kPa. Cap at -10 MPa for numerical hygiene.
    return float(max(-1.0e4, -psi_cm * 0.0981))


def water_stress_index(
    theta: float,
    soil_texture: str,
    crop_type: str,
    T_C: float,
    RH_pct: float,
    u2_m_s: float,
    R_n_MJ_m2_d: float,
    rooting_depth_m: float = 0.5,
) -> dict:
    """Returns stress index S in [0,1] plus its physical components."""
    et0 = penman_monteith_et0(T_C, RH_pct, u2_m_s, R_n_MJ_m2_d)
    kc = CROP_KC.get(crop_type, CROP_KC["default"])
    demand_mm = et0 * kc
    psi_kPa = soil_water_potential(theta, soil_texture)
    if psi_kPa >= -33.0:
        supply_factor = 1.0
    elif psi_kPa <= -1500.0:
        supply_factor = 0.0
    else:
        supply_factor = (psi_kPa - (-1500.0)) / ((-33.0) - (-1500.0))
    supply_mm = demand_mm * supply_factor
    stress_index = 1.0 - min(supply_mm / max(demand_mm, 1e-3), 1.0)
    return {
        "stress_index": float(np.clip(stress_index, 0.0, 1.0)),
        "demand_mm": float(demand_mm),
        "supply_mm": float(supply_mm),
        "soil_psi_kPa": float(psi_kPa),
    }
