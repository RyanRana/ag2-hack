"""Gaussian-plume short-range drift model for ground-level spray applications.

Reference: Pasquill stability classes; ISO 22866 spray drift methodology.
Simplified for low-altitude (~1m) ground-based spray applications, omitting
buoyancy/inversion effects that apply only to stack emissions.
"""

from __future__ import annotations

import numpy as np


# Pasquill-Gifford dispersion coefficients (open country, daytime).
# σ_y = a_y * x^b_y; σ_z = a_z * x^b_z, x in metres.
PASQUILL_C_OPEN = {"ay": 0.36, "by": 0.86, "az": 0.20, "bz": 0.81}
PASQUILL_D_OPEN = {"ay": 0.32, "by": 0.78, "az": 0.22, "bz": 0.78}
PASQUILL_E_OPEN = {"ay": 0.24, "by": 0.74, "az": 0.14, "bz": 0.74}


def gaussian_plume_concentration(
    x_m: np.ndarray,
    y_m: np.ndarray,
    z_m: np.ndarray,
    Q_g_per_s: float,
    u_m_per_s: float,
    H_m: float = 1.0,
    coefs: dict | None = None,
) -> np.ndarray:
    """Concentration in g/m^3 at each (x, y, z) sample.

    Pasquill-Gifford Gaussian plume:

        C(x,y,z) = Q / (2π u σ_y σ_z)
                   * exp(-y² / 2σ_y²)
                   * [exp(-(z-H)² / 2σ_z²) + exp(-(z+H)² / 2σ_z²)]

    The bracketed term is the image source for ground reflection.
    """
    coefs = coefs or PASQUILL_D_OPEN
    x_safe = np.maximum(x_m, 1.0)
    sigma_y = coefs["ay"] * x_safe ** coefs["by"]
    sigma_z = coefs["az"] * x_safe ** coefs["bz"]
    u = max(u_m_per_s, 0.5)
    norm = Q_g_per_s / (2 * np.pi * u * sigma_y * sigma_z + 1e-12)
    crosswind = np.exp(-(y_m ** 2) / (2 * sigma_y ** 2 + 1e-12))
    vertical = (
        np.exp(-((z_m - H_m) ** 2) / (2 * sigma_z ** 2 + 1e-12))
        + np.exp(-((z_m + H_m) ** 2) / (2 * sigma_z ** 2 + 1e-12))
    )
    C = norm * crosswind * vertical
    C = np.where(x_m < 0, 0.0, C)
    return C


def deposition_at_locations(
    source_xy: tuple[float, float],
    target_xy: list[tuple[float, float]],
    wind_dir_deg: float,
    wind_speed_m_s: float,
    application_rate_g_s: float,
    application_duration_s: float,
    deposition_velocity_m_s: float = 0.01,
) -> dict[int, float]:
    """Predict deposition (ppm-equivalent on canopy) at each target location.

    ``wind_dir_deg`` follows the meteorological convention: the direction
    the wind is *coming FROM*. So wind_dir_deg=270 means wind from the west
    (flowing east → +x).
    """
    wind_to_rad = np.deg2rad((wind_dir_deg + 180.0) % 360.0)
    cos_t = np.cos(wind_to_rad)
    sin_t = np.sin(wind_to_rad)
    sx, sy = source_xy
    deposition: dict[int, float] = {}
    for idx, (tx, ty) in enumerate(target_xy):
        dx = tx - sx
        dy = ty - sy
        x_along = dx * cos_t + dy * sin_t
        y_cross = -dx * sin_t + dy * cos_t
        C_g_m3 = float(
            gaussian_plume_concentration(
                np.array([x_along]),
                np.array([y_cross]),
                np.array([1.0]),
                Q_g_per_s=application_rate_g_s,
                u_m_per_s=max(wind_speed_m_s, 0.5),
            )[0]
        )
        dep_g_m2 = C_g_m3 * deposition_velocity_m_s * application_duration_s
        # Convert to ppm-equivalent assuming a small canopy mass exposure.
        ppm = (dep_g_m2 * 0.1) * 1e6 / 1000.0
        deposition[idx] = ppm
    return deposition


def soil_persistence_curve(
    initial_conc_g_kg: float,
    dt50_days: float,
    days: np.ndarray,
) -> np.ndarray:
    """First-order kinetics: C(t) = C0 * exp(-ln(2)/DT50 * t)."""
    k = np.log(2.0) / dt50_days
    return initial_conc_g_kg * np.exp(-k * days)
