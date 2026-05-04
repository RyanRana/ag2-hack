"""PesticideFateAgent — continuum-physics paradigm.

Same ConversableAgent + llm_config=False envelope as the ML agents. Per
spec §1, the unified envelope is the architectural moment: the *protocol*
is uniform; the *paradigm* is plural.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from autogen import ConversableAgent

from pesto.messages import InterventionAssessmentMessage
from pesto.physics.drift import deposition_at_locations


_CHEM_PATH = Path(__file__).resolve().parents[2] / "data" / "pesticide_chemistry.json"
CHEMISTRY_DB: dict[str, dict[str, Any]] = json.loads(_CHEM_PATH.read_text())

SPRAY_ACTIONS = {"targeted_spray", "targeted_fungicide"}


class PesticideFateAgent(ConversableAgent):
    """Physics, not ML. Vetoes spray candidates by predicted hazard."""

    def __init__(self, default_chemistry: str = "glyphosate") -> None:
        super().__init__(
            name="pesticide_fate",
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.default_chemistry = default_chemistry
        self.register_reply(
            trigger=[ConversableAgent, None],
            reply_func=self._assess_reply,
            position=0,
            remove_other_reply_funcs=True,
        )

    def _assess_reply(self, recipient, messages=None, sender=None, config=None):
        latest = messages[-1] if messages else {}
        payload = json.loads(latest.get("content", "{}"))
        plant_id = payload["plant_id"]
        action_type = payload["action_type"]
        action_params = payload.get("action_params", {})
        field = payload["field_state"]
        chemistry = payload.get("chemistry", self.default_chemistry)
        msg = self.assess_intervention(plant_id, action_type, action_params, field, chemistry)
        return True, {
            "role": "assistant",
            "name": self.name,
            "content": json.dumps(dataclasses.asdict(msg)),
        }

    def assess_intervention(
        self,
        plant_id: int,
        action_type: str,
        action_params: dict,
        field_state: dict,
        chemistry_id: str | None = None,
    ) -> InterventionAssessmentMessage:
        chemistry_id = chemistry_id or self.default_chemistry
        if action_type not in SPRAY_ACTIONS:
            return InterventionAssessmentMessage(
                sender=self.name,
                timestamp=time.time(),
                plant_id=plant_id,
                action_type=action_type,
                action_params=dict(action_params),
                off_target_deposition={},
                soil_half_life_days=0.0,
                time_to_offsite_hours=None,
                hazard_score=0.0,
                hazard_breakdown={"non_chemical_action": 1.0},
            )

        chem = CHEMISTRY_DB[chemistry_id]
        source = next(p for p in field_state["plants"] if p["plant_id"] == plant_id)
        source_xy = tuple(float(v) for v in source["xy_m"])
        non_target = [p for p in field_state["plants"] if p["plant_id"] != plant_id]
        target_xy = [tuple(float(v) for v in p["xy_m"]) for p in non_target]

        volume_ml = float(action_params.get("volume_ml", 5.0))
        conc_g_l = float(action_params.get("concentration_g_l", 360.0))
        duration_s = float(action_params.get("application_duration_s", 0.3))
        active_g = volume_ml * conc_g_l / 1000.0
        rate_g_s = active_g / max(duration_s, 1e-3)

        dep_idx = deposition_at_locations(
            source_xy=source_xy,
            target_xy=target_xy,
            wind_dir_deg=float(field_state["wind_dir_deg"]),
            wind_speed_m_s=float(field_state["wind_speed_m_s"]),
            application_rate_g_s=rate_g_s,
            application_duration_s=duration_s,
        )
        off_target_dep = {
            non_target[i]["plant_id"]: float(dep_idx[i]) for i in range(len(non_target))
        }

        threshold = float(chem["non_target_threshold_ppm"])
        deps = list(off_target_dep.values())
        max_dep = max(deps) if deps else 0.0
        n_above = sum(1 for d in deps if d > threshold)

        offsite_hours: float | None = None
        for p in non_target:
            if p["plant_id"] >= 90 and off_target_dep[p["plant_id"]] > threshold:
                dx = p["xy_m"][0] - source_xy[0]
                dy = p["xy_m"][1] - source_xy[1]
                dist = float(np.hypot(dx, dy))
                offsite_hours = dist / max(field_state["wind_speed_m_s"], 0.5) / 3600.0
                break

        dep_term = min(max_dep / (threshold + 1e-12), 5.0) / 5.0
        count_term = min(n_above / max(len(non_target), 1), 1.0)
        persist_term = min(chem["soil_dt50_days"] / 100.0, 1.0)
        hazard_score = float(0.5 * dep_term + 0.3 * count_term + 0.2 * persist_term)

        return InterventionAssessmentMessage(
            sender=self.name,
            timestamp=time.time(),
            plant_id=plant_id,
            action_type=action_type,
            action_params={
                "volume_ml": volume_ml,
                "concentration_g_l": conc_g_l,
                "application_duration_s": duration_s,
            },
            off_target_deposition=off_target_dep,
            soil_half_life_days=float(chem["soil_dt50_days"]),
            time_to_offsite_hours=offsite_hours,
            hazard_score=hazard_score,
            hazard_breakdown={
                "drift_max_ppm": float(max_dep),
                "neighbors_at_risk": float(n_above),
                "non_target_threshold_ppm": threshold,
                "ditch_arrival_hours": float(offsite_hours) if offsite_hours else -1.0,
                "soil_dt50_days": float(chem["soil_dt50_days"]),
            },
        )
