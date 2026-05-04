"""WaterBalanceAgent — biophysics paradigm.

Same ConversableAgent + llm_config=False envelope as ML and physics agents.
Internal computation: Penman-Monteith ET0 + van Genuchten soil retention.
Emits a ConstraintMessage that elevates ``water_stress`` and suppresses
``disease`` when the field is in genuine water deficit, and conversely
suppresses ``water_stress`` when soil water is plentiful — resolving the
canonical wilting-vs-disease ML ambiguity.
"""

from __future__ import annotations

import json
import time

import numpy as np
from autogen import ConversableAgent

from pesto.biophysics.water_balance import water_stress_index
from pesto.latent import CONDITION_LABELS
from pesto.messages import ConstraintMessage


class WaterBalanceAgent(ConversableAgent):
    def __init__(self, default_soil_texture: str = "loam") -> None:
        super().__init__(
            name="water_balance",
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.default_soil_texture = default_soil_texture
        self.register_reply(
            trigger=[ConversableAgent, None],
            reply_func=self._emit_reply,
            position=0,
            remove_other_reply_funcs=True,
        )

    def _emit_reply(self, recipient, messages=None, sender=None, config=None):
        latest = messages[-1] if messages else {}
        payload = json.loads(latest.get("content", "{}"))
        constraint = self.emit_constraint(payload)
        return True, {
            "role": "assistant",
            "name": self.name,
            "content": json.dumps(self._serialize(constraint)),
        }

    def emit_constraint(self, payload: dict) -> ConstraintMessage:
        latent = payload.get("latent", {})
        field = payload["field_state"]
        wsi = water_stress_index(
            theta=field["soil_moisture_m3_m3"],
            soil_texture=field.get("soil_texture", self.default_soil_texture),
            crop_type=field.get("crop_type", "default"),
            T_C=field["T_C"],
            RH_pct=field["RH_pct"],
            u2_m_s=field["u2_m_s"],
            R_n_MJ_m2_d=field["R_n_MJ_m2_d"],
        )
        S = wsi["stress_index"]
        log_lik = np.zeros(len(CONDITION_LABELS))
        i_water = CONDITION_LABELS.index("water_stress")
        i_disease = CONDITION_LABELS.index("disease")
        i_nutrient = CONDITION_LABELS.index("nutrient_stress")
        i_healthy = CONDITION_LABELS.index("healthy_crop")

        if S > 0.6:
            log_lik[i_water] = 2.0
            log_lik[i_disease] = -1.5
            log_lik[i_nutrient] = -1.0
        elif S > 0.3:
            log_lik[i_water] = 1.0
            log_lik[i_disease] = -0.5
        elif S < 0.1:
            log_lik[i_water] = -1.5
            log_lik[i_healthy] = 0.5

        plants = field.get("plants", [])
        per_plant_ll = {p["plant_id"]: log_lik.copy() for p in plants}
        per_plant_resid = {p["plant_id"]: 0.0 for p in plants}
        confidence = float(0.4 + 0.6 * abs(S - 0.5) * 2)
        per_plant_conf = {p["plant_id"]: min(confidence, 1.0) for p in plants}

        return ConstraintMessage(
            sender=self.name,
            timestamp=time.time(),
            iteration=int(latent.get("iteration", 0)),
            per_plant_log_likelihoods=per_plant_ll,
            per_plant_residual=per_plant_resid,
            per_plant_confidence=per_plant_conf,
            labels_discriminated=[
                "water_stress",
                "disease",
                "nutrient_stress",
                "healthy_crop",
            ],
            metadata={
                "stress_index": float(S),
                "demand_mm_per_day": float(wsi["demand_mm"]),
                "supply_mm_per_day": float(wsi["supply_mm"]),
                "soil_psi_kPa": float(wsi["soil_psi_kPa"]),
            },
        )

    @staticmethod
    def _serialize(c: ConstraintMessage) -> dict:
        return {
            "sender": c.sender,
            "timestamp": c.timestamp,
            "iteration": c.iteration,
            "per_plant_log_likelihoods": {
                int(k): v.tolist() for k, v in c.per_plant_log_likelihoods.items()
            },
            "per_plant_residual": {int(k): float(v) for k, v in c.per_plant_residual.items()},
            "per_plant_confidence": {int(k): float(v) for k, v in c.per_plant_confidence.items()},
            "labels_discriminated": c.labels_discriminated,
            "metadata": c.metadata,
        }
