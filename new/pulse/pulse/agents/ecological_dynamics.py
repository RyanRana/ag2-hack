"""EcologicalDynamicsAgent — population biology paradigm.

Same envelope as physics + ML + biophysics agents (ConversableAgent +
llm_config=False + register_reply). Internal computation: Lotka-Volterra
ODE with pesticide toxicity. Emits TrajectoryMessages.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time

import numpy as np
from autogen import ConversableAgent

from pulse.biology.lotka_volterra import simulate_population_trajectory
from pulse.messages import TrajectoryMessage


SPRAY_ACTIONS = {"targeted_spray", "targeted_fungicide"}


class EcologicalDynamicsAgent(ConversableAgent):
    def __init__(self) -> None:
        super().__init__(
            name="ecological_dynamics",
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.register_reply(
            trigger=[ConversableAgent, None],
            reply_func=self._assess_reply,
            position=0,
            remove_other_reply_funcs=True,
        )

    def _assess_reply(self, recipient, messages=None, sender=None, config=None):
        latest = messages[-1] if messages else {}
        payload = json.loads(latest.get("content", "{}"))
        msg = self.assess_intervention(payload)
        return True, {
            "role": "assistant",
            "name": self.name,
            "content": json.dumps(dataclasses.asdict(msg)),
        }

    def assess_intervention(self, payload: dict) -> TrajectoryMessage:
        plant_id = int(payload["plant_id"])
        action_type = payload["action_type"]
        action_params = payload.get("action_params", {})
        chemistry = payload.get("chemistry")
        pops = payload.get(
            "initial_populations",
            {"pest": 100.0, "predator": 20.0, "parasitoid": 15.0},
        )

        applied_chemical = action_type in SPRAY_ACTIONS and chemistry is not None
        if applied_chemical:
            volume_ml = float(action_params.get("volume_ml", 5.0))
            conc_g_l = float(action_params.get("concentration_g_l", 360.0))
            initial_C_ppm = volume_ml * conc_g_l * 1.0
            decay_k = math.log(2.0) / 23.0  # use a sensible default DT50
        else:
            chemistry = None
            initial_C_ppm = 0.0
            decay_k = 0.0

        traj = simulate_population_trajectory(
            initial_pest=float(pops["pest"]),
            initial_predator=float(pops["predator"]),
            initial_parasitoid=float(pops["parasitoid"]),
            chemistry_id=chemistry,
            initial_concentration_ppm=initial_C_ppm,
            decay_k_per_day=decay_k,
            horizon_days=30,
        )
        p_norm = [v / max(traj["pest"][0], 1e-9) for v in traj["pest"]]
        r_norm = [v / max(traj["predator"][0], 1e-9) for v in traj["predator"]]
        pa_norm = [v / max(traj["parasitoid"][0], 1e-9) for v in traj["parasitoid"]]

        predator_drop_d14 = max(0.0, 1.0 - r_norm[14])
        parasitoid_drop_d14 = max(0.0, 1.0 - pa_norm[14])
        pest_rebound_d30 = max(0.0, p_norm[30] - 1.0)

        # Ecological cost is the *additional* harm caused by the intervention
        # beyond the no-intervention baseline. When no chemical is applied
        # (laser zap, no_action, irrigation, …), the cost is zero — we never
        # penalize an action for the ambient pest-predator dynamics it didn't
        # cause. When a chemical is applied, predator/parasitoid drop is
        # weighted heavily and pest rebound only counts if it co-occurs with
        # predator depletion (the chemical-treadmill signature).
        if not applied_chemical:
            cost = 0.0
        else:
            rebound_when_predators_gone = (
                min(pest_rebound_d30 / 3.0, 1.0) * predator_drop_d14
            )
            cost = float(np.clip(
                0.45 * predator_drop_d14
                + 0.35 * parasitoid_drop_d14
                + 0.20 * rebound_when_predators_gone,
                0.0,
                1.0,
            ))

        return TrajectoryMessage(
            sender=self.name,
            timestamp=time.time(),
            plant_id=plant_id,
            action_type=action_type,
            action_params=dict(action_params),
            days=traj["days"],
            pest_trajectory=p_norm,
            predator_trajectory=r_norm,
            parasitoid_trajectory=pa_norm,
            ecological_cost_score=cost,
            cost_breakdown={
                "predator_drop_pct_d14": float(predator_drop_d14),
                "parasitoid_drop_pct_d14": float(parasitoid_drop_d14),
                "pest_rebound_factor_d30": float(p_norm[30]),
            },
        )
