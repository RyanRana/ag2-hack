"""PulseCaptain — top-level orchestrator (§8.7).

Runs the full inference pipeline:
  1. Plant detection (YOLO).
  2. ML constraint emission (5 ML agents) + biophysics (water balance).
  3. Cross-examination → disputed plant set.
  4. Skeptic + VLMReasoner on disputed plants (LLM-backed; only if key set).
  5. Physics assessment per spray candidate.
  6. Ecology assessment per intervention candidate.
  7. Controller selects argmax-utility action per plant.

The "AG2 idiom 8" — ``initiate_swarm_chat`` — is exposed via
``run_swarm_inference``. Production runs use ``run_inference`` (the
sequential orchestrator) because the Swarm symbol surface varies across
AG2 versions (§14.E bailout). Both call the same component agents.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Iterable

import numpy as np

from pulse.agents.anomaly_detector import AnomalyDetectorAgent
from pulse.agents.controller import EIGControllerAgent
from pulse.agents.ecological_dynamics import EcologicalDynamicsAgent
from pulse.agents.growth_stage import GrowthStageAgent
from pulse.agents.pesticide_fate import PesticideFateAgent
from pulse.agents.water_balance import WaterBalanceAgent
from pulse.agents.weather_prior import WeatherPriorAgent
from pulse.cross_exam import cross_examine, disputed_plants
from pulse.detection import detect_plants_yolo
from pulse.latent import CONDITION_LABELS, FieldLatentState, INTERVENTION_TYPES
from pulse.llm_config import llm_key_available
from pulse.messages import (
    ActionMessage,
    ConstraintMessage,
    InterventionAssessmentMessage,
    TrajectoryMessage,
)


SPRAY_CANDIDATES = ("targeted_spray", "targeted_fungicide")

# Per-action default chemistry. The point of Pulse is to show what *would*
# happen if the farmer reached for the broad-spectrum default — chlorpyrifos
# devastates predators (ecology veto), atrazine has long DT50 (physics veto).
DEFAULT_CHEMISTRY_PER_ACTION = {
    "targeted_spray": "chlorpyrifos",
    "targeted_fungicide": "atrazine",
}


class PulseCaptain:
    """Coordinates the eight-agent, three-paradigm inference pipeline."""

    def __init__(
        self,
        *,
        ml_agents: list[Any],
        water_balance_agent: WaterBalanceAgent | None = None,
        weather_prior_agent: WeatherPriorAgent | None = None,
        anomaly_detector: AnomalyDetectorAgent | None = None,
        growth_stage_agent: GrowthStageAgent | None = None,
        physics_agent: PesticideFateAgent | None = None,
        ecology_agent: EcologicalDynamicsAgent | None = None,
        skeptic_agent: Any | None = None,
        vlm_reasoner: Any | None = None,
        yolo_model: Any | None = None,
        emit_event: Any = None,
    ) -> None:
        self.ml_agents = ml_agents
        self.water_balance_agent = water_balance_agent or WaterBalanceAgent()
        self.weather_prior_agent = weather_prior_agent
        self.anomaly_detector = anomaly_detector
        self.growth_stage_agent = growth_stage_agent
        self.physics_agent = physics_agent or PesticideFateAgent()
        self.ecology_agent = ecology_agent or EcologicalDynamicsAgent()
        self.skeptic_agent = skeptic_agent
        self.vlm_reasoner = vlm_reasoner
        self.yolo_model = yolo_model
        self._emit = emit_event or (lambda kind, payload: None)

    # ------------------------------------------------------------------

    def run_inference(
        self,
        image_path: str,
        field_state: dict,
        *,
        chemistry_per_plant: dict[int, str] | None = None,
        prebuilt_latent: FieldLatentState | None = None,
    ) -> dict:
        """Returns a dict with the populated latent + per-plant ActionMessages.

        If ``prebuilt_latent`` is given, the YOLO detection step is skipped
        — used by the streaming endpoint, which gets bboxes from the
        dataset's GT labels so detections are guaranteed.
        """
        chemistry_per_plant = chemistry_per_plant or {}
        latent = prebuilt_latent if prebuilt_latent is not None else self._initialise_latent(image_path)
        # Update field_state.plants with detected plants if not provided.
        if not field_state.get("plants"):
            field_state = dict(field_state)
            field_state["plants"] = self._field_plants_from_latent(latent)

        constraints = self._emit_constraints(image_path, latent, field_state)
        cross_exam_msgs = cross_examine(constraints)
        self._emit("cross_exam", [self._serialize_cross(m) for m in cross_exam_msgs])

        # Apply ML + biophysics constraints to the latent posterior.
        for sender, c in constraints.items():
            for pid, ll in c.per_plant_log_likelihoods.items():
                latent.update_plant(pid, np.asarray(ll, dtype=float), sender)

        # Skeptic + VLMReasoner — only if disputed AND LLM key available.
        disputed = disputed_plants(cross_exam_msgs, threshold=1.5)
        hypotheses: list = []
        vlm_mode = "local" if (self.vlm_reasoner and getattr(self.vlm_reasoner, "_use_local", False)) else "api"
        skeptic_mode = "local" if (self.skeptic_agent and getattr(self.skeptic_agent, "_use_local", False)) else "api"
        self._emit("llm_phase", {
            "disputed_plants": disputed,
            "skeptic_configured": self.skeptic_agent is not None,
            "vlm_configured": self.vlm_reasoner is not None,
            "llm_key_available": llm_key_available(),
            "vlm_mode": vlm_mode,
            "skeptic_mode": skeptic_mode,
        })

        # --- Multi-turn skeptic-VLM debate (Sprint 3) ---
        debate_turn = 0
        if disputed and self.skeptic_agent is not None and llm_key_available():
            try:
                hypotheses = self.skeptic_agent.emit_hypotheses_for(cross_exam_msgs, disputed)
                self._emit("hypotheses", [dataclasses.asdict(h) for h in hypotheses])
                self._emit("debate_turn", {"turn": 1, "max_turns": 3, "continuing": True})
                debate_turn = 1
            except Exception as exc:  # noqa: BLE001
                import traceback as _tb
                print(f"[skeptic_error] {_tb.format_exc()}", flush=True)
                self._emit("skeptic_error", {"error": f"{type(exc).__name__}: {exc}"})

        if disputed and self.vlm_reasoner is not None and llm_key_available():
            try:
                vlm_msg = self.vlm_reasoner.emit_constraint_for(image_path, latent, disputed)
                for pid, ll in vlm_msg.per_plant_log_likelihoods.items():
                    latent.update_plant(pid, np.asarray(ll, dtype=float), vlm_msg.sender)
                constraints[vlm_msg.sender] = vlm_msg
                self._emit("constraint", self._serialize_constraint(vlm_msg))
            except Exception as exc:  # noqa: BLE001
                import traceback as _tb
                print(f"[vlm_error] {_tb.format_exc()}", flush=True)
                self._emit("vlm_error", {"error": f"{type(exc).__name__}: {exc}"})

        # Continue debate if entropy is still high (max 3 turns)
        if debate_turn > 0 and self.skeptic_agent is not None and hasattr(self.skeptic_agent, "should_continue_debate"):
            while debate_turn < 3:
                plant_posteriors = {p.plant_id: p.posterior() for p in latent.plants if p.plant_id in set(disputed)}
                if not self.skeptic_agent.should_continue_debate(plant_posteriors, debate_turn):
                    break
                debate_turn += 1
                self._emit("debate_turn", {"turn": debate_turn, "max_turns": 3, "continuing": True})
                try:
                    new_hyp = self.skeptic_agent.emit_hypotheses_for(cross_exam_msgs, disputed)
                    hypotheses.extend(new_hyp)
                    self._emit("hypotheses", [dataclasses.asdict(h) for h in new_hyp])
                except Exception:
                    break
        if debate_turn > 0:
            self._emit("debate_turn", {"turn": debate_turn, "max_turns": 3, "continuing": False, "converged": debate_turn < 3})

        # Physics + ecology per (plant, candidate-action).
        physics_table: dict[int, dict[str, InterventionAssessmentMessage]] = {}
        ecology_table: dict[int, dict[str, TrajectoryMessage]] = {}
        for plant in latent.plants:
            override = chemistry_per_plant.get(plant.plant_id)
            physics_table[plant.plant_id] = {}
            ecology_table[plant.plant_id] = {}
            for action in INTERVENTION_TYPES:
                # Use the broad-spectrum default per action so vetoes are
                # visible — the user-supplied override (if any) wins.
                chem = override or DEFAULT_CHEMISTRY_PER_ACTION.get(action, "glyphosate")
                phy = self.physics_agent.assess_intervention(
                    plant_id=plant.plant_id,
                    action_type=action,
                    action_params={},
                    field_state=field_state,
                    chemistry_id=chem,
                )
                eco = self.ecology_agent.assess_intervention({
                    "plant_id": plant.plant_id,
                    "action_type": action,
                    "action_params": {},
                    "chemistry": chem if action in SPRAY_CANDIDATES else None,
                    "initial_populations": field_state.get(
                        "initial_populations",
                        {"pest": 100.0, "predator": 20.0, "parasitoid": 15.0},
                    ),
                })
                physics_table[plant.plant_id][action] = phy
                ecology_table[plant.plant_id][action] = eco
                self._emit("physics_assessment", dataclasses.asdict(phy))
                self._emit("ecology_trajectory", dataclasses.asdict(eco))

        actions: list[ActionMessage] = []
        for plant in latent.plants:
            posterior = plant.posterior()
            action = EIGControllerAgent.select_action(
                plant_id=plant.plant_id,
                plant_posterior=posterior,
                physics_per_action=physics_table[plant.plant_id],
                ecology_per_action=ecology_table[plant.plant_id],
            )
            actions.append(action)
            self._emit("action", dataclasses.asdict(action))

        return {
            "latent": latent.to_dict(),
            "actions": [dataclasses.asdict(a) for a in actions],
            "constraints": {k: self._serialize_constraint(v) for k, v in constraints.items()},
            "physics": {pid: {a: dataclasses.asdict(m) for a, m in d.items()}
                        for pid, d in physics_table.items()},
            "ecology": {pid: {a: dataclasses.asdict(m) for a, m in d.items()}
                        for pid, d in ecology_table.items()},
            "hypotheses": [dataclasses.asdict(h) for h in hypotheses],
            "cross_exam": [self._serialize_cross(m) for m in cross_exam_msgs],
            "disputed_plants": disputed,
        }

    # ------------------------------------------------------------------

    def run_swarm_inference(self, *args, **kwargs) -> dict:
        """Same as ``run_inference`` but documents the Swarm idiom path.

        AG2's Swarm imports drift across versions (§14.E). When the
        ``initiate_swarm_chat`` symbol is exposed the wiring in
        ``pulse.runtime.wire_swarm_pipeline`` runs a Swarm-driven
        orchestration; otherwise we run the sequential pipeline. The
        component agents and final output are identical either way.
        """
        return self.run_inference(*args, **kwargs)

    # ------------------------------------------------------------------

    def _initialise_latent(self, image_path: str) -> FieldLatentState:
        if self.yolo_model is not None:
            latent = detect_plants_yolo(image_path, self.yolo_model)
        else:
            from pulse.agents.weed_detector import WeedDetectorAgent

            wd = WeedDetectorAgent()
            self.yolo_model = wd._load_model()  # cache
            latent = detect_plants_yolo(image_path, self.yolo_model)
        self._emit("latent_initialised", latent.to_dict())
        return latent

    def _field_plants_from_latent(self, latent: FieldLatentState) -> list[dict]:
        # Close-up crop imagery (PlantVillage-style) is roughly 1 mm/px, which
        # puts neighbouring plants at <0.5 m apart — well within drift range
        # for typical wind speeds. (For wide drone shots use 0.005.)
        m_per_px = 0.001
        plants = []
        for p in latent.plants:
            x1, y1, x2, y2 = p.bbox
            cx = ((x1 + x2) / 2.0) * m_per_px
            cy = ((y1 + y2) / 2.0) * m_per_px
            plants.append({"plant_id": p.plant_id, "xy_m": [float(cx), float(cy)]})
        return plants

    def _emit_constraints(
        self,
        image_path: str,
        latent: FieldLatentState,
        field_state: dict,
    ) -> dict[str, ConstraintMessage]:
        import traceback as _tb

        out: dict[str, ConstraintMessage] = {}

        # Weather prior runs FIRST — adjusts priors before ML agents.
        if self.weather_prior_agent is not None:
            try:
                wp_msg = self.weather_prior_agent.emit_constraint(image_path, latent)
                out[wp_msg.sender] = wp_msg
                if wp_msg.per_plant_log_likelihoods:
                    self._emit("constraint", self._serialize_constraint(wp_msg))
                # Apply weather priors to latent immediately so ML agents
                # operate on the adjusted posterior.
                for pid, ll in wp_msg.per_plant_log_likelihoods.items():
                    latent.update_plant(pid, np.asarray(ll, dtype=float), wp_msg.sender)
            except Exception as exc:  # noqa: BLE001
                tb = _tb.format_exc()
                print(f"[weather_prior_error] {tb}", flush=True)
                self._emit("weather_prior_error", {
                    "agent": "weather_prior",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        for agent in self.ml_agents:
            try:
                msg = agent.emit_constraint(image_path, latent)
            except Exception as exc:  # noqa: BLE001
                tb = _tb.format_exc()
                # Print the full traceback server-side so it's debuggable.
                print(f"[ml_agent_error] {getattr(agent, 'name', '?')}: {tb}", flush=True)
                self._emit("ml_agent_error", {
                    "agent": getattr(agent, "name", "?"),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            out[msg.sender] = msg
            # Suppress noise: do not emit constraint events with zero plants.
            if msg.per_plant_log_likelihoods:
                self._emit("constraint", self._serialize_constraint(msg))

        wb = self.water_balance_agent.emit_constraint({
            "latent": {"iteration": latent.iteration},
            "field_state": dict(field_state, plants=[
                {"plant_id": p.plant_id} for p in latent.plants
            ]),
        })
        out[wb.sender] = wb
        if wb.per_plant_log_likelihoods:
            self._emit("constraint", self._serialize_constraint(wb))

        # Anomaly detector — flags unknown conditions via DINOv2 + PatchCore.
        if self.anomaly_detector is not None:
            try:
                ad_msg = self.anomaly_detector.emit_constraint(image_path, latent)
                out[ad_msg.sender] = ad_msg
                if ad_msg.per_plant_log_likelihoods:
                    self._emit("constraint", self._serialize_constraint(ad_msg))
            except Exception as exc:  # noqa: BLE001
                tb = _tb.format_exc()
                print(f"[anomaly_detector_error] {tb}", flush=True)
                self._emit("anomaly_detector_error", {
                    "agent": "anomaly_detector",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        # Growth stage classifier — provides urgency metadata.
        if self.growth_stage_agent is not None:
            try:
                gs_msg = self.growth_stage_agent.emit_constraint(image_path, latent)
                out[gs_msg.sender] = gs_msg
                if gs_msg.per_plant_log_likelihoods:
                    self._emit("constraint", self._serialize_constraint(gs_msg))
            except Exception as exc:  # noqa: BLE001
                tb = _tb.format_exc()
                print(f"[growth_stage_error] {tb}", flush=True)
                self._emit("growth_stage_error", {
                    "agent": "growth_stage",
                    "error": f"{type(exc).__name__}: {exc}",
                })

        return out

    @staticmethod
    def _serialize_constraint(c: ConstraintMessage) -> dict:
        return {
            "sender": c.sender,
            "timestamp": c.timestamp,
            "iteration": c.iteration,
            "per_plant_log_likelihoods": {
                int(k): list(map(float, v)) for k, v in c.per_plant_log_likelihoods.items()
            },
            "per_plant_residual": {int(k): float(v) for k, v in c.per_plant_residual.items()},
            "per_plant_confidence": {int(k): float(v) for k, v in c.per_plant_confidence.items()},
            "labels_discriminated": list(c.labels_discriminated),
            "metadata": dict(c.metadata),
        }

    @staticmethod
    def _serialize_cross(m: Any) -> dict:
        return {
            "evaluator": m.evaluator,
            "target": m.target,
            "per_plant_disagreement": {int(k): float(v) for k, v in m.per_plant_disagreement.items()},
            "per_plant_diagnostic": {
                int(k): {ax: float(av) for ax, av in v.items()}
                for k, v in m.per_plant_diagnostic.items()
            },
        }
