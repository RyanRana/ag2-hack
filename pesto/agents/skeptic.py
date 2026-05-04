"""SkepticAgent — local LLM + external API fallback with multi-turn debate.

Enhanced with:
  1. **Local inference**: Uses same model as VLM (text-only mode) via
     pesto.local_model.get_shared_backend(). Zero API cost.
  2. **Multi-turn convergence**: If posterior entropy still high after VLM
     update, skeptic counter-argues. Max 3 total turns, then accept
     current posterior.
  3. **External API fallback**: Original AssistantAgent + register_function
     path when local model is unavailable.

AG2 idiom: ``AssistantAgent`` whose ONLY output is a typed tool call to
``propose_alternative_hypothesis``. The LLM never speaks prose. The tool
emits a HypothesisMessage; the controller fuses it into the per-plant
posterior.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Annotated

import numpy as np
from autogen import AssistantAgent, UserProxyAgent, register_function

from pesto.cross_exam import dominant_axes_for_plant
from pesto.latent import CONDITION_LABELS
from pesto.llm_config import openai_llm_config
from pesto.local_model import LocalModelBackend, get_shared_backend, parse_structured_output
from pesto.messages import CrossExamMessage, HypothesisMessage


SYSTEM_MESSAGE = (
    "You are an alternative-hypothesis proposer for a precision-agriculture "
    "pipeline. When invoked, you receive a list of disputed plants with "
    "per-axis disagreement magnitudes. For each disputed plant you MUST call "
    "propose_alternative_hypothesis exactly once with structured arguments. "
    "Pick a hypothesis_id like 'young_crop_misidentified_as_weed', "
    "'disease_masquerading_as_nutrient_stress', "
    "'water_stress_concurrent_with_pest', etc. NEVER respond in prose. "
    "NEVER explain. ONLY call the tool."
)

_LOCAL_PROMPT_TEMPLATE = """You are an alternative-hypothesis proposer for a precision-agriculture pipeline.

Disputed plants with their disagreement patterns:
{plant_data}

For each plant, propose an alternative hypothesis. Respond with a JSON array where each element has:
- "plant_id": int
- "hypothesis_id": string (e.g. "young_crop_misidentified_as_weed", "disease_masquerading_as_nutrient_stress")
- "log_posterior": float (your confidence, range -3 to 0)
- "evidence_axes": dict mapping condition labels to float evidence values

Example response:
[{{"plant_id": 0, "hypothesis_id": "water_stress_concurrent_with_pest", "log_posterior": -0.5, "evidence_axes": {{"healthy_crop": -0.2, "weed": 0.0, "disease": 0.1, "nutrient_stress": 0.0, "water_stress": 0.6, "pest_damage": 0.4, "ambiguous": 0.1}}}}]

Respond ONLY with the JSON array:"""

MAX_DEBATE_TURNS = 3
ENTROPY_CONVERGENCE_THRESHOLD = 1.2  # below this, consider converged


class SkepticAgent(AssistantAgent):
    """LLM-backed agent that emits HypothesisMessages.

    Supports local model (text-only mode, shared with VLM) with external
    API fallback. Implements multi-turn debate protocol.
    """

    def __init__(
        self,
        *,
        llm_config: dict | None = None,
        local_backend: LocalModelBackend | None = None,
    ) -> None:
        self._local_backend = local_backend
        self._use_local = False

        if self._local_backend is not None and self._local_backend.is_loaded:
            self._use_local = True

        if not self._use_local:
            try:
                cfg = llm_config or openai_llm_config()
            except Exception:
                cfg = False  # no LLM available at all
        else:
            cfg = False  # local model handles inference, not AG2 LLM

        super().__init__(
            name="skeptic",
            llm_config=cfg,
            system_message=SYSTEM_MESSAGE,
        )
        self._hypotheses: list[HypothesisMessage] = []

        agent_self = self

        def propose_alternative_hypothesis(
            plant_id: Annotated[int, "Which plant the hypothesis applies to"],
            hypothesis_id: Annotated[str, "Short identifier for the hypothesis"],
            log_posterior: Annotated[float, "Log posterior probability"],
            evidence_axis_healthy_crop: Annotated[float, "evidence on healthy_crop axis"],
            evidence_axis_weed: Annotated[float, "evidence on weed axis"],
            evidence_axis_disease: Annotated[float, "evidence on disease axis"],
            evidence_axis_nutrient_stress: Annotated[float, "evidence on nutrient_stress axis"],
            evidence_axis_water_stress: Annotated[float, "evidence on water_stress axis"],
            evidence_axis_pest_damage: Annotated[float, "evidence on pest_damage axis"],
            evidence_axis_ambiguous: Annotated[float, "evidence on ambiguous axis"],
        ) -> dict:
            return agent_self._record_hypothesis(
                plant_id=plant_id,
                hypothesis_id=hypothesis_id,
                log_posterior=log_posterior,
                evidence_axes={
                    "healthy_crop": float(evidence_axis_healthy_crop),
                    "weed": float(evidence_axis_weed),
                    "disease": float(evidence_axis_disease),
                    "nutrient_stress": float(evidence_axis_nutrient_stress),
                    "water_stress": float(evidence_axis_water_stress),
                    "pest_damage": float(evidence_axis_pest_damage),
                    "ambiguous": float(evidence_axis_ambiguous),
                },
            )

        # Only register tools with AG2 when using external API path.
        # Local mode bypasses AG2 tool calling entirely.
        if cfg and cfg is not False:
            register_function(
                propose_alternative_hypothesis,
                caller=self,
                executor=self,
                description=(
                    "Propose an alternative hypothesis to explain a cross-agent "
                    "disagreement on a single plant."
                ),
            )
        self.propose_alternative_hypothesis = propose_alternative_hypothesis

    # --- Public entry points ------------------------------------------------

    def emit_hypotheses_for(
        self,
        cross_exam_msgs: list[CrossExamMessage],
        disputed_plant_ids: list[int],
    ) -> list[HypothesisMessage]:
        """Generate alternative hypotheses for disputed plants.

        Uses local model if available, otherwise external API.
        """
        self._hypotheses.clear()

        if self._use_local:
            return self._emit_local(cross_exam_msgs, disputed_plant_ids)
        return self._emit_external(cross_exam_msgs, disputed_plant_ids)

    def should_continue_debate(
        self,
        plant_posteriors: dict[int, np.ndarray],
        turn: int,
    ) -> bool:
        """Check if the debate should continue based on posterior entropy.

        Parameters
        ----------
        plant_posteriors : dict of plant_id -> posterior probability array.
        turn : Current debate turn (0-indexed).

        Returns
        -------
        True if entropy is still high and we haven't hit max turns.
        """
        if turn >= MAX_DEBATE_TURNS:
            return False

        for pid, posterior in plant_posteriors.items():
            p = posterior / (posterior.sum() + 1e-12)
            entropy = float(-np.sum(p * np.log(p + 1e-12)))
            if entropy > ENTROPY_CONVERGENCE_THRESHOLD:
                return True

        return False

    # --- Local inference path -----------------------------------------------

    def _emit_local(
        self,
        cross_exam_msgs: list[CrossExamMessage],
        disputed_plant_ids: list[int],
    ) -> list[HypothesisMessage]:
        """Generate hypotheses using local LLM (text-only mode)."""
        plant_data_lines = []
        for pid in disputed_plant_ids:
            axes = dominant_axes_for_plant(cross_exam_msgs, pid)
            sorted_axes = sorted(axes.items(), key=lambda kv: -kv[1])[:3]
            axes_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted_axes)
            plant_data_lines.append(f"  plant_id={pid} dominant_axes: {axes_str}")

        prompt = _LOCAL_PROMPT_TEMPLATE.format(
            plant_data="\n".join(plant_data_lines)
        )

        try:
            response = self._local_backend.generate_text_only(
                prompt, max_new_tokens=512, temperature=0.2
            )
            parsed = parse_structured_output(response)

            if parsed is None:
                return []

            # Handle both single dict and list of dicts
            items = parsed if isinstance(parsed, list) else [parsed]

            for item in items:
                if not isinstance(item, dict):
                    continue
                pid = int(item.get("plant_id", -1))
                if pid not in disputed_plant_ids:
                    continue
                evidence = item.get("evidence_axes", {})
                self._record_hypothesis(
                    plant_id=pid,
                    hypothesis_id=str(item.get("hypothesis_id", "unknown")),
                    log_posterior=float(item.get("log_posterior", -1.0)),
                    evidence_axes={
                        label: float(evidence.get(label, 0.0))
                        for label in CONDITION_LABELS
                    },
                )
        except Exception:
            pass

        return list(self._hypotheses)

    # --- External API path (fallback) ---------------------------------------

    def _emit_external(
        self,
        cross_exam_msgs: list[CrossExamMessage],
        disputed_plant_ids: list[int],
    ) -> list[HypothesisMessage]:
        """Generate hypotheses using external API via AG2 AssistantAgent."""
        prompt_text = _build_prompt(cross_exam_msgs, disputed_plant_ids)
        driver = UserProxyAgent(
            name="skeptic_driver",
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config=False,
            max_consecutive_auto_reply=1 + len(disputed_plant_ids),
            is_termination_msg=lambda m: (
                m.get("name") == self.name
                and not m.get("tool_calls")
            ),
        )
        driver.register_function(function_map=dict(self.function_map))
        driver.initiate_chat(self, message=prompt_text, silent=True)
        return list(self._hypotheses)

    # --- Shared -----------------------------------------------------------

    def _record_hypothesis(
        self,
        plant_id: int,
        hypothesis_id: str,
        log_posterior: float,
        evidence_axes: dict[str, float],
    ) -> dict:
        msg = HypothesisMessage(
            sender="skeptic",
            timestamp=time.time(),
            plant_id=int(plant_id),
            hypothesis_id=str(hypothesis_id),
            log_posterior=float(log_posterior),
            evidence_axes={k: float(v) for k, v in evidence_axes.items()},
        )
        self._hypotheses.append(msg)
        return dataclasses.asdict(msg)


def _build_prompt(
    cross_exam_msgs: list[CrossExamMessage], disputed: list[int]
) -> str:
    lines = [
        "Disputed plants requiring alternative-hypothesis analysis.",
        "For each plant, call propose_alternative_hypothesis exactly once.",
        "",
    ]
    for pid in disputed:
        axes = dominant_axes_for_plant(cross_exam_msgs, pid)
        sorted_axes = sorted(axes.items(), key=lambda kv: -kv[1])[:3]
        axes_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted_axes)
        lines.append(f"plant_id={pid} dominant_axes: {axes_str}")
    return "\n".join(lines)
