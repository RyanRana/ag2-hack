"""SkepticAgent — LLM-backed alternative-hypothesis proposer.

AG2 idiom: ``AssistantAgent`` whose ONLY output is a typed tool call to
``propose_alternative_hypothesis``. The LLM never speaks prose. The tool
emits a HypothesisMessage; the controller fuses it into the per-plant
posterior.

Per project memory, the LLM is required (no heuristic fallback at the
agent level). Tests inject a fake llm_config and call the registered tool
directly to exercise the typed contract without an API key.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Annotated

from autogen import AssistantAgent, UserProxyAgent, register_function

from pulse.cross_exam import dominant_axes_for_plant
from pulse.llm_config import openai_llm_config
from pulse.messages import CrossExamMessage, HypothesisMessage


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


class SkepticAgent(AssistantAgent):
    """LLM-backed agent that emits HypothesisMessages via typed tool calls."""

    def __init__(self, *, llm_config: dict | None = None) -> None:
        cfg = llm_config or openai_llm_config()
        super().__init__(
            name="skeptic",
            llm_config=cfg,
            system_message=SYSTEM_MESSAGE,
        )
        self._hypotheses: list[HypothesisMessage] = []

        agent_self = self

        def propose_alternative_hypothesis(
            plant_id: Annotated[int, "Which plant the hypothesis applies to"],
            hypothesis_id: Annotated[
                str,
                "Short identifier such as 'young_crop_misidentified_as_weed', "
                "'disease_masquerading_as_nutrient', or "
                "'water_stress_concurrent_with_pest'.",
            ],
            log_posterior: Annotated[
                float, "Log posterior probability of this hypothesis given disagreement"
            ],
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

        register_function(
            propose_alternative_hypothesis,
            caller=self,
            executor=self,
            description=(
                "Propose an alternative hypothesis to explain a cross-agent "
                "disagreement on a single plant. All evidence axes must be "
                "numeric (KL-divergence magnitudes)."
            ),
        )
        self.propose_alternative_hypothesis = propose_alternative_hypothesis

    # --- Public entry point ----------------------------------------------

    def emit_hypotheses_for(
        self,
        cross_exam_msgs: list[CrossExamMessage],
        disputed_plant_ids: list[int],
    ) -> list[HypothesisMessage]:
        self._hypotheses.clear()
        prompt_text = _build_prompt(cross_exam_msgs, disputed_plant_ids)
        # Use a UserProxyAgent to drive the chat: it sends the prompt,
        # the skeptic LLM emits tool calls, the proxy executes them.
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
        # The driver is the agent receiving the tool_call message — it must
        # have the tool implementations in its function_map. Copy ours over.
        driver.register_function(function_map=dict(self.function_map))
        driver.initiate_chat(self, message=prompt_text, silent=True)
        return list(self._hypotheses)

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
