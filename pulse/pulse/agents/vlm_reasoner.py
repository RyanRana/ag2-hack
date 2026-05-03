"""VLMReasonerAgent — LLM-backed targeted cross-examination of disputed regions.

AG2 idiom: ``AssistantAgent`` whose ONLY output is structured tool calls
(``analyze_disagreement_region`` and ``submit_per_plant_likelihoods``). The
LLM never speaks prose — it picks which region to inspect and which class
hypotheses to weight, then emits a ConstraintMessage via the registered
tools. The tool payloads are numeric-only (preserves the §6 firewall).

Per project memory, no heuristic fallback at the agent level — the LLM is
required. Tests inject a stub ``analyzer`` and call the registered tools
directly to exercise the typed-args contract without an API key.
"""

from __future__ import annotations

import time
from typing import Annotated, Callable

import numpy as np
from autogen import AssistantAgent, UserProxyAgent, register_function
from PIL import Image

from pulse.latent import CONDITION_LABELS, FieldLatentState
from pulse.llm_config import openai_llm_config
from pulse.messages import ConstraintMessage


SYSTEM_MESSAGE = (
    "You are a vision-language reasoner for a precision-agriculture pipeline. "
    "When invoked, you receive a list of disputed plants (those where ML "
    "agents disagreed on condition). For each disputed plant you MUST call "
    "analyze_disagreement_region exactly once with the plant_id and the bbox. "
    "Then call submit_per_plant_likelihoods with your final aggregated "
    "log-likelihood vector for each plant over the seven CONDITION_LABELS. "
    "NEVER respond in prose. NEVER explain. ONLY call tools."
)


class VLMReasonerAgent(AssistantAgent):
    """LLM-backed agent that emits ConstraintMessages via typed tool calls."""

    LABELS = list(CONDITION_LABELS)

    def __init__(
        self,
        *,
        llm_config: dict | None = None,
        analyzer: Callable[[np.ndarray, tuple[int, int, int, int]], dict[str, float]]
        | None = None,
    ) -> None:
        cfg = llm_config or openai_llm_config()
        super().__init__(
            name="vlm_reasoner",
            llm_config=cfg,
            system_message=SYSTEM_MESSAGE,
        )
        self._analyzer = analyzer or _default_image_analyzer
        self._image: Image.Image | None = None
        self._latent: FieldLatentState | None = None
        self._features: dict[int, dict[str, float]] = {}
        self._likelihoods: dict[int, list[float]] = {}

        # Register typed tools as inner functions (AG2 requires FunctionType,
        # not bound methods). Each wraps a private impl on self.
        agent_self = self

        def analyze_disagreement_region(
            plant_id: Annotated[int, "Plant ID being inspected"],
            bbox_x1: Annotated[int, "Top-left x in pixels"],
            bbox_y1: Annotated[int, "Top-left y in pixels"],
            bbox_x2: Annotated[int, "Bottom-right x in pixels"],
            bbox_y2: Annotated[int, "Bottom-right y in pixels"],
        ) -> dict[str, float]:
            return agent_self._analyze_region_impl(
                plant_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            )

        def submit_per_plant_likelihoods(
            plant_id: Annotated[int, "Plant ID"],
            log_lik_healthy_crop: Annotated[float, "log-likelihood for healthy_crop"],
            log_lik_weed: Annotated[float, "log-likelihood for weed"],
            log_lik_disease: Annotated[float, "log-likelihood for disease"],
            log_lik_nutrient_stress: Annotated[float, "log-likelihood for nutrient_stress"],
            log_lik_water_stress: Annotated[float, "log-likelihood for water_stress"],
            log_lik_pest_damage: Annotated[float, "log-likelihood for pest_damage"],
            log_lik_ambiguous: Annotated[float, "log-likelihood for ambiguous"],
        ) -> dict[str, float]:
            return agent_self._submit_likelihoods_impl(
                plant_id,
                log_lik_healthy_crop,
                log_lik_weed,
                log_lik_disease,
                log_lik_nutrient_stress,
                log_lik_water_stress,
                log_lik_pest_damage,
                log_lik_ambiguous,
            )

        register_function(
            analyze_disagreement_region,
            caller=self,
            executor=self,
            description=(
                "Crop the disputed plant region and return numerical visual features "
                "(green_ratio, yellowness, edge_density, brightness, size_px)."
            ),
        )
        register_function(
            submit_per_plant_likelihoods,
            caller=self,
            executor=self,
            description=(
                "Submit final per-plant log-likelihood vector over the seven "
                "CONDITION_LABELS in fixed order: healthy_crop, weed, disease, "
                "nutrient_stress, water_stress, pest_damage, ambiguous."
            ),
        )
        # Expose the registered tools so tests can call them directly.
        self.analyze_disagreement_region = analyze_disagreement_region
        self.submit_per_plant_likelihoods = submit_per_plant_likelihoods

    # --- Public entry point ----------------------------------------------

    def emit_constraint_for(
        self,
        image_path: str,
        latent: FieldLatentState,
        disputed_plant_ids: list[int],
    ) -> ConstraintMessage:
        """Run the LLM loop on disputed plants, collecting structured outputs."""
        self._image = Image.open(image_path).convert("RGB")
        self._latent = latent
        self._features.clear()
        self._likelihoods.clear()

        plants = [p for p in latent.plants if p.plant_id in set(disputed_plant_ids)]
        prompt_text = _build_user_prompt(plants)
        driver = UserProxyAgent(
            name="vlm_driver",
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config=False,
            max_consecutive_auto_reply=2 * (1 + len(plants)),
            is_termination_msg=lambda m: (
                m.get("name") == self.name
                and not m.get("tool_calls")
            ),
        )
        # Driver receives the tool_call messages — it must have the tool
        # implementations in its function_map. Copy ours over.
        driver.register_function(function_map=dict(self.function_map))
        driver.initiate_chat(self, message=prompt_text, silent=True)

        per_ll: dict[int, np.ndarray] = {}
        per_resid: dict[int, float] = {}
        per_conf: dict[int, float] = {}
        for pid in disputed_plant_ids:
            ll = self._likelihoods.get(pid, [0.0] * len(self.LABELS))
            per_ll[pid] = np.asarray(ll, dtype=float)
            feats = self._features.get(pid, {})
            per_conf[pid] = float(feats.get("brightness", 0.5))
            per_resid[pid] = float(1.0 - feats.get("green_ratio", 0.0))
        return ConstraintMessage(
            sender=self.name,
            timestamp=time.time(),
            iteration=latent.iteration,
            per_plant_log_likelihoods=per_ll,
            per_plant_residual=per_resid,
            per_plant_confidence=per_conf,
            labels_discriminated=list(CONDITION_LABELS),
            metadata={"features": dict(self._features)},
        )

    # --- Internal tool implementations ------------------------------------

    def _analyze_region_impl(
        self,
        plant_id: int,
        bbox_x1: int,
        bbox_y1: int,
        bbox_x2: int,
        bbox_y2: int,
    ) -> dict[str, float]:
        if self._image is None:
            raise RuntimeError("emit_constraint_for must be called before tool use")
        crop = np.asarray(self._image.crop((bbox_x1, bbox_y1, bbox_x2, bbox_y2)))
        feats = self._analyzer(crop, (bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        self._features[int(plant_id)] = feats
        return feats

    def _submit_likelihoods_impl(
        self,
        plant_id: int,
        log_lik_healthy_crop: float,
        log_lik_weed: float,
        log_lik_disease: float,
        log_lik_nutrient_stress: float,
        log_lik_water_stress: float,
        log_lik_pest_damage: float,
        log_lik_ambiguous: float,
    ) -> dict[str, float]:
        ll = [
            float(log_lik_healthy_crop),
            float(log_lik_weed),
            float(log_lik_disease),
            float(log_lik_nutrient_stress),
            float(log_lik_water_stress),
            float(log_lik_pest_damage),
            float(log_lik_ambiguous),
        ]
        self._likelihoods[int(plant_id)] = ll
        return {"accepted": 1.0}


def _default_image_analyzer(
    crop: np.ndarray, bbox: tuple[int, int, int, int]
) -> dict[str, float]:
    """Numerical features for the LLM to reason over."""
    if crop.size == 0:
        return {"green_ratio": 0.0, "yellowness": 0.0, "edge_density": 0.0,
                "brightness": 0.0, "size_px": 0.0}
    rgb = crop.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    green = ((g > r + 0.05) & (g > b + 0.05) & (g > 0.18)).mean()
    yellow = ((r > 0.55) & (g > 0.55) & (b < 0.45)).mean()
    gx = np.diff(g, axis=1, prepend=g[:, :1])
    gy = np.diff(g, axis=0, prepend=g[:1, :])
    edge = np.sqrt(gx * gx + gy * gy)
    edges = (edge > 0.15).mean()
    bright = float(rgb.mean())
    h = bbox[3] - bbox[1]
    w = bbox[2] - bbox[0]
    return {
        "green_ratio": float(green),
        "yellowness": float(yellow),
        "edge_density": float(edges),
        "brightness": bright,
        "size_px": float(max(w * h, 0.0)),
    }


def _build_user_prompt(plants: list) -> str:
    lines = [
        "Disputed plants requiring vision-language re-examination:",
        "Call analyze_disagreement_region(plant_id, x1, y1, x2, y2) for each, then",
        "submit_per_plant_likelihoods with your final log-likelihood vector.",
        "",
    ]
    for p in plants:
        x1, y1, x2, y2 = p.bbox
        lines.append(f"plant_id={p.plant_id} bbox=({x1},{y1},{x2},{y2})")
    return "\n".join(lines)
