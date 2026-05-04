"""GrowthStageAgent — ViT growth-stage classifier (paradigm: ML).

Classifies each plant crop into one of four growth stages:
  seedling, vegetative, flowering, fruiting

The growth stage does NOT directly modify condition likelihoods — instead
it emits metadata that the controller uses as an urgency multiplier:
  - seedling + disease/pest  → act immediately (high urgency)
  - fruiting + disease       → act soon (medium urgency, protect yield)
  - vegetative + minor issue → can wait (low urgency)

The ConstraintMessage log-likelihoods are set to zero (neutral) since
growth stage is orthogonal to condition diagnosis. The signal is carried
in the metadata and per_plant_confidence fields.

Same ChannelAgent envelope, same ConstraintMessage protocol.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image

from pesto.agents.base import ChannelAgent
from pesto.latent import CONDITION_LABELS, FieldLatentState
from pesto.messages import ConstraintMessage


GROWTH_STAGES = ["seedling", "vegetative", "flowering", "fruiting"]

# Urgency multiplier per growth stage — used by the controller.
# Higher = more urgent to intervene.
URGENCY_MULTIPLIER = {
    "seedling": 1.5,    # Young plants are fragile, act fast
    "vegetative": 0.8,  # Can tolerate some stress
    "flowering": 1.2,   # Protect pollination
    "fruiting": 1.3,    # Protect yield
}


class GrowthStageClassifier:
    """Classifies plant crops into growth stages.

    Uses a ViT-based classifier if available, otherwise falls back to
    a heuristic based on green area ratio and aspect ratio.
    """

    def __init__(self, model: Any | None = None, processor: Any | None = None) -> None:
        self._model = model
        self._processor = processor

    @property
    def has_model(self) -> bool:
        return self._model is not None

    def classify(self, crop: Image.Image | np.ndarray) -> dict[str, float]:
        """Classify a single plant crop into growth stage probabilities.

        Returns dict mapping growth stage name -> probability.
        """
        if self._model is not None and self._processor is not None:
            return self._classify_with_model(crop)
        return self._classify_heuristic(crop)

    def _classify_with_model(self, crop: Image.Image | np.ndarray) -> dict[str, float]:
        """Classify using a loaded ViT model."""
        import torch

        if isinstance(crop, np.ndarray):
            crop = Image.fromarray(crop)

        inputs = self._processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        id2label = getattr(self._model.config, "id2label", {})
        result = {stage: 0.0 for stage in GROWTH_STAGES}
        for idx, p in enumerate(probs):
            label = id2label.get(idx, "").lower()
            for stage in GROWTH_STAGES:
                if stage in label:
                    result[stage] += float(p)
                    break
        # Normalise
        total = sum(result.values())
        if total > 1e-6:
            result = {k: v / total for k, v in result.items()}
        else:
            result = {stage: 1.0 / len(GROWTH_STAGES) for stage in GROWTH_STAGES}
        return result

    def _classify_heuristic(self, crop: Image.Image | np.ndarray) -> dict[str, float]:
        """Heuristic growth-stage classification from visual features.

        Simple rules based on plant size and greenness:
          - Small + sparse green → seedling
          - Medium + dense green → vegetative
          - Presence of bright colours (non-green) → flowering
          - Large + mixed colours → fruiting
        """
        if isinstance(crop, Image.Image):
            crop = np.asarray(crop)

        if crop.size == 0:
            return {stage: 1.0 / len(GROWTH_STAGES) for stage in GROWTH_STAGES}

        h, w = crop.shape[:2]
        area = h * w
        rgb = crop.astype(np.float32) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        # Green ratio
        green_mask = (g > r + 0.05) & (g > b + 0.05) & (g > 0.18)
        green_ratio = float(green_mask.sum()) / max(area, 1)

        # Bright non-green (flowers/fruit) — high saturation, not green-dominant
        bright_mask = ((r > 0.5) | (b > 0.5)) & (~green_mask) & ((r + g + b) > 0.8)
        bright_ratio = float(bright_mask.sum()) / max(area, 1)

        # Size heuristic (relative to expected crop size)
        size_score = min(area / 10000.0, 1.0)  # normalise assuming ~100x100 is full-size

        probs = {stage: 0.1 for stage in GROWTH_STAGES}  # baseline

        # Seedling: small and sparse
        if size_score < 0.3 and green_ratio < 0.4:
            probs["seedling"] += 0.6
        elif size_score < 0.5:
            probs["seedling"] += 0.3

        # Vegetative: good green coverage, no flowers
        if green_ratio > 0.4 and bright_ratio < 0.05:
            probs["vegetative"] += 0.5
        elif green_ratio > 0.25:
            probs["vegetative"] += 0.2

        # Flowering: bright non-green regions
        if bright_ratio > 0.05:
            probs["flowering"] += 0.4
        if bright_ratio > 0.15:
            probs["flowering"] += 0.3

        # Fruiting: large plant with mixed colours
        if size_score > 0.5 and bright_ratio > 0.1 and green_ratio > 0.2:
            probs["fruiting"] += 0.4

        # Normalise
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}


class GrowthStageAgent(ChannelAgent):
    """ChannelAgent that classifies growth stage and provides urgency metadata.

    The log-likelihoods are neutral (growth stage is orthogonal to condition).
    The signal is in metadata["growth_stage"] and metadata["urgency_multiplier"].
    """

    def __init__(
        self,
        classifier: GrowthStageClassifier | None = None,
    ) -> None:
        super().__init__(name="growth_stage")
        self._classifier = classifier or GrowthStageClassifier()
        # Side-channel: per-plant growth stage results
        self.growth_stages: dict[int, dict[str, float]] = {}

    def emit_constraint(
        self, image_path: str | None, latent: FieldLatentState
    ) -> ConstraintMessage:
        per_ll: dict[int, np.ndarray] = {}
        per_resid: dict[int, float] = {}
        per_conf: dict[int, float] = {}
        self.growth_stages = {}
        per_plant_meta: dict[int, dict] = {}

        if image_path is not None:
            full_img = Image.open(image_path).convert("RGB")
            for plant in latent.plants:
                crop = full_img.crop(plant.bbox)
                stage_probs = self._classifier.classify(crop)
                self.growth_stages[plant.plant_id] = stage_probs

                # Dominant stage
                dominant = max(stage_probs, key=stage_probs.get)
                urgency = URGENCY_MULTIPLIER[dominant]

                # Neutral log-likelihoods (growth stage is orthogonal to condition)
                per_ll[plant.plant_id] = np.zeros(len(CONDITION_LABELS))
                per_resid[plant.plant_id] = 0.0
                per_conf[plant.plant_id] = float(stage_probs[dominant])

                per_plant_meta[plant.plant_id] = {
                    "growth_stage": dominant,
                    "growth_stage_probs": stage_probs,
                    "urgency_multiplier": urgency,
                }
        else:
            for plant in latent.plants:
                per_ll[plant.plant_id] = np.zeros(len(CONDITION_LABELS))
                per_resid[plant.plant_id] = 0.0
                per_conf[plant.plant_id] = 0.25
                per_plant_meta[plant.plant_id] = {
                    "growth_stage": "vegetative",
                    "growth_stage_probs": {s: 0.25 for s in GROWTH_STAGES},
                    "urgency_multiplier": URGENCY_MULTIPLIER["vegetative"],
                }

        return ConstraintMessage(
            sender=self.name,
            timestamp=time.time(),
            iteration=latent.iteration,
            per_plant_log_likelihoods=per_ll,
            per_plant_residual=per_resid,
            per_plant_confidence=per_conf,
            labels_discriminated=[],  # does not discriminate conditions
            metadata={
                "per_plant_growth": {
                    int(k): v for k, v in per_plant_meta.items()
                },
            },
        )
