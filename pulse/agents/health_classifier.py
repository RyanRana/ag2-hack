"""HealthClassifierAgent — ViT binary healthy/unhealthy + severity (paradigm: ML).

Model: Diginsa/Plant-Disease-Detection-Project (small ViT). Per §2.1 this
agent is coarse — it can't distinguish *what's* wrong, just *whether*
something is wrong — but it's calibrated to discriminate healthy from
unhealthy with low compute, which makes it a useful tiebreaker.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
from PIL import Image

from pulse.agents.base import ChannelAgent
from pulse.latent import CONDITION_LABELS, FieldLatentState
from pulse.messages import ConstraintMessage


HF_MODEL_ID = "Diginsa/Plant-Disease-Detection-Project"


def _split_health_mass(probs: np.ndarray, id2label: dict[int, str]) -> tuple[float, float]:
    """Return (P_healthy, P_unhealthy) summed across all classes."""
    p_h = 0.0
    p_u = 0.0
    healthy_pat = re.compile(r"healthy", re.I)
    for idx, p in enumerate(probs):
        label = id2label.get(int(idx), str(idx))
        if healthy_pat.search(label):
            p_h += float(p)
        else:
            p_u += float(p)
    s = p_h + p_u
    if s <= 0:
        return 0.5, 0.5
    return p_h / s, p_u / s


class HealthClassifierAgent(ChannelAgent):
    """ViT binary healthy/unhealthy classifier with temperature scaling."""

    HF_MODEL_ID = HF_MODEL_ID

    def __init__(
        self,
        model: Any | None = None,
        processor: Any | None = None,
        temperature: float | None = None,
    ) -> None:
        super().__init__(name="health_classifier")
        self._model = model
        self._processor = processor
        self._temperature = temperature

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        from transformers import AutoModelForImageClassification

        from pulse.agents.disease_classifier import _load_processor_with_fallback

        self._processor = _load_processor_with_fallback(
            self.HF_MODEL_ID, fallback="vit"
        )
        self._model = AutoModelForImageClassification.from_pretrained(self.HF_MODEL_ID)
        self._model.eval()
        return self._model, self._processor

    def _get_temperature(self) -> float:
        if self._temperature is not None:
            return self._temperature
        from pulse.calibration import load_temperature
        self._temperature = load_temperature("health_classifier")
        return self._temperature

    def emit_constraint(
        self, image_path: str | None, latent: FieldLatentState
    ) -> ConstraintMessage:
        import torch

        model, processor = self._load_model()
        T = self._get_temperature()
        full = Image.open(image_path).convert("RGB")
        id2label = getattr(model.config, "id2label", {})
        per_ll: dict[int, np.ndarray] = {}
        per_resid: dict[int, float] = {}
        per_conf: dict[int, float] = {}
        i_healthy = CONDITION_LABELS.index("healthy_crop")
        i_disease = CONDITION_LABELS.index("disease")
        i_pest = CONDITION_LABELS.index("pest_damage")
        i_nutrient = CONDITION_LABELS.index("nutrient_stress")
        for plant in latent.plants:
            crop = full.crop(plant.bbox)
            inputs = processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits[0]
            # Temperature scaling
            calibrated_logits = logits / T
            probs = torch.softmax(calibrated_logits, dim=-1).cpu().numpy()
            p_h, p_u = _split_health_mass(probs, id2label)
            log_lik = np.zeros(len(CONDITION_LABELS))
            # Push the healthy mass to healthy_crop.
            log_lik[i_healthy] = float(np.log(p_h + 1e-3) - np.log(0.5))
            # Distribute unhealthy mass across disease/pest/nutrient evenly —
            # this agent is *not* calibrated to distinguish them.
            spread = float(np.log(p_u + 1e-3) - np.log(0.5))
            log_lik[i_disease] = spread
            log_lik[i_pest] = spread
            log_lik[i_nutrient] = spread * 0.5
            per_ll[plant.plant_id] = log_lik
            per_resid[plant.plant_id] = float(1.0 - probs.max())
            per_conf[plant.plant_id] = float(probs.max())
        return ConstraintMessage(
            sender=self.name,
            timestamp=time.time(),
            iteration=latent.iteration,
            per_plant_log_likelihoods=per_ll,
            per_plant_residual=per_resid,
            per_plant_confidence=per_conf,
            labels_discriminated=["healthy_crop", "disease", "pest_damage", "nutrient_stress"],
        )
