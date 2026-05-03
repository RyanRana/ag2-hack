"""DiseaseClassifierAgent — MobileNetV2 plant-disease classifier (paradigm: ML).

Model: linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification (38
PlantVillage classes). We map the 38 classes to Pulse's 7 CONDITION_LABELS
via a small heuristic on class-name suffixes.
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


HF_MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

# A coarse mapping of PlantVillage class-name patterns to our condition labels.
# Calibration is done downstream — this just routes mass to the right axis.
_CLASS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"healthy", re.I), "healthy_crop"),
    (re.compile(r"deficiency|nitrogen|nutrient", re.I), "nutrient_stress"),
    (re.compile(r"mite|aphid|insect|pest|borer|caterpillar|whitefly", re.I), "pest_damage"),
    (re.compile(r"drought|wilt", re.I), "water_stress"),
    # Anything with a recognizable disease keyword.
    (re.compile(
        r"rust|spot|blight|mildew|mold|virus|rot|scab|scorch|canker|wilt|"
        r"anthracnose|leaf|disease",
        re.I,
    ), "disease"),
]


def class_to_condition(name: str) -> str:
    for pat, cond in _CLASS_PATTERNS:
        if pat.search(name):
            return cond
    return "skip"  # Background / non-leaf — discard, never pollute "ambiguous".


def map_probs_to_conditions(
    probs: np.ndarray,
    id2label: dict[int, str],
) -> np.ndarray:
    """Aggregate per-class probabilities into per-CONDITION_LABEL log-likelihoods.

    Classes that don't map to any of our seven conditions (e.g.
    ``Background_without_leaves``) are dropped — they should NEVER push mass
    into ``ambiguous`` or the posterior collapses to "mixed-signal" on every
    out-of-distribution frame.
    """
    cond_mass = np.zeros(len(CONDITION_LABELS))
    accepted = 0.0
    for idx, p in enumerate(probs):
        label = id2label.get(int(idx), str(idx))
        cond = class_to_condition(label)
        if cond == "skip":
            continue
        cond_mass[CONDITION_LABELS.index(cond)] += float(p)
        accepted += float(p)
    if accepted < 1e-6:
        return np.zeros(len(CONDITION_LABELS))
    cond_mass = cond_mass / accepted
    # Additive smoothing — without this, axes that received zero probability
    # (e.g. "weed" — model has no weed class) become log(~0) = -∞, which
    # dominates the centered log-likelihoods and produces unreasonably
    # large positive values on the populated axes.
    cond_mass = cond_mass + 0.10
    cond_mass = cond_mass / cond_mass.sum()
    log_lik = np.log(cond_mass)
    # Hard-suppress ambiguous; the controller infers "review" from entropy.
    log_lik[CONDITION_LABELS.index("ambiguous")] = -5.0
    log_lik = log_lik - np.mean(log_lik)
    return log_lik


def _load_processor_with_fallback(repo: str, *, fallback: str):
    """Load an image processor from HF, tolerating missing image_processor_type.

    Newer ``transformers`` raises ``ValueError`` when ``preprocessor_config.json``
    is missing the ``image_processor_type`` field. Several plant-disease repos
    on HF predate that requirement; we resolve them by explicitly instantiating
    the processor class for the model family (``mobilenet_v2`` or ``vit``).
    """
    from transformers import AutoImageProcessor

    try:
        return AutoImageProcessor.from_pretrained(repo)
    except (ValueError, OSError):
        if fallback == "mobilenet_v2":
            from transformers import MobileNetV2ImageProcessor

            try:
                return MobileNetV2ImageProcessor.from_pretrained(repo)
            except (ValueError, OSError):
                return MobileNetV2ImageProcessor()
        if fallback == "vit":
            from transformers import ViTImageProcessor

            try:
                return ViTImageProcessor.from_pretrained(repo)
            except (ValueError, OSError):
                return ViTImageProcessor()
        raise


class DiseaseClassifierAgent(ChannelAgent):
    """ChannelAgent backed by the linkanjarad MobileNetV2 model."""

    HF_MODEL_ID = HF_MODEL_ID

    def __init__(self, model: Any | None = None, processor: Any | None = None) -> None:
        super().__init__(name="disease_classifier")
        self._model = model
        self._processor = processor

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        from transformers import AutoModelForImageClassification

        self._processor = _load_processor_with_fallback(
            self.HF_MODEL_ID, fallback="mobilenet_v2"
        )
        self._model = AutoModelForImageClassification.from_pretrained(self.HF_MODEL_ID)
        self._model.eval()
        return self._model, self._processor

    def emit_constraint(
        self, image_path: str | None, latent: FieldLatentState
    ) -> ConstraintMessage:
        import torch  # local import keeps the module importable without torch

        model, processor = self._load_model()
        full_img = Image.open(image_path).convert("RGB")
        per_ll: dict[int, np.ndarray] = {}
        per_resid: dict[int, float] = {}
        per_conf: dict[int, float] = {}
        id2label = getattr(model.config, "id2label", {})
        for plant in latent.plants:
            crop = full_img.crop(plant.bbox)
            inputs = processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            log_lik = map_probs_to_conditions(probs, id2label)
            per_ll[plant.plant_id] = log_lik
            top = float(probs.max())
            per_resid[plant.plant_id] = float(1.0 - top)
            per_conf[plant.plant_id] = top
        return ConstraintMessage(
            sender=self.name,
            timestamp=time.time(),
            iteration=latent.iteration,
            per_plant_log_likelihoods=per_ll,
            per_plant_residual=per_resid,
            per_plant_confidence=per_conf,
            labels_discriminated=["disease", "healthy_crop", "nutrient_stress", "pest_damage"],
        )
