"""SegmentationAgent — per-plant segmentation refinement.

Per §14.B bailout (no GPU), we skip SAM weights and use the YOLO bboxes as
mask proxies, computing simple per-plant green-pixel ratios as a calibrated
"is this actually leaf area" feature. The architecture story holds — same
ChannelAgent envelope, same ConstraintMessage protocol. When SAM weights
are available, this agent's ``_compute_mask`` can swap in a real SAM call
without touching the rest of the pipeline.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image

from pulse.agents.base import ChannelAgent
from pulse.latent import CONDITION_LABELS, FieldLatentState
from pulse.messages import ConstraintMessage


class SegmentationAgent(ChannelAgent):
    """Per-plant segmentation. Bailout uses the YOLO bbox crop directly."""

    def __init__(self, sam_predictor: Any | None = None) -> None:
        super().__init__(name="segmentation")
        self._sam = sam_predictor  # may be None — we have a deterministic fallback

    def emit_constraint(
        self, image_path: str | None, latent: FieldLatentState
    ) -> ConstraintMessage:
        img = np.asarray(Image.open(image_path).convert("RGB"))
        per_ll: dict[int, np.ndarray] = {}
        per_resid: dict[int, float] = {}
        per_conf: dict[int, float] = {}
        i_healthy = CONDITION_LABELS.index("healthy_crop")
        i_pest = CONDITION_LABELS.index("pest_damage")
        i_disease = CONDITION_LABELS.index("disease")
        i_water = CONDITION_LABELS.index("water_stress")
        for plant in latent.plants:
            x1, y1, x2, y2 = plant.bbox
            crop = img[max(0, y1):y2, max(0, x1):x2]
            mask = self._compute_mask(crop)
            features = _crop_features(crop, mask)
            log_lik = np.zeros(len(CONDITION_LABELS))
            # Healthy plants are leafy and uniformly green.
            green_mass = features["green_ratio"]
            yellowness = features["yellowness"]
            edginess = features["edge_density"]
            log_lik[i_healthy] = float(np.tanh((green_mass - 0.25) * 4))
            # Yellow-ish leaves implicate disease/water_stress over a healthy split.
            if yellowness > 0.05:
                log_lik[i_disease] = float(min(yellowness * 2, 1.0))
                log_lik[i_water] = float(min(yellowness * 1.2, 1.0))
                log_lik[i_healthy] -= float(min(yellowness * 1.5, 1.5))
            # High edge density inside the bbox → leaf perforations / pest damage.
            if edginess > 0.18:
                log_lik[i_pest] = float(min((edginess - 0.18) * 4, 1.5))
            per_ll[plant.plant_id] = log_lik
            per_resid[plant.plant_id] = float(1.0 - green_mass)
            per_conf[plant.plant_id] = float(min(0.3 + green_mass, 0.9))
        return ConstraintMessage(
            sender=self.name,
            timestamp=time.time(),
            iteration=latent.iteration,
            per_plant_log_likelihoods=per_ll,
            per_plant_residual=per_resid,
            per_plant_confidence=per_conf,
            labels_discriminated=["healthy_crop", "disease", "water_stress", "pest_damage"],
        )

    def _compute_mask(self, crop: np.ndarray) -> np.ndarray:
        """Return a binary leaf mask. Uses SAM if available, else green-channel HSV."""
        if self._sam is not None:
            return self._sam(crop)  # injection point; expected to return HxW bool
        # Fallback: HSV-based green-thresholding.
        if crop.size == 0:
            return np.zeros((1, 1), dtype=bool)
        rgb = crop.astype(np.float32) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return (g > r + 0.05) & (g > b + 0.05) & (g > 0.18)


def _crop_features(crop: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if crop.size == 0:
        return {"green_ratio": 0.0, "yellowness": 0.0, "edge_density": 0.0}
    rgb = crop.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    leaf = mask.astype(bool)
    leaf_area = float(leaf.sum())
    total = float(rgb.shape[0] * rgb.shape[1])
    green_ratio = leaf_area / max(total, 1.0)
    # Yellowness is computed over the full crop, not the leaf mask — a wholly
    # yellowed plant has zero green leaf area but the bbox is still yellow.
    yellow_pixels = float(((r > 0.55) & (g > 0.55) & (b < 0.45)).sum())
    yellowness = yellow_pixels / max(total, 1.0)
    # Cheap edge density via gradient magnitude
    gx = np.diff(g, axis=1, prepend=g[:, :1])
    gy = np.diff(g, axis=0, prepend=g[:1, :])
    edge = np.sqrt(gx * gx + gy * gy)
    edge_density = float((edge > 0.15).mean())
    return {
        "green_ratio": green_ratio,
        "yellowness": yellowness,
        "edge_density": edge_density,
    }
