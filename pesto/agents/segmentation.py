"""SegmentationAgent — per-plant OpenCV evidence with spatial mask retention.

Enhanced from the original bbox-proxy approach to retain spatial evidence
masks (HSV, edge, contour) alongside the scalar log-likelihoods. The masks
feed the visual explanation layer (``pesto.visual_explain``) so that every
overlay pixel comes from the SAME computation that produced the log-likelihood.

Explanation === Evidence.

Per §14.B bailout (no GPU), we skip SAM weights and use the YOLO bboxes as
mask proxies. When SAM weights are available, ``_compute_mask`` swaps in a
real SAM call without touching the rest of the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from pesto.agents.base import ChannelAgent
from pesto.latent import CONDITION_LABELS, FieldLatentState
from pesto.messages import ConstraintMessage


@dataclass
class PlantEvidenceMaps:
    """Spatial evidence masks for a single plant crop — produced by the SAME
    computation that generates log-likelihoods."""

    plant_id: int
    bbox: tuple[int, int, int, int]
    leaf_mask: np.ndarray          # bool HxW — green-channel HSV threshold
    yellow_mask: np.ndarray        # bool HxW — yellowed tissue
    edge_mask: np.ndarray          # bool HxW — high-gradient pixels (perforations)
    contour_mask: np.ndarray       # uint8 HxW — contour outlines drawn on black
    gradient_magnitude: np.ndarray # float32 HxW — raw gradient magnitude
    features: dict[str, float] = field(default_factory=dict)


class SegmentationAgent(ChannelAgent):
    """Per-plant OpenCV evidence. Retains spatial masks for visual explanation."""

    def __init__(self, sam_predictor: Any | None = None) -> None:
        super().__init__(name="segmentation")
        self._sam = sam_predictor
        # Side-channel: per-plant evidence maps from the most recent inference.
        # Keyed by plant_id. NOT sent through the protocol (no prose) — consumed
        # by visual_explain.py for overlay rendering.
        self.evidence_maps: dict[int, PlantEvidenceMaps] = {}

    def emit_constraint(
        self, image_path: str | None, latent: FieldLatentState
    ) -> ConstraintMessage:
        img = np.asarray(Image.open(image_path).convert("RGB"))
        per_ll: dict[int, np.ndarray] = {}
        per_resid: dict[int, float] = {}
        per_conf: dict[int, float] = {}
        self.evidence_maps = {}
        i_healthy = CONDITION_LABELS.index("healthy_crop")
        i_pest = CONDITION_LABELS.index("pest_damage")
        i_disease = CONDITION_LABELS.index("disease")
        i_water = CONDITION_LABELS.index("water_stress")
        for plant in latent.plants:
            x1, y1, x2, y2 = plant.bbox
            crop = img[max(0, y1):y2, max(0, x1):x2]
            mask = self._compute_mask(crop)
            features, evidence = _crop_features_with_evidence(crop, mask, plant.plant_id, plant.bbox)
            self.evidence_maps[plant.plant_id] = evidence
            log_lik = np.zeros(len(CONDITION_LABELS))
            green_mass = features["green_ratio"]
            yellowness = features["yellowness"]
            edginess = features["edge_density"]
            log_lik[i_healthy] = float(np.tanh((green_mass - 0.25) * 4))
            if yellowness > 0.05:
                log_lik[i_disease] = float(min(yellowness * 2, 1.0))
                log_lik[i_water] = float(min(yellowness * 1.2, 1.0))
                log_lik[i_healthy] -= float(min(yellowness * 1.5, 1.5))
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
            return self._sam(crop)
        if crop.size == 0:
            return np.zeros((1, 1), dtype=bool)
        rgb = crop.astype(np.float32) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return (g > r + 0.05) & (g > b + 0.05) & (g > 0.18)


def _crop_features_with_evidence(
    crop: np.ndarray,
    mask: np.ndarray,
    plant_id: int,
    bbox: tuple[int, int, int, int],
) -> tuple[dict[str, float], PlantEvidenceMaps]:
    """Compute scalar features AND retain the spatial evidence masks.

    The scalars feed the log-likelihood computation (same math as before).
    The masks are stored for visual explanation — same computation, zero divergence.
    """
    h, w = max(crop.shape[0], 1), max(crop.shape[1], 1)
    if crop.size == 0:
        empty = np.zeros((1, 1), dtype=np.uint8)
        evidence = PlantEvidenceMaps(
            plant_id=plant_id, bbox=bbox,
            leaf_mask=np.zeros((1, 1), dtype=bool),
            yellow_mask=np.zeros((1, 1), dtype=bool),
            edge_mask=np.zeros((1, 1), dtype=bool),
            contour_mask=empty,
            gradient_magnitude=np.zeros((1, 1), dtype=np.float32),
            features={"green_ratio": 0.0, "yellowness": 0.0, "edge_density": 0.0},
        )
        return {"green_ratio": 0.0, "yellowness": 0.0, "edge_density": 0.0}, evidence

    rgb = crop.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    leaf = mask.astype(bool)
    total = float(h * w)

    # Green ratio
    green_ratio = float(leaf.sum()) / max(total, 1.0)

    # Yellow mask — same threshold as before, now retained
    yellow_mask = (r > 0.55) & (g > 0.55) & (b < 0.45)
    yellowness = float(yellow_mask.sum()) / max(total, 1.0)

    # Edge detection — gradient magnitude + Canny for contours
    gx = np.diff(g, axis=1, prepend=g[:, :1])
    gy = np.diff(g, axis=0, prepend=g[:1, :])
    grad_mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    edge_mask = grad_mag > 0.15
    edge_density = float(edge_mask.mean())

    # Canny contour detection for visual overlay
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges_canny = cv2.Canny(gray, 50, 150)
    contour_mask = np.zeros_like(gray)
    contours, _ = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_mask, contours, -1, 255, 1)

    features = {
        "green_ratio": green_ratio,
        "yellowness": yellowness,
        "edge_density": edge_density,
    }

    evidence = PlantEvidenceMaps(
        plant_id=plant_id,
        bbox=bbox,
        leaf_mask=leaf,
        yellow_mask=yellow_mask,
        edge_mask=edge_mask,
        contour_mask=contour_mask,
        gradient_magnitude=grad_mag,
        features=features,
    )

    return features, evidence
