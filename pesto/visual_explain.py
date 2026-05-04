"""Visual explanation renderer — overlays evidence masks on the original image.

GUARANTEE: Every overlay pixel comes from the SAME computation that produced
the log-likelihood in SegmentationAgent. Explanation === Evidence.

Layers:
  - Red contours   = lesion regions (Canny-detected)
  - Yellow overlay  = yellowed tissue (HSV-detected)
  - Magenta edges   = perforation edges (gradient-detected)
  - Green tint      = healthy leaf area (HSV-detected)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pesto.agents.segmentation import PlantEvidenceMaps


def render_plant_explanation(
    image: np.ndarray,
    evidence: PlantEvidenceMaps,
    *,
    alpha: float = 0.4,
) -> np.ndarray:
    """Render evidence overlay for a single plant onto its bbox region.

    Parameters
    ----------
    image   : Full-frame image (HxWx3, uint8 RGB).
    evidence: PlantEvidenceMaps from SegmentationAgent.
    alpha   : Overlay transparency (0=invisible, 1=opaque).

    Returns
    -------
    Copy of ``image`` with the plant's bbox region overlaid with evidence.
    """
    out = image.copy()
    x1, y1, x2, y2 = evidence.bbox
    crop_h = max(y2 - max(0, y1), 1)
    crop_w = max(x2 - max(0, x1), 1)
    y1c, x1c = max(0, y1), max(0, x1)

    overlay = out[y1c:y2, x1c:x2].copy()
    oh, ow = overlay.shape[:2]

    def _resize_mask(mask: np.ndarray) -> np.ndarray:
        if mask.shape[:2] == (oh, ow):
            return mask
        return cv2.resize(mask.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Green tint on healthy leaf area
    leaf = _resize_mask(evidence.leaf_mask)
    overlay[leaf] = (
        overlay[leaf].astype(np.float32) * (1 - alpha * 0.5)
        + np.array([60, 180, 60], dtype=np.float32) * (alpha * 0.5)
    ).astype(np.uint8)

    # Yellow overlay on yellowed tissue
    yellow = _resize_mask(evidence.yellow_mask)
    overlay[yellow] = (
        overlay[yellow].astype(np.float32) * (1 - alpha)
        + np.array([220, 200, 50], dtype=np.float32) * alpha
    ).astype(np.uint8)

    # Magenta edges on high-gradient pixels (perforations)
    edges = _resize_mask(evidence.edge_mask)
    overlay[edges] = (
        overlay[edges].astype(np.float32) * (1 - alpha * 0.7)
        + np.array([200, 50, 200], dtype=np.float32) * (alpha * 0.7)
    ).astype(np.uint8)

    # Red contours from Canny
    contour = _resize_mask(evidence.contour_mask > 0)
    overlay[contour] = np.array([255, 40, 40], dtype=np.uint8)

    out[y1c:y2, x1c:x2] = overlay

    # Draw bounding box
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return out


def render_field_explanation(
    image_path: str,
    evidence_maps: dict[int, PlantEvidenceMaps],
    *,
    alpha: float = 0.4,
) -> np.ndarray:
    """Render evidence overlays for all plants in the field image.

    Parameters
    ----------
    image_path    : Path to the original field image.
    evidence_maps : dict plant_id -> PlantEvidenceMaps from SegmentationAgent.
    alpha         : Overlay transparency.

    Returns
    -------
    Annotated image (HxWx3, uint8 RGB).
    """
    img = np.asarray(Image.open(image_path).convert("RGB")).copy()
    for _pid, evidence in sorted(evidence_maps.items()):
        img = render_plant_explanation(img, evidence, alpha=alpha)
    return img


def save_explanation(
    image_path: str,
    evidence_maps: dict[int, PlantEvidenceMaps],
    output_path: str | Path,
    *,
    alpha: float = 0.4,
) -> Path:
    """Render and save the annotated explanation image."""
    annotated = render_field_explanation(image_path, evidence_maps, alpha=alpha)
    out = Path(output_path)
    # Convert RGB to BGR for cv2.imwrite
    cv2.imwrite(str(out), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    return out
