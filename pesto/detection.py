"""Initial plant detection — populate FieldLatentState with PlantInstances.

Supports two detection backends:
  1. **YOLO** (existing) — fixed-class detector from foduucom model.
  2. **Grounding DINO** (new) — open-vocabulary detector that catches things
     YOLO was never trained on ("wilted plant", "leaf with holes", etc.).

When both are available, detections are merged via NMS deduplication.
The output contract is the same: a FieldLatentState whose plants have
unique ``plant_id``, integer bboxes, and uniform priors over CONDITION_LABELS.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from pesto.latent import FieldLatentState, PlantInstance


# Default open-vocabulary prompts for Grounding DINO
DEFAULT_GROUNDING_PROMPTS = [
    "diseased leaf",
    "wilted plant",
    "insect damage",
    "healthy plant",
    "weed",
    "yellowed leaf",
]


def detect_plants_yolo(
    image_path: str,
    yolo_model: Any,
    *,
    grounding_dino_model: Any | None = None,
    grounding_prompts: list[str] | None = None,
    nms_iou_threshold: float = 0.5,
    max_plants: int = 64,
) -> FieldLatentState:
    """Run YOLO (+ optional Grounding DINO) and emit PlantInstances.

    When ``grounding_dino_model`` is provided, both detectors run and
    their outputs are merged via NMS deduplication.
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # YOLO detections
    results = yolo_model.predict(image_path)
    yolo_boxes = extract_boxes(results)

    # Grounding DINO detections (if available)
    gdino_boxes: list[dict] = []
    if grounding_dino_model is not None:
        prompts = grounding_prompts or DEFAULT_GROUNDING_PROMPTS
        gdino_boxes = detect_grounding_dino(
            image_path, grounding_dino_model, prompts
        )

    # Merge and deduplicate
    if gdino_boxes:
        all_boxes = merge_detections(yolo_boxes, gdino_boxes, nms_iou_threshold)
    else:
        all_boxes = yolo_boxes

    all_boxes.sort(key=lambda b: (-b["conf"], b["xyxy"][0], b["xyxy"][1]))
    field = FieldLatentState(image_shape=(height, width))
    for i, det in enumerate(all_boxes[:max_plants]):
        x1, y1, x2, y2 = det["xyxy"]
        field.plants.append(
            PlantInstance(plant_id=i, bbox=(int(x1), int(y1), int(x2), int(y2)))
        )
    return field


def detect_grounding_dino(
    image_path: str,
    model: Any,
    prompts: list[str],
    *,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
) -> list[dict]:
    """Run Grounding DINO with open-vocabulary text prompts.

    Parameters
    ----------
    image_path    : Path to the input image.
    model         : Loaded Grounding DINO model (from transformers).
    prompts       : List of text prompts to detect.
    box_threshold : Minimum box confidence to keep.
    text_threshold: Minimum text match confidence.

    Returns
    -------
    List of detection dicts with "xyxy", "cls_name", "conf", "source" keys.
    """
    try:
        from transformers import AutoProcessor

        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        prompt_text = ". ".join(prompts) + "."

        # Try processor-based pipeline (transformers GroundingDino)
        processor = getattr(model, "_processor", None)
        if processor is None:
            try:
                processor = AutoProcessor.from_pretrained(
                    "IDEA-Research/grounding-dino-base"
                )
            except Exception:
                return []

        import torch

        inputs = processor(images=img, text=prompt_text, return_tensors="pt")
        device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process: outputs contain pred_boxes (cx, cy, w, h normalised)
        # and logits
        logits = outputs.logits.sigmoid()[0]  # (n_queries, n_tokens)
        boxes_cxcywh = outputs.pred_boxes[0]  # (n_queries, 4)

        # Filter by box threshold
        max_logits = logits.max(dim=-1).values  # (n_queries,)
        keep = max_logits > box_threshold
        filtered_boxes = boxes_cxcywh[keep].cpu().numpy()
        filtered_scores = max_logits[keep].cpu().numpy()

        # Convert cxcywh normalised → xyxy pixel coordinates
        detections: list[dict] = []
        for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
            cx, cy, w, h = box
            x1 = (cx - w / 2) * width
            y1 = (cy - h / 2) * height
            x2 = (cx + w / 2) * width
            y2 = (cy + h / 2) * height
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Find which prompt matched (highest logit token)
            token_scores = logits[keep][i].cpu().numpy()
            # Map token index back to prompt (approximate: split by prompt count)
            prompt_idx = int(np.argmax(token_scores)) % len(prompts)
            cls_name = prompts[min(prompt_idx, len(prompts) - 1)]

            detections.append({
                "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                "cls_name": cls_name,
                "conf": float(score),
                "source": "grounding_dino",
            })

        return detections

    except Exception:
        return []


def merge_detections(
    yolo_boxes: list[dict],
    gdino_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """Merge YOLO and Grounding DINO detections via NMS deduplication.

    YOLO detections take priority (higher confidence assumed for known classes).
    Grounding DINO detections are kept only if they don't overlap significantly
    with existing YOLO detections.
    """
    # Tag sources
    for b in yolo_boxes:
        b.setdefault("source", "yolo")
    for b in gdino_boxes:
        b.setdefault("source", "grounding_dino")

    # Start with all YOLO boxes
    merged = list(yolo_boxes)

    # Add Grounding DINO boxes that don't overlap with YOLO
    for gd in gdino_boxes:
        overlaps = False
        for yb in merged:
            if _compute_iou(gd["xyxy"], yb["xyxy"]) > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            merged.append(gd)

    return merged


def _compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two xyxy bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / max(union, 1e-8)


def extract_boxes(results: Any) -> list[dict]:
    """Normalize ultralytics Results into dicts with xyxy/cls_name/conf.

    Tolerates the small variations between ``ultralytics`` versions and
    accepts simple test fakes that expose the same attribute surface.
    """
    boxes: list[dict] = []
    for r in results:
        names = getattr(r, "names", {})
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for box in r.boxes:
            xyxy = _scalar_list(box.xyxy)
            if xyxy is None or len(xyxy) < 4:
                continue
            cls_idx = _scalar(box.cls)
            conf = _scalar(box.conf)
            cls_name = (
                names.get(int(cls_idx), str(int(cls_idx)))
                if isinstance(names, dict)
                else str(int(cls_idx))
            )
            boxes.append(
                {
                    "xyxy": tuple(float(v) for v in xyxy[:4]),
                    "cls_name": cls_name,
                    "conf": float(conf),
                }
            )
    return boxes


def _scalar(x: Any) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    if hasattr(x, "__len__"):
        return float(x[0])
    return float(x)


def _scalar_list(x: Any) -> list[float] | None:
    if x is None:
        return None
    # box.xyxy is a tensor of shape (1, 4). Strip the leading dim.
    if hasattr(x, "tolist"):
        v = x.tolist()
    else:
        v = list(x)
    while isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
        v = v[0]
    return v if isinstance(v, list) else None
