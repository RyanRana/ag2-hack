"""Initial plant detection — populate FieldLatentState with PlantInstances.

Phase 2 uses YOLO directly. Phase 3 swaps in SAM (or skips per §14.B). The
output contract is the same: a FieldLatentState whose plants have unique
``plant_id``, integer bboxes, and uniform priors over CONDITION_LABELS.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from pulse.latent import FieldLatentState, PlantInstance


def detect_plants_yolo(
    image_path: str,
    yolo_model: Any,
    *,
    max_plants: int = 64,
) -> FieldLatentState:
    """Run YOLO and emit one PlantInstance per detection (uniform priors)."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    results = yolo_model.predict(image_path)
    boxes = extract_boxes(results)
    boxes.sort(key=lambda b: (-b["conf"], b["xyxy"][0], b["xyxy"][1]))
    field = FieldLatentState(image_shape=(height, width))
    for i, det in enumerate(boxes[:max_plants]):
        x1, y1, x2, y2 = det["xyxy"]
        field.plants.append(
            PlantInstance(plant_id=i, bbox=(int(x1), int(y1), int(x2), int(y2)))
        )
    return field


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
