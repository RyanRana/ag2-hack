"""WeedDetectorAgent — YOLO weed-vs-crop detection (paradigm: ML)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pulse.agents.base import ChannelAgent
from pulse.detection import extract_boxes
from pulse.latent import CONDITION_LABELS, FieldLatentState
from pulse.messages import ConstraintMessage


class WeedDetectorAgent(ChannelAgent):
    HF_MODEL_ID = "foduucom/plant-leaf-detection-and-classification"

    def __init__(self, model: Any | None = None) -> None:
        super().__init__(name="weed_detector")
        self._model = model

    def _load_model(self) -> Any:
        """Lazy-load the foduucom YOLO weights.

        Strategy: download ``best.pt`` from HF Hub directly, then load with
        base ``ultralytics.YOLO``. Avoids the version-pinning fight with
        ``ultralyticsplus`` and works with torch>=2.6 (§14.A).
        """
        if self._model is not None:
            return self._model
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO

        weights = hf_hub_download(repo_id=self.HF_MODEL_ID, filename="best.pt")
        self._model = YOLO(weights)
        if hasattr(self._model, "overrides"):
            self._model.overrides["conf"] = 0.25
            self._model.overrides["iou"] = 0.45
            self._model.overrides["max_det"] = 100
        return self._model

    def emit_constraint(
        self, image_path: str | None, latent: FieldLatentState
    ) -> ConstraintMessage:
        model = self._load_model()
        results = model.predict(image_path)
        detections = extract_boxes(results)
        per_ll: dict[int, np.ndarray] = {}
        per_resid: dict[int, float] = {}
        per_conf: dict[int, float] = {}
        weed_idx = CONDITION_LABELS.index("weed")
        healthy_idx = CONDITION_LABELS.index("healthy_crop")
        for plant in latent.plants:
            # Default: NO information (zeros) — never drown the other agents
            # just because YOLO missed. Only emit a confident constraint
            # when YOLO actually overlaps this plant.
            log_lik = np.zeros(len(CONDITION_LABELS))
            best = self._best_overlap(plant.bbox, detections)
            if best is None:
                # YOLO had no overlap. Soft nudge: mildly suppress weed,
                # gently raise healthy. NOT a strong claim either way.
                log_lik[healthy_idx] = 0.40
                log_lik[weed_idx] = -0.30
                per_conf[plant.plant_id] = 0.3
            else:
                cls_name = best["cls_name"].lower()
                conf = best["conf"]
                if "weed" in cls_name:
                    log_lik[weed_idx] = float(2.0 + 4.0 * conf)
                    log_lik[healthy_idx] = -1.5
                else:
                    log_lik[healthy_idx] = float(1.0 + 2.5 * conf)
                    log_lik[weed_idx] = -1.0
                per_conf[plant.plant_id] = float(conf)
            per_ll[plant.plant_id] = log_lik
            per_resid[plant.plant_id] = 0.0
        return ConstraintMessage(
            sender=self.name,
            timestamp=time.time(),
            iteration=latent.iteration,
            per_plant_log_likelihoods=per_ll,
            per_plant_residual=per_resid,
            per_plant_confidence=per_conf,
            labels_discriminated=["weed", "healthy_crop"],
        )

    @staticmethod
    def _best_overlap(
        plant_bbox: tuple[int, int, int, int],
        detections: list[dict],
        *,
        iou_threshold: float = 0.1,
    ) -> dict | None:
        if not detections:
            return None
        px1, py1, px2, py2 = plant_bbox
        plant_area = max(0, px2 - px1) * max(0, py2 - py1)
        best: dict | None = None
        best_iou = 0.0
        for det in detections:
            dx1, dy1, dx2, dy2 = det["xyxy"]
            ix1 = max(px1, dx1)
            iy1 = max(py1, dy1)
            ix2 = min(px2, dx2)
            iy2 = min(py2, dy2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            det_area = max(0.0, (dx2 - dx1) * (dy2 - dy1))
            union = plant_area + det_area - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best = det
        return best if best_iou >= iou_threshold else None
