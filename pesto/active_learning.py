"""Active Learning Loop — log hard cases, queue for labeling, fine-tune.

Identifies plants that the system struggled with and queues them for
human labeling. Over time, fine-tuning on these hard cases improves
the system where it's weakest.

Trigger conditions (any one is sufficient):
  - human_review was triggered (controller chose human_review action)
  - Cross-exam KL > 3.0 (severe disagreement between agents)
  - Anomaly score > 3σ from mean (truly unknown condition)

Queued crops are saved to disk with metadata. A periodic fine-tune
script can retrain the disease classifier on accumulated labeled data.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pesto.latent import CONDITION_LABELS, FieldLatentState
from pesto.messages import ActionMessage, ConstraintMessage


_QUEUE_DIR = Path(__file__).parent.parent / "data" / "active_learning_queue"


@dataclass
class ActiveLearningEntry:
    """A single plant flagged for human labeling."""

    entry_id: str
    plant_id: int
    image_path: str
    bbox: tuple[int, int, int, int]
    trigger_reason: str         # "human_review", "high_disagreement", "anomaly"
    trigger_score: float        # KL, anomaly score, or confidence
    model_posterior: list[float]  # current posterior over CONDITION_LABELS
    top_prediction: str
    top_confidence: float
    timestamp: float
    crop_saved_path: str | None = None
    human_label: str | None = None      # filled after labeling
    labeled_at: float | None = None


@dataclass
class ActiveLearningState:
    """Tracks the active learning queue and statistics."""

    entries: list[ActiveLearningEntry] = field(default_factory=list)
    total_queued: int = 0
    total_labeled: int = 0
    total_used_for_training: int = 0


class ActiveLearningManager:
    """Manages the active learning queue.

    Identifies hard cases from inference results, saves crops + metadata,
    and tracks labeling progress.
    """

    def __init__(
        self,
        queue_dir: Path | None = None,
        *,
        kl_threshold: float = 3.0,
        anomaly_sigma_threshold: float = 3.0,
    ) -> None:
        self._queue_dir = queue_dir or _QUEUE_DIR
        self._kl_threshold = kl_threshold
        self._anomaly_sigma_threshold = anomaly_sigma_threshold
        self._state = ActiveLearningState()

    @property
    def queue_size(self) -> int:
        return len(self._state.entries)

    @property
    def unlabeled_count(self) -> int:
        return sum(1 for e in self._state.entries if e.human_label is None)

    def process_inference_results(
        self,
        image_path: str,
        latent: FieldLatentState,
        actions: list[ActionMessage],
        cross_exam_kl: dict[int, float],
        anomaly_scores: dict[int, float] | None = None,
    ) -> list[ActiveLearningEntry]:
        """Scan inference results and queue hard cases for labeling.

        Parameters
        ----------
        image_path     : Path to the original field image.
        latent         : Final latent state after inference.
        actions        : Per-plant action recommendations.
        cross_exam_kl  : Max KL disagreement per plant_id.
        anomaly_scores : Per-plant anomaly scores (from AnomalyDetectorAgent).

        Returns
        -------
        List of newly queued entries.
        """
        anomaly_scores = anomaly_scores or {}
        new_entries: list[ActiveLearningEntry] = []

        # Compute anomaly mean + std for sigma threshold
        if anomaly_scores:
            scores = list(anomaly_scores.values())
            anom_mean = float(np.mean(scores))
            anom_std = float(np.std(scores)) if len(scores) > 1 else 1.0
        else:
            anom_mean, anom_std = 0.0, 1.0

        action_map = {a.plant_id: a for a in actions}
        plants = {p.plant_id: p for p in latent.plants}

        for plant in latent.plants:
            pid = plant.plant_id
            trigger_reason = None
            trigger_score = 0.0

            # Check trigger 1: human_review action
            action = action_map.get(pid)
            if action and action.action_type == "human_review":
                trigger_reason = "human_review"
                trigger_score = float(action.expected_utility)

            # Check trigger 2: severe cross-exam disagreement
            kl = cross_exam_kl.get(pid, 0.0)
            if kl > self._kl_threshold:
                if trigger_reason is None or kl > trigger_score:
                    trigger_reason = "high_disagreement"
                    trigger_score = kl

            # Check trigger 3: anomaly score > 3σ
            anom = anomaly_scores.get(pid, 0.0)
            if anom_std > 1e-8 and (anom - anom_mean) / anom_std > self._anomaly_sigma_threshold:
                if trigger_reason is None or anom > trigger_score:
                    trigger_reason = "anomaly"
                    trigger_score = anom

            if trigger_reason is None:
                continue

            posterior = plant.posterior()
            top_idx = int(np.argmax(posterior))
            entry = ActiveLearningEntry(
                entry_id=f"{int(time.time())}_{pid}",
                plant_id=pid,
                image_path=str(image_path),
                bbox=plant.bbox,
                trigger_reason=trigger_reason,
                trigger_score=trigger_score,
                model_posterior=posterior.tolist(),
                top_prediction=CONDITION_LABELS[top_idx],
                top_confidence=float(posterior[top_idx]),
                timestamp=time.time(),
            )

            # Save crop to disk
            crop_path = self._save_crop(image_path, plant.bbox, entry.entry_id)
            entry.crop_saved_path = str(crop_path) if crop_path else None

            self._state.entries.append(entry)
            self._state.total_queued += 1
            new_entries.append(entry)

        return new_entries

    def label_entry(self, entry_id: str, label: str) -> bool:
        """Apply a human label to a queued entry.

        Parameters
        ----------
        entry_id : The entry's unique ID.
        label    : Human-provided condition label (must be in CONDITION_LABELS).

        Returns
        -------
        True if the entry was found and labeled.
        """
        if label not in CONDITION_LABELS:
            raise ValueError(f"Label '{label}' not in CONDITION_LABELS: {CONDITION_LABELS}")

        for entry in self._state.entries:
            if entry.entry_id == entry_id:
                entry.human_label = label
                entry.labeled_at = time.time()
                self._state.total_labeled += 1
                return True
        return False

    def get_labeled_entries(self) -> list[ActiveLearningEntry]:
        """Return all entries that have been labeled by humans."""
        return [e for e in self._state.entries if e.human_label is not None]

    def get_unlabeled_entries(self) -> list[ActiveLearningEntry]:
        """Return all entries pending labeling."""
        return [e for e in self._state.entries if e.human_label is None]

    def get_training_data(self) -> list[dict]:
        """Export labeled entries as training data for fine-tuning.

        Returns list of dicts with "image_path", "label", "label_index" keys.
        """
        data = []
        for entry in self.get_labeled_entries():
            path = entry.crop_saved_path or entry.image_path
            data.append({
                "image_path": path,
                "label": entry.human_label,
                "label_index": CONDITION_LABELS.index(entry.human_label),
                "entry_id": entry.entry_id,
                "trigger_reason": entry.trigger_reason,
            })
        return data

    def export_queue(self, path: Path | None = None) -> Path:
        """Save the queue state to disk as JSON."""
        d = path or self._queue_dir
        d.mkdir(parents=True, exist_ok=True)
        out = d / "queue.json"
        data = {
            "total_queued": self._state.total_queued,
            "total_labeled": self._state.total_labeled,
            "total_used_for_training": self._state.total_used_for_training,
            "entries": [asdict(e) for e in self._state.entries],
        }
        out.write_text(json.dumps(data, indent=2))
        return out

    @classmethod
    def load_queue(cls, path: Path | None = None) -> "ActiveLearningManager":
        """Load queue state from disk."""
        d = path or _QUEUE_DIR
        f = d / "queue.json"
        mgr = cls(queue_dir=d)
        if f.exists():
            data = json.loads(f.read_text())
            mgr._state.total_queued = data.get("total_queued", 0)
            mgr._state.total_labeled = data.get("total_labeled", 0)
            mgr._state.total_used_for_training = data.get("total_used_for_training", 0)
            for ed in data.get("entries", []):
                mgr._state.entries.append(ActiveLearningEntry(
                    entry_id=ed["entry_id"],
                    plant_id=ed["plant_id"],
                    image_path=ed["image_path"],
                    bbox=tuple(ed["bbox"]),
                    trigger_reason=ed["trigger_reason"],
                    trigger_score=ed["trigger_score"],
                    model_posterior=ed["model_posterior"],
                    top_prediction=ed["top_prediction"],
                    top_confidence=ed["top_confidence"],
                    timestamp=ed["timestamp"],
                    crop_saved_path=ed.get("crop_saved_path"),
                    human_label=ed.get("human_label"),
                    labeled_at=ed.get("labeled_at"),
                ))
        return mgr

    def _save_crop(
        self,
        image_path: str,
        bbox: tuple[int, int, int, int],
        entry_id: str,
    ) -> Path | None:
        """Save a plant crop to the queue directory."""
        try:
            crops_dir = self._queue_dir / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            img = Image.open(image_path).convert("RGB")
            crop = img.crop(bbox)
            out = crops_dir / f"{entry_id}.jpg"
            crop.save(str(out))
            return out
        except Exception:
            return None

    def summary(self) -> dict:
        """Return a summary of the active learning state."""
        reasons = {}
        for e in self._state.entries:
            reasons[e.trigger_reason] = reasons.get(e.trigger_reason, 0) + 1

        return {
            "total_queued": self._state.total_queued,
            "total_labeled": self._state.total_labeled,
            "unlabeled": self.unlabeled_count,
            "total_used_for_training": self._state.total_used_for_training,
            "trigger_breakdown": reasons,
        }
