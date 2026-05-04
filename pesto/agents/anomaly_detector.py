"""AnomalyDetectorAgent — PatchCore on DINOv2 embeddings (paradigm: CV).

Catches unknown-unknowns that classifiers would force into a known label.
Trained offline on healthy plant crops: at inference, scores each plant's
distance from the healthy distribution using a memory bank of DINOv2 patch
embeddings (PatchCore approach).

If anomaly score > threshold (default 3 sigma), the agent flags the plant
as anomalous and pushes mass toward ``ambiguous`` in the shared posterior,
triggering human_review via the controller's entropy-based escalation.

Same ChannelAgent envelope, same ConstraintMessage protocol.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pesto.agents.base import ChannelAgent
from pesto.backbone import DINOv2Backbone
from pesto.latent import CONDITION_LABELS, FieldLatentState
from pesto.messages import ConstraintMessage


_MEMORY_BANK_DIR = Path(__file__).parent.parent.parent / "data" / "anomaly_memory_bank"


class PatchCoreMemoryBank:
    """Simple PatchCore-style memory bank of healthy-plant DINOv2 embeddings.

    Stores a (M, D) matrix of patch-level embeddings from healthy crops.
    At inference, anomaly score = min distance from any stored embedding.
    """

    def __init__(self, embeddings: np.ndarray | None = None) -> None:
        self._bank: np.ndarray | None = embeddings  # (M, D)

    @property
    def is_fitted(self) -> bool:
        return self._bank is not None and self._bank.shape[0] > 0

    def fit(self, patch_embeddings_list: list[np.ndarray]) -> None:
        """Build the memory bank from a list of per-crop patch embedding arrays.

        Parameters
        ----------
        patch_embeddings_list : list of (N_patches, D) arrays from healthy crops.
        """
        all_patches = np.concatenate(patch_embeddings_list, axis=0)  # (total, D)
        # Coreset subsampling: keep at most 10k patches for efficiency
        if all_patches.shape[0] > 10_000:
            rng = np.random.RandomState(42)
            idx = rng.choice(all_patches.shape[0], 10_000, replace=False)
            all_patches = all_patches[idx]
        self._bank = all_patches.astype(np.float32)

    def score(self, patch_embeddings: np.ndarray) -> float:
        """Compute anomaly score for a single plant crop.

        Uses Euclidean nearest-neighbour distance (standard PatchCore).

        Parameters
        ----------
        patch_embeddings : (N_patches, D) from DINOv2 backbone.

        Returns
        -------
        Anomaly score: mean of per-patch minimum L2 distances to the bank.
        Higher = more anomalous.
        """
        if not self.is_fitted:
            return 0.0

        query = patch_embeddings.astype(np.float32)  # (N, D)

        # Compute pairwise squared L2 distances efficiently:
        # ||q - b||^2 = ||q||^2 + ||b||^2 - 2*q.b
        q_sq = np.sum(query ** 2, axis=1, keepdims=True)   # (N, 1)
        b_sq = np.sum(self._bank ** 2, axis=1, keepdims=True).T  # (1, M)
        dist_sq = q_sq + b_sq - 2.0 * (query @ self._bank.T)  # (N, M)
        dist_sq = np.maximum(dist_sq, 0.0)  # numerical safety

        # Min distance per query patch to its nearest bank neighbour
        min_dist_per_patch = np.sqrt(dist_sq.min(axis=1))  # (N,)

        # Anomaly score = mean of top-k patch distances (top 10% most anomalous)
        k = max(1, int(0.1 * len(min_dist_per_patch)))
        top_k = np.sort(min_dist_per_patch)[-k:]
        return float(top_k.mean())

    def save(self, path: Path | None = None) -> Path:
        """Persist the memory bank to disk."""
        d = path or _MEMORY_BANK_DIR
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        out = d / "memory_bank.npy"
        if self._bank is not None:
            np.save(str(out), self._bank)
        return out

    @classmethod
    def load(cls, path: Path | None = None) -> "PatchCoreMemoryBank":
        """Load a persisted memory bank."""
        d = path or _MEMORY_BANK_DIR
        f = Path(d) / "memory_bank.npy"
        if f.exists():
            bank = np.load(str(f))
            return cls(embeddings=bank)
        return cls()


class AnomalyDetectorAgent(ChannelAgent):
    """ChannelAgent that flags unknown conditions via PatchCore anomaly scoring.

    Uses DINOv2 embeddings to compare each plant crop against a memory bank
    of healthy plant patches. Plants that are far from the healthy distribution
    get mass pushed toward ``ambiguous``, triggering review/rescan.
    """

    def __init__(
        self,
        backbone: DINOv2Backbone | None = None,
        memory_bank: PatchCoreMemoryBank | None = None,
        anomaly_threshold: float = 0.3,
    ) -> None:
        super().__init__(name="anomaly_detector")
        self._backbone = backbone
        self._memory_bank = memory_bank or PatchCoreMemoryBank.load()
        self._anomaly_threshold = anomaly_threshold
        # Store per-plant anomaly scores for downstream consumption
        self.anomaly_scores: dict[int, float] = {}

    def _get_backbone(self) -> DINOv2Backbone:
        if self._backbone is None:
            self._backbone = DINOv2Backbone()
        return self._backbone

    def emit_constraint(
        self, image_path: str | None, latent: FieldLatentState
    ) -> ConstraintMessage:
        backbone = self._get_backbone()
        per_ll: dict[int, np.ndarray] = {}
        per_resid: dict[int, float] = {}
        per_conf: dict[int, float] = {}
        self.anomaly_scores = {}

        i_ambig = CONDITION_LABELS.index("ambiguous")
        i_healthy = CONDITION_LABELS.index("healthy_crop")

        if image_path is not None and self._memory_bank.is_fitted:
            full_img = Image.open(image_path).convert("RGB")
            for plant in latent.plants:
                crop = full_img.crop(plant.bbox)
                if crop.size[0] < 14 or crop.size[1] < 14:
                    # Skip tiny crops
                    per_ll[plant.plant_id] = np.zeros(len(CONDITION_LABELS))
                    per_resid[plant.plant_id] = 0.0
                    per_conf[plant.plant_id] = 0.1
                    self.anomaly_scores[plant.plant_id] = 0.0
                    continue

                feats = backbone.extract_features(crop)
                score = self._memory_bank.score(feats["patch_tokens"])
                self.anomaly_scores[plant.plant_id] = score

                log_lik = np.zeros(len(CONDITION_LABELS))

                if score > self._anomaly_threshold:
                    # Anomalous: push toward ambiguous, suppress healthy
                    strength = min((score - self._anomaly_threshold) * 5.0, 3.0)
                    log_lik[i_ambig] = strength
                    log_lik[i_healthy] = -strength * 0.5
                else:
                    # Normal: mildly support healthy
                    log_lik[i_healthy] = 0.3 * (1.0 - score / max(self._anomaly_threshold, 1e-8))

                per_ll[plant.plant_id] = log_lik
                per_resid[plant.plant_id] = float(score)
                per_conf[plant.plant_id] = float(
                    min(0.5 + abs(score - self._anomaly_threshold) * 2.0, 0.9)
                )
        else:
            # No image or unfitted bank — emit neutral constraints
            for plant in latent.plants:
                per_ll[plant.plant_id] = np.zeros(len(CONDITION_LABELS))
                per_resid[plant.plant_id] = 0.0
                per_conf[plant.plant_id] = 0.1
                self.anomaly_scores[plant.plant_id] = 0.0

        return ConstraintMessage(
            sender=self.name,
            timestamp=time.time(),
            iteration=latent.iteration,
            per_plant_log_likelihoods=per_ll,
            per_plant_residual=per_resid,
            per_plant_confidence=per_conf,
            labels_discriminated=["ambiguous", "healthy_crop"],
            metadata={
                "memory_bank_fitted": self._memory_bank.is_fitted,
                "anomaly_threshold": self._anomaly_threshold,
                "per_plant_anomaly_scores": {
                    int(k): float(v) for k, v in self.anomaly_scores.items()
                },
            },
        )
