"""Temporal Diff Module — per-plant change detection across frames.

Given a current frame and one or more previous frames, computes per-plant
change scores via:
  1. **Frame differencing**: absolute pixel difference in the plant's bbox.
  2. **Optical flow** (optional): Farneback dense optical flow magnitude.

The change scores feed cross-examination as an urgency signal:
  - "Plant changed since last scan" is a strong disease-progression indicator.
  - High change + ML disagreement → auto-escalate in cross-exam.

This is a pre-processing module (not an AG2 agent, no AG2 envelope).
Requires multi-frame data (drone revisit imagery).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class PlantChangeScore:
    """Per-plant temporal change metrics."""

    plant_id: int
    bbox: tuple[int, int, int, int]
    pixel_diff_score: float       # Mean absolute difference (normalised 0-1)
    flow_magnitude: float         # Mean optical flow magnitude (px/frame)
    structural_change: float      # SSIM-based structural change (0=same, 1=different)
    combined_score: float         # Weighted combination
    changed: bool                 # True if combined_score > threshold


@dataclass
class TemporalDiffResult:
    """Result of temporal analysis across two frames."""

    per_plant_changes: dict[int, PlantChangeScore] = field(default_factory=dict)
    frame_diff_mean: float = 0.0
    escalated_plant_ids: list[int] = field(default_factory=list)


def compute_frame_diff(
    current_path: str,
    previous_path: str,
    bboxes: list[tuple[int, int, int, int]],
    plant_ids: list[int],
    *,
    change_threshold: float = 0.15,
    use_optical_flow: bool = True,
) -> TemporalDiffResult:
    """Compute per-plant change scores between two frames.

    Parameters
    ----------
    current_path   : Path to the current frame image.
    previous_path  : Path to the previous frame image.
    bboxes         : List of (x1, y1, x2, y2) bounding boxes for plants.
    plant_ids      : Corresponding plant IDs.
    change_threshold : Threshold for flagging a plant as "changed".
    use_optical_flow : Whether to compute Farneback optical flow.

    Returns
    -------
    TemporalDiffResult with per-plant change scores.
    """
    curr_img = np.asarray(Image.open(current_path).convert("RGB"))
    prev_img = np.asarray(Image.open(previous_path).convert("RGB"))

    # Resize previous to match current if dimensions differ
    if prev_img.shape[:2] != curr_img.shape[:2]:
        prev_img = cv2.resize(prev_img, (curr_img.shape[1], curr_img.shape[0]))

    # Convert to grayscale for flow and structural comparison
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)

    # Global frame diff
    global_diff = np.abs(curr_img.astype(np.float32) - prev_img.astype(np.float32)) / 255.0
    frame_diff_mean = float(global_diff.mean())

    # Optical flow (full frame, computed once)
    flow = None
    if use_optical_flow:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

    result = TemporalDiffResult(frame_diff_mean=frame_diff_mean)

    for pid, bbox in zip(plant_ids, bboxes):
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(curr_img.shape[1], x2)
        y2 = min(curr_img.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            result.per_plant_changes[pid] = PlantChangeScore(
                plant_id=pid, bbox=bbox,
                pixel_diff_score=0.0, flow_magnitude=0.0,
                structural_change=0.0, combined_score=0.0, changed=False,
            )
            continue

        # Per-plant pixel difference
        crop_diff = global_diff[y1:y2, x1:x2]
        pixel_diff_score = float(crop_diff.mean())

        # Per-plant optical flow magnitude
        flow_mag = 0.0
        if flow is not None:
            flow_crop = flow[y1:y2, x1:x2]
            mag = np.sqrt(flow_crop[..., 0] ** 2 + flow_crop[..., 1] ** 2)
            flow_mag = float(mag.mean())

        # Structural change (simplified SSIM approximation)
        curr_crop_gray = curr_gray[y1:y2, x1:x2].astype(np.float32)
        prev_crop_gray = prev_gray[y1:y2, x1:x2].astype(np.float32)
        structural_change = _simple_structural_diff(curr_crop_gray, prev_crop_gray)

        # Weighted combination
        combined = (
            0.4 * pixel_diff_score
            + 0.3 * min(flow_mag / 5.0, 1.0)  # normalise flow to [0,1]
            + 0.3 * structural_change
        )

        change = PlantChangeScore(
            plant_id=pid,
            bbox=bbox,
            pixel_diff_score=pixel_diff_score,
            flow_magnitude=flow_mag,
            structural_change=structural_change,
            combined_score=float(combined),
            changed=combined > change_threshold,
        )
        result.per_plant_changes[pid] = change

        if change.changed:
            result.escalated_plant_ids.append(pid)

    return result


def compute_multi_frame_diff(
    current_path: str,
    previous_paths: list[str],
    bboxes: list[tuple[int, int, int, int]],
    plant_ids: list[int],
    *,
    change_threshold: float = 0.15,
) -> TemporalDiffResult:
    """Compute change scores against multiple previous frames.

    Uses the maximum change score across all previous frames per plant,
    which captures both sudden changes (vs. last frame) and gradual
    progression (vs. earlier frames).
    """
    if not previous_paths:
        return TemporalDiffResult()

    all_results = []
    for prev_path in previous_paths:
        r = compute_frame_diff(
            current_path, prev_path, bboxes, plant_ids,
            change_threshold=change_threshold,
        )
        all_results.append(r)

    # Merge: take max change per plant across all comparisons
    merged = TemporalDiffResult(
        frame_diff_mean=max(r.frame_diff_mean for r in all_results),
    )

    for pid in plant_ids:
        best: PlantChangeScore | None = None
        for r in all_results:
            score = r.per_plant_changes.get(pid)
            if score is None:
                continue
            if best is None or score.combined_score > best.combined_score:
                best = score
        if best is not None:
            merged.per_plant_changes[pid] = best
            if best.changed:
                merged.escalated_plant_ids.append(pid)

    return merged


def _simple_structural_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Simplified structural difference (SSIM-inspired).

    Returns a value in [0, 1] where 0 = identical and 1 = completely different.
    """
    if a.size == 0 or b.size == 0:
        return 0.0

    # Ensure same shape
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    mu_a = float(a.mean())
    mu_b = float(b.mean())
    sigma_a = float(a.std())
    sigma_b = float(b.std())
    sigma_ab = float(((a - mu_a) * (b - mu_b)).mean())

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / (
        (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a ** 2 + sigma_b ** 2 + C2)
    )

    return float(1.0 - max(0.0, min(1.0, ssim)))
