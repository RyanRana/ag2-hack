"""Tests for temporal diff module — uses synthetic images."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pulse.temporal import (
    PlantChangeScore,
    compute_frame_diff,
    compute_multi_frame_diff,
    _simple_structural_diff,
)


def _make_green_image(w=200, h=200):
    return np.full((h, w, 3), [60, 130, 40], dtype=np.uint8)


def _make_yellow_image(w=200, h=200):
    return np.full((h, w, 3), [180, 180, 60], dtype=np.uint8)


def test_identical_frames_low_change(tmp_path):
    img = Image.fromarray(_make_green_image())
    img.save(tmp_path / "a.jpg")
    img.save(tmp_path / "b.jpg")

    result = compute_frame_diff(
        str(tmp_path / "a.jpg"),
        str(tmp_path / "b.jpg"),
        bboxes=[(10, 10, 190, 190)],
        plant_ids=[0],
        use_optical_flow=False,
    )

    assert 0 in result.per_plant_changes
    assert result.per_plant_changes[0].pixel_diff_score < 0.01
    assert not result.per_plant_changes[0].changed


def test_different_frames_high_change(tmp_path):
    Image.fromarray(_make_green_image()).save(tmp_path / "curr.jpg")
    Image.fromarray(_make_yellow_image()).save(tmp_path / "prev.jpg")

    result = compute_frame_diff(
        str(tmp_path / "curr.jpg"),
        str(tmp_path / "prev.jpg"),
        bboxes=[(10, 10, 190, 190)],
        plant_ids=[0],
        change_threshold=0.10,  # lower threshold for this clear difference
        use_optical_flow=False,
    )

    assert result.per_plant_changes[0].pixel_diff_score > 0.1
    assert result.per_plant_changes[0].changed


def test_partial_change_detected(tmp_path):
    """Only part of the plant changed."""
    curr = _make_green_image()
    prev = _make_green_image()
    # Change bottom half to yellow in previous
    prev[100:, :] = [180, 180, 60]
    Image.fromarray(curr).save(tmp_path / "curr.jpg")
    Image.fromarray(prev).save(tmp_path / "prev.jpg")

    result = compute_frame_diff(
        str(tmp_path / "curr.jpg"),
        str(tmp_path / "prev.jpg"),
        bboxes=[(0, 0, 200, 200)],
        plant_ids=[0],
        use_optical_flow=False,
    )

    score = result.per_plant_changes[0].pixel_diff_score
    # Partial change — should be between 0 and full change
    assert 0.05 < score < 0.5


def test_multiple_plants(tmp_path):
    curr = _make_green_image(400, 200)
    prev = _make_green_image(400, 200)
    # Change only the right half (plant 1)
    prev[:, 200:] = [180, 180, 60]
    Image.fromarray(curr).save(tmp_path / "curr.jpg")
    Image.fromarray(prev).save(tmp_path / "prev.jpg")

    result = compute_frame_diff(
        str(tmp_path / "curr.jpg"),
        str(tmp_path / "prev.jpg"),
        bboxes=[(0, 0, 190, 190), (210, 0, 390, 190)],
        plant_ids=[0, 1],
        use_optical_flow=False,
    )

    assert result.per_plant_changes[0].pixel_diff_score < result.per_plant_changes[1].pixel_diff_score


def test_escalated_ids(tmp_path):
    Image.fromarray(_make_green_image()).save(tmp_path / "curr.jpg")
    Image.fromarray(_make_yellow_image()).save(tmp_path / "prev.jpg")

    result = compute_frame_diff(
        str(tmp_path / "curr.jpg"),
        str(tmp_path / "prev.jpg"),
        bboxes=[(10, 10, 190, 190)],
        plant_ids=[0],
        change_threshold=0.05,
        use_optical_flow=False,
    )

    assert 0 in result.escalated_plant_ids


def test_optical_flow_runs(tmp_path):
    """Optical flow should run without error."""
    curr = _make_green_image()
    prev = _make_green_image()
    # Shift pattern slightly
    prev[5:, 5:] = curr[:-5, :-5]
    Image.fromarray(curr).save(tmp_path / "curr.jpg")
    Image.fromarray(prev).save(tmp_path / "prev.jpg")

    result = compute_frame_diff(
        str(tmp_path / "curr.jpg"),
        str(tmp_path / "prev.jpg"),
        bboxes=[(10, 10, 190, 190)],
        plant_ids=[0],
        use_optical_flow=True,
    )

    assert result.per_plant_changes[0].flow_magnitude >= 0


def test_multi_frame_diff(tmp_path):
    Image.fromarray(_make_green_image()).save(tmp_path / "curr.jpg")
    Image.fromarray(_make_green_image()).save(tmp_path / "prev1.jpg")
    Image.fromarray(_make_yellow_image()).save(tmp_path / "prev2.jpg")

    result = compute_multi_frame_diff(
        str(tmp_path / "curr.jpg"),
        [str(tmp_path / "prev1.jpg"), str(tmp_path / "prev2.jpg")],
        bboxes=[(10, 10, 190, 190)],
        plant_ids=[0],
        change_threshold=0.05,
    )

    # Should pick the max change (vs prev2 which is yellow)
    assert result.per_plant_changes[0].pixel_diff_score > 0.1


def test_structural_diff_identical():
    a = np.full((50, 50), 128.0, dtype=np.float32)
    assert _simple_structural_diff(a, a.copy()) < 0.01


def test_structural_diff_different():
    a = np.full((50, 50), 50.0, dtype=np.float32)
    b = np.full((50, 50), 200.0, dtype=np.float32)
    diff = _simple_structural_diff(a, b)
    assert diff > 0.3
