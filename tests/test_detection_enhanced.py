"""Tests for enhanced detection (Grounding DINO + NMS merge)."""

from __future__ import annotations

from pulse.detection import _compute_iou, merge_detections


def _yolo_box(x1, y1, x2, y2, conf=0.9, name="plant"):
    return {"xyxy": (x1, y1, x2, y2), "cls_name": name, "conf": conf, "source": "yolo"}


def _gdino_box(x1, y1, x2, y2, conf=0.6, name="wilted plant"):
    return {"xyxy": (x1, y1, x2, y2), "cls_name": name, "conf": conf, "source": "grounding_dino"}


# --- IoU tests ---

def test_iou_identical_boxes():
    iou = _compute_iou((10, 10, 50, 50), (10, 10, 50, 50))
    assert abs(iou - 1.0) < 1e-6


def test_iou_no_overlap():
    iou = _compute_iou((0, 0, 10, 10), (20, 20, 30, 30))
    assert iou == 0.0


def test_iou_partial_overlap():
    iou = _compute_iou((0, 0, 20, 20), (10, 10, 30, 30))
    # Intersection = 10x10 = 100. Union = 2*400 - 100 = 700.
    assert abs(iou - 100 / 700) < 1e-6


def test_iou_one_inside_other():
    iou = _compute_iou((0, 0, 100, 100), (25, 25, 75, 75))
    # Intersection = 50x50 = 2500. Union = 10000 + 2500 - 2500 = 10000.
    assert abs(iou - 0.25) < 1e-6


# --- Merge detections tests ---

def test_merge_no_overlap():
    yolo = [_yolo_box(0, 0, 50, 50)]
    gdino = [_gdino_box(100, 100, 150, 150)]
    merged = merge_detections(yolo, gdino, iou_threshold=0.5)
    assert len(merged) == 2


def test_merge_overlapping_removed():
    yolo = [_yolo_box(0, 0, 50, 50)]
    gdino = [_gdino_box(0, 0, 50, 50)]  # exact overlap
    merged = merge_detections(yolo, gdino, iou_threshold=0.5)
    assert len(merged) == 1
    assert merged[0]["source"] == "yolo"


def test_merge_partial_overlap_above_threshold():
    yolo = [_yolo_box(0, 0, 50, 50)]
    gdino = [_gdino_box(5, 5, 55, 55)]  # large overlap
    merged = merge_detections(yolo, gdino, iou_threshold=0.3)
    assert len(merged) == 1


def test_merge_preserves_all_yolo():
    yolo = [_yolo_box(0, 0, 50, 50), _yolo_box(100, 100, 150, 150)]
    gdino = [_gdino_box(200, 200, 250, 250)]
    merged = merge_detections(yolo, gdino, iou_threshold=0.5)
    assert len(merged) == 3


def test_merge_empty_gdino():
    yolo = [_yolo_box(0, 0, 50, 50)]
    merged = merge_detections(yolo, [], iou_threshold=0.5)
    assert len(merged) == 1


def test_merge_empty_yolo():
    gdino = [_gdino_box(0, 0, 50, 50)]
    merged = merge_detections([], gdino, iou_threshold=0.5)
    assert len(merged) == 1
