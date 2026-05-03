"""WeedDetectorAgent unit tests with an injected fake YOLO model.

These tests do not require torch/ultralyticsplus or the foduucom HF model
— they verify the agent's bbox matching, label mapping, and ConstraintMessage
shape against a deterministic fake.
"""

from __future__ import annotations

import numpy as np

from pulse.agents.weed_detector import WeedDetectorAgent
from pulse.detection import detect_plants_yolo, extract_boxes
from pulse.latent import CONDITION_LABELS, FieldLatentState, PlantInstance
from pulse.messages import ConstraintMessage


# --- Fake YOLO model -------------------------------------------------------


class _FakeTensorScalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _FakeTensorList:
    """Mimics tensor.xyxy: a (1, 4) shape that exposes .tolist()."""

    def __init__(self, xyxy):
        self._v = [list(xyxy)]

    def tolist(self):
        return self._v


class _FakeBox:
    def __init__(self, xyxy, cls_idx, conf):
        self.xyxy = _FakeTensorList(xyxy)
        self.cls = _FakeTensorScalar(cls_idx)
        self.conf = _FakeTensorScalar(conf)


class _FakeResults:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeYOLO:
    def __init__(self, detections):
        # detections: [(xyxy, cls_name, conf)]
        unique_names = []
        for _, cls, _ in detections:
            if cls not in unique_names:
                unique_names.append(cls)
        names = {i: n for i, n in enumerate(unique_names)}
        cls_to_idx = {n: i for i, n in names.items()}
        boxes = [_FakeBox(xyxy, cls_to_idx[cls], conf) for xyxy, cls, conf in detections]
        self._results = [_FakeResults(boxes, names)]

    def predict(self, image_path):
        return self._results


# --- Helpers ---------------------------------------------------------------


def _make_latent(boxes):
    f = FieldLatentState(image_shape=(480, 640))
    for i, b in enumerate(boxes):
        f.plants.append(PlantInstance(plant_id=i, bbox=b))
    return f


# --- Tests -----------------------------------------------------------------


def test_extract_boxes_normalizes_fake_yolo():
    yolo = _FakeYOLO([((10, 20, 100, 200), "weed", 0.85)])
    out = extract_boxes(yolo.predict("x"))
    assert len(out) == 1
    assert out[0]["cls_name"] == "weed"
    assert out[0]["conf"] == 0.85
    assert out[0]["xyxy"] == (10.0, 20.0, 100.0, 200.0)


def test_emit_constraint_returns_message_for_each_plant():
    yolo = _FakeYOLO([((10, 10, 100, 100), "weed", 0.85)])
    latent = _make_latent([(10, 10, 100, 100), (200, 200, 280, 280)])
    agent = WeedDetectorAgent(model=yolo)
    msg = agent.emit_constraint("dummy.jpg", latent)
    assert isinstance(msg, ConstraintMessage)
    assert msg.sender == "weed_detector"
    assert msg.iteration == latent.iteration
    assert set(msg.per_plant_log_likelihoods.keys()) == {0, 1}
    assert set(msg.labels_discriminated) == {"weed", "healthy_crop"}


def test_overlapping_weed_detection_concentrates_on_weed_label():
    yolo = _FakeYOLO([((10, 10, 100, 100), "weed", 0.85)])
    latent = _make_latent([(10, 10, 100, 100)])
    agent = WeedDetectorAgent(model=yolo)
    msg = agent.emit_constraint("dummy.jpg", latent)
    log_lik = msg.per_plant_log_likelihoods[0]
    weed_idx = CONDITION_LABELS.index("weed")
    healthy_idx = CONDITION_LABELS.index("healthy_crop")
    assert log_lik[weed_idx] > log_lik[healthy_idx]
    assert msg.per_plant_confidence[0] == 0.85


def test_overlapping_crop_class_concentrates_on_healthy():
    yolo = _FakeYOLO([((10, 10, 100, 100), "tomato", 0.7)])
    latent = _make_latent([(10, 10, 100, 100)])
    agent = WeedDetectorAgent(model=yolo)
    msg = agent.emit_constraint("dummy.jpg", latent)
    log_lik = msg.per_plant_log_likelihoods[0]
    weed_idx = CONDITION_LABELS.index("weed")
    healthy_idx = CONDITION_LABELS.index("healthy_crop")
    assert log_lik[healthy_idx] > log_lik[weed_idx]


def test_no_overlap_falls_back_to_healthy_with_low_confidence():
    yolo = _FakeYOLO([((10, 10, 100, 100), "weed", 0.85)])
    # Plant nowhere near any detection.
    latent = _make_latent([(500, 500, 580, 580)])
    agent = WeedDetectorAgent(model=yolo)
    msg = agent.emit_constraint("dummy.jpg", latent)
    healthy_idx = CONDITION_LABELS.index("healthy_crop")
    weed_idx = CONDITION_LABELS.index("weed")
    log_lik = msg.per_plant_log_likelihoods[0]
    assert log_lik[healthy_idx] > log_lik[weed_idx]
    assert msg.per_plant_confidence[0] == 0.3


def test_constraint_post_update_makes_weed_dominant():
    yolo = _FakeYOLO([((10, 10, 100, 100), "weed", 0.95)])
    latent = _make_latent([(10, 10, 100, 100)])
    agent = WeedDetectorAgent(model=yolo)
    msg = agent.emit_constraint("dummy.jpg", latent)
    latent.update_plant(0, msg.per_plant_log_likelihoods[0], "weed_detector")
    top = latent.plants[0].top_k(1)
    assert top[0][0] == "weed"
    assert top[0][1] > 0.5


def test_detect_plants_yolo_creates_uniform_priors(tmp_path):
    # Real PIL image → real (height, width).
    from PIL import Image

    img_path = tmp_path / "field.jpg"
    Image.new("RGB", (640, 480), color=(120, 80, 40)).save(img_path)

    yolo = _FakeYOLO(
        [
            ((10, 10, 100, 100), "weed", 0.9),
            ((300, 200, 400, 380), "tomato", 0.8),
        ]
    )
    latent = detect_plants_yolo(str(img_path), yolo)
    assert latent.image_shape == (480, 640)  # (h, w)
    assert len(latent.plants) == 2
    # Plants are sorted by descending confidence, so plant 0 is the weed.
    assert latent.plants[0].bbox == (10, 10, 100, 100)
    # Uniform priors mean all entries equal.
    np.testing.assert_allclose(
        latent.plants[0].posterior(),
        np.ones(len(CONDITION_LABELS)) / len(CONDITION_LABELS),
    )
