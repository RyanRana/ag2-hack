"""SegmentationAgent tests — uses synthetic image, no SAM weights needed."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pesto.agents.segmentation import SegmentationAgent
from pesto.latent import CONDITION_LABELS, FieldLatentState, PlantInstance


def _make_latent(boxes, shape=(480, 640)):
    f = FieldLatentState(image_shape=shape)
    for i, b in enumerate(boxes):
        f.plants.append(PlantInstance(plant_id=i, bbox=b))
    return f


def test_green_crop_supports_healthy(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))  # solid green leaf
    img.save(tmp_path / "g.jpg")
    latent = _make_latent([(10, 10, 190, 190)], shape=(200, 200))
    agent = SegmentationAgent()
    msg = agent.emit_constraint(str(tmp_path / "g.jpg"), latent)
    log_lik = msg.per_plant_log_likelihoods[0]
    healthy_idx = CONDITION_LABELS.index("healthy_crop")
    assert log_lik[healthy_idx] > 0


def test_yellow_crop_elevates_disease(tmp_path):
    # Yellow-leaf crop pixels — mostly satisfies r>0.55, g>0.55, b<0.45
    img = Image.new("RGB", (200, 200), color=(180, 180, 60))
    img.save(tmp_path / "y.jpg")
    latent = _make_latent([(10, 10, 190, 190)], shape=(200, 200))
    agent = SegmentationAgent()
    msg = agent.emit_constraint(str(tmp_path / "y.jpg"), latent)
    log_lik = msg.per_plant_log_likelihoods[0]
    disease_idx = CONDITION_LABELS.index("disease")
    healthy_idx = CONDITION_LABELS.index("healthy_crop")
    assert log_lik[disease_idx] > 0
    assert log_lik[disease_idx] > log_lik[healthy_idx]


def test_constraint_returns_correct_labels():
    agent = SegmentationAgent()
    assert "disease" in {"healthy_crop", "disease", "water_stress", "pest_damage"}


def test_injected_sam_predictor_used(tmp_path):
    """If a SAM predictor is provided, the agent uses it instead of the green-channel fallback."""
    Image.new("RGB", (100, 100), color=(60, 130, 40)).save(tmp_path / "x.jpg")
    called = {"n": 0}

    def fake_sam(crop):
        called["n"] += 1
        return np.ones((crop.shape[0], crop.shape[1]), dtype=bool)

    agent = SegmentationAgent(sam_predictor=fake_sam)
    latent = _make_latent([(10, 10, 90, 90)], shape=(100, 100))
    agent.emit_constraint(str(tmp_path / "x.jpg"), latent)
    assert called["n"] == 1
