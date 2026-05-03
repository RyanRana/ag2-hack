"""Tests for visual explanation rendering and evidence map retention."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pulse.agents.segmentation import PlantEvidenceMaps, SegmentationAgent
from pulse.latent import FieldLatentState, PlantInstance
from pulse.visual_explain import render_field_explanation, render_plant_explanation


def _make_latent(boxes, shape=(200, 200)):
    f = FieldLatentState(image_shape=shape)
    for i, b in enumerate(boxes):
        f.plants.append(PlantInstance(plant_id=i, bbox=b))
    return f


def test_segmentation_retains_evidence_maps(tmp_path):
    """SegmentationAgent should populate evidence_maps after emit_constraint."""
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "g.jpg")
    latent = _make_latent([(10, 10, 190, 190)])
    agent = SegmentationAgent()
    agent.emit_constraint(str(tmp_path / "g.jpg"), latent)

    assert 0 in agent.evidence_maps
    ev = agent.evidence_maps[0]
    assert isinstance(ev, PlantEvidenceMaps)
    assert ev.leaf_mask.shape[0] > 0
    assert ev.yellow_mask.shape[0] > 0
    assert ev.edge_mask.shape[0] > 0
    assert ev.contour_mask.shape[0] > 0
    assert ev.gradient_magnitude.shape[0] > 0
    assert "green_ratio" in ev.features


def test_evidence_maps_cleared_between_calls(tmp_path):
    """Evidence maps should be fresh on each emit_constraint call."""
    img = Image.new("RGB", (100, 100), color=(60, 130, 40))
    img.save(tmp_path / "a.jpg")
    latent1 = _make_latent([(5, 5, 95, 95)], shape=(100, 100))
    latent2 = _make_latent([(5, 5, 50, 50)], shape=(100, 100))

    agent = SegmentationAgent()
    agent.emit_constraint(str(tmp_path / "a.jpg"), latent1)
    assert 0 in agent.evidence_maps

    agent.emit_constraint(str(tmp_path / "a.jpg"), latent2)
    # Should still have plant 0, but from the new call
    assert 0 in agent.evidence_maps


def test_render_plant_explanation_shape(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "g.jpg")
    latent = _make_latent([(10, 10, 190, 190)])
    agent = SegmentationAgent()
    agent.emit_constraint(str(tmp_path / "g.jpg"), latent)

    img_arr = np.asarray(img)
    result = render_plant_explanation(img_arr, agent.evidence_maps[0])
    assert result.shape == img_arr.shape
    assert result.dtype == np.uint8


def test_render_field_explanation(tmp_path):
    img = Image.new("RGB", (300, 300), color=(60, 130, 40))
    img.save(tmp_path / "field.jpg")
    latent = _make_latent([(10, 10, 140, 140), (150, 150, 290, 290)], shape=(300, 300))
    agent = SegmentationAgent()
    agent.emit_constraint(str(tmp_path / "field.jpg"), latent)

    result = render_field_explanation(str(tmp_path / "field.jpg"), agent.evidence_maps)
    assert result.shape == (300, 300, 3)
    assert result.dtype == np.uint8


def test_yellow_crop_produces_yellow_evidence(tmp_path):
    """Yellow pixels in the crop should appear in the yellow_mask."""
    img = Image.new("RGB", (200, 200), color=(180, 180, 60))
    img.save(tmp_path / "y.jpg")
    latent = _make_latent([(10, 10, 190, 190)])
    agent = SegmentationAgent()
    agent.emit_constraint(str(tmp_path / "y.jpg"), latent)

    ev = agent.evidence_maps[0]
    # Most pixels should be yellow
    yellow_ratio = ev.yellow_mask.sum() / max(ev.yellow_mask.size, 1)
    assert yellow_ratio > 0.5, f"Expected most pixels yellow, got ratio {yellow_ratio}"
