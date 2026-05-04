"""Tests for GrowthStageAgent — uses synthetic images, no real ViT model."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pesto.agents.growth_stage import (
    GROWTH_STAGES,
    URGENCY_MULTIPLIER,
    GrowthStageAgent,
    GrowthStageClassifier,
)
from pesto.latent import CONDITION_LABELS, FieldLatentState, PlantInstance


def _make_latent(n_plants: int = 2, shape=(200, 200)):
    f = FieldLatentState(image_shape=shape)
    for i in range(n_plants):
        f.plants.append(PlantInstance(
            plant_id=i,
            bbox=(i * 80 + 10, 10, i * 80 + 90, 90),
        ))
    return f


# --- GrowthStageClassifier heuristic tests ---

def test_classifier_returns_all_stages():
    clf = GrowthStageClassifier()
    img = Image.new("RGB", (100, 100), color=(60, 130, 40))
    probs = clf.classify(img)
    assert set(probs.keys()) == set(GROWTH_STAGES)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_green_medium_image_favors_vegetative():
    """Dense green coverage with medium size → vegetative."""
    clf = GrowthStageClassifier()
    # Create a mostly green image
    img = np.full((100, 100, 3), [60, 160, 40], dtype=np.uint8)
    probs = clf.classify(img)
    assert probs["vegetative"] > probs["seedling"]


def test_small_sparse_image_favors_seedling():
    """Small image with little green → seedling."""
    clf = GrowthStageClassifier()
    # Small brown/soil-coloured image
    img = np.full((30, 30, 3), [140, 110, 80], dtype=np.uint8)
    # Add a tiny bit of green
    img[10:20, 10:20] = [60, 140, 40]
    probs = clf.classify(img)
    assert probs["seedling"] > probs["fruiting"]


def test_bright_spots_favor_flowering():
    """Bright non-green regions → flowering."""
    clf = GrowthStageClassifier()
    img = np.full((100, 100, 3), [60, 130, 40], dtype=np.uint8)  # green base
    # Add bright red/pink "flowers"
    img[30:60, 30:60] = [220, 80, 120]
    probs = clf.classify(img)
    assert probs["flowering"] > probs["vegetative"]


def test_empty_image_returns_uniform():
    clf = GrowthStageClassifier()
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    probs = clf.classify(img)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


# --- GrowthStageAgent tests ---

def test_agent_emits_neutral_log_likelihoods(tmp_path):
    """Growth stage log-likelihoods should be zero (orthogonal to condition)."""
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    agent = GrowthStageAgent()
    latent = _make_latent(2)
    msg = agent.emit_constraint(str(tmp_path / "test.jpg"), latent)

    assert msg.sender == "growth_stage"
    for pid in range(2):
        np.testing.assert_allclose(msg.per_plant_log_likelihoods[pid], 0.0)


def test_agent_provides_growth_metadata(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    agent = GrowthStageAgent()
    latent = _make_latent(2)
    msg = agent.emit_constraint(str(tmp_path / "test.jpg"), latent)

    per_plant = msg.metadata["per_plant_growth"]
    assert 0 in per_plant
    assert "growth_stage" in per_plant[0]
    assert "urgency_multiplier" in per_plant[0]
    assert per_plant[0]["growth_stage"] in GROWTH_STAGES


def test_agent_stores_growth_stages(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    agent = GrowthStageAgent()
    latent = _make_latent(2)
    agent.emit_constraint(str(tmp_path / "test.jpg"), latent)

    assert 0 in agent.growth_stages
    assert 1 in agent.growth_stages
    assert set(agent.growth_stages[0].keys()) == set(GROWTH_STAGES)


def test_agent_handles_no_image():
    agent = GrowthStageAgent()
    latent = _make_latent(2)
    msg = agent.emit_constraint(None, latent)

    assert len(msg.per_plant_log_likelihoods) == 2
    per_plant = msg.metadata["per_plant_growth"]
    assert per_plant[0]["growth_stage"] == "vegetative"  # default


def test_urgency_multipliers_defined():
    for stage in GROWTH_STAGES:
        assert stage in URGENCY_MULTIPLIER
        assert URGENCY_MULTIPLIER[stage] > 0


def test_labels_discriminated_is_empty(tmp_path):
    """Growth stage should not discriminate any condition labels."""
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    agent = GrowthStageAgent()
    latent = _make_latent(1)
    msg = agent.emit_constraint(str(tmp_path / "test.jpg"), latent)
    assert msg.labels_discriminated == []
