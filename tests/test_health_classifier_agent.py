"""HealthClassifierAgent tests with injected fake ViT model."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pulse.agents.health_classifier import HealthClassifierAgent, _split_health_mass
from pulse.latent import CONDITION_LABELS, FieldLatentState, PlantInstance


def test_split_health_mass_healthy_dominant():
    id2label = {0: "Tomato_healthy", 1: "Tomato_diseased"}
    probs = np.array([0.85, 0.15])
    p_h, p_u = _split_health_mass(probs, id2label)
    assert p_h > p_u
    assert abs(p_h + p_u - 1.0) < 1e-9


def test_split_health_mass_unhealthy_dominant():
    id2label = {0: "Tomato_healthy", 1: "Tomato_blight", 2: "Apple_rust"}
    probs = np.array([0.1, 0.5, 0.4])
    p_h, p_u = _split_health_mass(probs, id2label)
    assert p_u > p_h
    assert p_u > 0.85


# --- Agent end-to-end with fake model -------------------------------------


class _FakeLogits:
    def __init__(self, logits: np.ndarray) -> None:
        import torch

        self.logits = torch.tensor(logits[None, :])


class _FakeConfig:
    def __init__(self, id2label: dict[int, str]) -> None:
        self.id2label = id2label


class _FakeModel:
    def __init__(self, logits: np.ndarray, id2label: dict[int, str]) -> None:
        self.config = _FakeConfig(id2label)
        self._logits = logits

    def eval(self) -> "_FakeModel":
        return self

    def __call__(self, **kwargs):
        return _FakeLogits(self._logits)


class _FakeProcessor:
    def __call__(self, *, images, return_tensors):
        return {"pixel_values": None}


def test_agent_concentrates_unhealthy_on_unhealthy_axis(tmp_path):
    Image.new("RGB", (200, 200), color=(120, 80, 40)).save(tmp_path / "f.jpg")
    id2label = {0: "Plant_healthy", 1: "Plant_diseased"}
    logits = np.array([-2.0, 3.0])  # strongly unhealthy
    agent = HealthClassifierAgent(model=_FakeModel(logits, id2label), processor=_FakeProcessor())
    latent = FieldLatentState(image_shape=(200, 200))
    latent.plants.append(PlantInstance(plant_id=0, bbox=(10, 10, 190, 190)))
    msg = agent.emit_constraint(str(tmp_path / "f.jpg"), latent)
    log_lik = msg.per_plant_log_likelihoods[0]
    healthy_idx = CONDITION_LABELS.index("healthy_crop")
    disease_idx = CONDITION_LABELS.index("disease")
    pest_idx = CONDITION_LABELS.index("pest_damage")
    # Each unhealthy axis should be elevated; healthy_crop suppressed.
    assert log_lik[disease_idx] > 0
    assert log_lik[pest_idx] > 0
    assert log_lik[healthy_idx] < 0


def test_agent_concentrates_healthy_on_healthy_axis(tmp_path):
    Image.new("RGB", (200, 200), color=(80, 160, 50)).save(tmp_path / "f.jpg")
    id2label = {0: "Plant_healthy", 1: "Plant_diseased"}
    logits = np.array([3.0, -2.0])  # strongly healthy
    agent = HealthClassifierAgent(model=_FakeModel(logits, id2label), processor=_FakeProcessor())
    latent = FieldLatentState(image_shape=(200, 200))
    latent.plants.append(PlantInstance(plant_id=0, bbox=(10, 10, 190, 190)))
    msg = agent.emit_constraint(str(tmp_path / "f.jpg"), latent)
    log_lik = msg.per_plant_log_likelihoods[0]
    healthy_idx = CONDITION_LABELS.index("healthy_crop")
    disease_idx = CONDITION_LABELS.index("disease")
    assert log_lik[healthy_idx] > 0
    assert log_lik[disease_idx] < 0
