"""DiseaseClassifierAgent tests with injected fake transformers model."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pulse.agents.disease_classifier import (
    DiseaseClassifierAgent,
    class_to_condition,
    map_probs_to_conditions,
)
from pulse.latent import CONDITION_LABELS, FieldLatentState, PlantInstance
from pulse.messages import ConstraintMessage


# --- Class → condition mapping ---------------------------------------------


def test_class_to_condition_healthy():
    assert class_to_condition("Tomato___healthy") == "healthy_crop"
    assert class_to_condition("Apple___healthy") == "healthy_crop"


def test_class_to_condition_disease():
    assert class_to_condition("Tomato___Early_blight") == "disease"
    assert class_to_condition("Apple___Cedar_apple_rust") == "disease"
    assert class_to_condition("Grape___Black_rot") == "disease"


def test_class_to_condition_pest():
    assert class_to_condition("Tomato___Spider_mites") == "pest_damage"
    assert class_to_condition("Cotton___aphid_damage") == "pest_damage"


def test_class_to_condition_nutrient():
    assert class_to_condition("Corn___Nitrogen_deficiency") == "nutrient_stress"


# --- Probability mapping ---------------------------------------------------


def test_map_probs_concentrates_on_disease():
    id2label = {0: "Tomato___healthy", 1: "Tomato___Late_blight", 2: "Apple___Black_rot"}
    probs = np.array([0.05, 0.85, 0.10])
    log_lik = map_probs_to_conditions(probs, id2label)
    disease_idx = CONDITION_LABELS.index("disease")
    healthy_idx = CONDITION_LABELS.index("healthy_crop")
    assert log_lik[disease_idx] > log_lik[healthy_idx]


def test_map_probs_concentrates_on_healthy():
    id2label = {0: "Tomato___healthy", 1: "Apple___healthy", 2: "Grape___Black_rot"}
    probs = np.array([0.6, 0.3, 0.1])
    log_lik = map_probs_to_conditions(probs, id2label)
    disease_idx = CONDITION_LABELS.index("disease")
    healthy_idx = CONDITION_LABELS.index("healthy_crop")
    assert log_lik[healthy_idx] > log_lik[disease_idx]


# --- Agent end-to-end with fake model -------------------------------------


class _FakeLogits:
    def __init__(self, logits: np.ndarray) -> None:
        import torch

        self.logits = torch.tensor(logits[None, :])  # batch dim


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


def _make_latent(boxes):
    f = FieldLatentState(image_shape=(480, 640))
    for i, b in enumerate(boxes):
        f.plants.append(PlantInstance(plant_id=i, bbox=b))
    return f


def test_agent_emits_constraint_with_fake_model(tmp_path):
    Image.new("RGB", (640, 480), color=(120, 80, 40)).save(tmp_path / "f.jpg")
    id2label = {0: "Tomato___healthy", 1: "Tomato___Late_blight", 2: "Apple___Black_rot"}
    # Logits favor late_blight (index 1)
    logits = np.array([0.1, 4.0, 0.5])
    agent = DiseaseClassifierAgent(model=_FakeModel(logits, id2label), processor=_FakeProcessor())
    latent = _make_latent([(10, 10, 100, 100)])
    msg = agent.emit_constraint(str(tmp_path / "f.jpg"), latent)
    assert isinstance(msg, ConstraintMessage)
    assert msg.sender == "disease_classifier"
    log_lik = msg.per_plant_log_likelihoods[0]
    disease_idx = CONDITION_LABELS.index("disease")
    healthy_idx = CONDITION_LABELS.index("healthy_crop")
    assert log_lik[disease_idx] > log_lik[healthy_idx]
