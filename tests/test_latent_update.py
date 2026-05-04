"""Latent state mechanics — posterior aggregation, normalization, disagreement."""

import numpy as np
import pytest

from pesto.latent import CONDITION_LABELS, FieldLatentState, PlantInstance


def _make_field(n_plants: int = 3) -> FieldLatentState:
    field = FieldLatentState(image_shape=(480, 640))
    for i in range(n_plants):
        field.plants.append(
            PlantInstance(plant_id=i, bbox=(10 * i, 10 * i, 10 * i + 50, 10 * i + 50))
        )
    return field


def test_uniform_prior():
    plant = PlantInstance(plant_id=0, bbox=(0, 0, 10, 10))
    p = plant.posterior()
    assert p.shape == (len(CONDITION_LABELS),)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(p, np.ones(len(CONDITION_LABELS)) / len(CONDITION_LABELS))


def test_update_plant_aggregates_log_likelihoods():
    field = _make_field(2)
    delta = np.zeros(len(CONDITION_LABELS))
    weed_idx = CONDITION_LABELS.index("weed")
    delta[weed_idx] = 5.0  # strong evidence for weed

    field.update_plant(plant_id=1, log_likelihood_delta=delta, source_agent="weed_detector")

    plant = field.plants[1]
    assert plant.constraint_history == ["weed_detector"]
    p = plant.posterior()
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-9)
    assert np.argmax(p) == weed_idx
    assert p[weed_idx] > 0.9

    # Other plant untouched.
    p0 = field.plants[0].posterior()
    np.testing.assert_allclose(p0, np.ones(len(CONDITION_LABELS)) / len(CONDITION_LABELS))


def test_update_plant_compounds_evidence():
    field = _make_field(1)
    disease_idx = CONDITION_LABELS.index("disease")
    d1 = np.zeros(len(CONDITION_LABELS))
    d1[disease_idx] = 2.0
    d2 = np.zeros(len(CONDITION_LABELS))
    d2[disease_idx] = 2.0

    field.update_plant(0, d1, "disease_classifier")
    field.update_plant(0, d2, "vlm_reasoner")

    plant = field.plants[0]
    assert plant.constraint_history == ["disease_classifier", "vlm_reasoner"]
    # Two independent +2.0 likelihoods compound to +4.0 — same as one +4 update.
    expected = np.zeros(len(CONDITION_LABELS))
    expected[disease_idx] = 4.0
    expected_post = np.exp(expected - np.logaddexp.reduce(expected - np.log(len(CONDITION_LABELS))))
    p = plant.posterior()
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-9)
    assert np.argmax(p) == disease_idx


def test_update_plant_unknown_id_raises():
    field = _make_field(2)
    delta = np.zeros(len(CONDITION_LABELS))
    with pytest.raises(StopIteration):
        field.update_plant(plant_id=99, log_likelihood_delta=delta, source_agent="x")


def test_top_k_orders_by_probability():
    plant = PlantInstance(plant_id=0, bbox=(0, 0, 10, 10))
    plant.log_posterior = np.array([0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    top = plant.top_k(2)
    # Only the top-2 are unambiguously ordered; remaining indices are tied at 0.0.
    assert [t[0] for t in top] == ["weed", "nutrient_stress"]
    assert top[0][1] > top[1][1]
    assert sum(t[1] for t in top) <= 1.0 + 1e-9


def test_entropy_bounds():
    plant = PlantInstance(plant_id=0, bbox=(0, 0, 10, 10))
    h_uniform = plant.entropy()
    np.testing.assert_allclose(h_uniform, np.log(len(CONDITION_LABELS)), atol=1e-9)

    # Concentrate mass on one label → entropy ≈ 0.
    plant.log_posterior = np.array([0.0] * len(CONDITION_LABELS))
    plant.log_posterior[0] = 50.0
    assert plant.entropy() < 1e-3


def test_disagreement_score_zero_when_aligned():
    field = _make_field(1)
    a = np.zeros(len(CONDITION_LABELS))
    a[CONDITION_LABELS.index("weed")] = 5.0
    b = a.copy()
    score = field.disagreement_score(0, {"agent_a": a, "agent_b": b})
    assert score < 1e-6


def test_disagreement_score_positive_when_opposed():
    field = _make_field(1)
    a = np.zeros(len(CONDITION_LABELS))
    a[CONDITION_LABELS.index("disease")] = 5.0
    b = np.zeros(len(CONDITION_LABELS))
    b[CONDITION_LABELS.index("water_stress")] = 5.0
    score = field.disagreement_score(0, {"a": a, "b": b})
    assert score > 1.0


def test_disagreement_score_handles_singleton():
    field = _make_field(1)
    a = np.zeros(len(CONDITION_LABELS))
    assert field.disagreement_score(0, {"only": a}) == 0.0
    assert field.disagreement_score(0, {}) == 0.0


def test_from_dict_roundtrip():
    field = _make_field(2)
    delta = np.zeros(len(CONDITION_LABELS))
    delta[CONDITION_LABELS.index("disease")] = 4.0
    field.update_plant(0, delta, "disease_classifier")
    field.iteration = 3

    rehydrated = FieldLatentState.from_dict(field.to_dict())
    assert rehydrated.iteration == 3
    assert rehydrated.image_shape == field.image_shape
    assert len(rehydrated.plants) == 2
    np.testing.assert_allclose(
        rehydrated.plants[0].log_posterior, field.plants[0].log_posterior
    )
    assert rehydrated.plants[0].constraint_history == ["disease_classifier"]
    assert rehydrated.plants[0].bbox == field.plants[0].bbox


def test_to_dict_roundtrip_shape():
    field = _make_field(2)
    delta = np.zeros(len(CONDITION_LABELS))
    delta[CONDITION_LABELS.index("weed")] = 3.0
    field.update_plant(0, delta, "weed_detector")
    field.iteration = 7

    d = field.to_dict()
    assert d["iteration"] == 7
    assert d["image_shape"] == [480, 640]
    assert len(d["plants"]) == 2
    p0 = d["plants"][0]
    assert p0["plant_id"] == 0
    assert len(p0["log_posterior"]) == len(CONDITION_LABELS)
    assert p0["constraint_history"] == ["weed_detector"]
    assert p0["top_k"][0][0] == "weed"
    assert p0["entropy"] >= 0.0
