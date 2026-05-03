"""Tests for active learning loop."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pulse.active_learning import ActiveLearningManager
from pulse.latent import CONDITION_LABELS, FieldLatentState, PlantInstance
from pulse.messages import ActionMessage


def _make_latent(n_plants=3):
    f = FieldLatentState(image_shape=(200, 200))
    for i in range(n_plants):
        p = PlantInstance(plant_id=i, bbox=(i * 60, 0, i * 60 + 50, 50))
        # Give plant 0 a high-entropy posterior (ambiguous)
        if i == 0:
            p.log_posterior = np.zeros(len(CONDITION_LABELS))
        # Plant 1 is clearly diseased
        elif i == 1:
            p.log_posterior = np.full(len(CONDITION_LABELS), -3.0)
            p.log_posterior[CONDITION_LABELS.index("disease")] = 2.0
        # Plant 2 is healthy
        else:
            p.log_posterior = np.full(len(CONDITION_LABELS), -3.0)
            p.log_posterior[CONDITION_LABELS.index("healthy_crop")] = 2.0
        f.plants.append(p)
    return f


def _make_actions(latent):
    actions = []
    for p in latent.plants:
        a_type = "no_action"
        if p.plant_id == 0:
            a_type = "human_review"
        elif p.plant_id == 1:
            a_type = "targeted_fungicide"
        actions.append(ActionMessage(
            sender="controller", timestamp=0.0, plant_id=p.plant_id,
            action_type=a_type, action_params={},
            expected_information_gain=0.5, expected_utility=0.3,
        ))
    return actions


def test_queues_human_review_plant(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    mgr = ActiveLearningManager(queue_dir=tmp_path / "al")
    latent = _make_latent()
    actions = _make_actions(latent)

    entries = mgr.process_inference_results(
        str(tmp_path / "test.jpg"), latent, actions,
        cross_exam_kl={0: 1.0, 1: 1.0, 2: 0.5},
    )

    # Plant 0 should be queued (human_review triggered)
    assert any(e.plant_id == 0 for e in entries)
    assert any(e.trigger_reason == "human_review" for e in entries)


def test_queues_high_disagreement(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    mgr = ActiveLearningManager(queue_dir=tmp_path / "al")
    latent = _make_latent()
    actions = _make_actions(latent)

    entries = mgr.process_inference_results(
        str(tmp_path / "test.jpg"), latent, actions,
        cross_exam_kl={0: 1.0, 1: 5.0, 2: 0.5},  # plant 1 has high KL
    )

    assert any(e.plant_id == 1 and e.trigger_reason == "high_disagreement" for e in entries)


def test_queues_anomaly(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    mgr = ActiveLearningManager(queue_dir=tmp_path / "al")
    latent = _make_latent()
    actions = _make_actions(latent)

    # Plant 2 has a very high anomaly score (>3σ from mean of [0.1, 0.1, 50.0])
    # mean ≈ 16.7, std ≈ 23.5 → (50 - 16.7)/23.5 ≈ 1.4 still not enough
    # Use lower sigma threshold for this test
    mgr._anomaly_sigma_threshold = 1.0
    entries = mgr.process_inference_results(
        str(tmp_path / "test.jpg"), latent, actions,
        cross_exam_kl={0: 1.0, 1: 1.0, 2: 0.5},
        anomaly_scores={0: 0.1, 1: 0.15, 2: 5.0},
    )

    assert any(e.plant_id == 2 and e.trigger_reason == "anomaly" for e in entries)


def test_no_queue_for_easy_cases(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    mgr = ActiveLearningManager(queue_dir=tmp_path / "al")
    latent = _make_latent(1)
    actions = [ActionMessage(
        sender="controller", timestamp=0.0, plant_id=0,
        action_type="no_action", action_params={},
        expected_information_gain=0.1, expected_utility=0.8,
    )]

    entries = mgr.process_inference_results(
        str(tmp_path / "test.jpg"), latent, actions,
        cross_exam_kl={0: 0.5},
    )

    assert len(entries) == 0


def test_label_entry(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    mgr = ActiveLearningManager(queue_dir=tmp_path / "al")
    latent = _make_latent()
    actions = _make_actions(latent)

    entries = mgr.process_inference_results(
        str(tmp_path / "test.jpg"), latent, actions,
        cross_exam_kl={0: 1.0, 1: 1.0, 2: 0.5},
    )

    assert len(entries) > 0
    eid = entries[0].entry_id
    assert mgr.label_entry(eid, "disease")
    assert mgr.get_labeled_entries()[0].human_label == "disease"


def test_label_invalid_label(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    mgr = ActiveLearningManager(queue_dir=tmp_path / "al")
    latent = _make_latent()
    actions = _make_actions(latent)

    entries = mgr.process_inference_results(
        str(tmp_path / "test.jpg"), latent, actions,
        cross_exam_kl={0: 1.0, 1: 1.0, 2: 0.5},
    )

    import pytest
    with pytest.raises(ValueError):
        mgr.label_entry(entries[0].entry_id, "not_a_real_label")


def test_export_load_roundtrip(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    mgr = ActiveLearningManager(queue_dir=tmp_path / "al")
    latent = _make_latent()
    actions = _make_actions(latent)
    mgr.process_inference_results(
        str(tmp_path / "test.jpg"), latent, actions,
        cross_exam_kl={0: 1.0, 1: 1.0, 2: 0.5},
    )
    mgr.export_queue(tmp_path / "al")

    loaded = ActiveLearningManager.load_queue(tmp_path / "al")
    assert loaded.queue_size == mgr.queue_size


def test_get_training_data(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    mgr = ActiveLearningManager(queue_dir=tmp_path / "al")
    latent = _make_latent()
    actions = _make_actions(latent)
    entries = mgr.process_inference_results(
        str(tmp_path / "test.jpg"), latent, actions,
        cross_exam_kl={0: 1.0, 1: 1.0, 2: 0.5},
    )

    # Label one
    mgr.label_entry(entries[0].entry_id, "disease")
    data = mgr.get_training_data()
    assert len(data) == 1
    assert data[0]["label"] == "disease"
    assert data[0]["label_index"] == CONDITION_LABELS.index("disease")


def test_summary(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    mgr = ActiveLearningManager(queue_dir=tmp_path / "al")
    latent = _make_latent()
    actions = _make_actions(latent)
    mgr.process_inference_results(
        str(tmp_path / "test.jpg"), latent, actions,
        cross_exam_kl={0: 1.0, 1: 1.0, 2: 0.5},
    )

    s = mgr.summary()
    assert s["total_queued"] > 0
    assert "trigger_breakdown" in s


def test_crop_saved(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    mgr = ActiveLearningManager(queue_dir=tmp_path / "al")
    latent = _make_latent()
    actions = _make_actions(latent)
    entries = mgr.process_inference_results(
        str(tmp_path / "test.jpg"), latent, actions,
        cross_exam_kl={0: 1.0, 1: 1.0, 2: 0.5},
    )

    for e in entries:
        if e.crop_saved_path:
            from pathlib import Path
            assert Path(e.crop_saved_path).exists()
