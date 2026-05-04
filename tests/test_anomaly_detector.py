"""Tests for AnomalyDetectorAgent — uses synthetic embeddings, no real DINOv2."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pesto.agents.anomaly_detector import (
    AnomalyDetectorAgent,
    PatchCoreMemoryBank,
)
from pesto.latent import CONDITION_LABELS, FieldLatentState, PlantInstance


def _make_latent(n_plants: int = 2, shape=(200, 200)):
    f = FieldLatentState(image_shape=shape)
    for i in range(n_plants):
        f.plants.append(PlantInstance(
            plant_id=i,
            bbox=(i * 80, 0, i * 80 + 70, 70),
        ))
    return f


class FakeBackbone:
    """Returns synthetic DINOv2-like features.

    Normal plants return embeddings drawn from the SAME distribution as
    the healthy memory bank (seed 0, std 0.1). Anomalous plants return
    embeddings far away from that distribution.
    """

    def __init__(self, anomalous_plants: set[int] | None = None):
        self._anomalous = anomalous_plants or set()
        self._call_count = 0

    def extract_features(self, crop):
        plant_idx = self._call_count
        self._call_count += 1
        n_patches, dim = 100, 768
        rng = np.random.RandomState(plant_idx + 10)

        if plant_idx in self._anomalous:
            # Anomalous: fixed direction far from healthy distribution
            patch_tokens = np.ones((n_patches, dim), dtype=np.float32) * 50.0
        else:
            # Normal: draw from the exact same distribution as the bank
            # (seed 0 with std 0.1 — bank patches are from this distribution)
            rng_healthy = np.random.RandomState(0)
            patch_tokens = rng_healthy.randn(n_patches, dim).astype(np.float32) * 0.1

        return {
            "cls_token": rng.randn(dim).astype(np.float32),
            "patch_tokens": patch_tokens,
            "attention_map": rng.rand(10, 10).astype(np.float32),
            "patch_grid": (10, 10),
        }


def _make_healthy_memory_bank(n_crops: int = 20, dim: int = 768):
    """Build a memory bank from synthetic 'healthy' embeddings."""
    rng = np.random.RandomState(0)
    patches_list = []
    for _ in range(n_crops):
        patches = rng.randn(100, dim).astype(np.float32) * 0.1
        patches_list.append(patches)
    bank = PatchCoreMemoryBank()
    bank.fit(patches_list)
    return bank


# --- PatchCoreMemoryBank tests ---

def test_memory_bank_fit_and_score():
    bank = _make_healthy_memory_bank()
    assert bank.is_fitted

    rng = np.random.RandomState(99)
    # Score a healthy sample — close to bank distribution, should be low
    healthy = rng.randn(100, 768).astype(np.float32) * 0.1
    score_healthy = bank.score(healthy)

    # Score an anomalous sample — constant offset far from origin, should be high
    # Use a fixed direction to ensure cosine distance is large
    anomalous = np.ones((100, 768), dtype=np.float32) * 3.0
    anomalous[:, :384] = -3.0  # opposite direction from typical healthy
    score_anomalous = bank.score(anomalous)

    assert score_anomalous > score_healthy


def test_memory_bank_unfitted_returns_zero():
    bank = PatchCoreMemoryBank()
    assert not bank.is_fitted
    rng = np.random.RandomState(0)
    score = bank.score(rng.randn(50, 768).astype(np.float32))
    assert score == 0.0


def test_memory_bank_save_load_roundtrip(tmp_path):
    bank = _make_healthy_memory_bank()
    bank.save(tmp_path)

    loaded = PatchCoreMemoryBank.load(tmp_path)
    assert loaded.is_fitted

    # Scores should match
    test = np.random.randn(50, 768).astype(np.float32)
    assert abs(bank.score(test) - loaded.score(test)) < 1e-6


def test_memory_bank_coreset_limits_size():
    """Memory bank should subsample to at most 10k patches."""
    rng = np.random.RandomState(0)
    big_list = [rng.randn(1000, 768).astype(np.float32) for _ in range(15)]
    bank = PatchCoreMemoryBank()
    bank.fit(big_list)
    assert bank._bank.shape[0] <= 10_000


# --- AnomalyDetectorAgent tests ---

def test_agent_flags_anomalous_plant(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    bank = _make_healthy_memory_bank()
    backbone = FakeBackbone(anomalous_plants={1})  # plant 1 is anomalous
    agent = AnomalyDetectorAgent(
        backbone=backbone,
        memory_bank=bank,
        anomaly_threshold=0.2,
    )
    latent = _make_latent(2)
    msg = agent.emit_constraint(str(tmp_path / "test.jpg"), latent)

    assert msg.sender == "anomaly_detector"
    assert len(msg.per_plant_log_likelihoods) == 2

    # Plant 1 (anomalous) should have higher ambiguous log-likelihood
    ll_0 = msg.per_plant_log_likelihoods[0]
    ll_1 = msg.per_plant_log_likelihoods[1]
    i_ambig = CONDITION_LABELS.index("ambiguous")
    assert ll_1[i_ambig] > ll_0[i_ambig]


def test_agent_stores_anomaly_scores(tmp_path):
    img = Image.new("RGB", (200, 200), color=(60, 130, 40))
    img.save(tmp_path / "test.jpg")

    bank = _make_healthy_memory_bank()
    backbone = FakeBackbone()
    agent = AnomalyDetectorAgent(backbone=backbone, memory_bank=bank)
    latent = _make_latent(2)
    agent.emit_constraint(str(tmp_path / "test.jpg"), latent)

    assert 0 in agent.anomaly_scores
    assert 1 in agent.anomaly_scores


def test_agent_handles_unfitted_bank(tmp_path):
    img = Image.new("RGB", (200, 200))
    img.save(tmp_path / "test.jpg")

    agent = AnomalyDetectorAgent(
        backbone=FakeBackbone(),
        memory_bank=PatchCoreMemoryBank(),
    )
    latent = _make_latent(2)
    msg = agent.emit_constraint(str(tmp_path / "test.jpg"), latent)

    # Should emit neutral constraints
    for pid in range(2):
        np.testing.assert_allclose(msg.per_plant_log_likelihoods[pid], 0.0)


def test_agent_handles_no_image():
    agent = AnomalyDetectorAgent(
        backbone=FakeBackbone(),
        memory_bank=_make_healthy_memory_bank(),
    )
    latent = _make_latent(2)
    msg = agent.emit_constraint(None, latent)

    assert len(msg.per_plant_log_likelihoods) == 2
    for pid in range(2):
        np.testing.assert_allclose(msg.per_plant_log_likelihoods[pid], 0.0)


def test_protocol_firewall_compliance():
    import dataclasses
    from pesto.messages import ConstraintMessage

    forbidden = {"text", "message", "prose", "explanation", "commentary",
                 "description", "narrative", "reasoning"}
    field_names = {f.name for f in dataclasses.fields(ConstraintMessage)}
    assert not (field_names & forbidden)
