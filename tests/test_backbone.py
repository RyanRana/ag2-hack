"""Tests for DINOv2 backbone — uses mock model to avoid downloading weights."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pulse.backbone import DINOv2Backbone


def _make_mock_outputs(n_patches: int = 1369, dim: int = 768, n_heads: int = 12):
    """Create mock DINOv2 model outputs."""
    import torch

    seq_len = 1 + n_patches  # CLS + patches
    last_hidden = torch.randn(1, seq_len, dim)
    attn = torch.softmax(torch.randn(1, n_heads, seq_len, seq_len), dim=-1)

    outputs = MagicMock()
    outputs.last_hidden_state = last_hidden
    outputs.attentions = (attn,)  # tuple of layer attentions
    return outputs


def _make_mock_backbone():
    """Create a DINOv2Backbone with mocked model and processor."""
    import torch

    model = MagicMock()
    model.return_value = _make_mock_outputs()

    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.randn(1, 3, 518, 518)}

    backbone = DINOv2Backbone(model=model, processor=processor)
    return backbone


def test_extract_features_returns_correct_keys():
    backbone = _make_mock_backbone()
    img = Image.new("RGB", (224, 224), color=(60, 130, 40))

    feats = backbone.extract_features(img)

    assert "cls_token" in feats
    assert "patch_tokens" in feats
    assert "attention_map" in feats
    assert "patch_grid" in feats


def test_cls_token_shape():
    backbone = _make_mock_backbone()
    img = Image.new("RGB", (224, 224))
    feats = backbone.extract_features(img)
    assert feats["cls_token"].shape == (768,)


def test_patch_tokens_shape():
    backbone = _make_mock_backbone()
    img = Image.new("RGB", (224, 224))
    feats = backbone.extract_features(img)
    assert feats["patch_tokens"].shape == (1369, 768)


def test_attention_map_is_2d():
    backbone = _make_mock_backbone()
    img = Image.new("RGB", (224, 224))
    feats = backbone.extract_features(img)
    assert feats["attention_map"].ndim == 2


def test_extract_features_accepts_numpy():
    backbone = _make_mock_backbone()
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    feats = backbone.extract_features(img)
    assert "cls_token" in feats


def test_extract_per_plant(tmp_path):
    backbone = _make_mock_backbone()
    img = Image.new("RGB", (400, 400), color=(60, 130, 40))
    img.save(tmp_path / "field.jpg")

    results = backbone.extract_per_plant(
        str(tmp_path / "field.jpg"),
        bboxes=[(10, 10, 190, 190), (200, 200, 390, 390)],
        plant_ids=[0, 1],
    )

    assert 0 in results
    assert 1 in results
    for pid in [0, 1]:
        assert "cls_token" in results[pid]
        assert "patch_tokens" in results[pid]


def test_extract_per_plant_skips_tiny_crops(tmp_path):
    backbone = _make_mock_backbone()
    img = Image.new("RGB", (100, 100))
    img.save(tmp_path / "tiny.jpg")

    results = backbone.extract_per_plant(
        str(tmp_path / "tiny.jpg"),
        bboxes=[(0, 0, 5, 5)],  # 5x5 crop — too small
        plant_ids=[0],
    )

    assert 0 not in results
