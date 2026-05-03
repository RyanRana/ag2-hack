"""VLMReasonerAgent tests — verify the typed tool contract.

We do NOT exercise the LLM loop here (no API key). We test that the tools
are registered with correct signatures, that ``analyze_disagreement_region``
returns numerical-only payloads, and that ``submit_per_plant_likelihoods``
correctly populates the agent's output buffers.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pulse.agents.vlm_reasoner import VLMReasonerAgent, _default_image_analyzer
from pulse.llm_config import LLMKeyMissingError


FAKE_LLM_CONFIG = {
    "config_list": [{"model": "stub", "api_key": "sk-fake-for-tests"}],
    "temperature": 0.0,
    "cache_seed": None,
}


def test_construction_requires_llm_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMKeyMissingError):
        VLMReasonerAgent()


def test_default_image_analyzer_returns_only_numerics():
    crop = np.zeros((20, 20, 3), dtype=np.uint8) + 100
    feats = _default_image_analyzer(crop, (0, 0, 20, 20))
    for v in feats.values():
        assert isinstance(v, (int, float, np.floating, np.integer))


def test_default_analyzer_detects_yellowness():
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    crop[..., 0] = 200  # red channel high
    crop[..., 1] = 200  # green channel high
    crop[..., 2] = 50   # blue channel low → yellow
    feats = _default_image_analyzer(crop, (0, 0, 20, 20))
    assert feats["yellowness"] > 0.5


def test_analyze_disagreement_region_populates_features(tmp_path):
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (100, 100), color=(60, 130, 40)).save(img_path)

    # Inject a stub analyzer so we don't need the LLM path.
    captured = {}

    def stub_analyzer(crop, bbox):
        captured["bbox"] = bbox
        return {"green_ratio": 0.9, "yellowness": 0.0, "edge_density": 0.0,
                "brightness": 0.5, "size_px": 4900.0}

    agent = VLMReasonerAgent(llm_config=FAKE_LLM_CONFIG, analyzer=stub_analyzer)
    agent._image = Image.open(img_path).convert("RGB")
    out = agent.analyze_disagreement_region(plant_id=3, bbox_x1=10, bbox_y1=10,
                                            bbox_x2=80, bbox_y2=80)
    assert captured["bbox"] == (10, 10, 80, 80)
    assert out["green_ratio"] == pytest.approx(0.9)
    assert agent._features[3]["green_ratio"] == pytest.approx(0.9)


def test_submit_per_plant_likelihoods_records_vector():
    agent = VLMReasonerAgent(llm_config=FAKE_LLM_CONFIG, analyzer=_default_image_analyzer)
    agent.submit_per_plant_likelihoods(
        plant_id=2,
        log_lik_healthy_crop=0.0,
        log_lik_weed=0.0,
        log_lik_disease=2.0,
        log_lik_nutrient_stress=-0.5,
        log_lik_water_stress=-1.0,
        log_lik_pest_damage=0.0,
        log_lik_ambiguous=-0.2,
    )
    ll = agent._likelihoods[2]
    assert len(ll) == 7
    assert ll[2] == 2.0
    assert all(isinstance(v, float) for v in ll)


def test_tool_signatures_use_typed_annotations():
    """Sanity-check that AG2's typed tool registration is in place."""
    from typing import get_type_hints

    agent = VLMReasonerAgent(llm_config=FAKE_LLM_CONFIG, analyzer=_default_image_analyzer)
    hints = get_type_hints(agent.analyze_disagreement_region, include_extras=True)
    # Each annotated parameter should expose its description string.
    assert "plant_id" in hints
    assert "bbox_x1" in hints
