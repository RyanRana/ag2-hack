"""Tests for SkepticAgent local model path — uses mock backend."""

from __future__ import annotations

import json

import numpy as np

from pesto.agents.skeptic import (
    MAX_DEBATE_TURNS,
    SkepticAgent,
)
from pesto.latent import CONDITION_LABELS
from pesto.local_model import LocalModelBackend
from pesto.messages import CrossExamMessage


def _make_cross_exam_msgs(disputed_ids: list[int]) -> list[CrossExamMessage]:
    """Create synthetic cross-exam messages for testing."""
    msgs = []
    for pid in disputed_ids:
        diag = {label: float(i * 0.1) for i, label in enumerate(CONDITION_LABELS)}
        msgs.append(CrossExamMessage(
            evaluator="disease_classifier",
            target="health_classifier",
            per_plant_disagreement={pid: 2.5},
            per_plant_diagnostic={pid: diag},
        ))
    return msgs


class MockLocalBackend(LocalModelBackend):
    """Backend that returns canned JSON responses."""

    def __init__(self, response_json: str = "[]"):
        super().__init__(model="mock", processor="mock", model_id="mock-model")
        self._response = response_json
        self._loaded = True

    def generate_text_only(self, prompt, *, max_new_tokens=512, temperature=0.1):
        return self._response


def test_skeptic_local_emits_hypotheses():
    response = json.dumps([{
        "plant_id": 0,
        "hypothesis_id": "disease_masquerading_as_nutrient_stress",
        "log_posterior": -0.5,
        "evidence_axes": {
            "healthy_crop": -0.2, "weed": 0.0, "disease": 0.6,
            "nutrient_stress": 0.3, "water_stress": 0.0,
            "pest_damage": 0.0, "ambiguous": 0.1,
        },
    }])
    backend = MockLocalBackend(response)
    agent = SkepticAgent(local_backend=backend)
    msgs = _make_cross_exam_msgs([0])

    hypotheses = agent.emit_hypotheses_for(msgs, [0])

    assert len(hypotheses) == 1
    assert hypotheses[0].plant_id == 0
    assert hypotheses[0].hypothesis_id == "disease_masquerading_as_nutrient_stress"


def test_skeptic_local_handles_invalid_json():
    backend = MockLocalBackend("This is not JSON at all")
    agent = SkepticAgent(local_backend=backend)
    msgs = _make_cross_exam_msgs([0])

    hypotheses = agent.emit_hypotheses_for(msgs, [0])
    assert hypotheses == []


def test_skeptic_local_handles_empty_response():
    backend = MockLocalBackend("[]")
    agent = SkepticAgent(local_backend=backend)
    msgs = _make_cross_exam_msgs([0])

    hypotheses = agent.emit_hypotheses_for(msgs, [0])
    assert hypotheses == []


def test_skeptic_local_multiple_plants():
    response = json.dumps([
        {"plant_id": 0, "hypothesis_id": "h0", "log_posterior": -0.3,
         "evidence_axes": {l: 0.1 for l in CONDITION_LABELS}},
        {"plant_id": 1, "hypothesis_id": "h1", "log_posterior": -0.7,
         "evidence_axes": {l: 0.2 for l in CONDITION_LABELS}},
    ])
    backend = MockLocalBackend(response)
    agent = SkepticAgent(local_backend=backend)
    msgs = _make_cross_exam_msgs([0, 1])

    hypotheses = agent.emit_hypotheses_for(msgs, [0, 1])
    assert len(hypotheses) == 2
    assert {h.plant_id for h in hypotheses} == {0, 1}


# --- Multi-turn debate tests ---

def test_should_continue_debate_converged():
    agent = SkepticAgent(local_backend=MockLocalBackend())
    # Low entropy posterior — should NOT continue
    converged = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01])
    assert not agent.should_continue_debate({0: converged}, turn=1)


def test_should_continue_debate_high_entropy():
    agent = SkepticAgent(local_backend=MockLocalBackend())
    # High entropy posterior — should continue
    uniform = np.ones(len(CONDITION_LABELS)) / len(CONDITION_LABELS)
    assert agent.should_continue_debate({0: uniform}, turn=1)


def test_should_continue_debate_max_turns():
    agent = SkepticAgent(local_backend=MockLocalBackend())
    uniform = np.ones(len(CONDITION_LABELS)) / len(CONDITION_LABELS)
    assert not agent.should_continue_debate({0: uniform}, turn=MAX_DEBATE_TURNS)
