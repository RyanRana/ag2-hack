"""Cross-examination + skeptic tests."""

from __future__ import annotations

import time

import numpy as np
import pytest

from pulse.agents.skeptic import SkepticAgent
from pulse.cross_exam import (
    cross_examine,
    disputed_plants,
    dominant_axes_for_plant,
    max_disagreement_per_plant,
)
from pulse.cross_exam_groupchat import CrossExamGroupChat
from pulse.latent import CONDITION_LABELS, FieldLatentState, PlantInstance
from pulse.llm_config import LLMKeyMissingError
from pulse.messages import ConstraintMessage


FAKE_LLM_CONFIG = {
    "config_list": [{"model": "stub", "api_key": "sk-fake-for-tests"}],
    "temperature": 0.0,
    "cache_seed": None,
}


# --- Cross-examination math ----------------------------------------------


def _make_constraint(sender: str, ll_per_plant: dict[int, list[float]]) -> ConstraintMessage:
    return ConstraintMessage(
        sender=sender,
        timestamp=time.time(),
        iteration=0,
        per_plant_log_likelihoods={pid: np.asarray(ll, dtype=float) for pid, ll in ll_per_plant.items()},
        per_plant_residual={pid: 0.0 for pid in ll_per_plant},
        per_plant_confidence={pid: 0.5 for pid in ll_per_plant},
        labels_discriminated=list(CONDITION_LABELS),
    )


def test_aligned_constraints_low_disagreement():
    base = [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # weed
    a = _make_constraint("a", {0: base})
    b = _make_constraint("b", {0: base})
    msgs = cross_examine({"a": a, "b": b})
    kl = msgs[0].per_plant_disagreement[0]
    assert kl < 1e-6


def test_opposed_constraints_high_disagreement():
    a = _make_constraint("a", {0: [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]})  # disease
    b = _make_constraint("b", {0: [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]})  # water_stress
    msgs = cross_examine({"a": a, "b": b})
    kl = msgs[0].per_plant_disagreement[0]
    assert kl > 1.0


def test_dominant_axis_concentrates_on_disputed_label():
    a = _make_constraint("a", {0: [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]})
    b = _make_constraint("b", {0: [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]})
    msgs = cross_examine({"a": a, "b": b})
    axes = dominant_axes_for_plant(msgs, 0)
    sorted_axes = sorted(axes.items(), key=lambda kv: -kv[1])
    top_two = {sorted_axes[0][0], sorted_axes[1][0]}
    assert top_two == {"disease", "water_stress"}


def test_disputed_plants_threshold():
    a = _make_constraint("a", {
        0: [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        1: [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    })
    b = _make_constraint("b", {
        0: [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],  # opposed → disputed
        1: [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],  # aligned → not disputed
    })
    msgs = cross_examine({"a": a, "b": b})
    disputed = disputed_plants(msgs, threshold=1.5)
    assert disputed == [0]


def test_max_disagreement_per_plant_aggregates_pairs():
    a = _make_constraint("a", {0: [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]})
    b = _make_constraint("b", {0: [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]})
    c = _make_constraint("c", {0: [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]})
    msgs = cross_examine({"a": a, "b": b, "c": c})
    maxes = max_disagreement_per_plant(msgs)
    assert maxes[0] > 1.0


# --- GroupChat speaker selection -----------------------------------------


def test_groupchat_round_robins_until_constraints_collected():
    a, b, c = _real_stub_agents("a", "b", "c")
    latent = FieldLatentState(image_shape=(100, 100))
    chat = CrossExamGroupChat([a, b, c], latent, max_round=4)
    nxt = chat._select_most_contested(a, chat)
    assert nxt is b


def test_groupchat_picks_unspoken_agent_first():
    a, b, c = _real_stub_agents("a", "b", "c")
    latent = FieldLatentState(image_shape=(100, 100))
    chat = CrossExamGroupChat([a, b, c], latent, max_round=4)
    chat.record_constraint("a", _make_constraint("a", {0: [0.0] * 7}))
    chat.record_constraint("b", _make_constraint("b", {0: [0.0] * 7}))
    nxt = chat._select_most_contested(b, chat)
    assert nxt is c


# --- Skeptic agent --------------------------------------------------------


def test_skeptic_requires_llm_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMKeyMissingError):
        SkepticAgent()


def test_skeptic_records_typed_hypothesis():
    agent = SkepticAgent(llm_config=FAKE_LLM_CONFIG)
    out = agent.propose_alternative_hypothesis(
        plant_id=2,
        hypothesis_id="disease_masquerading_as_water_stress",
        log_posterior=-1.2,
        evidence_axis_healthy_crop=0.0,
        evidence_axis_weed=0.0,
        evidence_axis_disease=0.7,
        evidence_axis_nutrient_stress=0.0,
        evidence_axis_water_stress=0.7,
        evidence_axis_pest_damage=0.0,
        evidence_axis_ambiguous=0.0,
    )
    assert out["plant_id"] == 2
    assert out["hypothesis_id"] == "disease_masquerading_as_water_stress"
    assert agent._hypotheses[0].plant_id == 2
    # Evidence axes are numeric only.
    for v in agent._hypotheses[0].evidence_axes.values():
        assert isinstance(v, float)


# --- Helpers --------------------------------------------------------------


def _real_stub_agents(*names: str):
    """Build real ConversableAgent instances — newer AG2 GroupChat requires Agent type."""
    from autogen import ConversableAgent

    return tuple(
        ConversableAgent(
            name=n,
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        for n in names
    )
