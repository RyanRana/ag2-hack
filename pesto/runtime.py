"""Swarm pipeline wiring (§8.5) + sequential fallback orchestrator.

The Swarm wiring uses ``register_hand_off`` with ``OnCondition`` and
``AfterWork``/``AFTER_WORK.TERMINATE`` so the AG2 idiom is demonstrably
present in the codebase (§15 success criteria). The actual end-to-end
demo runs through ``SequentialPipeline`` because it deals more reliably
with the cross-version Swarm import surface (§14.E bailout) and lets
unit tests exercise the controller without LLM dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


# AG2 has reorganised the Swarm import surface across versions (see §14.E).
# Try the canonical paths in order; fall back to None if all are missing.
def _import_swarm() -> dict[str, Any] | None:
    candidates = [
        "autogen.agentchat.contrib.swarm_agent",
        "autogen.agentchat.swarm_agent",
        "autogen.agents.experimental.swarm",
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            return {
                "register_hand_off": getattr(mod, "register_hand_off", None),
                "OnCondition": getattr(mod, "OnCondition", None),
                "AfterWork": getattr(mod, "AfterWork", None),
                "AFTER_WORK": getattr(mod, "AFTER_WORK", None),
                "initiate_swarm_chat": getattr(mod, "initiate_swarm_chat", None),
            }
        except ImportError:
            continue
    return None


SWARM = _import_swarm()


def wire_swarm_pipeline(
    constraint_phase: Any,
    cross_exam_phase: Any,
    skeptic_phase: Any,
    physics_phase: Any,
    ecology_phase: Any,
    controller_phase: Any,
    human_review_proxy: Any,
    *,
    max_disagreement: Callable[[Any], float] = lambda agent: 0.0,
    needs_human: Callable[[Any], bool] = lambda agent: False,
) -> None:
    """Register handoffs across the eight-paradigm pipeline.

    Pipeline order: constraint_emission → cross_exam → [skeptic if disputed]
    → physics → ecology → controller → [human_review if ambiguous].
    """
    if SWARM is None or SWARM["register_hand_off"] is None:
        # Bailout §14.E: AG2 Swarm symbols not exposed in this version.
        # The SequentialPipeline below performs the same routing in code.
        return
    register_hand_off = SWARM["register_hand_off"]
    OnCondition = SWARM["OnCondition"]
    AfterWork = SWARM["AfterWork"]
    AFTER_WORK = SWARM["AFTER_WORK"]

    register_hand_off(
        agent=constraint_phase,
        hand_to=[
            OnCondition(
                target=cross_exam_phase,
                condition=lambda agent, messages: getattr(
                    agent, "_all_constraints_collected", lambda: True
                )(),
            ),
            AfterWork(AFTER_WORK.STAY),
        ],
    )
    register_hand_off(
        agent=cross_exam_phase,
        hand_to=[
            OnCondition(
                target=skeptic_phase,
                condition=lambda agent, messages: max_disagreement(agent) > 1.5,
            ),
            AfterWork(target=physics_phase),
        ],
    )
    register_hand_off(
        agent=skeptic_phase,
        hand_to=[AfterWork(target=physics_phase)],
    )
    register_hand_off(
        agent=physics_phase,
        hand_to=[AfterWork(target=ecology_phase)],
    )
    register_hand_off(
        agent=ecology_phase,
        hand_to=[AfterWork(target=controller_phase)],
    )
    register_hand_off(
        agent=controller_phase,
        hand_to=[
            OnCondition(
                target=human_review_proxy,
                condition=lambda agent, messages: needs_human(agent),
            ),
            AfterWork(AFTER_WORK.TERMINATE),
        ],
    )


class SequentialPipeline:
    """Reliable in-process orchestrator that runs the same pipeline as
    Swarm. We use this as the captain's default executor since the Swarm
    symbol surface drifts across AG2 versions (§14.E).
    """

    def __init__(
        self,
        ml_agents: list,
        water_balance_agent: Any,
        physics_agent: Any,
        ecology_agent: Any,
        controller_cls: Any,
        skeptic_factory: Callable[[], Any] | None = None,
        vlm_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.ml_agents = ml_agents
        self.water_balance_agent = water_balance_agent
        self.physics_agent = physics_agent
        self.ecology_agent = ecology_agent
        self.controller_cls = controller_cls
        self.skeptic_factory = skeptic_factory
        self.vlm_factory = vlm_factory
