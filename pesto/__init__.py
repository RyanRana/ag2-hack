"""Pesto — multi-agent inference SDK for precision agriculture.

Top-level surface:

* :class:`Pipeline` and :class:`PipelineConfig` — high-level entry points.
* :func:`register_agent` / :func:`register_intervention` — extend the SDK
  with your own agents or intervention types.
* :class:`PestoCaptain` — the orchestrator that the SDK wraps; use this if
  you need direct control over agent assembly.
* :class:`FieldLatentState`, ``CONDITION_LABELS``, ``INTERVENTION_TYPES``
  — core data types and label sets shared across agents.
* Message dataclasses (:class:`ConstraintMessage`, :class:`ActionMessage`,
  …) — the typed inter-agent protocol.
"""

from pesto.latent import (
    CONDITION_LABELS,
    FieldLatentState,
    INTERVENTION_TYPES,
    PlantInstance,
)
from pesto.messages import (
    ActionMessage,
    ConstraintMessage,
    CrossExamMessage,
    HypothesisMessage,
    InterventionAssessmentMessage,
    TrajectoryMessage,
)
from pesto.registry import (
    AGENT_REGISTRY,
    INTERVENTION_REGISTRY,
    AgentSpec,
    list_agents,
    list_interventions,
    register_agent,
    register_intervention,
)


def __getattr__(name: str):
    """Lazy attribute access for heavy imports.

    ``Pipeline``, ``PipelineConfig``, and ``PestoCaptain`` pull in autogen
    and the agent factories. Importing ``pesto`` shouldn't pay that cost
    until the user actually constructs a pipeline.
    """
    if name == "Pipeline":
        from pesto.sdk import Pipeline
        return Pipeline
    if name == "PipelineConfig":
        from pesto.sdk import PipelineConfig
        return PipelineConfig
    if name == "PestoCaptain":
        from pesto.captain import PestoCaptain
        return PestoCaptain
    raise AttributeError(f"module 'pesto' has no attribute {name!r}")


__version__ = "0.1.0"

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "PestoCaptain",
    "register_agent",
    "register_intervention",
    "list_agents",
    "list_interventions",
    "AGENT_REGISTRY",
    "INTERVENTION_REGISTRY",
    "AgentSpec",
    "FieldLatentState",
    "PlantInstance",
    "CONDITION_LABELS",
    "INTERVENTION_TYPES",
    "ActionMessage",
    "ConstraintMessage",
    "CrossExamMessage",
    "HypothesisMessage",
    "InterventionAssessmentMessage",
    "TrajectoryMessage",
    "__version__",
]
