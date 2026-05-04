"""Agent registry for the Pesto SDK.

Two registries are exposed:

* ``AGENT_REGISTRY`` — name → ``AgentSpec`` for every built-in and user-added
  agent. ``Pipeline`` looks up specs here when assembling a captain.
* ``INTERVENTION_REGISTRY`` — name → utility-tag metadata. Built-ins are
  Pesto's eight intervention types from :mod:`pesto.latent`; users can
  register additional ones (the captain's controller still has to know how
  to score them — see ``EIGControllerAgent.compute_utility``).

Custom agents are added with the :func:`register_agent` decorator. They are
plugged into a pipeline either by including their name in
``PipelineConfig.agents`` (when toggleable) or by passing instances via
``PipelineConfig.custom_agents``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pesto.latent import INTERVENTION_TYPES


AgentFactory = Callable[..., Any]


@dataclass
class AgentSpec:
    """Description of how to instantiate an agent inside a pipeline."""

    name: str
    factory: AgentFactory
    role: str  # one of: "ml", "weather", "biophysics", "physics", "ecology",
               #         "anomaly", "growth", "skeptic", "vlm", "custom"
    requires_llm: bool = False
    description: str = ""

    def build(self, **kwargs: Any) -> Any:
        return self.factory(**kwargs)


AGENT_REGISTRY: dict[str, AgentSpec] = {}
INTERVENTION_REGISTRY: dict[str, dict[str, Any]] = {
    name: {"default_chemistry": None} for name in INTERVENTION_TYPES
}


def register_agent(
    name: str,
    *,
    role: str = "custom",
    requires_llm: bool = False,
    description: str = "",
) -> Callable[[AgentFactory], AgentFactory]:
    """Register an agent factory under ``name``.

    Used as a class decorator on a ``ChannelAgent`` subclass, or on any
    callable that returns an agent instance compatible with the captain's
    expected role.

    Example:
        @register_agent("blight_detector", role="ml",
                        description="Detects late blight.")
        class BlightDetectorAgent(ChannelAgent):
            def emit_constraint(self, image_path, latent): ...
    """

    def deco(factory: AgentFactory) -> AgentFactory:
        AGENT_REGISTRY[name] = AgentSpec(
            name=name,
            factory=factory,
            role=role,
            requires_llm=requires_llm,
            description=description,
        )
        return factory

    return deco


def register_intervention(
    name: str,
    *,
    default_chemistry: str | None = None,
) -> None:
    """Register a new intervention name and its default chemistry.

    The pipeline only uses this entry to know the action exists and to map
    it to a chemistry id. The controller's utility function still needs to
    score it — see ``EIGControllerAgent.compute_utility`` for how to add
    a utility branch.
    """
    INTERVENTION_REGISTRY[name] = {"default_chemistry": default_chemistry}


def list_agents() -> list[str]:
    """Sorted list of registered agent names."""
    return sorted(AGENT_REGISTRY.keys())


def list_interventions() -> list[str]:
    """Sorted list of registered interventions."""
    return sorted(INTERVENTION_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in agent registration.
#
# The factory functions are registered lazily so importing ``pesto.registry``
# alone does not pull in heavyweight ML deps. ``ensure_builtins_registered``
# is idempotent and is called from ``Pipeline.__init__`` and the SDK
# top-level imports.
# ---------------------------------------------------------------------------

_BUILTINS_REGISTERED = False


def ensure_builtins_registered() -> None:
    """Register Pesto's twelve bundled agents in ``AGENT_REGISTRY``.

    Safe to call multiple times. Imports happen lazily so this can run in
    contexts that haven't installed the optional ML extras (the import
    failures are surfaced when the agent is *used*, not at registration
    time).
    """
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    def _make(modpath: str, classname: str) -> AgentFactory:
        def factory(**kwargs: Any) -> Any:
            mod = __import__(modpath, fromlist=[classname])
            return getattr(mod, classname)(**kwargs)

        return factory

    builtins: list[tuple[str, str, str, str, bool, str]] = [
        # name,                modpath,                            class,                role,         llm,   doc
        ("weed_detector",      "pesto.agents.weed_detector",       "WeedDetectorAgent",    "ml",        False, "YOLO weed/crop detection."),
        ("disease_classifier", "pesto.agents.disease_classifier",  "DiseaseClassifierAgent","ml",       False, "MobileNetV2 disease ID."),
        ("health_classifier",  "pesto.agents.health_classifier",   "HealthClassifierAgent","ml",        False, "ViT healthy/unhealthy."),
        ("segmentation",       "pesto.agents.segmentation",        "SegmentationAgent",    "ml",        False, "OpenCV evidence + spatial masks."),
        ("anomaly_detector",   "pesto.agents.anomaly_detector",    "AnomalyDetectorAgent", "anomaly",   False, "DINOv2 + PatchCore."),
        ("growth_stage",       "pesto.agents.growth_stage",        "GrowthStageAgent",     "growth",    False, "Seedling/vegetative/flowering/fruiting."),
        ("weather_prior",      "pesto.agents.weather_prior",       "WeatherPriorAgent",    "weather",   False, "Open-Meteo prior adjuster."),
        ("water_balance",      "pesto.agents.water_balance",       "WaterBalanceAgent",    "biophysics",False, "FAO-56 + van Genuchten."),
        ("pesticide_fate",     "pesto.agents.pesticide_fate",      "PesticideFateAgent",   "physics",   False, "Gaussian plume drift + DT50."),
        ("ecological_dynamics","pesto.agents.ecological_dynamics", "EcologicalDynamicsAgent","ecology", False, "Lotka-Volterra population ODE."),
        ("skeptic",            "pesto.agents.skeptic",             "SkepticAgent",         "skeptic",   True,  "Devil's-advocate hypotheses."),
        ("vlm_reasoner",       "pesto.agents.vlm_reasoner",        "VLMReasonerAgent",     "vlm",       True,  "VLM disambiguation on disputed plants."),
    ]
    for name, mod, cls, role, llm, doc in builtins:
        AGENT_REGISTRY[name] = AgentSpec(
            name=name,
            factory=_make(mod, cls),
            role=role,
            requires_llm=llm,
            description=doc,
        )

    _BUILTINS_REGISTERED = True
