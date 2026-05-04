"""Pesto SDK — high-level Pipeline + configuration surface.

Wraps :class:`pesto.captain.PestoCaptain` so callers don't have to
hand-assemble agents. Use it when you want to:

* run inference end-to-end on an image with sensible defaults,
* toggle individual agents on/off (e.g. drop the LLM agents on a CI box),
* restrict the action space (e.g. forbid ``targeted_spray`` in organic mode),
* plug in a custom agent without forking the captain.

Public surface (re-exported from :mod:`pesto`):

* :class:`Pipeline`
* :class:`PipelineConfig`
* :func:`register_agent` / :func:`register_intervention`

Example::

    from pesto import Pipeline, PipelineConfig

    pipe = Pipeline(PipelineConfig(
        agents={"vlm_reasoner": False, "skeptic": False},
        interventions=["no_action", "laser_zap", "targeted_irrigation",
                       "human_review", "rescan_higher_res"],
    ))
    result = pipe.run("frame.jpg", field_state={
        "wind_dir_deg": 270.0, "wind_speed_m_s": 2.0,
        "soil_moisture_m3_m3": 0.30, "soil_texture": "loam",
        "T_C": 26.0, "RH_pct": 45.0, "u2_m_s": 2.0,
        "R_n_MJ_m2_d": 18.0, "crop_type": "tomato",
    })
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pesto.latent import FieldLatentState, INTERVENTION_TYPES
from pesto.registry import (
    AGENT_REGISTRY,
    INTERVENTION_REGISTRY,
    AgentSpec,
    ensure_builtins_registered,
    register_agent,
    register_intervention,
)

if TYPE_CHECKING:
    from pesto.captain import PestoCaptain


# Toggleable built-ins. ``water_balance``, ``pesticide_fate`` and
# ``ecological_dynamics`` are intentionally absent: the captain calls them
# unconditionally (math-only, ~ms latency), so they're always on. Everything
# below can be turned off with ``PipelineConfig.agents``.
_DEFAULT_AGENT_TOGGLES: dict[str, bool] = {
    "weed_detector": True,
    "disease_classifier": True,
    "health_classifier": True,
    "segmentation": True,
    "anomaly_detector": True,
    "growth_stage": True,
    "weather_prior": True,
    # LLM agents auto-skip when no API key is available, so leaving these on
    # in the default config is safe.
    "skeptic": True,
    "vlm_reasoner": True,
}


@dataclass
class PipelineConfig:
    """Declarative configuration for :class:`Pipeline`.

    All fields have sensible defaults; the captain is assembled by reading
    this config at ``Pipeline`` construction time.
    """

    # Per-agent on/off toggles. Unspecified names fall back to the default
    # toggle (True for built-ins, True for any registered custom agent).
    agents: dict[str, bool] = field(default_factory=dict)

    # Subset of intervention names to consider in the action argmax. ``None``
    # means "all built-in interventions".
    interventions: list[str] | None = None

    # Override the default chemistry per intervention (e.g. swap chlorpyrifos
    # for spinosad on ``targeted_spray``). Missing entries fall back to
    # ``DEFAULT_CHEMISTRY_PER_ACTION`` in :mod:`pesto.captain`.
    chemistry_per_action: dict[str, str] = field(default_factory=dict)

    # Pre-built custom agent instances appended to the constraint phase.
    # Use this when you don't want to register your factory globally.
    custom_agents: list[Any] = field(default_factory=list)

    # Optional event sink invoked as ``emit(kind, payload)``. The dashboard
    # uses this to pump WebSocket events; SDK consumers usually leave it None.
    emit_event: Any = None

    # If True, the pipeline raises when a built-in LLM agent is enabled but
    # no API key is configured. Default behaviour is to silently skip them.
    require_llm: bool = False

    def with_agents(self, **toggles: bool) -> "PipelineConfig":
        """Return a copy with extra ``agents`` toggles merged in."""
        merged = {**self.agents, **toggles}
        return dataclasses.replace(self, agents=merged)

    def disabled(self, *names: str) -> "PipelineConfig":
        """Return a copy with each ``name`` toggled off."""
        return self.with_agents(**{n: False for n in names})

    def only(self, *names: str) -> "PipelineConfig":
        """Return a copy with every toggleable agent off except ``names``."""
        toggles = {n: False for n in _DEFAULT_AGENT_TOGGLES}
        for n in names:
            toggles[n] = True
        return dataclasses.replace(self, agents=toggles)


class Pipeline:
    """High-level entry point for running a Pesto inference pass."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        ensure_builtins_registered()
        self.config = config or PipelineConfig()
        self._captain: "PestoCaptain | None" = None

    # ------------------------------------------------------------------
    # Construction / mutation helpers
    # ------------------------------------------------------------------

    def disable_agent(self, name: str) -> "Pipeline":
        self.config = self.config.with_agents(**{name: False})
        self._captain = None
        return self

    def enable_agent(self, name: str) -> "Pipeline":
        self.config = self.config.with_agents(**{name: True})
        self._captain = None
        return self

    def disable_intervention(self, name: str) -> "Pipeline":
        active = self._active_interventions()
        if name in active:
            active.remove(name)
        self.config = dataclasses.replace(self.config, interventions=active)
        self._captain = None
        return self

    def enable_intervention(self, name: str) -> "Pipeline":
        active = self._active_interventions()
        if name not in active:
            active.append(name)
        self.config = dataclasses.replace(self.config, interventions=active)
        self._captain = None
        return self

    def add_agent(self, agent: Any) -> "Pipeline":
        """Append a constructed agent instance to the constraint phase."""
        agents = list(self.config.custom_agents) + [agent]
        self.config = dataclasses.replace(self.config, custom_agents=agents)
        self._captain = None
        return self

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def active_agents(self) -> list[str]:
        """Names of agents that will be instantiated on next ``run``."""
        return [name for name, on in self._resolved_toggles().items() if on]

    def active_interventions(self) -> list[str]:
        return list(self._active_interventions())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def captain(self) -> "PestoCaptain":
        """Lazily build (and cache) the underlying :class:`PestoCaptain`."""
        if self._captain is None:
            self._captain = self._build_captain()
        return self._captain

    def run(
        self,
        image_path: str,
        field_state: dict | None = None,
        *,
        chemistry_per_plant: dict[int, str] | None = None,
        prebuilt_latent: FieldLatentState | None = None,
    ) -> dict:
        """Run the full pipeline on one image.

        Returns the same dict shape as :meth:`PestoCaptain.run_inference`.
        """
        return self.captain().run_inference(
            image_path,
            field_state or {},
            chemistry_per_plant=chemistry_per_plant,
            prebuilt_latent=prebuilt_latent,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolved_toggles(self) -> dict[str, bool]:
        toggles = dict(_DEFAULT_AGENT_TOGGLES)
        # Any non-built-in registered agents default ON unless opted out.
        for name in AGENT_REGISTRY:
            toggles.setdefault(name, True)
        toggles.update(self.config.agents)
        return toggles

    def _active_interventions(self) -> list[str]:
        if self.config.interventions is not None:
            return list(self.config.interventions)
        return list(INTERVENTION_TYPES)

    def _build_captain(self) -> "PestoCaptain":
        from pesto.captain import PestoCaptain
        from pesto.llm_config import llm_key_available

        toggles = self._resolved_toggles()

        def maybe(name: str) -> Any:
            if not toggles.get(name, False):
                return None
            spec = AGENT_REGISTRY.get(name)
            if spec is None:
                return None
            if spec.requires_llm and not llm_key_available():
                if self.config.require_llm:
                    raise RuntimeError(
                        f"Agent {name!r} requires an LLM key but none is "
                        "configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
                    )
                return None
            return spec.build()

        ml_agents: list[Any] = []
        for name in ("weed_detector", "disease_classifier", "segmentation",
                     "health_classifier"):
            agent = maybe(name)
            if agent is not None:
                ml_agents.append(agent)

        # Append any non-built-in registered agents whose toggle is ON.
        builtin_names = set(_DEFAULT_AGENT_TOGGLES)
        for name, spec in AGENT_REGISTRY.items():
            if name in builtin_names:
                continue
            if not toggles.get(name, True):
                continue
            try:
                ml_agents.append(spec.build())
            except Exception as exc:  # noqa: BLE001
                if self.config.emit_event:
                    self.config.emit_event(
                        "custom_agent_error",
                        {"name": name, "error": f"{type(exc).__name__}: {exc}"},
                    )

        ml_agents.extend(self.config.custom_agents)

        return PestoCaptain(
            ml_agents=ml_agents,
            weather_prior_agent=maybe("weather_prior"),
            anomaly_detector=maybe("anomaly_detector"),
            growth_stage_agent=maybe("growth_stage"),
            skeptic_agent=maybe("skeptic"),
            vlm_reasoner=maybe("vlm_reasoner"),
            interventions=self._active_interventions(),
            chemistry_per_action=self.config.chemistry_per_action or None,
            emit_event=self.config.emit_event,
        )


__all__ = [
    "Pipeline",
    "PipelineConfig",
    "register_agent",
    "register_intervention",
    "AGENT_REGISTRY",
    "INTERVENTION_REGISTRY",
    "AgentSpec",
]
