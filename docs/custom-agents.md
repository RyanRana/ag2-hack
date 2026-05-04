# Custom Agents

Plug your own model into the constraint phase by subclassing
`ChannelAgent` and registering it.

## The interface

A code-only agent must:

1. Subclass `pesto.agents.base.ChannelAgent`.
2. Pass a `name` to `super().__init__()` — used as the registry key and
   as the `sender` field on emitted messages.
3. Implement `emit_constraint(image_path, latent) -> ConstraintMessage`.

```python
import time
import numpy as np

from pesto import register_agent
from pesto.agents.base import ChannelAgent
from pesto.latent import CONDITION_LABELS, FieldLatentState
from pesto.messages import ConstraintMessage


@register_agent(
    "late_blight_detector",
    role="ml",
    description="Specialist late-blight CNN.",
)
class LateBlightDetectorAgent(ChannelAgent):
    def __init__(self):
        super().__init__(name="late_blight_detector")
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForImageClassification
            self._model = AutoModelForImageClassification.from_pretrained(
                "username/late-blight-cnn"
            )
        return self._model

    def emit_constraint(
        self, image_path: str | None, latent: FieldLatentState
    ) -> ConstraintMessage:
        model = self._load()
        per_ll: dict[int, np.ndarray] = {}
        per_resid: dict[int, float] = {}
        per_conf: dict[int, float] = {}
        i_disease = CONDITION_LABELS.index("disease")
        i_healthy = CONDITION_LABELS.index("healthy_crop")

        for plant in latent.plants:
            score = self._score_plant(image_path, plant)  # 0..1
            log_lik = np.zeros(len(CONDITION_LABELS))
            log_lik[i_disease] = float(2.5 * score)
            log_lik[i_healthy] = float(-1.5 * score)
            per_ll[plant.plant_id] = log_lik
            per_resid[plant.plant_id] = 0.0
            per_conf[plant.plant_id] = float(score)

        return ConstraintMessage(
            sender=self.name,
            timestamp=time.time(),
            iteration=latent.iteration,
            per_plant_log_likelihoods=per_ll,
            per_plant_residual=per_resid,
            per_plant_confidence=per_conf,
            labels_discriminated=["disease", "healthy_crop"],
        )

    def _score_plant(self, image_path, plant):
        ...  # crop, preprocess, run model, return scalar
```

After the import-time decorator runs, the agent is in `AGENT_REGISTRY`
and `Pipeline()` will pick it up automatically the next time it builds
a captain.

## What each `ConstraintMessage` field is for

| Field | Purpose |
|-------|---------|
| `sender` | Used by the cross-exam to attribute disagreement |
| `iteration` | Frame counter — copy from `latent.iteration` |
| `per_plant_log_likelihoods` | The actual constraint, one vector of length 7 per plant |
| `per_plant_residual` | OOD score the skeptic consumes (0 = in-distribution) |
| `per_plant_confidence` | Self-confidence — also surfaces in the dashboard |
| `labels_discriminated` | Subset of `CONDITION_LABELS` your model can actually distinguish |
| `metadata` | Free-form dict — not used by the loop, surfaces in dashboard event payloads |

The cross-examiner only weighs disagreement on labels listed in
`labels_discriminated` — a binary healthy-vs-unhealthy classifier
should not "vote" on whether something is `weed` vs. `disease`.

## Roles

The `role` argument to `@register_agent` is metadata for introspection.
Recognized values:

* `ml` — pattern-recognition model (joins the parallel constraint phase)
* `weather` — pre-inference context adjuster
* `biophysics` — Penman-Monteith-style mechanistic agent
* `physics` — drift / fate / dispersion model
* `ecology` — population-dynamics ODE
* `anomaly` — out-of-distribution flagger
* `growth` — phenology classifier
* `skeptic` / `vlm` — LLM-backed disambiguator (set `requires_llm=True`)
* `custom` — anything else

The Pipeline currently treats everything except the seven slot-specific
roles as "ml" — they're appended to the constraint phase. To plug into
a specialized slot (replace the weather agent, for example), pass a
pre-built instance via `PipelineConfig.custom_agents` or directly to
`PestoCaptain.__init__`.

## LLM-backed agents

Set `requires_llm=True` on the decorator. The pipeline silently skips
the agent when no API key is configured, unless `PipelineConfig.require_llm`
is set. See `pesto/agents/skeptic.py` and `pesto/agents/vlm_reasoner.py`
for examples that combine `AssistantAgent + register_function` with a
local-model fallback.

## Testing your agent in isolation

```python
from pesto.latent import FieldLatentState, PlantInstance
from your_module import LateBlightDetectorAgent

latent = FieldLatentState(image_shape=(640, 640))
latent.plants.append(PlantInstance(plant_id=0, bbox=(0, 0, 640, 640)))

agent = LateBlightDetectorAgent()
msg = agent.emit_constraint("path/to/leaf.jpg", latent)

assert msg.sender == "late_blight_detector"
assert 0 in msg.per_plant_log_likelihoods
```

Then plug it into a pipeline:

```python
from pesto import Pipeline
import your_module  # registers the agent at import time

pipe = Pipeline()
assert "late_blight_detector" in pipe.active_agents()
result = pipe.run("frame.jpg", field_state={...})
```

## Adding a new intervention

If your agent introduces a new action that the controller should pick
between (say, `targeted_uv_treatment`), register it and extend the
controller's utility function:

```python
from pesto import register_intervention

register_intervention("targeted_uv_treatment", default_chemistry=None)
```

You also need to add a utility branch in
`EIGControllerAgent.compute_utility` so the controller knows how to
score it. See [Interventions](interventions.md) for the recipe.

## Inter-agent firewall

The `tests/test_protocol_firewall.py` test scans every dataclass in
`pesto.messages` for prose-y field names (`text`, `message`, `prose`,
`explanation`, `commentary`, `description`, `narrative`, `reasoning`).
Custom agents that introduce new typed messages must keep that
discipline — the LLM agents must emit only structured tool-call
arguments, not free-form prose. The firewall is what makes the
multi-paradigm orchestration auditable.
