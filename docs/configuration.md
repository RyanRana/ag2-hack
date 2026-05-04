# Configuration

`PipelineConfig` is the single source of truth for what a pipeline does.

```python
from pesto import Pipeline, PipelineConfig

cfg = PipelineConfig(
    agents={"vlm_reasoner": False},                # toggle individual agents
    interventions=["no_action", "laser_zap",       # restrict the action space
                   "targeted_irrigation",
                   "human_review", "rescan_higher_res"],
    chemistry_per_action={"targeted_spray":        # swap default chemistry
                          "spinosad"},
    custom_agents=[],                              # pre-built instances to plug in
    emit_event=None,                               # event sink (dashboard hook)
    require_llm=False,                             # raise vs. silently skip LLM
)
pipe = Pipeline(cfg)
```

## Field reference

### `agents: dict[str, bool]`

Per-agent on/off map keyed by registry name. Anything not in the dict
falls back to its default (true for built-ins and any registered custom
agent). The toggleable built-ins are:

* `weed_detector`, `disease_classifier`, `health_classifier`,
  `segmentation`, `anomaly_detector`, `growth_stage`, `weather_prior`,
  `skeptic`, `vlm_reasoner`

`water_balance`, `pesticide_fate`, and `ecological_dynamics` are *not*
toggleable — the captain calls them unconditionally because they're
math-only and load-bearing for the action veto.

#### Helpers

```python
PipelineConfig().disabled("vlm_reasoner", "skeptic")
PipelineConfig().with_agents(weed_detector=True, disease_classifier=False)
PipelineConfig().only("weed_detector", "segmentation")  # everything else off
```

### `interventions: list[str] | None`

Subset of `INTERVENTION_TYPES` to consider in the controller's argmax.
`None` (the default) means all eight built-in actions:

```
no_action  laser_zap  targeted_spray   targeted_fungicide
targeted_irrigation   foliar_nutrient  human_review  rescan_higher_res
```

Restricting this list does **not** disable the physics or ecology
assessments — only the controller's action space. Vetoes still fire on
whatever spray candidates remain.

### `chemistry_per_action: dict[str, str]`

Chemistry id used by the physics + ecology agents per spray action.
Defaults are intentionally broad-spectrum so the vetoes are visible
(chlorpyrifos for `targeted_spray` causes predator collapse; atrazine
for `targeted_fungicide` has long DT50). Override to demonstrate a less
hazardous alternative:

```python
PipelineConfig(chemistry_per_action={
    "targeted_spray": "spinosad",
    "targeted_fungicide": "copper_hydroxide",
})
```

The chemistry id must exist in `data/pesticide_chemistry.json` (PPDB +
PAN-derived parameters). Unknown ids fall through to the agent's
internal default.

### `custom_agents: list[Any]`

Pre-built agent instances appended to the constraint phase. Use this
for one-off experiments where you don't want to register a factory in
the global registry:

```python
from pesto.agents.base import ChannelAgent

class MyExperimentAgent(ChannelAgent):
    def __init__(self):
        super().__init__(name="my_experiment")
    def emit_constraint(self, image_path, latent):
        ...

cfg = PipelineConfig(custom_agents=[MyExperimentAgent()])
```

For reusable agents, prefer `@register_agent` — see
[Custom Agents](custom-agents.md).

### `emit_event: Callable[[str, dict], None] | None`

Event sink invoked at every pipeline phase. Called as
`emit_event(kind, payload)`. Used by the dashboard to push WebSocket
events; SDK consumers usually leave it `None`. Useful event kinds:

`weather_prior_started`, `constraint`, `cross_exam`, `llm_phase`,
`hypotheses`, `debate_turn`, `physics_assessment`, `ecology_trajectory`,
`action`, plus `*_error` variants.

### `require_llm: bool`

If true, the pipeline raises `RuntimeError` when a built-in LLM agent is
toggled on but no API key is configured. Default `False` (silent skip).
Use this in tests to detect missing-key regressions.

## Mutating after construction

```python
pipe = Pipeline()
pipe.disable_agent("vlm_reasoner")
pipe.disable_intervention("targeted_spray")
pipe.enable_intervention("foliar_nutrient")
pipe.add_agent(MyExperimentAgent())  # appends to custom_agents
```

All mutators return `self` so they chain. The captain is rebuilt lazily
on the next `run()`.

## Environment variables

Set in `.env` at the repo root or export before running:

| Var | Purpose |
|-----|---------|
| `OPENAI_API_KEY` | OpenAI key for Skeptic + VLM Reasoner |
| `ANTHROPIC_API_KEY` | Anthropic key (preferred when both are set) |
| `OPENAI_BASE_URL` / `OPENROUTER_BASE_URL` | OpenAI-compatible gateway base URL |
| `PESTO_LLM_MODEL` | Override the chat model (default: provider default) |
| `HF_TOKEN` | HuggingFace token for higher download rate limits |

The Weather Prior agent fetches Open-Meteo without a key.
