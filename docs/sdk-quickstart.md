# SDK Quickstart

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e .

# Pre-cache HuggingFace weights (~1 GB, one-time)
python scripts/download_models.py
```

Optional: drop an LLM key in `.env` to enable the Skeptic + VLM Reasoner.
Without a key, the pipeline silently skips those two agents and the other
ten still run.

```ini
# .env
ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY=sk-...
```

## Run inference

```python
from pesto import Pipeline

pipe = Pipeline()
result = pipe.run(
    "frame.jpg",
    field_state={
        "wind_dir_deg": 270.0,
        "wind_speed_m_s": 2.0,
        "soil_moisture_m3_m3": 0.30,
        "soil_texture": "loam",
        "T_C": 26.0,
        "RH_pct": 45.0,
        "u2_m_s": 2.0,
        "R_n_MJ_m2_d": 18.0,
        "crop_type": "tomato",
    },
)

for action in result["actions"]:
    print(action["plant_id"], action["action_type"], action["expected_utility"])
```

The `result` dict mirrors :meth:`PestoCaptain.run_inference`:

| Key | Shape | Meaning |
|-----|-------|---------|
| `actions` | list of `ActionMessage` dicts | per-plant intervention recommendation |
| `latent` | dict | populated `FieldLatentState` posterior |
| `constraints` | dict[agent_name, ConstraintMessage] | log-likelihoods each agent contributed |
| `cross_exam` | list of `CrossExamMessage` dicts | pairwise KL divergence diagnostics |
| `disputed_plants` | list[int] | plant IDs sent to the LLM debate |
| `physics` | dict | per-plant × action drift / hazard scores |
| `ecology` | dict | per-plant × action 30-day population trajectories |
| `hypotheses` | list of `HypothesisMessage` dicts | skeptic alternatives (debate runs) |

## Config recipes

Disable the LLM agents (CI box, no key, or strict latency budget):

```python
from pesto import Pipeline, PipelineConfig

pipe = Pipeline(PipelineConfig().disabled("vlm_reasoner", "skeptic"))
```

Restrict the action space (organic-mode, no chemical sprays):

```python
pipe = Pipeline(PipelineConfig(
    interventions=[
        "no_action", "laser_zap", "targeted_irrigation",
        "foliar_nutrient", "human_review", "rescan_higher_res",
    ],
))
```

Swap the chemistry default for a spray action:

```python
pipe = Pipeline(PipelineConfig(
    chemistry_per_action={"targeted_spray": "spinosad"},
))
```

Run only a minimal ML stack:

```python
pipe = Pipeline(PipelineConfig().only("weed_detector", "segmentation"))
```

Mutate after construction:

```python
pipe = Pipeline()
pipe.disable_intervention("targeted_spray").disable_agent("vlm_reasoner")
print(pipe.active_agents())
print(pipe.active_interventions())
```

## Add a custom agent

See [Custom Agents](custom-agents.md). The TL;DR:

```python
from pesto import register_agent
from pesto.agents.base import ChannelAgent

@register_agent("late_blight_detector", role="ml",
                description="Specialist late-blight CNN.")
class LateBlightDetectorAgent(ChannelAgent):
    def __init__(self):
        super().__init__(name="late_blight_detector")

    def emit_constraint(self, image_path, latent):
        ...  # return a ConstraintMessage
```

It joins the constraint phase automatically the next time `Pipeline()` is
constructed.
