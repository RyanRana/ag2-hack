# Pesto

Multi-agent inference SDK for precision agriculture, built on
[AG2](https://github.com/ag2ai/ag2).

**12 agents. 4 paradigms. 10 local + 2 LLM. AG2 orchestrates all.**

> Modern agriculture runs on chemical blanketing because no single model
> is trustworthy enough to drive per-plant decisions, and even when one
> is, identifying a weed doesn't tell you whether spraying is safe — or
> whether it will create tomorrow's pest explosion. Pesto fuses four
> scientific paradigms in one AG2 orchestration graph: machine learning,
> computer vision, continuum physics, and population biology. The
> combination is what produces the recommendation.

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e .

# Pre-cache HuggingFace weights (~1 GB, one-time)
python scripts/download_models.py
```

```python
from pesto import Pipeline

pipe = Pipeline()
result = pipe.run("frame.jpg", field_state={
    "wind_dir_deg": 270.0, "wind_speed_m_s": 2.0,
    "soil_moisture_m3_m3": 0.30, "soil_texture": "loam",
    "T_C": 26.0, "RH_pct": 45.0, "u2_m_s": 2.0,
    "R_n_MJ_m2_d": 18.0, "crop_type": "tomato",
})
for action in result["actions"]:
    print(action["plant_id"], action["action_type"])
```

Run the dashboard:

```bash
pesto serve --port 8000
# or: .venv/bin/uvicorn pesto.dashboard.server:app --port 8000
```

## SDK at a glance

```python
from pesto import Pipeline, PipelineConfig, register_agent

# Toggle agents
Pipeline(PipelineConfig().disabled("vlm_reasoner", "skeptic"))

# Restrict the action space
Pipeline(PipelineConfig(interventions=[
    "no_action", "laser_zap", "targeted_irrigation",
    "human_review", "rescan_higher_res",
]))

# Plug in your own agent
@register_agent("my_blight_detector", role="ml")
class MyBlightDetector(ChannelAgent):
    def emit_constraint(self, image_path, latent):
        ...
```

Full guide: [`docs/`](docs/README.md).

## Documentation

| Topic | File |
|-------|------|
| 5-line install + first run | [docs/sdk-quickstart.md](docs/sdk-quickstart.md) |
| Pipeline phases, paradigms, file layout | [docs/architecture.md](docs/architecture.md) |
| Every built-in agent | [docs/agents.md](docs/agents.md) |
| `PipelineConfig` field reference | [docs/configuration.md](docs/configuration.md) |
| Adding a custom agent | [docs/custom-agents.md](docs/custom-agents.md) |
| Adding a new intervention | [docs/interventions.md](docs/interventions.md) |
| Live dashboard | [docs/dashboard.md](docs/dashboard.md) |
| `pesto` CLI | [docs/cli.md](docs/cli.md) |
| Test suite | [docs/testing.md](docs/testing.md) |

## LLM agents (optional)

The Skeptic and VLM Reasoner require an API key. Without one, the
other ten agents still run.

```ini
# .env
ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY=sk-...
```

## Performance

| Metric | Value |
|--------|-------|
| Chemical reduction vs. blanket spray | ~91% |
| Agents (total / local / LLM) | 12 / 10 / 2 |
| API cost per undisputed frame | $0.00 |
| Latency (undisputed, no VLM) | ~7–8 s |
| Latency (disputed, with VLM debate) | ~12–15 s |

## Data provenance

* **Pesticide chemistry** — EPA Pesticide Properties DataBase (PPDB,
  AERU University of Hertfordshire) and PAN Pesticide Database
* **Soil van Genuchten parameters** — USDA-NRCS via Carsel & Parrish 1988
* **Predator / parasitoid LC50 values** — EPA ECOTOX
* **Weather data** — Open-Meteo (free, no API key required)
* **Agronomic knowledge base** — Extension service guidelines, IPM bulletins
