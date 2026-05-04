# CLI

Pesto installs a `pesto` console script.

## Subcommands

```bash
pesto agents                  # list registered agents
pesto interventions           # list registered interventions
pesto run IMAGE [flags]       # run the pipeline on one image, print actions
pesto serve [--port 8000]     # run the dashboard with uvicorn
```

## `pesto run`

```bash
pesto run frame.jpg                                       # run with defaults
pesto run frame.jpg --disable vlm_reasoner skeptic        # turn agents off
pesto run frame.jpg --only weed_detector segmentation     # turn most agents off
pesto run frame.jpg --disable-intervention targeted_spray
pesto run frame.jpg --field-state '{"wind_dir_deg":270,"wind_speed_m_s":2}'
```

Output is JSON:

```json
{
  "actions": [
    {
      "plant_id": 0,
      "action_type": "laser_zap",
      "expected_utility": 0.612,
      "expected_information_gain": 0.81,
      "physics_hazard_score": 0.0
    }
  ],
  "disputed_plants": [],
  "active_agents": ["weed_detector", "segmentation", "..."],
  "active_interventions": ["no_action", "laser_zap", "..."]
}
```

## `pesto serve`

Wraps `uvicorn pesto.dashboard.server:app`. See [Dashboard](dashboard.md)
for what the UI shows.

```bash
pesto serve --host 0.0.0.0 --port 8000 --reload
```

## Programmatic use

The CLI is a thin shell around the SDK; everything it does is also one-liners
in Python:

```python
import pesto
pesto.list_agents()
pesto.list_interventions()
pesto.Pipeline().run("frame.jpg", field_state={...})
```
