# Dashboard

A live FastAPI + WebSocket dashboard for visualizing the pipeline.

## Run it

```bash
# Pre-cache model weights (~1 GB, one-time)
python scripts/download_models.py

# Start the dashboard
.venv/bin/uvicorn pesto.dashboard.server:app --host 0.0.0.0 --port 8000

# Or via the CLI
pesto serve --port 8000
```

Open `http://localhost:8000` and click **START STREAM**.

## Panels

* Video stream with bounding-box overlays (color-coded by condition)
* Heatmap canvas (additive-blended halos per plant)
* Drift cone visualization (spray dispersion path)
* Evidence overlay toggle (visual explanation from segmentation)
* 12-agent status panel with paradigm badges and LOCAL / API indicators
* Weather context panel (prior adjustments from Open-Meteo)
* Growth stage & anomaly panel
* Agronomic knowledge panel (RAG-retrieved treatment references)
* Intervention summary (cumulative bar chart)
* Biophysics readout (stress index, ET demand / supply, soil ψ)
* Recommended actions with per-plant cropped previews
* Debate indicator (turn dots with convergence status)
* KPI strip (weeds detected, zapped, disease treated, drift vetoes,
  predators protected, review queue)
* Message stream (real-time event log)

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/` | Static dashboard |
| `POST` | `/api/run` | Single-frame inference on an uploaded image |
| `WS`   | `/ws` | Live event stream from the running captain |

## How the dashboard hooks into the SDK

`pesto/dashboard/server.py` calls `_build_captain()` which currently
constructs a `PestoCaptain` directly. To run the dashboard against an
SDK-configured pipeline (e.g. with custom agents loaded), instantiate a
`Pipeline` and pass the captain through:

```python
from pesto import Pipeline, PipelineConfig

pipe = Pipeline(PipelineConfig(
    chemistry_per_action={"targeted_spray": "spinosad"},
))
captain = pipe.captain()  # use this where _build_captain() returns
```

The captain's `emit_event` callback is what pumps messages into the
WebSocket; the dashboard wires it up automatically.
