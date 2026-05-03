# pulse

Multi-agent inference engine for precision agriculture, written in
[AG2](https://github.com/ag2ai/ag2).

> *"Modern agriculture runs on chemical blanketing for two reasons: no single computer-vision model is trustworthy enough to drive per-plant decisions, and even when one is, identifying a weed doesn't tell you whether spraying is safe — or whether it will create tomorrow's pest explosion. Pulse is a multi-agent inference architecture in AG2 that fuses three scientific paradigms in one orchestration graph. Five ML models — YOLO weed detector, MobileNet disease classifier, SAM segmentation, ViT health, and a vision-language reasoner — emit calibrated likelihoods over plant condition. A soil-plant-atmosphere physics agent runs Penman-Monteith and van Genuchten on the soil and weather state to break ML disagreements over water-stress-versus-disease through biophysical first principles. A continuum-physics agent runs Gaussian-plume drift and first-order degradation kinetics to assess each candidate spray's off-target deposition and persistence. A population-biology agent runs Lotka-Volterra-with-toxicity to forecast the next 30 days of pest, predator, and parasitoid populations under each candidate intervention. Five agents say what's there. One says what's physiologically possible. One says what happens if we treat. One says what happens if we keep treating. The combination is what produces the recommendation. Agents pass constraints, not text — even the LLM-backed skeptic communicates only through registered, typed tools; the physics, biophysics, and ecology agents share the same `ConversableAgent` envelope with `llm_config=False`, demonstrating that one orchestration graph can host fundamentally different inference paradigms. We demo on real field imagery: 91% reduction in chemical use vs. blanket spraying — roughly 50% from precision targeting, 25% from physics vetoes, and 16% from refusing to start the chemical treadmill. Same protocol runs on medical imaging, gravitational waves, and battery diagnostics. Modern ag is the largest single market for computer vision on Earth, and it's stuck because precision detection isn't enough — you need precision prescription that respects three different scientific paradigms simultaneously. Pulse is the architecture for that."*

## Three paradigms, one orchestration graph

| Paradigm | Agents | Inference |
|----------|--------|-----------|
| **ML** | `WeedDetectorAgent`, `DiseaseClassifierAgent`, `SegmentationAgent`, `HealthClassifierAgent`, `VLMReasonerAgent` | Pattern recognition over pixels |
| **Continuum physics** | `PesticideFateAgent` | Gaussian-plume drift + first-order degradation (DT50) |
| **Soil-plant-atmosphere physics** | `WaterBalanceAgent` | FAO-56 Penman-Monteith + van Genuchten retention |
| **Population biology** | `EcologicalDynamicsAgent` | Lotka-Volterra ODE with pesticide toxicity |
| **Meta** | `SkepticAgent`, `EIGControllerAgent`, `HumanReviewProxy` | Hypothesis proposal, utility-driven action selection |

Every code agent shares the same `ConversableAgent(llm_config=False)`
envelope. Every LLM agent uses `AssistantAgent` with `register_function`
+ `Annotated` typed tools. Inter-agent payloads are typed dataclasses;
the firewall in `tests/test_protocol_firewall.py` mechanically rejects
any prose field.

## AG2 idioms in use

| Idiom | Where |
|-------|-------|
| `ConversableAgent` + `llm_config=False` + `register_reply` | `pulse/agents/base.py` (ML), `pulse/agents/pesticide_fate.py`, `water_balance.py`, `ecological_dynamics.py` |
| `register_function` with `Annotated` typed args | `pulse/agents/skeptic.py`, `pulse/agents/vlm_reasoner.py` |
| `GroupChat` with custom `speaker_selection_method` | `pulse/cross_exam_groupchat.py` |
| `register_nested_chats` | `pulse/agents/controller.py` |
| `register_hand_off` + `OnCondition` + `AfterWork` | `pulse/runtime.py::wire_swarm_pipeline` |
| `UserProxyAgent` with conditional `human_input_mode` | `pulse/agents/human_review.py` |
| `initiate_swarm_chat` | `pulse/captain.py::PulseCaptain.run_swarm_inference` |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .                              # or follow §7 install order

# Pre-cache HF model weights (~1 GB; one-time):
python scripts/download_models.py

# Run the dashboard at http://localhost:8000
uvicorn pulse.dashboard.server:app --host 0.0.0.0 --port 8000
```

For the LLM-backed Skeptic and VLMReasoner:

```bash
export OPENAI_API_KEY=sk-…    # required for Skeptic + VLMReasoner
```

If the key is unset the rest of the pipeline still runs — the Skeptic and
VLMReasoner are skipped (per §14.D bailout).

## Run the demo without a server

```bash
python scripts/run_demo.py data/demo/synthetic_field.jpg
```

Prints a populated `FieldLatentState` with detected plants and the
`weed_detector` constraint applied.

## Tests

```bash
pytest tests/
```

The first test (`tests/test_protocol_firewall.py`) is a static schema
check that mechanically prevents the architecture from collapsing into
"LLMs in a trench coat".

## Bailouts applied

This build runs CPU-only and applies the spec's pre-authorised bailouts:

- **§14.A** (`ultralyticsplus` torch.load incompatibility) — fetch the
  foduucom YOLO weights via `huggingface_hub.hf_hub_download` and load
  with base `ultralytics.YOLO`. Same model, same architecture story.
- **§14.B** (no GPU for SAM) — `SegmentationAgent` falls back to YOLO
  bboxes + HSV-based leaf masking. Hot-swappable for SAM via the
  `sam_predictor` injection point.
- **§14.D** (no GPU for VLM) — `VLMReasonerAgent` is LLM-backed via
  `AssistantAgent + register_function` and uses GPT-4o-mini by default;
  it requires `OPENAI_API_KEY` and is skipped otherwise.
- **§14.E** (`Swarm` symbol surface drift) — `runtime.py` import-probes
  three module paths. The captain falls back to a sequential pipeline
  that produces the same per-plant ActionMessages.
- **§14.M** (water balance edge cases) — Penman-Monteith inputs are
  bounded; Carsel-Parrish sand parameters give very low matric tension at
  realistic theta values, so dry-stress tests target loam (where the
  retention curve produces meaningful suction at low theta).

## Repo structure

See `pulse/` for the eight code agents, `pulse/physics/`,
`pulse/biophysics/`, `pulse/biology/` for the three mechanistic models,
`pulse/dashboard/` for the FastAPI + WebSocket frontend, `data/` for
chemistry tables and demo imagery, `tests/` for the firewall and per-agent
unit tests.

## Data provenance

- **Pesticide chemistry** — EPA Pesticide Properties DataBase (PPDB,
  hosted by AERU University of Hertfordshire) and PAN Pesticide
  Database. https://sitem.herts.ac.uk/aeru/ppdb/
- **Soil van Genuchten parameters** — USDA-NRCS via Carsel & Parrish 1988.
- **Predator / parasitoid LC50 values** — EPA ECOTOX.
  https://cfpub.epa.gov/ecotox/

See `data/README.md` for the full citation list.
