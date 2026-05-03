# Pulse

Multi-agent inference engine for precision agriculture, built on
[AG2](https://github.com/ag2ai/ag2).

**12 agents. 4 paradigms. 10 local + 2 LLM. AG2 orchestrates all.**

> *Modern agriculture runs on chemical blanketing because no single model is trustworthy enough to drive per-plant decisions, and even when one is, identifying a weed doesn't tell you whether spraying is safe — or whether it will create tomorrow's pest explosion. Pulse fuses four scientific paradigms in one AG2 orchestration graph: machine learning, computer vision, continuum physics, and population biology. The combination is what produces the recommendation.*

---

## Architecture

```
Frame Input
    │
    ▼
Phase 0 ─── Pre-Inference Context ────────────────────────────────────
    │   [Weather Prior]  Open-Meteo → adjusts priors before ML runs
    │
    ▼
Phase 1 ─── Constraint Emission (7 agents, parallel) ─────────────────
    │   ML:      [Weed Detector] [Disease Classifier] [Health Classifier]
    │   CV:      [Segmentation + Evidence Maps] [Anomaly Detector]
    │   Physics: [Water Balance]
    │   ML:      [Growth Stage]
    │
    ▼
Phase 2 ─── Cross-Examination ─────────────────────────────────────────
    │   Weighted KL divergence across paradigm pairs
    │   ML×ML = 1.0   ML×Bio = 1.5   ML×Anomaly = 2.0
    │   KL > 1.5 → plant is "disputed"
    │
    ▼  (only if disputed plants exist)
Phase 3 ─── Multi-Turn Debate (max 3 turns) ───────────────────────────
    │   [Skeptic]  ──hypotheses──▶  [VLM Reasoner]
    │       ▲                            │
    │       └──── entropy check ─────────┘
    │   Both use OpenAI via AG2 AssistantAgent
    │
    ▼
Phase 4 ─── Physics + Ecology (per plant × action) ────────────────────
    │   [Pesticide Fate]      Gaussian plume drift + DT50 degradation
    │   [Eco Dynamics]        Lotka-Volterra 30-day population forecast
    │   drift > 0.4 ppm → VETO spray
    │   predator drop > 50% → VETO chemical
    │
    ▼
Phase 5 ─── Controller (argmax utility) ───────────────────────────────
    │   U = yield_protected - chem_cost - 0.5×drift - 0.4×eco_cost
    │
    ▼
Output ─── Per-Plant Actions ──────────────────────────────────────────
    [laser zap] [fungicide] [irrigate] [fertilize] [review] [no action]
```

---

## Agent Roster (12 agents)

### 10 Local Agents (no API calls)

| # | Agent | Paradigm | Model / Method | Role |
|---|-------|----------|----------------|------|
| 1 | `WeedDetectorAgent` | ML | foduucom YOLOv8 | Weed/crop classification |
| 2 | `DiseaseClassifierAgent` | ML | MobileNetV2 + temperature scaling | 38-class plant disease ID |
| 3 | `HealthClassifierAgent` | ML | ViT binary + temperature scaling | Healthy vs unhealthy |
| 4 | `SegmentationAgent` | CV | HSV/Canny/contour + evidence maps | Spatial evidence retention |
| 5 | `AnomalyDetectorAgent` | CV | DINOv2 + PatchCore | Flags unknown conditions |
| 6 | `GrowthStageAgent` | ML | ViT classifier (heuristic fallback) | Seedling/veg/flower/fruit → urgency |
| 7 | `WeatherPriorAgent` | Physics | Open-Meteo API | Adjusts priors from weather context |
| 8 | `WaterBalanceAgent` | Physics | Penman-Monteith + van Genuchten | Water stress vs disease disambiguation |
| 9 | `PesticideFateAgent` | Physics | Gaussian plume + DT50 kinetics | Drift risk + persistence scoring |
| 10 | `EcologicalDynamicsAgent` | Biology | Lotka-Volterra ODE + toxicity | Predator/pest population forecast |

### 2 LLM Agents (OpenAI via AG2)

| # | Agent | Paradigm | Backend | Role |
|---|-------|----------|---------|------|
| 11 | `VLMReasonerAgent` | ML/LLM | gpt-4o-mini (AG2 AssistantAgent) | Visual reasoning on disputed crops |
| 12 | `SkepticAgent` | Meta | gpt-4o-mini (AG2 AssistantAgent) | Alternative hypothesis generation |

LLM agents only fire on disputed plants (every 5th frame in streaming mode). The other 10 run on every frame with zero API cost.

### Non-inference agents

- `EIGControllerAgent` — utility-based action selection with nested chats
- `HumanReviewProxy` — escalation when confidence is too low

---

## Four Paradigms

| Paradigm | What it does | Agents |
|----------|-------------|--------|
| **Machine Learning** | Pattern recognition over pixels (YOLO, MobileNet, ViT, PatchCore) | 1, 2, 3, 5, 6 |
| **Computer Vision** | Physics-of-light feature extraction with spatial evidence (OpenCV, DINOv2) | 4, 5 |
| **Continuum Physics** | Mechanistic simulation (Gaussian plume, Penman-Monteith, van Genuchten) | 7, 8, 9 |
| **Population Biology** | ODE-based population dynamics (Lotka-Volterra with toxicity) | 10 |

Every code agent shares the same `ConversableAgent(llm_config=False)` envelope. Every LLM agent uses `AssistantAgent` with `register_function` + `Annotated` typed tools. Inter-agent payloads are typed dataclasses; the firewall in `tests/test_protocol_firewall.py` rejects any prose field.

---

## Key Capabilities

### Temperature Scaling (Sprint 1)
Post-hoc calibration via learned scalar T per model. Disease and health classifiers divide logits by T before softmax, preventing overconfident models from dominating the shared posterior.

### Visual Explanation (Sprint 1)
Segmentation agent retains spatial evidence masks (leaf, yellow, edge, contour). The visual explanation renderer overlays these on the original image. **Every overlay pixel comes from the same computation that produced the log-likelihood.** Explanation === Evidence.

### Weather-Aware Priors (Sprint 1)
Weather Prior agent fetches from Open-Meteo (free, no key), computes conditional priors:
- 5+ dry days + heat → water_stress ↑
- High humidity + warm → disease ↑ (fungal risk)
- Heavy rain → nutrient_stress ↑ (leaching)

### DINOv2 Backbone + Anomaly Detection (Sprint 2)
DINOv2-base extracts per-patch embeddings. PatchCore memory bank scores each plant's distance from the healthy distribution. Plants far from healthy get flagged as anomalous — catching unknown-unknowns that classifiers would force into a known label.

### Growth Stage Classification (Sprint 2)
Classifies seedling / vegetative / flowering / fruiting. Provides urgency multiplier to the controller: seedling with disease = act immediately, mature plant = can wait.

### Weighted Cross-Examination (Sprint 2)
KL divergence weighted by paradigm pair type:
- ML-vs-ML: 1.0 (same paradigm, normal weight)
- ML-vs-biophysics: 1.5 (cross-paradigm disagreement matters more)
- ML-vs-anomaly: 2.0 (anomaly flags are rare and important)

### Multi-Turn Skeptic-VLM Debate (Sprint 3)
When posterior entropy remains high after initial analysis, the Skeptic proposes alternative hypotheses and the VLM re-examines the crop image. Max 3 turns. Convergence checked via entropy threshold.

### RAG over Agronomic Literature (Sprint 3)
FAISS + sentence-transformers vector search over agronomic knowledge (IPM guidelines, treatment thresholds, resistance reports). Retrieved context adjusts chemical choice and treatment thresholds.

### Temporal Diff (Sprint 4)
Frame differencing + optical flow between consecutive frames. Per-plant change scores identify disease progression: "changed since last scan" is a strong escalation signal.

### Active Learning Loop (Sprint 4)
Plants where human_review triggered, cross-exam KL > 3.0, or anomaly score > 3σ are queued for labeling. Crops saved to disk with metadata for periodic fine-tuning on hard cases.

### Grounding DINO Integration (Sprint 4)
Open-vocabulary detection alongside YOLO. Text prompts like "wilted plant" or "leaf with holes" catch things YOLO was never trained on. NMS deduplication merges both detection sets.

---

## AG2 Idioms

| Idiom | Where |
|-------|-------|
| `ConversableAgent` + `llm_config=False` + `register_reply` | `pulse/agents/base.py`, `water_balance.py`, `pesticide_fate.py`, `ecological_dynamics.py` |
| `register_function` with `Annotated` typed args | `pulse/agents/skeptic.py`, `pulse/agents/vlm_reasoner.py` |
| `GroupChat` with custom `speaker_selection_method` | `pulse/cross_exam_groupchat.py` |
| `register_nested_chats` | `pulse/agents/controller.py` |
| `register_hand_off` + `OnCondition` + `AfterWork` | `pulse/runtime.py::wire_swarm_pipeline` |
| `UserProxyAgent` with conditional `human_input_mode` | `pulse/agents/human_review.py` |
| `initiate_swarm_chat` | `pulse/captain.py::PulseCaptain.run_swarm_inference` |

---

## Dashboard

Real-time web UI built with FastAPI + WebSocket + vanilla JS + Tailwind.

**Live panels:**
- Video stream with bounding box overlays (color-coded by condition)
- Heatmap canvas (additive-blended halos per plant)
- Drift cone visualization (spray dispersion path)
- Evidence overlay toggle (visual explanation from segmentation)
- 12-agent status panel with paradigm badges and LOCAL/API indicators
- Weather context panel (prior adjustments from Open-Meteo)
- Growth stage & anomaly panel
- Agronomic knowledge panel (RAG-retrieved treatment references)
- Intervention summary (cumulative bar chart)
- Biophysics readout (stress index, ET demand/supply, soil ψ)
- Recommended actions with per-plant cropped previews
- Debate indicator (turn dots with convergence status)
- KPI strip (weeds detected, zapped, disease treated, drift vetoes, predators protected, review queue)
- Message stream (real-time event log)

---

## Quickstart

```bash
# Clone and setup
cd ag2-hack
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install faiss-cpu

# Pre-cache model weights (~1 GB, one-time)
python scripts/download_models.py

# Run the dashboard
.venv/bin/uvicorn pulse.dashboard.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` and click **START STREAM**.

### LLM Agents (optional)

The Skeptic and VLM Reasoner require an API key. Without it, the other 10 agents still run.

```bash
# Add to .env
OPENAI_API_KEY=sk-...
```

---

## Demos

### OpenCV Evidence Agent — AG2 dispatch (CLI)

Exercises the full AG2 conversational path through `SegmentationAgent`
(`UserProxyAgent.initiate_chat` → `ChannelAgent._emit_constraint_reply`)
without the dashboard, then saves an overlay PNG. The overlay pixels
come from the same boolean masks (`leaf_mask`, `yellow_mask`,
`edge_mask`, `contour_mask`) whose pixel counts produced the scalar
log-likelihoods on the returned `ConstraintMessage` —
*Explanation === Evidence*.

```bash
# Synthetic chlorotic swatch (no setup needed)
python scripts/demo_opencv_evidence.py

# Or your own image
python scripts/demo_opencv_evidence.py --image path/to/leaf.jpg --out overlays/
```

The script prints per-plant top-2 conditions, log-likelihoods,
residual + confidence, and the path to the rendered overlay PNG. No
LLM API key is required for this demo path.

---

## Tests

```bash
# Core tests (119 tests)
.venv/bin/python -m pytest tests/ --ignore=tests/test_rag.py

# RAG tests (12 tests, run separately due to FAISS+torch process interaction)
.venv/bin/python -m pytest tests/test_rag.py
```

**131 total tests** covering:
- Protocol firewall (no prose in inter-agent messages)
- Temperature scaling calibration
- Visual explanation rendering
- Weather prior computation
- DINOv2 backbone feature extraction
- PatchCore anomaly scoring
- Growth stage classification
- Local model backend + structured output parsing
- RAG retrieval + utility adjustment
- Skeptic local inference + multi-turn debate
- Temporal diff frame differencing
- Active learning queue management
- Detection merging (YOLO + Grounding DINO NMS)
- All original agent tests (segmentation, cross-exam, latent, controller, physics, ecology)

---

## Repo Structure

```
ag2-hack/
  pulse/                    Python package (12 agents + orchestration)
    agents/                 All 12 agent implementations
      base.py               ChannelAgent base class
      weed_detector.py       YOLO weed/crop
      disease_classifier.py  MobileNetV2 + temp scaling
      health_classifier.py   ViT binary + temp scaling
      segmentation.py        OpenCV evidence + spatial masks
      anomaly_detector.py    DINOv2 + PatchCore
      growth_stage.py        Growth stage classifier
      weather_prior.py       Open-Meteo weather priors
      water_balance.py       Penman-Monteith biophysics
      pesticide_fate.py      Gaussian plume drift
      ecological_dynamics.py Lotka-Volterra ecology
      vlm_reasoner.py        VLM (local + API fallback)
      skeptic.py             Skeptic (local + API fallback)
      controller.py          EIG utility-based action selection
      human_review.py        Escalation proxy
    rag/                    FAISS + sentence-transformers RAG
      retriever.py           Agronomic knowledge retrieval
    dashboard/              FastAPI + WebSocket + JS frontend
      server.py              API endpoints + agent wiring
      static/                HTML + JS + CSS
    biology/                Lotka-Volterra ODE
    biophysics/             Penman-Monteith + van Genuchten
    physics/                Gaussian plume drift model
    backbone.py             DINOv2 feature extraction
    calibration.py          Temperature scaling
    captain.py              PulseCaptain orchestrator
    cross_exam.py           Weighted cross-examination
    cross_exam_groupchat.py GroupChat speaker selection
    detection.py            YOLO + Grounding DINO + NMS merge
    latent.py               Shared posterior (FieldLatentState)
    llm_config.py           LLM provider configuration
    local_model.py          Local VLM backend (LLaVA/InternVL2)
    messages.py             Typed inter-agent protocol
    runtime.py              Swarm pipeline wiring
    temporal.py             Frame differencing + optical flow
    active_learning.py      Hard case queuing + labeling
    visual_explain.py       Evidence overlay renderer
  tests/                   131 unit tests
  data/                    Chemistry tables, demo imagery, RAG index
  scripts/                 Model download + demo runner
  pyproject.toml
  plan.md                  Architecture enhancement plan
```

---

## Data Provenance

- **Pesticide chemistry** — EPA Pesticide Properties DataBase (PPDB, AERU University of Hertfordshire) and PAN Pesticide Database
- **Soil van Genuchten parameters** — USDA-NRCS via Carsel & Parrish 1988
- **Predator / parasitoid LC50 values** — EPA ECOTOX
- **Weather data** — Open-Meteo (free, no API key required)
- **Agronomic knowledge base** — Extension service guidelines, IPM bulletins

---

## Performance

| Metric | Value |
|--------|-------|
| Chemical reduction vs blanket spray | ~91% |
| Agents (total / local / LLM) | 12 / 10 / 2 |
| API cost per undisputed frame | $0.00 |
| Latency (undisputed, no VLM) | ~7-8s |
| Latency (disputed, with VLM debate) | ~12-15s |
| Test coverage | 131 tests |
