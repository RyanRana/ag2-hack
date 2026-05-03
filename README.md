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

| # | Agent | Paradigm | Model / Method |
|---|-------|----------|----------------|
| 1 | `WeedDetectorAgent` | ML | `foduucom/plant-leaf-detection-and-classification` (YOLOv8, ~25MB) |
| 2 | `DiseaseClassifierAgent` | ML | `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` (MobileNetV2, ~14MB) |
| 3 | `HealthClassifierAgent` | ML | `Diginsa/Plant-Disease-Detection-Project` (ViT, ~90MB) |
| 4 | `SegmentationAgent` | CV | OpenCV HSV/Canny/contour (no neural net) |
| 5 | `AnomalyDetectorAgent` | CV | `facebook/dinov2-base` + PatchCore (~350MB, lazy-loaded) |
| 6 | `GrowthStageAgent` | ML | Heuristic (green ratio + size + bright regions) |
| 7 | `WeatherPriorAgent` | Physics | Open-Meteo HTTP API + rule-based priors |
| 8 | `WaterBalanceAgent` | Physics | FAO-56 Penman-Monteith + van Genuchten (math only) |
| 9 | `PesticideFateAgent` | Physics | Gaussian plume drift + first-order DT50 (math only) |
| 10 | `EcologicalDynamicsAgent` | Biology | Lotka-Volterra ODE solver + toxicity (math only) |

### 2 LLM Agents (OpenAI via AG2)

| # | Agent | Paradigm | Backend |
|---|-------|----------|---------|
| 11 | `VLMReasonerAgent` | ML/LLM | OpenAI gpt-4o-mini via AG2 AssistantAgent |
| 12 | `SkepticAgent` | Meta | OpenAI gpt-4o-mini via AG2 AssistantAgent |

### What Each Agent Does

**1. WeedDetectorAgent** — Runs YOLOv8 on the full frame to detect and classify plants as weed or crop. Produces bounding boxes and initial weed-vs-crop probabilities. This is Phase 0 detection — everything downstream operates on these crop regions.

**2. DiseaseClassifierAgent** — Crops each plant's bounding box and runs MobileNetV2 trained on 38 PlantVillage disease classes (early blight, late blight, powdery mildew, rust, etc.). Maps the 38 fine-grained classes to Pulse's 7 condition labels. Temperature scaling (learned scalar T) calibrates the logits before softmax so the model doesn't dominate the posterior with overconfident scores.

**3. HealthClassifierAgent** — Coarse binary classifier: "is something wrong with this plant?" Can't tell *what's* wrong, but is calibrated to distinguish healthy from unhealthy with low compute. Acts as a tiebreaker when the disease classifier and segmentation disagree. Also temperature-scaled.

**4. SegmentationAgent** — Pure OpenCV analysis on each plant crop. Computes HSV-based green leaf masks, yellow tissue detection, Canny edge detection for perforations, and contour analysis. Produces both scalar features (green_ratio, yellowness, edge_density) for log-likelihood computation AND retains the spatial masks for the visual explanation overlay. This is the "Explanation === Evidence" guarantee — every overlay pixel comes from the same computation that produced the log-likelihood.

**5. AnomalyDetectorAgent** — Catches *unknown-unknowns* that classifiers would force into a known label. Extracts DINOv2 patch embeddings from each plant crop and scores them against a PatchCore memory bank of healthy plant patches. Plants that are far from the healthy distribution get mass pushed toward "ambiguous", triggering human review. The classifiers say "this is probably X"; the anomaly detector says "this doesn't look like anything I've seen before."

**6. GrowthStageAgent** — Classifies each plant as seedling, vegetative, flowering, or fruiting using visual heuristics (plant size, green coverage, presence of bright non-green regions). The growth stage is orthogonal to condition — it doesn't change the disease/weed diagnosis. Instead, it provides an urgency multiplier to the controller: a seedling with disease needs immediate action (fragile), a mature plant with the same disease can wait.

**7. WeatherPriorAgent** — Runs BEFORE all ML agents. Fetches 7-day weather history from Open-Meteo (free, no API key), then adjusts the shared prior:
- 5+ dry days + high temp → P(water_stress) ↑
- High humidity + warm → P(fungal disease) ↑
- Heavy recent rain → P(nutrient_stress) ↑ (leaching)
- Cool + dry → all stress priors ↓

This means the ML agents start from a weather-informed prior instead of a uniform one. If it hasn't rained in a week and it's 35°C, the system is already suspicious of water stress before the first pixel is analyzed.

**8. WaterBalanceAgent** — Resolves the canonical ML ambiguity: "is this plant wilting from disease or from water stress?" ML models can't tell — both look like drooping leaves. The water balance agent runs FAO-56 Penman-Monteith evapotranspiration demand against van Genuchten soil water retention to compute a physics-based stress index. If the soil is genuinely dry (high demand, low supply, low matric potential), it pushes water_stress UP and disease DOWN. If the soil is wet, it pushes water_stress DOWN. No pixels involved — pure biophysics.

**9. PesticideFateAgent** — Evaluates "what happens if we spray?" for each candidate action. Runs a Gaussian plume atmospheric dispersion model to compute off-target deposition on neighboring plants and waterways, using the current wind speed/direction. Also computes soil persistence via first-order DT50 degradation kinetics. Outputs a hazard score: drift > 0.4 ppm → VETO the spray. This is why the system sometimes recommends laser zap over herbicide even when it's confident about a weed — the wind is wrong.

**10. EcologicalDynamicsAgent** — Evaluates "what happens to the ecosystem if we treat?" Runs a Lotka-Volterra predator-prey-parasitoid ODE with pesticide toxicity parameters for the next 30 days. Predicts the population trajectory of pests, predators, and parasitoid wasps under each candidate intervention. If chlorpyrifos would crash the predator population by >50% in 14 days → VETO the chemical. This prevents the "pesticide treadmill" — killing predators causes a worse pest rebound than the original problem.

**11. VLMReasonerAgent** — Only fires on *disputed* plants (where ML agents disagreed, KL divergence > 1.5). Receives the actual crop image plus structured prompt. Looks for specific visual cues: concentric rings → fungal, water-soaked margins → bacterial, clean tears → mechanical damage. Outputs calibrated log-likelihoods per condition. Uses OpenAI gpt-4o-mini via AG2's AssistantAgent with typed tool calls (`analyze_disagreement_region`, `submit_per_plant_likelihoods`). Supports local VLM fallback (LLaVA/InternVL2) when configured.

**12. SkepticAgent** — The devil's advocate. When agents disagree, the Skeptic proposes alternative hypotheses: "what if this disease is actually nutrient stress?" or "what if this weed is a young crop?" Uses the same OpenAI backend as the VLM but in text-only mode. Generates structured hypotheses with evidence axes. Engages in a multi-turn debate with the VLM (max 3 rounds) — if posterior entropy remains high after the VLM's assessment, the Skeptic counter-argues and the VLM re-examines. Converges when entropy drops below threshold.

### Non-inference agents

- **`EIGControllerAgent`** — Utility-based action selection. Computes expected utility for all 8 intervention types per plant: `U = yield_protected - chem_cost - 0.5×drift_hazard - 0.4×eco_cost`. Picks the argmax. Uses AG2 `register_nested_chats` to spawn one ActionEvaluator per intervention type.
- **`HumanReviewProxy`** — Escalation agent. When the controller's best action is `human_review` (entropy too high, no confident diagnosis), this agent flags the plant for expert inspection.

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
