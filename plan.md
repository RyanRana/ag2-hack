# Pulse — Proposed Architecture Enhancement Plan

## Overview

Enhance the existing 8-agent, 3-paradigm precision agriculture inference engine with calibration, visual explanations, contextual priors, anomaly detection, and local VLM reasoning.

**Target**: 12 agents (from 8), 4 paradigms (from 3), with visual explanation guarantees.
**Constraint**: No external LLM/Vision API calls. All reasoning runs on local models.

**The four paradigms:**
1. **Machine Learning** — pattern recognition over pixels (YOLO, MobileNet, ViT, PatchCore)
2. **Computer Vision** — physics-of-light feature extraction with spatial evidence (OpenCV, DINOv2)
3. **Continuum Physics** — mechanistic simulation (Gaussian plume, Penman-Monteith, van Genuchten)
4. **Population Biology** — ODE-based population dynamics (Lotka-Volterra with toxicity)

---

## Current Architecture (Baseline)

```
Single Frame Input
       │
       ▼
Phase 1: YOLO Detection (foduucom model, fixed classes)
       │
       ▼
Phase 2: Constraint Emission (5 ML + 1 biophysics, parallel)
  • WeedDetector (YOLO overlap)
  • DiseaseClassifier (MobileNetV2)
  • HealthClassifier (ViT binary)
  • Segmentation (HSV fallback — spatial info DISCARDED)
  • WaterBalance (Penman-Monteith + van Genuchten)
       │
       ▼
Phase 3: Cross-Examination (pairwise KL-divergence, threshold 1.5)
       │
       ▼
Phase 4: Skeptic + VLM (GPT-4o-mini, one-shot, numeric features only)
       │
       ▼
Phase 5-6: Physics + Ecology (per plant × action cartesian product)
  • PesticideFate (Gaussian plume + DT50)
  • EcologicalDynamics (Lotka-Volterra ODE)
       │
       ▼
Phase 7: Controller (U = base - 0.5×hazard - 0.4×eco → argmax)
       │
       ▼
Output: Text-only action card ("Plant #3: disease → fungicide")
```

### Current Limitations

- Single frame, no temporal context
- Spatial evidence discarded after scalar reduction
- VLM gets numbers, not the actual image (and relies on external API)
- No calibration — loud models dominate posterior
- No anomaly detection for unknown conditions
- No weather/growth-stage context
- No visual explanation of WHY

---

## Proposed Architecture (Enhanced)

### Pre-Inference Context Layer (NEW)

```
Current Frame ──┐            Weather API ──┐
                │                          │
                ▼                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PRE-INFERENCE CONTEXT                          │
│                                                                  │
│  Weather Prior Agent              Growth Stage Agent              │
│  • Open-Meteo fetch               • ViT classifier               │
│  • 5-day rain=0 →                 • seedling/veg/flower/fruit    │
│    P(water_stress)↑               • Modifies urgency multiplier  │
│  • Humidity+temp →                  in utility function           │
│    P(fungal)↑                                                    │
│                                                                  │
│  Output: Adjusted Priors for Ψ (before any ML agent runs)        │
└──────────────────────────────────────────────────────────────────┘

Optional (Sprint 4, requires multi-frame data):
  • Temporal Diff Module — frame differencing + optical flow
  • "Plant changed since last scan" → auto-escalate in cross-exam
```

### Phase 1: Detection (Enhanced)

```
┌──────────────────────────────────────────────────────────────────┐
│  YOLO (existing)          Grounding DINO (NEW)                   │
│  foduucom model           IDEA-Research/grounding-dino-base      │
│  Fixed classes            Open-vocab: "diseased leaf",           │
│                           "wilted plant", "insect damage"        │
│                                                                  │
│  Output: Union of detections, NMS dedup → PlantInstance list     │
└──────────────────────────────────────────────────────────────────┘
```

### Phase 2: Constraint Emission (Enhanced)

```
┌──────────────────────────────────────────────────────────────────┐
│  Two independent feature paths (NOT a shared backbone):          │
│                                                                  │
│  PATH A: Existing ML models (own backbones, own forward passes)  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ Disease      │ │ Health       │ │ Weed         │             │
│  │ Classifier   │ │ Classifier   │ │ Detector     │             │
│  │ MobileNetV2  │ │ ViT          │ │ (YOLO)       │             │
│  │ + Temp Scale │ │ + Temp Scale │ │              │             │
│  │ (calibrated) │ │ (calibrated) │ │              │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│                                                                  │
│  PATH B: DINOv2 (NEW — runs ONCE per frame, separate from above)│
│  ┌────────────────────────────────────────────────────────┐      │
│  │ facebook/dinov2-base — 768-dim patch features           │      │
│  │ Feeds into:                                             │      │
│  │   • Anomaly Detector (PatchCore on DINOv2 embeddings)   │      │
│  │   • Attention maps → Visual Explanation layer           │      │
│  │ Does NOT replace MobileNet/ViT — supplements them       │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  PATH C: OpenCV Evidence Agent (ENHANCED segmentation)           │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ • HSV color segmentation, Canny edge detection          │      │
│  │ • Contour analysis, gradient magnitude                  │      │
│  │ • RETAINS spatial masks (not just scalars)              │      │
│  │ • Same scalars → Ψ as constraint                        │      │
│  │ • Same masks → Visual Explanation layer as evidence     │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  PATH D: Biophysics                                              │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ Water Balance (ENHANCED)                                │      │
│  │ Penman-Monteith + van Genuchten + weather forecast data │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  All emit ConstraintMessage (CALIBRATED via temperature scaling) │
└──────────────────────────────────────────────────────────────────┘
```

**Key clarification**: DINOv2 does NOT replace MobileNet or ViT. Those models keep their own backbones and produce their own predictions. DINOv2 is a separate feature path that feeds the anomaly detector and provides attention maps for visual explanation. The existing models just get calibrated via temperature scaling.

### Phase 3: Cross-Examination (Enhanced)

```
┌──────────────────────────────────────────────────────────────────┐
│  KL-divergence (existing) + NEW weighted paradigm pairs:         │
│                                                                  │
│  • ML-vs-ML disagreement (weight 1.0)                            │
│  • ML-vs-biophysics disagreement (weight 1.5 — cross-paradigm)   │
│  • ML-vs-anomaly disagreement (weight 2.0 — anomaly flags rare)  │
│                                                                  │
│  Disputed threshold: KL > 1.5 (same) but weighted by pair type   │
└──────────────────────────────────────────────────────────────────┘
```

### Phase 4: Multi-Turn Resolution (Local Models Only)

```
┌──────────────────────────────────────────────────────────────────┐
│  Skeptic Agent (local LLM — text-only, same weights as VLM)      │
│  • Receives cross-exam scores + constraint summaries             │
│  • Generates 2-3 alternative hypotheses as structured data       │
│       │                                                          │
│       ▼                                                          │
│  Local VLM Reasoner (NO external API)                            │
│  • Receives ACTUAL crop image + skeptic hypotheses               │
│  • Model: LLaVA-1.6-7B (GPU) or InternVL2-2B (lighter)          │
│  • Both Skeptic and VLM share the same local model weights       │
│    (multimodal model used in text-only mode for Skeptic,         │
│     image+text mode for VLM)                                     │
│  • Structured output parsing (JSON schema enforcement):          │
│    "concentric rings? → fungal"                                  │
│    "water-soaked margin? → bacterial"                             │
│    "clean tear? → mechanical damage"                              │
│       │                                                          │
│       ▼                                                          │
│  Convergence check:                                              │
│    If posterior entropy still high after VLM update:              │
│    → Skeptic counter-argues (1 more turn)                        │
│    → VLM re-examines with new hypothesis                         │
│    → Max 3 total turns, then accept current posterior            │
│    AG2 GroupChat with custom speaker_selection                    │
└──────────────────────────────────────────────────────────────────┘
```

### Phase 5-6: Physics + Ecology (Enhanced)

```
┌──────────────────────────────────────────────────────────────────┐
│  Pesticide Fate (ENHANCED)         Ecological Dynamics (ENHANCED)│
│  + weather forecast integration    + seasonal predator baselines │
│  + wind prediction from Open-Meteo                               │
│                                                                  │
│  "Wind shifts in 2h — delay       "Spring: parasitoid pop        │
│   spray advised"                    rebuilding, extra penalty     │
│                                     for chemical intervention"    │
└──────────────────────────────────────────────────────────────────┘
```

### Phase 7: Controller (Enhanced Utility Function)

```
U(action) = E[yield_protected]
           − chem_cost(action)
           − 0.5 × physics_hazard
           − 0.4 × ecological_cost
           − 0.3 × temporal_urgency_penalty        ← NEW (delay cost)
           + 0.2 × growth_stage_multiplier         ← NEW (seedling = urgent)
           + RAG_context_adjustment                ← NEW (regional knowledge)

RAG Lookup (NEW — fully local, FAISS + sentence-transformers):
  Input: "Early blight + tomato + July + central valley"
  Retrieved: "Resistance to mancozeb documented 2022. Use chlorothalonil.
              Threshold: 2 lesions/leaf before treatment justified."
  → adjusts action_params + chemical choice + treatment threshold
```

### Phase 8: Visual Explanation (NEW)

```
┌──────────────────────────────────────────────────────────────────┐
│  Per-Plant Annotated Output                                      │
│                                                                  │
│  Layer 1: OpenCV evidence masks (from Phase 2 — SAME data)       │
│           • Red contours = lesion regions (HSV-detected)          │
│           • Magenta = perforation edges (Canny-detected)          │
│           • Blue heatmap = wilt/saturation loss                   │
│                                                                  │
│  Layer 2: DINOv2 attention overlay                                │
│           • Where the feature extractor focused                   │
│           • Validates ML models attended to same regions           │
│                                                                  │
│  Layer 3: Physics drift cone visualization                        │
│           • Translucent cone showing spray dispersion path         │
│           • Red dots = at-risk neighbor plants                    │
│                                                                  │
│  Layer 4 (optional, Sprint 4): Temporal diff highlight            │
│           • Yellow border = "changed since last scan"             │
│                                                                  │
│  GUARANTEE: Every overlay pixel comes from the SAME computation  │
│  that produced the log-likelihood. Explanation === Evidence.      │
└──────────────────────────────────────────────────────────────────┘
```

### Active Learning Loop (NEW)

```
Plants where:
  • human_review was triggered
  • cross-exam KL > 3.0 (severe disagreement)
  • anomaly score > 3σ
→ Queued for labeling
→ Periodic fine-tune of disease classifier on hard cases
→ System improves where it's weakest over time
```

---

## Agent Roster: Current 8 → Proposed 12

| # | Agent | Paradigm | Role | Status |
|---|-------|----------|------|--------|
| 1 | WeedDetector | ML | YOLO weed/crop classification | existing |
| 2 | DiseaseClassifier | ML | MobileNetV2 + temperature scaling | enhanced |
| 3 | HealthClassifier | ML | ViT binary healthy/unhealthy + temperature scaling | enhanced |
| 4 | OpenCVEvidence | CV | HSV/edge/contour analysis with spatial mask retention | **enhanced** |
| 5 | AnomalyDetector | CV | PatchCore on DINOv2 embeddings, flags unknown conditions | **new** |
| 6 | GrowthStage | ML | ViT growth-stage classifier, modifies urgency | **new** |
| 7 | WeatherPrior | Physics | Open-Meteo fetch, adjusts priors before ML runs | **new** |
| 8 | WaterBalance | Physics | Penman-Monteith + van Genuchten + forecast | enhanced |
| 9 | PesticideFate | Physics | Gaussian plume + DT50 + wind forecast | enhanced |
| 10 | EcologicalDynamics | Biology | Lotka-Volterra ODE + seasonal baselines | enhanced |
| 11 | LocalVLMReasoner | ML/CV | LLaVA/InternVL on-device, image + text reasoning | **enhanced** |
| 12 | Skeptic | ML | Same local model as VLM (text-only mode), hypothesis generation | enhanced |

**Non-inference agents** (not counted above):
- Controller (EIGControllerAgent) — utility-based action selection
- HumanReviewProxy — escalation when confidence too low

**Pre-processing modules** (not agents, no AG2 envelope):
- DINOv2 backbone — runs once per frame, feeds AnomalyDetector + visual explanation
- Temporal Diff (Sprint 4) — frame differencing, feeds cross-examination
- RAG retriever — FAISS lookup, feeds controller utility adjustment

---

## New Models Required

| Model | Purpose | Size | Runs On |
|-------|---------|------|---------|
| `facebook/dinov2-base` | Feature extraction for anomaly detection + attention maps | ~350MB | CPU/GPU |
| `IDEA-Research/grounding-dino-base` | Open-vocabulary plant detection | ~700MB | GPU preferred |
| PatchCore (anomalib) | Anomaly detection trained on healthy plant patches | ~50MB | CPU |
| Growth stage ViT (fine-tuned) | Seedling/vegetative/flowering/fruiting classification | ~90MB | CPU |
| Temperature scaling parameters | Learned scalar T per existing model | <1KB | N/A |
| `llava-hf/llava-v1.6-mistral-7b-hf` | Local VLM + Skeptic (shared weights) | ~14GB | GPU (8GB+ VRAM) |
| OR `OpenGVLab/InternVL2-2B` | Lighter alternative for VLM + Skeptic | ~4GB | GPU (4GB+) / CPU |
| `sentence-transformers/all-MiniLM-L6-v2` | Embeddings for RAG vector search | ~80MB | CPU |
| `depth-anything/Depth-Anything-V2-Small` | Canopy structure analysis (optional) | ~100MB | CPU/GPU |

**Total new model weight**: ~15-16GB (with LLaVA) or ~5-6GB (with InternVL2-2B)

## External Services

| Service | Purpose | Cost | Type |
|---------|---------|------|------|
| Open-Meteo | Weather current + 7-day forecast | Free, no key | HTTP API |

**Everything else is local:**
- FAISS vector DB (local library, not a service)
- All ML inference (HuggingFace Transformers / llama.cpp)
- All reasoning (local LLM, no OpenAI/Anthropic API)

---

## Implementation Order (Priority)

### Sprint 1: Foundation (Low effort, high impact)

1. **Temperature Scaling** — Learn scalar T per model on calibration set
   - File: `pulse/calibration.py`
   - What: Post-hoc calibration. Collect model outputs on a held-out set, learn T via NLL minimization
   - Impact: All downstream decisions improve when likelihoods are honest
   - Effort: ~50 LOC

2. **OpenCV Evidence Retention** — Keep spatial masks in SegmentationAgent
   - File: `pulse/agents/segmentation.py` (modify), `pulse/visual_explain.py` (new)
   - What: Stop discarding the HSV/edge/contour masks after computing scalars. Store per-plant evidence maps. Add a renderer.
   - Impact: Visual explanations with zero divergence from diagnosis
   - Effort: ~150 LOC

3. **Weather Prior Agent** — One API call adjusts all priors
   - File: `pulse/agents/weather_prior.py` (new)
   - What: Fetch from Open-Meteo, compute conditional priors (drought → water_stress↑, humid+warm → fungal↑)
   - Impact: Context-aware priors before ML runs
   - Effort: ~100 LOC

### Sprint 2: Better Vision (Medium effort, high impact)

4. **DINOv2 Feature Extraction + Attention Maps**
   - File: `pulse/backbone.py` (new)
   - What: Run DINOv2 once per frame. Extract per-patch features (for anomaly detection) and attention maps (for visualization). Does NOT modify existing agents.
   - Impact: Enables anomaly detection + free saliency visualization
   - Effort: ~120 LOC

5. **Anomaly Detector Agent**
   - File: `pulse/agents/anomaly_detector.py` (new)
   - What: PatchCore on DINOv2 embeddings. Trained offline on healthy plant crops. At inference, scores each plant's distance from the healthy distribution.
   - Impact: Catches unknown-unknowns before classifiers force a label
   - Effort: ~150 LOC

6. **Growth Stage Classifier**
   - File: `pulse/agents/growth_stage.py` (new)
   - What: Small ViT classifying seedling/vegetative/flowering/fruiting. Output feeds the controller's urgency multiplier.
   - Impact: Seedling with disease = act immediately. Mature plant = can wait.
   - Effort: ~120 LOC

### Sprint 3: Reasoning & Context (Medium effort, medium-high impact)

7. **Local VLM Reasoner (replaces GPT-4o-mini VLM)**
   - File: `pulse/agents/vlm_reasoner.py` (rewrite)
   - Model: LLaVA-1.6-7B or InternVL2-2B loaded via transformers
   - What: Receives actual crop image + structured prompt. Outputs JSON with per-condition log-likelihoods. No external API.
   - Impact: Visual reasoning fully offline, no API cost, no key dependency
   - Effort: ~200 LOC (model loading, image preprocessing, structured output parsing)

8. **Skeptic on Local LLM (replaces GPT-4o-mini Skeptic)**
   - File: `pulse/agents/skeptic.py` (rewrite)
   - What: Uses same loaded model as VLM (text-only mode). Generates structured hypotheses from cross-exam data.
   - Shared inference: VLM and Skeptic share one model instance in memory (14GB loaded once, not twice)
   - Impact: Better resolution of ambiguous cases, zero API cost
   - Effort: ~100 LOC

9. **RAG over Agronomic Literature**
   - File: `pulse/rag/` (new directory), `pulse/agents/controller.py` (modify)
   - What: FAISS index over extension service bulletins + IPM guidelines. Embedding model: sentence-transformers/all-MiniLM-L6-v2. Controller queries with (disease, crop, region, season) → gets treatment context.
   - Impact: Regional knowledge adjusts chemical choice + treatment thresholds
   - Effort: ~300 LOC + data preparation

### Sprint 4: Temporal & Advanced (Higher effort)

10. **Grounding DINO Integration**
    - File: `pulse/detection.py` (modify)
    - What: Run alongside YOLO. Text prompts like "wilted plant" or "leaf with holes" catch things YOLO was never trained on. NMS dedup merges both detection sets.
    - Impact: Open-vocab detection for novel threats
    - Effort: ~150 LOC

11. **Temporal Diff Module**
    - File: `pulse/temporal.py` (new)
    - What: Given current + previous frame(s), compute per-plant change scores via frame differencing or optical flow. Feeds cross-examination as an urgency signal.
    - Prerequisite: Multi-frame data pipeline (drone revisit imagery)
    - Impact: "Changed since last scan" is a strong disease-progression signal
    - Effort: ~200 LOC + data pipeline work

12. **Active Learning Loop**
    - File: `pulse/active_learning.py` (new)
    - What: Log plants where human_review triggered, KL > 3.0, or anomaly > 3σ. Store crops + eventual labels. Periodically fine-tune disease classifier on hard cases.
    - Impact: System improves over time on its weakest cases
    - Effort: ~250 LOC + labeling infrastructure

---

## Key Architectural Principles

1. **Explanation === Evidence**: Every visual overlay pixel comes from the same computation that produced the log-likelihood. No post-hoc independent analysis that could diverge from the diagnosis.

2. **Calibration Before Fusion**: Temperature-scaled outputs ensure no single model dominates the shared posterior Ψ just because it produces overconfident scores.

3. **Priors From Physics, Not Guesses**: Weather and soil signals adjust priors through principled Bayesian mechanisms (conditional probabilities from environmental data) — not arbitrary heuristic rules.

4. **Multi-Turn Only When Needed**: The skeptic-VLM debate loop only fires on disputed plants (KL > 1.5), keeping expensive local VLM inference proportional to case difficulty.

5. **Same AG2 Envelope**: All new agents use the same `ChannelAgent` (ConversableAgent + llm_config=False + register_reply) pattern. The protocol is uniform; the paradigm is plural. The LocalVLM and Skeptic use AssistantAgent with local llm_config pointing to a local inference server.

6. **Protocol Firewall Preserved**: No prose fields in inter-agent messages. New agents emit `ConstraintMessage` or store artifacts via side-channels (evidence maps, attention maps). The firewall test still passes.

7. **Fully Offline**: No external LLM APIs. The system runs air-gapped after model weights are downloaded. Only optional external call is Open-Meteo for weather (can be replaced with local weather station data).

---

## Performance Expectations

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Chemical reduction vs blanket spray | 91% | 94%+ | Better calibration + anomaly detection reduce false positives |
| False negative rate (missed disease) | Unknown | < 5% | Anomaly detector catches what classifiers miss |
| Explanation fidelity | N/A (no overlay) | 100% | Guaranteed by architecture (same computation) |
| Unknown condition detection | 0% (forced into labels) | > 80% flagged | PatchCore anomaly scoring |
| API cost per frame | ~$0.01 (GPT-4o-mini) | $0.00 | Fully local inference |
| Latency (Sprint 1-2, no VLM) | ~8s | ~7s | Calibration doesn't add latency; DINOv2 adds ~1s but anomaly replaces some classifier work |
| Latency (Sprint 3+, with local VLM on disputes) | ~8s | ~12-15s on disputed frames | Local VLM inference is slow (~3-5s per plant); only runs on disputed plants |
| Latency (undisputed frames, no VLM triggered) | ~8s | ~7-8s | VLM doesn't run if no disputes — majority of frames |
| GPU VRAM requirement | ~2GB (YOLO + MobileNet + ViT) | ~10-16GB (+ DINOv2 + VLM) | InternVL2-2B reduces to ~8GB total |

**Latency note**: The local VLM is slower than GPT-4o-mini API calls. However, it only fires on disputed plants (typically <10% of frames). Undisputed frames process at roughly the same speed as today. The tradeoff is: zero API cost + offline capability + data privacy vs. higher latency on hard cases.

---

## Dependency Graph

```
Sprint 1 (no dependencies, can start immediately):
  [1] Temperature Scaling
  [2] OpenCV Evidence Retention
  [3] Weather Prior Agent

Sprint 2 (depends on Sprint 1 for calibration):
  [4] DINOv2 Backbone ← needed by [5]
  [5] Anomaly Detector ← depends on [4]
  [6] Growth Stage ← independent

Sprint 3 (depends on Sprint 2 for full pipeline):
  [7] Local VLM ← depends on [4] for attention maps in prompt
  [8] Skeptic on Local LLM ← depends on [7] (shared model)
  [9] RAG ← independent of [7,8], depends on [6] for growth context

Sprint 4 (depends on Sprint 1-3 being stable):
  [10] Grounding DINO ← independent
  [11] Temporal Diff ← independent, but needs multi-frame data
  [12] Active Learning ← depends on [5] for anomaly scores
```
