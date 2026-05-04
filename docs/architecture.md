# Architecture

Pesto threads four scientific paradigms through one AG2 orchestration graph.
Each paradigm contributes a constraint on the same per-plant posterior; the
controller picks the argmax-utility action subject to physics and ecology
vetoes.

## Pipeline phases

```
Frame Input
    │
    ▼
Phase 0 — Pre-Inference Context
    │   [Weather Prior]  Open-Meteo → adjusts priors before ML runs
    │
    ▼
Phase 1 — Constraint Emission (parallel)
    │   ML:      [Weed Detector] [Disease Classifier] [Health Classifier]
    │   CV:      [Segmentation + Evidence Maps] [Anomaly Detector]
    │   Physics: [Water Balance]
    │   ML:      [Growth Stage]
    │
    ▼
Phase 2 — Cross-Examination
    │   Weighted KL divergence across paradigm pairs
    │   ML×ML = 1.0   ML×Bio = 1.5   ML×Anomaly = 2.0
    │   KL > 1.5 → plant is "disputed"
    │
    ▼ (only if disputed plants exist)
Phase 3 — Multi-Turn Debate (max 3 turns)
    │   [Skeptic] ──hypotheses──▶ [VLM Reasoner]
    │       ▲                          │
    │       └──── entropy check ───────┘
    │
    ▼
Phase 4 — Physics + Ecology (per plant × action)
    │   [Pesticide Fate]  Gaussian plume drift + DT50 degradation
    │   [Eco Dynamics]    Lotka-Volterra 30-day population forecast
    │   drift > 0.4 ppm    → VETO spray
    │   predator drop > 50% → VETO chemical
    │
    ▼
Phase 5 — Controller (argmax utility)
    │   U = yield_protected − chem_cost − 0.5×drift − 0.4×eco_cost
    │
    ▼
Output — Per-Plant Actions
    [laser zap] [fungicide] [irrigate] [fertilize] [review] [no action]
```

Phases 0–4 produce a `FieldLatentState` (the shared posterior over
`CONDITION_LABELS`); phase 5 maps that to an `ActionMessage` per plant
over `INTERVENTION_TYPES`.

## Why each paradigm

| Paradigm | Failure mode it fixes |
|----------|------------------------|
| **ML** | Pattern recognition over pixels — fast, but overconfident on unfamiliar inputs |
| **CV** | Physics-of-light feature extraction with spatial evidence — anchors ML claims |
| **Continuum physics** | Resolves "wilting from disease vs. water stress" — same image, different cause |
| **Population biology** | Catches the "pesticide treadmill" — predator collapse causing worse rebound |

Mixing paradigms via the cross-examination KL is what lets the system veto
a high-confidence ML decision when the physics or ecology disagrees.

## Inter-agent protocol

Every payload exchanged inside the inference loop is a typed dataclass in
`pesto.messages`:

* `ConstraintMessage` — one agent's per-plant log-likelihood vector
* `CrossExamMessage` — pairwise KL diagnostic between two agents
* `HypothesisMessage` — skeptic's alternative explanation
* `InterventionAssessmentMessage` — physics agent's hazard score per action
* `TrajectoryMessage` — ecology agent's 30-day population forecast
* `ActionMessage` — controller's final per-plant pick

Free-form prose fields (`text`, `message`, `prose`, `explanation`,
`commentary`, `description`, `narrative`, `reasoning`) are forbidden by
`tests/test_protocol_firewall.py`. The LLM agents (Skeptic, VLM Reasoner)
emit only typed tool calls — the LLM never speaks prose into the loop.

## AG2 idioms in use

| Idiom | Where |
|-------|-------|
| `ConversableAgent` + `llm_config=False` + `register_reply` | `pesto/agents/base.py` (and every code-only agent) |
| `register_function` with `Annotated` typed args | `pesto/agents/skeptic.py`, `pesto/agents/vlm_reasoner.py` |
| `GroupChat` with custom `speaker_selection_method` | `pesto/cross_exam_groupchat.py` |
| `register_nested_chats` | `pesto/agents/controller.py` |
| `register_hand_off` + `OnCondition` + `AfterWork` | `pesto/runtime.py::wire_swarm_pipeline` |
| `UserProxyAgent` with conditional `human_input_mode` | `pesto/agents/human_review.py` |
| `initiate_swarm_chat` | `pesto/captain.py::PestoCaptain.run_swarm_inference` |

## File layout

```
pesto/                  Python package (SDK + 12 agents + orchestration)
  __init__.py           Public re-exports (Pipeline, PipelineConfig, …)
  sdk.py                Pipeline class + config helpers
  registry.py           Agent + intervention registries; @register_agent
  cli.py                `pesto` console-script entry point
  captain.py            PestoCaptain orchestrator
  runtime.py            Swarm wiring + sequential fallback pipeline
  cross_exam.py         Weighted cross-examination
  cross_exam_groupchat.py  GroupChat speaker selection
  detection.py          YOLO + Grounding DINO + NMS merge
  latent.py             Shared posterior (FieldLatentState)
  llm_config.py         Anthropic / OpenAI provider configuration
  local_model.py        Local VLM backend (LLaVA / InternVL2)
  messages.py           Typed inter-agent protocol
  backbone.py           DINOv2 feature extraction
  calibration.py        Temperature scaling
  active_learning.py    Hard-case queuing
  visual_explain.py     Evidence overlay renderer
  temporal.py           Frame differencing + optical flow
  agents/               Twelve agent implementations
  biology/              Lotka-Volterra ODE
  biophysics/           Penman-Monteith + van Genuchten
  physics/              Gaussian plume drift model
  rag/                  FAISS + sentence-transformers RAG
  dashboard/            FastAPI + WebSocket + JS frontend

tests/                  Unit + integration tests
scripts/                Model download + demo runner
docs/                   This documentation
data/                   Chemistry tables, demo imagery, RAG index (gitignored)
```
