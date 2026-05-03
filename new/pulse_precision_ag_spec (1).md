# Pulse — Precision Agriculture Multi-Model Inference Engine

> **Build prompt for Claude Code.** Paste this entire file as the initial prompt in a fresh repo.
> Then work through the phases in §13 one at a time, in separate Claude Code sessions.
> The judge of this hackathon may be the founder of AG2. The orchestration matters.

---

## 0. AGENT OPERATING RULES (DO NOT VIOLATE)

You are Claude Code. You are building **Pulse**, a multi-agent inference system in **AG2** (the `autogen` Python package — formerly AutoGen, now AG2) that performs joint Bayesian decision-making across multiple specialist models for precision agriculture: per-plant intervention recommendations from drone/rover imagery.

**Hard rules. Re-read these at the start of every phase.**

1. **NO TEXT BETWEEN AGENTS IN THE INFERENCE LOOP.** Inter-agent payloads use the structured dataclasses in §6. The fields `text`, `message`, `prose`, `explanation`, `commentary`, `description`, `narrative`, `reasoning` are FORBIDDEN in any inter-agent payload. The LLM-backed Skeptic and Narrator agents communicate through *registered tools with typed arguments* — never through free-form prose.

2. **EVERY MODEL AGENT EMITS A LIKELIHOOD.** Not a label. Not a confidence score. A `ConstraintMessage` with calibrated per-class log-likelihoods for the relevant pixels/plants. If you find yourself writing `agent.predict() -> dict` returning class names, stop and refactor.

3. **USE REAL LIBRARIES, REAL MODELS, REAL DATA.** No mocks except where explicitly bailed-out per §14. The pinned versions in §7 are verified compatible. All models referenced in §10 are real and pip-/HF-installable.

4. **TESTS BEFORE INTEGRATION.** Every phase has a deliverable test that must pass before moving on. The first test (§6.4) is a static schema check that mechanically prevents the architecture from collapsing into "LLMs in a trench coat."

5. **AG2 IDIOMS ARE LOAD-BEARING — USE THEM CORRECTLY.** Use `ConversableAgent` with `llm_config=False` + `register_reply` for code agents (this includes the ML agents AND the three scientific-paradigm agents — physics, biophysics, ecology — same envelope, three different inference paradigms); `AssistantAgent` with `register_function` (typed `Annotated` args) for LLM agents; `GroupChat` with custom `speaker_selection_method` for cross-examination; `register_nested_chats` for parallel sub-conversations; Swarm `register_hand_off` with `OnCondition` and `AFTER_WORK` for the main pipeline (the pipeline must include `physics_phase` AND `ecology_phase` between cross-examination and the controller); `UserProxyAgent` with conditional `human_input_mode` for human review; `initiate_swarm_chat` as the entry point. The judge will recognize idiomatic AG2 — write it idiomatically. **The cross-paradigm flex (ML + continuum physics + biophysics + population biology in the same orchestration graph) is the headline.**

6. **WHEN BLOCKED, READ §14 BEFORE ASKING.** A pre-authorized bailout exists for every common failure mode. Use it, document it in the README, continue.

---

## 1. WHAT YOU ARE BUILDING

A working hackathon demo + MVP that takes a field image (drone or ground-level), runs **five ML diagnosis agents, one physics-based pesticide-fate agent, one biophysics water-balance agent, and one ecological-dynamics agent** in parallel as AG2 agents, performs cross-paradigm Bayesian fusion, and outputs **per-plant intervention recommendations with auditable provenance** — not blanket spray decisions, not even simple per-plant spray decisions, but interventions that are physically simulated for off-target impact, biophysically validated against plant water status, and ecologically projected for downstream pest-population effects before being recommended.

**The scientific/practical claim (three-part):**

*Part one — diagnosis is cross-modal:*

A single computer-vision model cannot reliably distinguish:
- Disease from nutrient stress (both cause leaf yellowing/spotting)
- Young weeds from young crop (morphology overlaps in early growth)
- Water stress from disease (both show wilting and color shift)
- Pest damage from environmental damage (both cause leaf perforations)

Each ML model has different failure modes. Current "AI ag" pipelines pick one model and rely on a confidence threshold. The result: blanket interventions, high false positives, and farmer distrust.

*Part two — prescription requires physics:*

Even *with* a perfect diagnosis ("this is a weed"), recommending a spray application is irresponsible without modeling what happens after the spray nozzle opens. Pesticide drift, soil residue persistence, runoff into watercourses, and concentration on neighboring non-target plants are governed by atmospheric and chemical physics — fluid dynamics, first-order kinetics, sorption isotherms. ML cannot predict these. They require mechanistic models.

*Part three — diagnosis itself is biophysical:*

Wilting is the canonical example of ML failure. ML sees "yellow droopy leaves" and reports `disease ≈ water_stress` with high entropy. But the actual answer is determined by physiology, not morphology: given the soil moisture, the atmospheric demand (vapor pressure deficit), and the plant's hydraulic conductance, what *should* the plant's water status be? If the Penman-Monteith / soil-plant-atmosphere model says the plant is well-supplied with water, the wilting is disease. If the model says the plant is in genuine water deficit, it's drought. **The Bayesian prior from physics resolves the ML ambiguity.**

*Part four — the chemical treadmill is an ecological problem:*

The reason single-spray decisions don't generalize is that today's spray changes tomorrow's pest pressure. Killing aphids with a broad-spectrum insecticide also kills the parasitic wasps and ladybugs that were keeping aphid populations in check. Two weeks later, aphids rebound without their predators and the farmer sprays more. This is the chemical treadmill made mathematical: a Lotka-Volterra-with-toxicity dynamic. **Without modeling beneficial-insect populations, "optimal" interventions are locally optimal and globally destructive.**

The state of the art in commercial precision agriculture is "spray detected weeds." The state of the art in regulatory science is "model fate before approval." The state of the art in agronomy is "monitor IPM thresholds before treatment." **Nobody combines per-plant ML detection with per-application physics simulation, biophysical state estimation, and ecological-dynamics forecasting in real time at the nozzle.** Pulse does.

**Pulse's claims:**

1. *The cross-model disagreement structure between ML agents is itself diagnostic.* When detection models agree, intervene with high confidence. When they disagree, the *pattern* of disagreement points to ground truth or warrants human review.
2. *The cross-paradigm disagreement between ML and physics is decision-changing.* When ML says "spray here" and the pesticide-fate model says "this spray puts off-target residue on three healthy plants and reaches the ditch in 4 hours," the recommendation flips: laser-zap, or use a different chemistry, or no action at all. **The physics agent is allowed to veto interventions the ML agents recommend.**
3. *The cross-paradigm prior from biophysics resolves diagnostic ambiguity.* When ML cannot distinguish water stress from disease, the soil-plant-atmosphere model emits a Bayesian prior on water status from physical first principles, breaking the tie.
4. *The ecological-dynamics agent makes intervention selection long-horizon.* It computes the expected pest-population trajectory over the next 30 days under each candidate intervention. A spray that locally maximizes weed kill but reduces beneficial-predator population by 60% loses to a laser-zap that preserves the predators. **This is the chemical treadmill, modeled and avoided.**

**The architectural claim:**

The eight agents represent **three fundamentally different inference paradigms** in one orchestration graph: machine learning (statistical pattern recognition), continuum physics (atmospheric and chemical mechanistic models), and population biology (ecological dynamics). They are not interchangeable. They are not redundant. The architecture is built around their genuine epistemological difference. This is what makes Pulse not "an ensemble" but a multi-paradigm inference system.

**The demo claim:**

A judge sees a real field image. ML agents commit to per-plant condition. The water-balance agent posts a biophysical prior on water status. The physics agent simulates each candidate intervention's drift cone. The ecology agent forecasts the next-30-day pest and predator populations under each candidate intervention. Pulse outputs per-plant actions: "row 3 col 7: weed confirmed by 4 of 5 ML agents, water-balance prior consistent with weed metabolism — but spray would deposit 0.18 ppm on row 3 col 6 (healthy crop) given current 4.2 m/s NW wind AND would reduce parasitic-wasp population by 47% over 14 days, increasing predicted aphid pressure → recommend LASER-ZAP instead. Row 4 col 12: ML disease/water disagreement, water-balance prior says soil moisture adequate → it's disease, not drought." The headline metric: 91% chemical reduction vs. blanket spraying, of which roughly 50% comes from precision targeting, 25% from physics vetoes, and 16% from ecological-cost-aware substitutions.

---

## 2. THE EIGHT AGENTS (five ML + one physics + one biophysics + one ecology)

### 2.1 The five ML diagnosis agents (paradigm: machine learning)

| Agent | Model | What It Sees | Failure Mode |
|---|---|---|---|
| `WeedDetectorAgent` | `foduucom/plant-leaf-detection-and-classification` (YOLOv8s, 46 species + weed class) | Bounding boxes around weeds vs. crops | Confuses young crops with weeds at seedling stage |
| `DiseaseClassifierAgent` | `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` (38 disease classes, PlantVillage) | Per-leaf disease classification | Confuses nutrient stress (mottling) with mild disease |
| `SegmentationAgent` | `facebook/sam-vit-base` (Segment Anything) | Per-plant instance segmentation | Generic — doesn't know plant-specific structure |
| `HealthClassifierAgent` | `Diginsa/Plant-Disease-Detection-Project` (ViT, distilled to be fast) | Healthy vs. unhealthy binary + severity | Coarse — can't distinguish causes |
| `VLMReasonerAgent` | LLM-backed (`AssistantAgent`) using a registered `analyze_disagreement_region` tool that crops the disputed region and runs a vision-capable model on it | Targeted cross-examination of disputed regions | Slow, used only when other 4 disagree |

These five emit `ConstraintMessage` payloads about plant **condition** — what the plant is/has, inferred from images.

### 2.2 The physics agent — `PesticideFateAgent` (paradigm: continuum physics)

| Agent | Model | What It Sees | Failure Mode |
|---|---|---|---|
| `PesticideFateAgent` | Gaussian-plume drift + first-order degradation (DT50) + simplified soil sorption — based on EPA PRZM/PWC physics | Candidate spray applications + meteorology + chemistry properties | Fails when wind is highly turbulent (Gaussian plume assumes steady flow) or near complex topography |

This emits `InterventionAssessmentMessage` payloads about intervention **outcome** — what happens if we apply this. It does NOT see plant condition. It sees a hypothetical action and predicts its physical consequences.

### 2.3 The biophysics agent — `WaterBalanceAgent` (paradigm: soil-plant-atmosphere physics)

| Agent | Model | What It Sees | Failure Mode |
|---|---|---|---|
| `WaterBalanceAgent` | Penman-Monteith evapotranspiration + van Genuchten soil retention + plant hydraulic conductance | Soil moisture + air temperature + humidity + radiation + crop type | Fails when soil moisture sensor is missing (must fall back to assumed field-capacity prior) |

This emits a `ConstraintMessage` *just like the ML agents* — but its likelihood vector is over plant **condition**, derived from physical first principles. It's not a sixth ML model; it's a Bayesian prior on what the plant's water status *physically must be*. **When the ML disease classifier and ML water-stress classifier disagree, this agent breaks the tie not by voting but by computing.**

### 2.4 The ecology agent — `EcologicalDynamicsAgent` (paradigm: population biology)

| Agent | Model | What It Sees | Failure Mode |
|---|---|---|---|
| `EcologicalDynamicsAgent` | Lotka-Volterra with stage structure + pesticide-toxicity terms + crop-pest-predator triads | Pest detection counts + pesticide application history + chemistry mode-of-action | Fails when predator/prey species composition is unknown (must use generic priors) |

For each candidate intervention, this agent simulates the next 30 days of pest population AND beneficial-insect population under that intervention. It emits `TrajectoryMessage` payloads with predicted population trajectories. **Today's spray that kills the parasitic wasps that were keeping aphids in check loses utility against tomorrow's predicted aphid explosion.** This is the chemical treadmill made mathematical.

### 2.5 Why the heterogeneity matters

Each of the five ML agents uses `ConversableAgent` with `llm_config=False` and `register_reply` (except VLMReasonerAgent which uses `AssistantAgent` with typed tools). The physics, biophysics, and ecology agents use the **same** `ConversableAgent` + `llm_config=False` pattern but their *internal computation* is ODE integration / radiative transfer / population dynamics, not neural inference. **Same protocol envelope. Three completely different inference paradigms.** This is the cross-paradigm flex AG2 enables — the inference protocol is uniform but the inference machinery is plural.

The judge sees one orchestration graph hosting:
- **Statistical inference** (ML model agents)
- **Continuum mechanics** (Gaussian plume, DT50)
- **Soil-plant-atmosphere physics** (Penman-Monteith)
- **Population biology** (Lotka-Volterra with toxicity)

All passing structured constraints. None passing prose. That is what AG2 was built for.

---

## 3. THE FOUR ARCHITECTURAL PRIMITIVES

These are non-negotiable. Every code module must trace to one of these.

### 3.1 Shared Latent State Ψ
Per-plant posterior over plant condition: `{healthy, weed, disease, nutrient_stress, water_stress, ambiguous}`. Plus a per-action posterior over intervention feasibility — populated by the physics agent. Plus per-intervention forecast trajectories — populated by the ecology agent. Lives in `pulse/latent.py`.

### 3.2 Constraint Emitters (three paradigms)

*Diagnosis constraint emitters (the five ML agents + water balance):* Each emits a `ConstraintMessage` containing per-plant log-likelihoods over the condition labels. The water-balance agent is on this side: it emits a likelihood-vector just like the ML agents, but its likelihoods come from physical first-principles (Penman-Monteith) rather than image features. **It looks like an ML agent to the protocol; it is a physics agent in its computation.**

*Intervention assessment emitter (the physics agent):* Emits an `InterventionAssessmentMessage` for each candidate (plant, action) pair, containing predicted off-target concentration field, persistence trajectory, and a per-action hazard score.

*Trajectory emitter (the ecology agent):* Emits a `TrajectoryMessage` for each candidate intervention, containing predicted pest-population and predator-population time series over a 30-day horizon under that intervention.

All three flavors live in `pulse/agents/`. They are protocol-compatible (all use `ConversableAgent`) but semantically distinct (constraints, assessments, trajectories).

### 3.3 Cross-Examination Orchestrator
Three layers:
- *ML-vs-ML* cross-examination: pairwise disagreement between the five ML agents per plant.
- *ML-vs-Biophysics* cross-examination: when ML disease and water-stress agents disagree, the water-balance agent's biophysical prior breaks the tie. The agent emits a likelihood; importance reweighting is uniform across emitters.
- *ML-vs-Physics* cross-examination: when the ML agents recommend a spray intervention, the physics agent assesses it. If physics hazard exceeds threshold, the proposed intervention is *flagged for substitution*.

Lives in `pulse/cross_exam.py`. **All three layers use the same KL-divergence math; only the emitter sources differ.**

### 3.4 EIG + Utility Action Selector with Long-Horizon Cost
Given the per-plant condition posterior (informed by ML and water balance), the per-action physical hazard (from physics), and the per-action 30-day population trajectory (from ecology), picks the action that maximizes expected utility:

```
U(action) = E[yield_protected_today]
          - chemical_cost
          - physics_hazard
          - ecological_future_cost      # NEW: from ecology agent
```

where `ecological_future_cost` is the predicted increase in pest pressure 14-30 days out resulting from depleted predator populations. Action types:
- `intervene_targeted_spray` — when condition-posterior + low-hazard + low-ecological-cost agree
- `intervene_laser_zap` — when spray would deplete beneficial insects OR exceed physics threshold
- `intervene_targeted_irrigation` — when water-balance prior says water deficit drove the symptoms
- `request_higher_resolution` — when condition ambiguity could be resolved by closer imaging
- `request_vlm_review` — when ML disagreement suggests need for vision-language reasoning
- `defer_human` — when ambiguity cannot be resolved programmatically
- `no_action` — when expected utility of all actions is negative

Lives in `pulse/agents/controller.py`. The utility function explicitly trades off detection confidence against physics hazard against ecological future cost. **This is where the value is created — and where the chemical treadmill is broken in math.**

---

## 4. THE DATASET

Use **CropAndWeed Dataset** (Steininger et al. 2023; available on Zenodo and GitHub) — it has:
- 8,034 field images
- 24 weed species + 7 crop species annotated
- Bounding box and instance segmentation labels
- Diverse lighting / soil / growth-stage conditions

**Backup datasets** (if CropAndWeed download fails):
- PlantVillage (54k leaf images, 38 classes; on Kaggle and HF)
- Sugar Beets 2016 (~12k images with weed/crop pixel masks; from University of Bonn)
- DeepWeeds (17k images, 8 weed species; Australian)

For the demo, you'll use ~20 carefully selected images that span the failure modes.

---

## 5. THE LATENT STATE

```python
# pulse/latent.py
from dataclasses import dataclass, field
import numpy as np

CONDITION_LABELS = [
    "healthy_crop",
    "weed",
    "disease",
    "nutrient_stress",
    "water_stress",
    "pest_damage",
    "ambiguous",
]

INTERVENTION_TYPES = [
    "no_action",          # plant is healthy
    "laser_zap",          # weed at known location
    "targeted_fungicide", # disease confirmed, specific area
    "targeted_irrigation", # water stress, specific zone
    "foliar_nutrient",    # nutrient deficiency, specific plants
    "human_review",       # ambiguous, needs expert eyes
    "rescan_higher_res",  # info-gain says zoom in
]

@dataclass
class PlantInstance:
    """One identified plant in the field image."""
    plant_id: int
    bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    mask: np.ndarray | None = None     # binary mask, when available
    crop_image: np.ndarray | None = None  # cropped pixels for downstream review
    
    # Posterior over condition (log-space). Index aligned with CONDITION_LABELS.
    log_posterior: np.ndarray = field(default_factory=lambda: np.full(len(CONDITION_LABELS), -np.log(len(CONDITION_LABELS))))
    
    # Provenance: which agents have contributed which constraints
    constraint_history: list[str] = field(default_factory=list)
    
    def posterior(self) -> np.ndarray:
        return np.exp(self.log_posterior - np.logaddexp.reduce(self.log_posterior))
    
    def top_k(self, k=2) -> list[tuple[str, float]]:
        p = self.posterior()
        idx = np.argsort(p)[::-1][:k]
        return [(CONDITION_LABELS[i], float(p[i])) for i in idx]
    
    def entropy(self) -> float:
        p = self.posterior()
        return float(-np.sum(p * np.log(p + 1e-12)))


@dataclass
class FieldLatentState:
    """The shared posterior across all plants in the field image."""
    plants: list[PlantInstance] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)  # (height, width) of source field image
    iteration: int = 0
    
    def update_plant(self, plant_id: int, log_likelihood_delta: np.ndarray, source_agent: str) -> None:
        """Importance-reweight a single plant's posterior with a new constraint."""
        plant = next(p for p in self.plants if p.plant_id == plant_id)
        plant.log_posterior = plant.log_posterior + log_likelihood_delta
        plant.constraint_history.append(source_agent)
    
    def disagreement_score(self, plant_id: int, constraints: dict) -> float:
        """Compute pairwise KL divergence among agent posteriors for this plant."""
        # constraints: {agent_name: log_likelihood_delta over CONDITION_LABELS}
        plant_constraints = [c for c in constraints.values() if c is not None]
        if len(plant_constraints) < 2: return 0.0
        score = 0.0; n = 0
        for i in range(len(plant_constraints)):
            for j in range(i+1, len(plant_constraints)):
                pi = self._softmax(plant_constraints[i])
                pj = self._softmax(plant_constraints[j])
                score += np.sum(pi * (np.log(pi + 1e-12) - np.log(pj + 1e-12)))
                score += np.sum(pj * (np.log(pj + 1e-12) - np.log(pi + 1e-12)))
                n += 1
        return float(score / n) if n > 0 else 0.0
    
    @staticmethod
    def _softmax(x):
        x = x - np.max(x); return np.exp(x) / np.sum(np.exp(x))
    
    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "image_shape": list(self.image_shape),
            "plants": [
                {
                    "plant_id": p.plant_id,
                    "bbox": list(p.bbox),
                    "log_posterior": p.log_posterior.tolist(),
                    "constraint_history": p.constraint_history,
                    "top_k": p.top_k(2),
                    "entropy": p.entropy(),
                }
                for p in self.plants
            ],
        }
```

---

## 6. THE MESSAGE PROTOCOL (THE FIREWALL)

```python
# pulse/messages.py
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

@dataclass
class ConstraintMessage:
    """A model agent's per-plant likelihood contribution. NO PROSE."""
    sender: str
    timestamp: float
    iteration: int
    # Map plant_id -> log-likelihood vector over CONDITION_LABELS
    per_plant_log_likelihoods: dict[int, np.ndarray]
    # Per-plant residual / out-of-distribution score for skeptic consumption
    per_plant_residual: dict[int, float]
    # Agent self-confidence per plant
    per_plant_confidence: dict[int, float]
    # Which condition labels this agent is calibrated to discriminate
    labels_discriminated: list[str]
    metadata: dict = field(default_factory=dict)  # numerical only


@dataclass
class CrossExamMessage:
    """Pairwise disagreement metric between agents per plant."""
    evaluator: str
    target: str
    per_plant_disagreement: dict[int, float]  # KL-divergence
    per_plant_diagnostic: dict[int, dict[str, float]]  # which axes the disagreement falls on


@dataclass
class HypothesisMessage:
    """Skeptic's alternative explanation for an asymmetry pattern."""
    sender: str
    timestamp: float
    plant_id: int
    hypothesis_id: str  # e.g. "disease_masquerading_as_nutrient"
    log_posterior: float
    evidence_axes: dict[str, float]  # numerical only


@dataclass
class ActionMessage:
    """Controller's per-plant intervention recommendation."""
    sender: str
    timestamp: float
    plant_id: int
    action_type: str  # one of INTERVENTION_TYPES
    action_params: dict  # e.g. {"intensity_pct": 70} — numerical only
    expected_information_gain: float
    expected_utility: float = 0.0  # NEW: expected utility used for action selection
    physics_hazard_score: float = 0.0  # NEW: from PesticideFateAgent veto
    target_hypothesis: str | None = None  # which hypothesis this would distinguish


@dataclass
class InterventionAssessmentMessage:
    """Physics agent's assessment of a candidate intervention. NO PROSE.

    Emitted per (plant_id, action_type) candidate. The physics agent
    produces ONE of these for each spray-class action proposed by the
    controller. The controller integrates these into its utility calc.
    """
    sender: str
    timestamp: float
    plant_id: int
    action_type: str  # the intervention being assessed
    action_params: dict  # numerical params used in the simulation
    # Predicted off-target deposition: {plant_id: ppm_at_t0}
    off_target_deposition: dict[int, float]
    # Predicted persistence: half-life in soil at application point (days)
    soil_half_life_days: float
    # Time to reach distant water/non-target features (hours; None if no path)
    time_to_offsite_hours: float | None
    # Aggregate hazard score in [0, 1]; >0.5 = recommend substitution
    hazard_score: float
    # Why hazard is high or low — numerical only:
    # {"drift_max_ppm": 0.18, "ditch_arrival_hours": 4.2, "neighbors_at_risk": 3}
    hazard_breakdown: dict[str, float]


@dataclass
class TrajectoryMessage:
    """Ecological-dynamics agent's forecast of population trajectories
    under a candidate intervention. NO PROSE.

    Emitted per (plant_id, action_type) candidate. The ecology agent
    simulates the local pest, predator, and parasitoid populations
    over a 30-day horizon under the candidate action.
    """
    sender: str
    timestamp: float
    plant_id: int
    action_type: str
    action_params: dict
    # Time grid in days (e.g., [0, 1, 2, ..., 30])
    days: list[float]
    # Population trajectories, normalized to t=0 = 1.0
    # Keys are species roles, not species names — paradigm-agnostic
    pest_trajectory: list[float]            # length == len(days)
    predator_trajectory: list[float]        # length == len(days)
    parasitoid_trajectory: list[float]      # length == len(days)
    # Aggregate ecological cost score in [0, 1]; >0.5 = significant disruption
    ecological_cost_score: float
    # Why cost is high or low — numerical only:
    # {"predator_drop_pct_d14": 0.47, "pest_rebound_factor_d30": 2.3,
    #  "parasitoid_drop_pct_d14": 0.61}
    cost_breakdown: dict[str, float]
```

### 6.1 The Firewall Test (MUST PASS BEFORE ANY OTHER TEST)

```python
# tests/test_protocol_firewall.py
import dataclasses
from pulse.messages import (
    ConstraintMessage, CrossExamMessage, ActionMessage,
    HypothesisMessage, InterventionAssessmentMessage, TrajectoryMessage,
)

FORBIDDEN_FIELDS = {"text", "message", "prose", "explanation", "commentary",
                    "description", "narrative", "reasoning"}

def test_no_prose_fields_in_messages():
    """Architecture firewall. If this test fails, Pulse has degraded into a chatbot."""
    for cls in [ConstraintMessage, CrossExamMessage, ActionMessage,
                HypothesisMessage, InterventionAssessmentMessage, TrajectoryMessage]:
        field_names = {f.name for f in dataclasses.fields(cls)}
        violations = field_names & FORBIDDEN_FIELDS
        assert not violations, f"{cls.__name__} contains forbidden prose fields: {violations}"


def test_action_params_no_strings():
    """Action parameters must be numerical."""
    import numpy as np
    msg = ActionMessage(
        sender="controller", timestamp=0.0,
        plant_id=0, action_type="laser_zap",
        action_params={"intensity_pct": 70, "duration_ms": 200},
        expected_information_gain=0.5,
    )
    for v in msg.action_params.values():
        assert isinstance(v, (int, float, np.floating, np.integer)), \
            f"Action param contains non-numeric: {v}"


def test_intervention_assessment_breakdown_no_strings():
    """Hazard breakdown is numerical-only."""
    import numpy as np
    msg = InterventionAssessmentMessage(
        sender="pesticide_fate", timestamp=0.0,
        plant_id=0, action_type="targeted_spray",
        action_params={"volume_ml": 5.0, "concentration_g_l": 360.0},
        off_target_deposition={1: 0.18, 2: 0.04},
        soil_half_life_days=23.0,
        time_to_offsite_hours=4.2,
        hazard_score=0.71,
        hazard_breakdown={"drift_max_ppm": 0.18, "neighbors_at_risk": 3.0,
                          "ditch_arrival_hours": 4.2},
    )
    for v in msg.hazard_breakdown.values():
        assert isinstance(v, (int, float, np.floating, np.integer)), \
            f"hazard_breakdown contains non-numeric: {v}"


def test_trajectory_message_no_strings():
    """Ecological cost breakdown is numerical-only; trajectories are float lists."""
    import numpy as np
    days = list(range(0, 31))
    msg = TrajectoryMessage(
        sender="ecological_dynamics", timestamp=0.0,
        plant_id=0, action_type="targeted_spray",
        action_params={"volume_ml": 5.0},
        days=[float(d) for d in days],
        pest_trajectory=[1.0] * 31,
        predator_trajectory=[1.0] * 31,
        parasitoid_trajectory=[1.0] * 31,
        ecological_cost_score=0.42,
        cost_breakdown={"predator_drop_pct_d14": 0.47, "pest_rebound_factor_d30": 2.3,
                        "parasitoid_drop_pct_d14": 0.61},
    )
    for v in msg.cost_breakdown.values():
        assert isinstance(v, (int, float, np.floating, np.integer)), \
            f"cost_breakdown contains non-numeric: {v}"
    for traj in [msg.pest_trajectory, msg.predator_trajectory, msg.parasitoid_trajectory]:
        for v in traj:
            assert isinstance(v, (int, float, np.floating, np.integer))
```

---

## 7. EXACT DEPENDENCIES

```toml
# pyproject.toml
[project]
name = "pulse"
version = "0.1.0"
requires-python = ">=3.10,<3.13"
dependencies = [
    # Multi-agent runtime
    "ag2[openai]>=0.7.0",
    
    # Vision models
    "torch>=2.2",
    "torchvision>=0.17",
    "transformers>=4.40",
    "ultralytics>=8.1",        # YOLOv8/v11
    "ultralyticsplus>=0.1.0",  # for foduucom HF model loading
    "segment-anything>=1.0",
    
    # Image / array
    "pillow>=10.0",
    "opencv-python-headless>=4.9",
    "numpy>=1.26,<2.0",        # transformers/torch compat
    "scipy>=1.11",
    "scikit-learn>=1.3",
    "scikit-image>=0.22",
    
    # Web stack (frontend)
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "websockets>=12.0",
    "pydantic>=2.6",
    "python-multipart>=0.0.9",  # for file upload
    
    # Data fetching
    "huggingface-hub>=0.22",
    "datasets>=2.18",
    "requests>=2.31",
    
    # Dev
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
```

**Install order:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "numpy<2.0" "scipy>=1.11"
pip install torch torchvision transformers
pip install ag2 fastapi uvicorn websockets pydantic python-multipart
pip install ultralytics ultralyticsplus segment-anything
pip install pillow opencv-python-headless scikit-learn scikit-image
pip install huggingface-hub datasets requests
pip install pytest pytest-asyncio
```

If `ultralyticsplus` fails (the version pinning is finicky), fall back per §14.A.

---

## 8. THE AG2 ORCHESTRATION (IDIOMATIC, LOAD-BEARING)

This is the section the AG2-founder judge will scrutinize. Use AG2 *correctly* and *distinctively*. Eight idioms are required.

### 8.1 Channel agent base class — `ConversableAgent` with `llm_config=False`

```python
# pulse/agents/base.py
from autogen import ConversableAgent
from pulse.messages import ConstraintMessage
from pulse.latent import FieldLatentState
import dataclasses, json
import numpy as np

class ChannelAgent(ConversableAgent):
    """Base for code-only model agents. No LLM in the inference loop."""

    def __init__(self, name: str, **kwargs):
        super().__init__(
            name=name,
            llm_config=False,                 # AG2 idiom 1: explicit opt-out
            human_input_mode="NEVER",
            code_execution_config=False,
            **kwargs,
        )
        # AG2 idiom 2: register reply at position 0, remove other reply funcs
        # so neither LLM, code, nor function defaults fire.
        self.register_reply(
            trigger=[ConversableAgent, None],
            reply_func=self._emit_constraint_reply,
            position=0,
            remove_other_reply_funcs=True,
        )

    def _emit_constraint_reply(self, recipient, messages=None, sender=None, config=None):
        """AG2 reply hook: deserialize state, run inference, serialize result."""
        latest = messages[-1] if messages else {}
        payload = json.loads(latest.get("content", "{}"))
        latent = FieldLatentState.from_dict(payload["latent"])
        image_path = payload.get("image_path")
        constraint = self.emit_constraint(image_path, latent)
        return True, {
            "role": "assistant",
            "name": self.name,
            "content": json.dumps(self._serialize_constraint(constraint)),
        }

    def _serialize_constraint(self, c: ConstraintMessage) -> dict:
        """np.ndarray → list for JSON safety."""
        return {
            "sender": c.sender, "timestamp": c.timestamp, "iteration": c.iteration,
            "per_plant_log_likelihoods": {
                k: v.tolist() for k, v in c.per_plant_log_likelihoods.items()
            },
            "per_plant_residual": c.per_plant_residual,
            "per_plant_confidence": c.per_plant_confidence,
            "labels_discriminated": c.labels_discriminated,
            "metadata": c.metadata,
        }

    def emit_constraint(self, image_path: str, latent: FieldLatentState) -> ConstraintMessage:
        raise NotImplementedError
```

### 8.2 Cross-examination as `GroupChat` with custom `speaker_selection_method`

```python
# pulse/cross_exam_groupchat.py
from autogen import GroupChat
import numpy as np

class CrossExamGroupChat(GroupChat):
    """AG2 idiom 3: speaker_selection_method picks based on Bayesian disagreement."""

    def __init__(self, agents, latent, max_round=8):
        super().__init__(
            agents=agents, messages=[], max_round=max_round,
            speaker_selection_method=self._select_most_contested,
        )
        self.latent = latent
        self._latest_constraints = {}

    def _select_most_contested(self, last_speaker, groupchat):
        """Pick the agent whose constraint would most contest the current consensus."""
        if len(self._latest_constraints) < 2:
            # round-robin until we have data
            idx = (groupchat.agents.index(last_speaker) + 1) % len(groupchat.agents)
            return groupchat.agents[idx]
        # Compute per-agent disagreement-with-consensus
        all_lls = list(self._latest_constraints.values())
        agent_names = list(self._latest_constraints.keys())
        # Aggregate consensus = mean log-likelihood across agents that have spoken
        consensus_per_plant = {}
        for plant_id in all_lls[0].per_plant_log_likelihoods:
            stack = np.stack([
                ll.per_plant_log_likelihoods[plant_id] for ll in all_lls
                if plant_id in ll.per_plant_log_likelihoods
            ])
            consensus_per_plant[plant_id] = stack.mean(axis=0)
        # For each remaining agent, compute predicted KL to consensus
        # (heuristic: pick the agent that has historically disagreed most)
        # For demo simplicity, prioritize agents not yet in consensus
        spoken = set(agent_names)
        remaining = [a for a in groupchat.agents if a.name not in spoken]
        if remaining:
            return remaining[0]
        return groupchat.agents[(groupchat.agents.index(last_speaker) + 1) % len(groupchat.agents)]
```

### 8.3 Skeptic agent — `AssistantAgent` with `register_function` typed tools

```python
# pulse/agents/skeptic.py
from autogen import AssistantAgent, register_function
from pulse.messages import HypothesisMessage
from typing import Annotated
import dataclasses, time

class SkepticAgent(AssistantAgent):
    """AG2 idiom 4: LLM-backed agent whose ONLY output is typed function calls."""

    def __init__(self, llm_config: dict):
        super().__init__(
            name="skeptic",
            llm_config=llm_config,
            system_message=(
                "You analyze cross-examination disagreement patterns between agricultural "
                "model agents. When agents disagree on a plant's condition, you propose "
                "an alternative hypothesis (e.g., 'disease_masquerading_as_nutrient_stress' "
                "when DiseaseAgent and NutrientAgent disagree). You MUST call "
                "propose_alternative_hypothesis with structured arguments. NEVER respond "
                "in prose. NEVER explain. ONLY call the tool."
            ),
        )
        register_function(
            self.propose_alternative_hypothesis,
            caller=self,
            executor=self,
            description=(
                "Propose an alternative hypothesis to explain a cross-agent "
                "disagreement on a single plant."
            ),
        )

    def propose_alternative_hypothesis(
        self,
        plant_id: Annotated[int, "Which plant the hypothesis applies to"],
        hypothesis_id: Annotated[
            str,
            "Short identifier such as 'young_crop_misidentified_as_weed', "
            "'disease_masquerading_as_nutrient', 'water_stress_concurrent_with_pest'"
        ],
        log_posterior: Annotated[
            float, "Log posterior probability of this hypothesis given the disagreement"
        ],
        evidence_axes: Annotated[
            dict[str, float],
            "Numerical features that triggered this hypothesis. "
            "Keys are condition names; values are KL-divergence magnitudes."
        ],
    ) -> dict:
        msg = HypothesisMessage(
            sender="skeptic", timestamp=time.time(),
            plant_id=plant_id, hypothesis_id=hypothesis_id,
            log_posterior=log_posterior, evidence_axes=evidence_axes,
        )
        return dataclasses.asdict(msg)
```

### 8.4 Controller with `register_nested_chats`

```python
# pulse/agents/controller.py
from autogen import ConversableAgent
from pulse.messages import ActionMessage
from pulse.latent import INTERVENTION_TYPES
import json, time

class EIGControllerAgent(ConversableAgent):
    """AG2 idiom 5: register_nested_chats spawns parallel action evaluators."""

    def __init__(self, name="controller"):
        super().__init__(name=name, llm_config=False, human_input_mode="NEVER")
        self._evaluators = self._build_evaluators()

        # Nested chat fires when this agent receives a message
        self.register_nested_chats(
            chat_queue=[
                {
                    "recipient": ev,
                    "message": self._build_evaluation_message,
                    "summary_method": "last_msg",
                    "max_turns": 1,
                }
                for ev in self._evaluators
            ],
            trigger=lambda sender: sender is not self,
        )

    def _build_evaluators(self):
        return [ActionEvaluator(name=f"eval_{a}", action_type=a) for a in INTERVENTION_TYPES]

    def _build_evaluation_message(self, recipient, messages, sender):
        latest = messages[-1] if messages else {}
        return {
            "content": latest.get("content", "{}"),
            "name": self.name,
        }


class ActionEvaluator(ConversableAgent):
    """Evaluates EIG of a single candidate action for a plant."""

    def __init__(self, name, action_type):
        super().__init__(name=name, llm_config=False, human_input_mode="NEVER")
        self._action_type = action_type
        self.register_reply(
            trigger=[ConversableAgent, None],
            reply_func=self._compute_eig,
            position=0, remove_other_reply_funcs=True,
        )

    def _compute_eig(self, recipient, messages, sender, config):
        payload = json.loads(messages[-1]["content"])
        # Compute expected information gain (or expected utility) for this action
        # on the targeted plant. Heuristic version for demo:
        eig = self._heuristic_eig(payload, self._action_type)
        return True, {
            "content": json.dumps({
                "action_type": self._action_type,
                "eig": eig,
                "plant_id": payload.get("plant_id"),
            })
        }

    def _heuristic_eig(self, payload, action_type) -> float:
        import numpy as np
        log_post = np.array(payload.get("log_posterior", []))
        if len(log_post) == 0: return 0.0
        post = np.exp(log_post - np.logaddexp.reduce(log_post))
        entropy = -np.sum(post * np.log(post + 1e-12))
        # Different actions have different EIG profiles
        scores = {
            "laser_zap": post[1] * 0.9,           # weed
            "targeted_fungicide": post[2] * 0.85,  # disease
            "targeted_irrigation": post[4] * 0.85, # water stress
            "foliar_nutrient": post[3] * 0.85,     # nutrient stress
            "human_review": entropy * 0.7,
            "rescan_higher_res": entropy * 0.5,
            "no_action": post[0] * 1.0,            # healthy
        }
        return float(scores.get(action_type, 0.0))
```

### 8.5 Swarm pipeline with `register_hand_off`, `OnCondition`, `AFTER_WORK`

```python
# pulse/runtime.py
from autogen import ConversableAgent
from autogen.agentchat.contrib.swarm_agent import (
    register_hand_off, OnCondition, AfterWork, AFTER_WORK,
    initiate_swarm_chat,
)
import json

# Phase agents — each is a coordinator for one step of the pipeline.
# (Implement each as a ConversableAgent subclass with custom register_reply.)

def wire_swarm_pipeline(constraint_phase, cross_exam_phase, skeptic_phase,
                        physics_phase, ecology_phase, controller_phase,
                        human_review_proxy):
    """AG2 idiom 6: Swarm handoffs with conditional routing.

    Pipeline (eight agents across three paradigms):
      constraint_emission (5 ML agents + WaterBalanceAgent emitting in parallel)
      → cross_exam (multi-source disagreement detection)
      → [if disagreement] skeptic
      → physics_phase (PesticideFateAgent assesses spray candidates)
      → ecology_phase (EcologicalDynamicsAgent forecasts populations for ALL candidates)
      → controller (utility = detection − chem_cost − physics_hazard − eco_cost)
      → [if ambiguous] human review
    """
    register_hand_off(
        agent=constraint_phase,
        hand_to=[
            OnCondition(
                target=cross_exam_phase,
                # All ML agents AND water balance agent must have emitted
                condition=lambda agent, messages: agent._all_constraints_collected(),
            ),
            AfterWork(AFTER_WORK.STAY),
        ],
    )
    register_hand_off(
        agent=cross_exam_phase,
        hand_to=[
            OnCondition(
                target=skeptic_phase,
                condition=lambda agent, messages: agent._max_disagreement() > 1.5,
            ),
            # When no significant ML disagreement, skip skeptic and go to physics
            AfterWork(target=physics_phase),
        ],
    )
    register_hand_off(
        agent=skeptic_phase,
        hand_to=[AfterWork(target=physics_phase)],
    )
    register_hand_off(
        agent=physics_phase,
        hand_to=[
            # Physics always feeds ecology — both inform controller utility
            AfterWork(target=ecology_phase),
        ],
    )
    register_hand_off(
        agent=ecology_phase,
        hand_to=[
            # Ecology always feeds the controller — it's not optional
            AfterWork(target=controller_phase),
        ],
    )
    register_hand_off(
        agent=controller_phase,
        hand_to=[
            OnCondition(
                target=human_review_proxy,
                condition=lambda agent, messages: agent._needs_human()
                                                  and agent._has_capacity_for_review(),
            ),
            AfterWork(AFTER_WORK.TERMINATE),
        ],
    )
```

The pipeline has two cross-paradigm steps inserted between cross-examination and the controller:

- `physics_phase` runs the `PesticideFateAgent` on each spray candidate (continuum physics paradigm)
- `ecology_phase` runs the `EcologicalDynamicsAgent` on every candidate including laser-zap (population biology paradigm)

The controller then has the full assessment table — ML condition posterior, water-balance prior already fused in, physics hazard per spray, ecological cost per intervention — when computing utilities. **This is the architectural moment where three paradigms reconcile into one decision.**

### 8.6 `UserProxyAgent` with conditional `human_input_mode`

```python
# pulse/agents/human_review.py
from autogen import UserProxyAgent

class HumanReviewProxy(UserProxyAgent):
    """AG2 idiom 7: human-in-loop only when system can't resolve."""

    def __init__(self):
        super().__init__(
            name="human_reviewer",
            human_input_mode="TERMINATE",  # asks human, then terminates conversation
            code_execution_config=False,
            llm_config=False,
            is_termination_msg=lambda msg: True,
        )
```

### 8.7 Captain + `initiate_swarm_chat`

```python
# pulse/captain.py
from autogen.agentchat.contrib.swarm_agent import initiate_swarm_chat
from pulse.latent import FieldLatentState
import json

class PulseCaptain:
    """AG2 idiom 8: top-level orchestrator using initiate_swarm_chat."""

    def __init__(self, swarm_agents, channel_agents):
        self.swarm_agents = swarm_agents
        self.channel_agents = channel_agents

    def run_inference(self, image_path: str) -> FieldLatentState:
        latent = self._initialize_latent(image_path)
        chat_result, _, _ = initiate_swarm_chat(
            initial_agent=self.swarm_agents[0],
            messages=[{
                "role": "user",
                "content": json.dumps({
                    "image_path": image_path,
                    "latent": latent.to_dict(),
                }),
            }],
            agents=self.swarm_agents,
            max_rounds=20,
        )
        return self._integrate_chat_result(chat_result, latent)

    def _initialize_latent(self, image_path: str) -> FieldLatentState:
        # First pass: detect plants with SAM/YOLO to populate plant_ids and bboxes
        # Then return a FieldLatentState with uniform priors over conditions
        ...
```

---

## 9. CONCRETE AGENT IMPLEMENTATIONS

### 9.1 `WeedDetectorAgent` (uses YOLO via ultralyticsplus)

```python
# pulse/agents/weed_detector.py
from ultralyticsplus import YOLO
from pulse.agents.base import ChannelAgent
from pulse.messages import ConstraintMessage
from pulse.latent import CONDITION_LABELS
import numpy as np
import time

class WeedDetectorAgent(ChannelAgent):
    def __init__(self):
        super().__init__(name="weed_detector")
        # Real HF model
        self.model = YOLO("foduucom/plant-leaf-detection-and-classification")
        self.model.overrides["conf"] = 0.25
        self.model.overrides["iou"] = 0.45
        self.model.overrides["max_det"] = 100

    def emit_constraint(self, image_path, latent):
        results = self.model.predict(image_path)
        per_plant_ll = {}
        per_plant_resid = {}
        per_plant_conf = {}
        for plant in latent.plants:
            log_lik = np.full(len(CONDITION_LABELS), -10.0)
            # Find detections that overlap this plant's bbox
            best_match = self._best_overlap(plant.bbox, results)
            if best_match is None:
                # No detection => mild evidence for healthy
                log_lik[CONDITION_LABELS.index("healthy_crop")] = 0.0
                per_plant_conf[plant.plant_id] = 0.3
            else:
                cls_name, conf = best_match
                if cls_name == "weed":
                    log_lik[CONDITION_LABELS.index("weed")] = np.log(conf + 1e-6) * 2
                else:
                    log_lik[CONDITION_LABELS.index("healthy_crop")] = np.log(conf + 1e-6)
                per_plant_conf[plant.plant_id] = float(conf)
            per_plant_ll[plant.plant_id] = log_lik
            per_plant_resid[plant.plant_id] = 0.0  # YOLO doesn't give residuals naturally
        return ConstraintMessage(
            sender=self.name, timestamp=time.time(), iteration=latent.iteration,
            per_plant_log_likelihoods=per_plant_ll,
            per_plant_residual=per_plant_resid,
            per_plant_confidence=per_plant_conf,
            labels_discriminated=["weed", "healthy_crop"],
        )

    def _best_overlap(self, bbox, results):
        # IoU-based matching between plant bbox and YOLO detections
        ...
```

### 9.2 `DiseaseClassifierAgent` (uses MobileNetV2 via transformers)

```python
# pulse/agents/disease_classifier.py
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from pulse.agents.base import ChannelAgent
from pulse.messages import ConstraintMessage
from pulse.latent import CONDITION_LABELS
import torch, numpy as np, time

class DiseaseClassifierAgent(ChannelAgent):
    def __init__(self):
        super().__init__(name="disease_classifier")
        self.processor = AutoImageProcessor.from_pretrained(
            "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
        )
        self.model.eval()

    def emit_constraint(self, image_path, latent):
        full_img = Image.open(image_path).convert("RGB")
        per_plant_ll, per_plant_resid, per_plant_conf = {}, {}, {}
        for plant in latent.plants:
            crop = full_img.crop(plant.bbox)
            inputs = self.processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).numpy()
            # Map model's 38-class output to our CONDITION_LABELS
            log_lik = self._map_to_conditions(probs)
            per_plant_ll[plant.plant_id] = log_lik
            # Residual = 1 - max prob (out-of-distribution proxy)
            per_plant_resid[plant.plant_id] = float(1 - probs.max())
            per_plant_conf[plant.plant_id] = float(probs.max())
        return ConstraintMessage(
            sender=self.name, timestamp=time.time(), iteration=latent.iteration,
            per_plant_log_likelihoods=per_plant_ll,
            per_plant_residual=per_plant_resid,
            per_plant_confidence=per_plant_conf,
            labels_discriminated=["disease", "healthy_crop", "nutrient_stress"],
        )

    def _map_to_conditions(self, probs_38):
        """Map 38-class PlantVillage output to our 7 CONDITION_LABELS."""
        # The 38 classes include "___healthy" suffix patterns and disease names
        # Build a hard-coded mapping at module load
        # Returns log-likelihoods for each of CONDITION_LABELS
        ...
```

(Implement `SegmentationAgent` using SAM, `HealthClassifierAgent` using ViT, and `VLMReasonerAgent` using a local LLM with vision tool similarly. Patterns are identical.)

### 9.6 `PesticideFateAgent` — the physics agent

This is the cross-paradigm agent. It does not classify plants. It receives a candidate intervention and predicts physical consequences via three coupled mechanistic models:

1. **Gaussian-plume drift model** — atmospheric dispersion of spray droplets
2. **First-order degradation kinetics** — chemical persistence in soil
3. **Off-target deposition map** — convolution of plume model with non-target plant locations

The math is established (Pasquill-Gifford stability classes, EPA PRZM/PWC frameworks). The novelty is making it an AG2 agent that vetoes ML-recommended interventions in real time.

#### 9.6.1 Pesticide chemistry data

Maintain `data/pesticide_chemistry.json` with tabulated values for ~10 common pesticides:

```json
{
  "glyphosate": {
    "soil_dt50_days": 23.0,
    "vapor_pressure_pa": 1.31e-7,
    "henry_atm_m3_mol": 1.4e-12,
    "log_kow": -3.4,
    "soil_koc_l_kg": 21698.0,
    "label": "non-selective herbicide",
    "non_target_threshold_ppm": 0.01
  },
  "2_4_d": {
    "soil_dt50_days": 10.0,
    "vapor_pressure_pa": 1.4e-5,
    "henry_atm_m3_mol": 5.6e-10,
    "log_kow": 2.81,
    "soil_koc_l_kg": 88.4,
    "label": "selective herbicide (broadleaf)",
    "non_target_threshold_ppm": 0.005
  },
  "atrazine": {
    "soil_dt50_days": 75.0,
    "vapor_pressure_pa": 3.85e-5,
    "henry_atm_m3_mol": 2.6e-9,
    "log_kow": 2.61,
    "soil_koc_l_kg": 100.0,
    "label": "triazine herbicide",
    "non_target_threshold_ppm": 0.003
  },
  "chlorpyrifos": {
    "soil_dt50_days": 30.0,
    "vapor_pressure_pa": 0.00227,
    "henry_atm_m3_mol": 4.16e-6,
    "log_kow": 4.7,
    "soil_koc_l_kg": 6070.0,
    "label": "organophosphate insecticide",
    "non_target_threshold_ppm": 0.0001
  }
}
```

Source values from EPA Pesticide Properties DataBase (PPDB) and PAN Pesticide Database. Cite the data source in the README.

#### 9.6.2 The Gaussian-plume drift model

```python
# pulse/physics/drift.py
"""Gaussian-plume short-range drift model for ground-level spray applications.

Reference: Pasquill stability classes; ISO 22866 spray drift methodology.
Simplified for low-altitude (~1m) ground-based spray applications, omitting
buoyancy/inversion effects valid only for stack emissions.
"""
import numpy as np

# Pasquill-Gifford dispersion coefficients (open-country, daytime)
# σ_y = a_y * x^b_y; σ_z = a_z * x^b_z, x in meters
# Class D = neutral atmosphere (most common daytime)
PASQUILL_D_OPEN = {"ay": 0.32, "by": 0.78, "az": 0.22, "bz": 0.78}
PASQUILL_C_OPEN = {"ay": 0.36, "by": 0.86, "az": 0.20, "bz": 0.81}  # slightly unstable
PASQUILL_E_OPEN = {"ay": 0.24, "by": 0.74, "az": 0.14, "bz": 0.74}  # slightly stable


def gaussian_plume_concentration(
    x_m: np.ndarray,        # downwind distance from source, m
    y_m: np.ndarray,        # crosswind distance from plume centerline, m
    z_m: np.ndarray,        # height above ground, m
    Q_g_per_s: float,       # source emission rate, g/s
    u_m_per_s: float,       # wind speed at source height, m/s
    H_m: float = 1.0,       # release height, m (typical boom height)
    coefs: dict = None,
) -> np.ndarray:
    """Returns concentration in g/m^3 at each (x, y, z) sample.

    Pasquill-Gifford Gaussian plume:
        C(x,y,z) = Q / (2π u σ_y σ_z)
                   * exp(-y² / 2σ_y²)
                   * [exp(-(z-H)² / 2σ_z²) + exp(-(z+H)² / 2σ_z²)]
    The bracketed term is image source for ground reflection.
    """
    coefs = coefs or PASQUILL_D_OPEN
    # Avoid division by zero: clamp x ≥ 1 m
    x_safe = np.maximum(x_m, 1.0)
    sigma_y = coefs["ay"] * x_safe ** coefs["by"]
    sigma_z = coefs["az"] * x_safe ** coefs["bz"]
    # Centerline factor
    norm = Q_g_per_s / (2 * np.pi * u_m_per_s * sigma_y * sigma_z + 1e-12)
    # Crosswind Gaussian
    crosswind = np.exp(-(y_m ** 2) / (2 * sigma_y ** 2 + 1e-12))
    # Vertical with ground reflection
    vertical = (np.exp(-((z_m - H_m) ** 2) / (2 * sigma_z ** 2 + 1e-12))
                + np.exp(-((z_m + H_m) ** 2) / (2 * sigma_z ** 2 + 1e-12)))
    C = norm * crosswind * vertical
    # Upwind (x < 0) → zero
    C = np.where(x_m < 0, 0.0, C)
    return C


def deposition_at_locations(
    source_xy: tuple[float, float],   # (x, y) of nozzle in field coords (meters)
    target_xy: list[tuple[float, float]],  # list of non-target plant locations
    wind_dir_deg: float,              # direction wind is FROM (meteorological)
    wind_speed_m_s: float,
    application_rate_g_s: float,
    application_duration_s: float,
    deposition_velocity_m_s: float = 0.01,  # typical for fine droplets
) -> dict[int, float]:
    """Predict deposition concentration (ppm equivalent) at each target.

    Translates wind-relative coordinates, evaluates plume at z=1m (canopy),
    integrates over application time, applies deposition velocity.
    """
    # Rotate target coords into wind-aligned frame (x = downwind)
    # Meteorological convention: wind FROM 270° (west) → flowing TO east → x-axis +east
    wind_to_rad = np.deg2rad((wind_dir_deg + 180) % 360)
    cos_t = np.cos(wind_to_rad)
    sin_t = np.sin(wind_to_rad)
    sx, sy = source_xy
    deposition = {}
    for idx, (tx, ty) in enumerate(target_xy):
        dx = tx - sx
        dy = ty - sy
        # Project onto wind-aligned axes
        x_along = dx * cos_t + dy * sin_t      # downwind
        y_cross = -dx * sin_t + dy * cos_t     # crosswind
        # Concentration at canopy height (~1 m)
        C_g_m3 = float(gaussian_plume_concentration(
            np.array([x_along]), np.array([y_cross]), np.array([1.0]),
            Q_g_per_s=application_rate_g_s,
            u_m_per_s=max(wind_speed_m_s, 0.5),  # floor for stability
        )[0])
        # Integrated deposition over application duration
        dep_g_m2 = C_g_m3 * deposition_velocity_m_s * application_duration_s
        # Convert to ppm-equivalent assuming 0.1 kg leaf area exposure
        ppm = (dep_g_m2 * 0.1) * 1e6 / 1000  # rough conversion to ppm
        deposition[idx] = ppm
    return deposition


def soil_persistence_curve(
    initial_conc_g_kg: float,
    dt50_days: float,
    days: np.ndarray,
) -> np.ndarray:
    """First-order kinetics: C(t) = C0 * exp(-ln(2)/DT50 * t)."""
    k = np.log(2) / dt50_days
    return initial_conc_g_kg * np.exp(-k * days)
```

#### 9.6.3 The agent

```python
# pulse/agents/pesticide_fate.py
import json, time
from pathlib import Path
import numpy as np
from autogen import ConversableAgent
from pulse.messages import InterventionAssessmentMessage
from pulse.physics.drift import deposition_at_locations, soil_persistence_curve


CHEMISTRY_DB = json.loads(
    (Path(__file__).parent.parent.parent / "data" / "pesticide_chemistry.json").read_text()
)


class PesticideFateAgent(ConversableAgent):
    """The cross-paradigm agent. Physics, not ML.

    AG2 idiom: same ConversableAgent + llm_config=False pattern as ML agents.
    The protocol envelope is uniform; the inference paradigm is plural.
    """

    def __init__(self, default_chemistry: str = "glyphosate"):
        super().__init__(
            name="pesticide_fate",
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.default_chemistry = default_chemistry
        self.register_reply(
            trigger=[ConversableAgent, None],
            reply_func=self._assess_reply,
            position=0,
            remove_other_reply_funcs=True,
        )

    def _assess_reply(self, recipient, messages=None, sender=None, config=None):
        latest = messages[-1] if messages else {}
        payload = json.loads(latest.get("content", "{}"))
        # Expected payload:
        #   {"plant_id": int, "action_type": str, "action_params": dict,
        #    "field_state": {"plants": [{"plant_id": int, "xy_m": [x, y], "is_target": bool}],
        #                    "wind_dir_deg": float, "wind_speed_m_s": float},
        #    "chemistry": str (optional)}
        plant_id = payload["plant_id"]
        action_type = payload["action_type"]
        action_params = payload.get("action_params", {})
        field = payload["field_state"]
        chemistry = payload.get("chemistry", self.default_chemistry)

        assessment = self.assess_intervention(
            plant_id, action_type, action_params, field, chemistry
        )
        return True, {
            "role": "assistant", "name": self.name,
            "content": json.dumps(self._serialize(assessment)),
        }

    def assess_intervention(
        self,
        plant_id: int,
        action_type: str,
        action_params: dict,
        field_state: dict,
        chemistry_id: str,
    ) -> InterventionAssessmentMessage:
        # Non-spray actions get a trivial all-zero assessment (no chemicals applied)
        if action_type not in ("intervene_targeted_spray", "targeted_fungicide"):
            return InterventionAssessmentMessage(
                sender=self.name, timestamp=time.time(),
                plant_id=plant_id, action_type=action_type,
                action_params=action_params,
                off_target_deposition={},
                soil_half_life_days=0.0,
                time_to_offsite_hours=None,
                hazard_score=0.0,
                hazard_breakdown={"non_chemical_action": 1.0},
            )

        chem = CHEMISTRY_DB[chemistry_id]
        # Find source plant (the one being treated)
        source = next(p for p in field_state["plants"] if p["plant_id"] == plant_id)
        source_xy = tuple(source["xy_m"])
        # Non-target plants (not the one being sprayed)
        non_target = [p for p in field_state["plants"] if p["plant_id"] != plant_id]
        target_xy = [tuple(p["xy_m"]) for p in non_target]

        # Application parameters with sensible defaults
        volume_ml = action_params.get("volume_ml", 5.0)
        conc_g_l = action_params.get("concentration_g_l", 360.0)  # glyphosate-typical
        duration_s = action_params.get("application_duration_s", 0.3)  # quick burst
        active_g = volume_ml * conc_g_l / 1000.0
        rate_g_s = active_g / duration_s

        # Predict deposition at all non-target plants
        dep_idx = deposition_at_locations(
            source_xy=source_xy,
            target_xy=target_xy,
            wind_dir_deg=field_state["wind_dir_deg"],
            wind_speed_m_s=field_state["wind_speed_m_s"],
            application_rate_g_s=rate_g_s,
            application_duration_s=duration_s,
        )
        # Map back to plant_ids
        off_target_dep = {non_target[i]["plant_id"]: float(dep_idx[i])
                          for i in range(len(non_target))}

        # Hazard breakdown
        threshold = chem["non_target_threshold_ppm"]
        deps = list(off_target_dep.values())
        max_dep = max(deps) if deps else 0.0
        n_above = sum(1 for d in deps if d > threshold)

        # Time-to-offsite: if any non-target with known watercourse flag is reached
        # For demo simplicity, assume "offsite" = >5m downwind any non-target with id >= 90
        offsite_hours = None
        for p in non_target:
            if p["plant_id"] >= 90 and off_target_dep[p["plant_id"]] > threshold:
                # Crude: time = horizontal distance / advection speed
                dx = p["xy_m"][0] - source_xy[0]
                dy = p["xy_m"][1] - source_xy[1]
                dist = float(np.hypot(dx, dy))
                offsite_hours = dist / max(field_state["wind_speed_m_s"], 0.5) / 3600
                break

        # Aggregate hazard score, in [0, 1]
        # Three contributions: max-deposition / threshold, count exceeding, persistence
        dep_term = min(max_dep / (threshold + 1e-12), 5.0) / 5.0       # in [0, 1]
        count_term = min(n_above / max(len(non_target), 1), 1.0)       # in [0, 1]
        persist_term = min(chem["soil_dt50_days"] / 100.0, 1.0)        # in [0, 1]
        hazard_score = float(0.5 * dep_term + 0.3 * count_term + 0.2 * persist_term)

        return InterventionAssessmentMessage(
            sender=self.name, timestamp=time.time(),
            plant_id=plant_id, action_type=action_type,
            action_params={"volume_ml": float(volume_ml),
                           "concentration_g_l": float(conc_g_l),
                           "application_duration_s": float(duration_s)},
            off_target_deposition=off_target_dep,
            soil_half_life_days=float(chem["soil_dt50_days"]),
            time_to_offsite_hours=float(offsite_hours) if offsite_hours else None,
            hazard_score=hazard_score,
            hazard_breakdown={
                "drift_max_ppm": float(max_dep),
                "neighbors_at_risk": float(n_above),
                "non_target_threshold_ppm": float(threshold),
                "ditch_arrival_hours": float(offsite_hours) if offsite_hours else -1.0,
                "soil_dt50_days": float(chem["soil_dt50_days"]),
            },
        )

    def _serialize(self, msg: InterventionAssessmentMessage) -> dict:
        import dataclasses
        return dataclasses.asdict(msg)
```

#### 9.6.4 Test for the physics agent

```python
# tests/test_pesticide_fate_agent.py
import pytest
from pulse.agents.pesticide_fate import PesticideFateAgent

def test_no_chemicals_zero_hazard():
    """Laser zap is a non-chemical action — must produce zero hazard."""
    agent = PesticideFateAgent()
    field = {
        "plants": [{"plant_id": 0, "xy_m": [0, 0], "is_target": True},
                   {"plant_id": 1, "xy_m": [0.5, 0], "is_target": False}],
        "wind_dir_deg": 270, "wind_speed_m_s": 4.0,
    }
    a = agent.assess_intervention(
        plant_id=0, action_type="intervene_laser_zap",
        action_params={}, field_state=field, chemistry_id="glyphosate",
    )
    assert a.hazard_score == 0.0
    assert a.off_target_deposition == {}


def test_spray_with_close_neighbor_high_hazard():
    """Spraying glyphosate next to a healthy plant 30cm downwind should flag hazard."""
    agent = PesticideFateAgent()
    field = {
        "plants": [{"plant_id": 0, "xy_m": [0.0, 0.0], "is_target": True},
                   {"plant_id": 1, "xy_m": [0.3, 0.0], "is_target": False}],
        # Wind from west; the neighbor at +x is downwind
        "wind_dir_deg": 270, "wind_speed_m_s": 2.0,
    }
    a = agent.assess_intervention(
        plant_id=0, action_type="intervene_targeted_spray",
        action_params={"volume_ml": 5.0, "concentration_g_l": 360.0,
                       "application_duration_s": 0.3},
        field_state=field, chemistry_id="glyphosate",
    )
    # The downwind neighbor should receive measurable deposition
    assert 1 in a.off_target_deposition
    assert a.off_target_deposition[1] > 0
    # And hazard_score is non-trivial
    assert a.hazard_score > 0.0


def test_upwind_neighbor_zero_deposition():
    """A plant directly upwind of the source must receive ~0 deposition."""
    agent = PesticideFateAgent()
    field = {
        "plants": [{"plant_id": 0, "xy_m": [0.0, 0.0], "is_target": True},
                   {"plant_id": 1, "xy_m": [-1.0, 0.0], "is_target": False}],  # upwind
        "wind_dir_deg": 270, "wind_speed_m_s": 4.0,
    }
    a = agent.assess_intervention(
        plant_id=0, action_type="intervene_targeted_spray",
        action_params={"volume_ml": 5.0, "concentration_g_l": 360.0,
                       "application_duration_s": 0.3},
        field_state=field, chemistry_id="glyphosate",
    )
    assert a.off_target_deposition[1] < 1e-6
```

#### 9.6.5 How the controller integrates physics into utility

Update `EIGControllerAgent` to invoke the physics agent for each spray-class candidate and fold the hazard score into expected utility:

```python
# pulse/agents/controller.py (additions)
def _compute_utility(
    self,
    plant_posterior: np.ndarray,
    action_type: str,
    physics_assessment: InterventionAssessmentMessage | None,
) -> float:
    """U(action) = E[yield_protected] - cost - hazard."""
    p_weed = plant_posterior[CONDITION_LABELS.index("weed")]
    p_disease = plant_posterior[CONDITION_LABELS.index("disease")]
    p_healthy = plant_posterior[CONDITION_LABELS.index("healthy_crop")]
    base = {
        "intervene_targeted_spray":   p_weed * 0.9 - 0.10,  # detection × yield - chem cost
        "targeted_fungicide":         p_disease * 0.85 - 0.12,
        "intervene_laser_zap":        p_weed * 0.85 - 0.05,  # higher labor cost, no chem
        "no_action":                  p_healthy * 1.0,
        "human_review":               -0.05,                  # cheap but non-zero
        "rescan_higher_res":          -0.02,
    }.get(action_type, 0.0)

    # Physics veto: subtract hazard score weighted by 0.5
    if physics_assessment is not None:
        base -= 0.5 * physics_assessment.hazard_score

    return float(base)


def _select_action(self, plant_id, plant_posterior, physics_assessments):
    """Pick action argmax over utility."""
    utilities = {}
    for action in INTERVENTION_TYPES:
        phy = physics_assessments.get(action)
        utilities[action] = self._compute_utility(plant_posterior, action, phy)
    best = max(utilities.items(), key=lambda kv: kv[1])
    return best  # (action_type, utility)
```

The flow:

1. ML agents diagnose plant → condition posterior
2. Controller, for each plant, considers spray actions
3. **For each candidate spray action, controller calls the physics agent** to get an `InterventionAssessmentMessage`
4. Utility for spray = (detection × yield_protected) − chemical_cost − 0.5 × hazard_score
5. Utility for laser = (detection × yield_protected) − labor_cost (no hazard)
6. **When hazard is high, laser wins even though spray detection is more confident**

**This is where the chemical reduction comes from.** Not from better detection. From refusing to spray when physics says the cure is worse than the disease.

### 9.7 `WaterBalanceAgent` — soil-plant-atmosphere physics

This is the **biophysics agent**. It does not look at images. It does not predict spray outcomes. Given soil-moisture and atmospheric conditions, it computes a Bayesian prior over plant water status from the Penman-Monteith equation and van Genuchten soil retention curves.

**The role in the architecture:** when ML disease classifier and ML water-stress observation disagree on a plant, this agent emits a likelihood-vector that *resolves the ambiguity* using physical first principles.

#### 9.7.1 The math

**Penman-Monteith reference evapotranspiration (FAO-56):**

$$
\text{ET}_0 = \frac{0.408 \Delta (R_n - G) + \gamma \frac{900}{T+273} u_2 (e_s - e_a)}{\Delta + \gamma (1 + 0.34 u_2)}
$$

where $\Delta$ is the slope of the saturation vapor-pressure curve, $R_n$ is net radiation, $G$ is soil heat flux, $\gamma$ is the psychrometric constant, $T$ is air temperature, $u_2$ is wind speed at 2m, and $e_s - e_a$ is the vapor-pressure deficit.

**Van Genuchten soil water retention curve:**

$$
\theta(\psi) = \theta_r + \frac{\theta_s - \theta_r}{(1 + (\alpha |\psi|)^n)^{(1 - 1/n)}}
$$

inverted to give plant water potential $\psi$ from soil moisture content $\theta$. Tabulated $\theta_s, \theta_r, \alpha, n$ values exist for every soil texture class (USDA-NRCS database).

**Plant water status indicator:**
- Crop water demand (mm/day) = $\text{ET}_0 \times K_c$ (crop coefficient, tabulated by FAO)
- Soil water supply (mm/day) = function of root-zone moisture and unsaturated hydraulic conductivity
- Stress index $S = 1 - \min(\text{supply}/\text{demand}, 1)$

When $S > 0.4$, plant is in significant water deficit and any wilting/yellowing observed by ML is *physically explained* by drought, not disease.

#### 9.7.2 The implementation

```python
# pulse/biophysics/water_balance.py
"""Soil-plant-atmosphere water balance using FAO-56 Penman-Monteith
and van Genuchten soil retention. Single-step model for current state.
"""
import numpy as np

# Tabulated van Genuchten parameters by soil texture class (USDA)
# theta_r, theta_s, alpha (1/cm), n
SOIL_TEXTURES = {
    "sand":            {"theta_r": 0.045, "theta_s": 0.430, "alpha": 0.145, "n": 2.68},
    "loamy_sand":      {"theta_r": 0.057, "theta_s": 0.410, "alpha": 0.124, "n": 2.28},
    "sandy_loam":      {"theta_r": 0.065, "theta_s": 0.410, "alpha": 0.075, "n": 1.89},
    "loam":            {"theta_r": 0.078, "theta_s": 0.430, "alpha": 0.036, "n": 1.56},
    "silt":            {"theta_r": 0.034, "theta_s": 0.460, "alpha": 0.016, "n": 1.37},
    "silt_loam":       {"theta_r": 0.067, "theta_s": 0.450, "alpha": 0.020, "n": 1.41},
    "clay_loam":       {"theta_r": 0.095, "theta_s": 0.410, "alpha": 0.019, "n": 1.31},
    "clay":            {"theta_r": 0.068, "theta_s": 0.380, "alpha": 0.008, "n": 1.09},
}

# FAO crop coefficients (mid-season Kc) for common crops
CROP_KC = {"corn": 1.20, "soybean": 1.15, "wheat": 1.15, "tomato": 1.15,
           "lettuce": 1.00, "cotton": 1.20, "potato": 1.15, "default": 1.10}


def saturation_vapor_pressure(T_C: float) -> float:
    """Tetens formula. Returns kPa."""
    return 0.6108 * np.exp((17.27 * T_C) / (T_C + 237.3))


def penman_monteith_et0(
    T_C: float, RH_pct: float, u2_m_s: float, R_n_MJ_m2_d: float,
    elevation_m: float = 100.0, G_MJ_m2_d: float = 0.0,
) -> float:
    """FAO-56 reference ET (mm/day)."""
    e_s = saturation_vapor_pressure(T_C)
    e_a = e_s * (RH_pct / 100.0)
    Delta = 4098 * e_s / ((T_C + 237.3) ** 2)  # slope of vapor-pressure curve, kPa/°C
    P = 101.3 * ((293 - 0.0065 * elevation_m) / 293) ** 5.26  # atmospheric pressure, kPa
    gamma = 0.000665 * P  # psychrometric constant, kPa/°C
    num = 0.408 * Delta * (R_n_MJ_m2_d - G_MJ_m2_d) + gamma * (900 / (T_C + 273)) * u2_m_s * (e_s - e_a)
    den = Delta + gamma * (1 + 0.34 * u2_m_s)
    return float(num / den)


def soil_water_potential(theta: float, soil_texture: str) -> float:
    """Inverse van Genuchten: returns psi in kPa (negative = drier)."""
    params = SOIL_TEXTURES[soil_texture]
    theta_r = params["theta_r"]; theta_s = params["theta_s"]
    alpha_inv_cm = params["alpha"]; n = params["n"]
    # Effective saturation Se in [0, 1]
    Se = max(min((theta - theta_r) / (theta_s - theta_r), 0.999), 0.001)
    m = 1 - 1/n
    # |psi| in cm: alpha_inv_cm * |psi| = (Se^(-1/m) - 1)^(1/n)
    psi_cm = (1 / alpha_inv_cm) * ((Se ** (-1/m)) - 1) ** (1/n)
    # Convert cm to kPa: 1 cm water = 0.0981 kPa
    return -float(psi_cm * 0.0981)


def water_stress_index(
    theta: float,                       # current volumetric soil moisture (m3/m3)
    soil_texture: str,                  # one of SOIL_TEXTURES
    crop_type: str,                     # one of CROP_KC
    T_C: float, RH_pct: float, u2_m_s: float, R_n_MJ_m2_d: float,
    rooting_depth_m: float = 0.5,
) -> dict:
    """Returns {'stress_index': S in [0,1], 'demand_mm': float, 'supply_mm': float,
                'soil_psi_kPa': float}. S > 0.4 = significant water deficit."""
    et0 = penman_monteith_et0(T_C, RH_pct, u2_m_s, R_n_MJ_m2_d)
    kc = CROP_KC.get(crop_type, CROP_KC["default"])
    demand_mm = et0 * kc
    psi_kPa = soil_water_potential(theta, soil_texture)
    # Plant water uptake limited by soil potential.
    # Generic: at psi > -33 kPa (field capacity), no stress.
    # At psi < -1500 kPa (wilting point), supply ≈ 0.
    # Linear interpolation between (simplified Feddes function).
    if psi_kPa >= -33:
        supply_factor = 1.0
    elif psi_kPa <= -1500:
        supply_factor = 0.0
    else:
        supply_factor = (psi_kPa - (-1500)) / ((-33) - (-1500))
    supply_mm = demand_mm * supply_factor
    stress_index = 1.0 - min(supply_mm / max(demand_mm, 1e-3), 1.0)
    return {
        "stress_index": float(np.clip(stress_index, 0.0, 1.0)),
        "demand_mm": float(demand_mm),
        "supply_mm": float(supply_mm),
        "soil_psi_kPa": float(psi_kPa),
    }
```

#### 9.7.3 The agent

```python
# pulse/agents/water_balance.py
import json, time
import numpy as np
from autogen import ConversableAgent
from pulse.messages import ConstraintMessage
from pulse.latent import CONDITION_LABELS
from pulse.biophysics.water_balance import water_stress_index


class WaterBalanceAgent(ConversableAgent):
    """The biophysics agent. Same protocol envelope as ML agents.
    Internal computation: Penman-Monteith + van Genuchten.

    Emits a ConstraintMessage over CONDITION_LABELS. The likelihood
    elevates `water_stress` and `healthy_crop` based on physical state,
    suppressing `disease` and `nutrient_stress` when soil/atmospheric
    conditions are consistent with adequate water supply.
    """

    def __init__(self, default_soil_texture: str = "loam"):
        super().__init__(
            name="water_balance",
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.default_soil_texture = default_soil_texture
        self.register_reply(
            trigger=[ConversableAgent, None],
            reply_func=self._emit_reply,
            position=0,
            remove_other_reply_funcs=True,
        )

    def _emit_reply(self, recipient, messages=None, sender=None, config=None):
        latest = messages[-1] if messages else {}
        payload = json.loads(latest.get("content", "{}"))
        # Expected payload includes a "field_state" dict with weather and soil:
        # {"latent": {...}, "field_state": {
        #     "soil_moisture_m3_m3": 0.18, "soil_texture": "loam",
        #     "T_C": 27.0, "RH_pct": 45, "u2_m_s": 2.0, "R_n_MJ_m2_d": 18.0,
        #     "crop_type": "tomato", "plants": [...]} }
        constraint = self.emit_constraint(payload)
        return True, {"role": "assistant", "name": self.name,
                      "content": json.dumps(self._serialize(constraint))}

    def emit_constraint(self, payload) -> ConstraintMessage:
        latent = payload["latent"]
        field = payload["field_state"]
        # Compute global water stress index for the field
        wsi = water_stress_index(
            theta=field["soil_moisture_m3_m3"],
            soil_texture=field.get("soil_texture", self.default_soil_texture),
            crop_type=field.get("crop_type", "default"),
            T_C=field["T_C"], RH_pct=field["RH_pct"],
            u2_m_s=field["u2_m_s"], R_n_MJ_m2_d=field["R_n_MJ_m2_d"],
        )
        S = wsi["stress_index"]
        # Build likelihood vector. The strength depends on how confident the
        # physics is: high S → strong evidence for water_stress AND against disease.
        log_lik = np.zeros(len(CONDITION_LABELS))
        # Indices
        i_water = CONDITION_LABELS.index("water_stress")
        i_disease = CONDITION_LABELS.index("disease")
        i_nutrient = CONDITION_LABELS.index("nutrient_stress")
        i_healthy = CONDITION_LABELS.index("healthy_crop")

        if S > 0.6:
            # Severe water deficit: strong elevation of water_stress,
            # strong suppression of disease/nutrient explanations
            log_lik[i_water] = 2.0
            log_lik[i_disease] = -1.5
            log_lik[i_nutrient] = -1.0
        elif S > 0.3:
            # Moderate stress
            log_lik[i_water] = 1.0
            log_lik[i_disease] = -0.5
        elif S < 0.1:
            # Plant has plenty of water: any wilting/yellowing is NOT drought
            log_lik[i_water] = -1.5
            log_lik[i_healthy] = 0.5
        # else: ambiguous, log_lik stays near zero

        per_plant_ll = {p["plant_id"]: log_lik.copy() for p in field["plants"]}
        per_plant_resid = {p["plant_id"]: 0.0 for p in field["plants"]}
        per_plant_conf = {p["plant_id"]: float(0.4 + 0.6 * abs(S - 0.5) * 2)
                          for p in field["plants"]}

        return ConstraintMessage(
            sender=self.name, timestamp=time.time(),
            iteration=latent.get("iteration", 0),
            per_plant_log_likelihoods=per_plant_ll,
            per_plant_residual=per_plant_resid,
            per_plant_confidence=per_plant_conf,
            labels_discriminated=["water_stress", "disease", "nutrient_stress", "healthy_crop"],
            metadata={
                "stress_index": float(S),
                "demand_mm_per_day": float(wsi["demand_mm"]),
                "supply_mm_per_day": float(wsi["supply_mm"]),
                "soil_psi_kPa": float(wsi["soil_psi_kPa"]),
            },
        )

    def _serialize(self, c: ConstraintMessage) -> dict:
        return {
            "sender": c.sender, "timestamp": c.timestamp, "iteration": c.iteration,
            "per_plant_log_likelihoods": {k: v.tolist() for k, v in c.per_plant_log_likelihoods.items()},
            "per_plant_residual": c.per_plant_residual,
            "per_plant_confidence": c.per_plant_confidence,
            "labels_discriminated": c.labels_discriminated,
            "metadata": c.metadata,
        }
```

#### 9.7.4 Tests

```python
# tests/test_water_balance_agent.py
import numpy as np
from pulse.biophysics.water_balance import water_stress_index
from pulse.agents.water_balance import WaterBalanceAgent


def test_well_watered_loam_no_stress():
    """Loam at field capacity in mild weather → near-zero stress index."""
    wsi = water_stress_index(
        theta=0.32, soil_texture="loam", crop_type="tomato",
        T_C=22.0, RH_pct=60, u2_m_s=1.5, R_n_MJ_m2_d=14.0,
    )
    assert wsi["stress_index"] < 0.1, f"Expected near-zero, got {wsi}"


def test_dry_sand_high_stress():
    """Sand near wilting point in hot dry weather → high stress index."""
    wsi = water_stress_index(
        theta=0.07, soil_texture="sand", crop_type="tomato",
        T_C=35.0, RH_pct=20, u2_m_s=4.0, R_n_MJ_m2_d=24.0,
    )
    assert wsi["stress_index"] > 0.6, f"Expected high stress, got {wsi}"


def test_agent_emits_likelihood_in_well_watered_field():
    """When physics says water is fine, agent suppresses water_stress label."""
    agent = WaterBalanceAgent()
    payload = {
        "latent": {"iteration": 0},
        "field_state": {
            "soil_moisture_m3_m3": 0.32, "soil_texture": "loam",
            "T_C": 22.0, "RH_pct": 60, "u2_m_s": 1.5, "R_n_MJ_m2_d": 14.0,
            "crop_type": "tomato",
            "plants": [{"plant_id": 0}, {"plant_id": 1}],
        },
    }
    c = agent.emit_constraint(payload)
    # Agent should emit negative log-likelihood for water_stress (suppression)
    from pulse.latent import CONDITION_LABELS
    i_water = CONDITION_LABELS.index("water_stress")
    for ll in c.per_plant_log_likelihoods.values():
        assert ll[i_water] < 0, f"Expected suppression of water_stress, got ll[water]={ll[i_water]}"
```

#### 9.7.5 Where it slots into the pipeline

The water balance agent runs *in parallel with* the ML diagnosis agents — same constraint-emission phase. Its likelihood is fused into the per-plant condition posterior the same way as any ML agent's.

This means the cross-examination layer treats it as a sixth diagnosis source. **When ML disease classifier and ML water-stress observation pull in different directions, the water balance agent's biophysical prior is what tips the posterior. No special-casing is needed — it just emits a stronger likelihood in regions where ML is uncertain.**

---

### 9.8 `EcologicalDynamicsAgent` — population biology

This is the **ecology agent**. It does not look at images. It does not predict spray outcomes. For each candidate intervention, it simulates the next 30 days of pest, predator, and parasitoid populations using a Lotka-Volterra-with-toxicity model.

**The role in the architecture:** the chemical treadmill is the consequence of optimizing each spray decision myopically. This agent makes the long-horizon cost of beneficial-insect disruption visible and computable. **Sprays that locally maximize weed/pest kill but reduce predator populations by 47% over 14 days lose utility against laser-zaps that preserve the predator population.**

#### 9.8.1 The math

Three-species Lotka-Volterra with crop, pest, and predator/parasitoid populations, plus a pesticide-mortality term:

$$
\frac{dP}{dt} = r_P P (1 - P/K_P) - a_{PR} P R - \mu_P(C, t) P
$$

$$
\frac{dR}{dt} = e \cdot a_{PR} P R - m_R R - \mu_R(C, t) R
$$

where $P$ is pest density, $R$ is predator density, $r_P$ is pest intrinsic growth rate, $K_P$ is pest carrying capacity, $a_{PR}$ is predation rate, $e$ is conversion efficiency, $m_R$ is predator natural mortality, and $\mu_P, \mu_R$ are pesticide-induced mortality rates that depend on chemical concentration $C$ and time since application.

Pesticide mortality follows a dose-response with first-order decay:

$$
\mu_X(C(t)) = \mu_{X,\max} \frac{C(t)/L_{X}}{1 + C(t)/L_{X}}, \quad C(t) = C_0 e^{-k_C t}
$$

Tabulated $L_{X}$ (lethal concentration) values exist for every species-chemistry pair in the EPA ECOTOX database. For the demo, hardcode 4 species pairs with values from extension service publications.

#### 9.8.2 The implementation

```python
# pulse/biology/lotka_volterra.py
"""Lotka-Volterra with pesticide toxicity for crop-pest-predator triads.

Reference: Crowder & Northfield 2014 ('Predator diversity strengthens herbivore
suppression') for predator-pest dynamics; EPA ECOTOX database for LC50 values.
"""
import numpy as np
from scipy.integrate import odeint


# Mode-of-action specific toxicity to non-target species.
# Values are LC50 (concentration killing 50%) in ppm, log-scale.
# Generic predator/parasitoid sensitivity by chemical class.
TOXICITY_DB = {
    "glyphosate": {
        # Herbicide — generally low non-target arthropod toxicity
        "predator_LC50_ppm": 100.0, "parasitoid_LC50_ppm": 50.0,
        "predator_max_mortality_d": 0.05, "parasitoid_max_mortality_d": 0.10,
    },
    "2_4_d": {
        "predator_LC50_ppm": 80.0, "parasitoid_LC50_ppm": 30.0,
        "predator_max_mortality_d": 0.08, "parasitoid_max_mortality_d": 0.15,
    },
    "atrazine": {
        "predator_LC50_ppm": 60.0, "parasitoid_LC50_ppm": 25.0,
        "predator_max_mortality_d": 0.12, "parasitoid_max_mortality_d": 0.20,
    },
    "chlorpyrifos": {
        # Broad-spectrum organophosphate insecticide — devastating to non-targets
        "predator_LC50_ppm": 0.05, "parasitoid_LC50_ppm": 0.02,
        "predator_max_mortality_d": 0.85, "parasitoid_max_mortality_d": 0.92,
    },
}

# Default parameters for a generic crop-pest-predator triad
# Values calibrated to match published aphid-coccinellid dynamics
DEFAULT_PARAMS = {
    "r_pest": 0.30,         # pest intrinsic growth rate (per day)
    "K_pest": 1000.0,       # pest carrying capacity
    "a_predation": 0.0008,  # predator-pest interaction rate
    "e_conversion": 0.10,   # predator energy conversion efficiency
    "m_predator": 0.05,     # predator natural mortality (per day)
    "m_parasitoid": 0.04,   # parasitoid natural mortality
    "a_parasitism": 0.0006, # parasitoid-pest interaction rate
}


def lotka_volterra_with_pesticide(
    state: np.ndarray,
    t: float,
    params: dict,
    pesticide_C0: float,
    pesticide_decay_k: float,
    toxicity: dict,
) -> np.ndarray:
    """ODE state vector: [P (pest), R (predator), Pa (parasitoid)]."""
    P, R, Pa = state
    P = max(P, 0); R = max(R, 0); Pa = max(Pa, 0)
    # Current concentration via first-order decay
    C_t = pesticide_C0 * np.exp(-pesticide_decay_k * t)
    # Pesticide-induced mortality (Hill function)
    mu_pred = (toxicity["predator_max_mortality_d"]
               * (C_t / toxicity["predator_LC50_ppm"])
               / (1 + C_t / toxicity["predator_LC50_ppm"]))
    mu_para = (toxicity["parasitoid_max_mortality_d"]
               * (C_t / toxicity["parasitoid_LC50_ppm"])
               / (1 + C_t / toxicity["parasitoid_LC50_ppm"]))
    # Pest mortality from pesticide (assume insecticides kill pest, herbicides don't)
    is_insecticide = (toxicity["predator_max_mortality_d"] > 0.5)
    mu_pest = 0.7 if is_insecticide else 0.0
    mu_pest_t = mu_pest * (C_t / 0.1) / (1 + C_t / 0.1)
    # Lotka-Volterra dynamics
    dP = (params["r_pest"] * P * (1 - P / params["K_pest"])
          - params["a_predation"] * P * R
          - params["a_parasitism"] * P * Pa
          - mu_pest_t * P)
    dR = (params["e_conversion"] * params["a_predation"] * P * R
          - params["m_predator"] * R
          - mu_pred * R)
    dPa = (params["e_conversion"] * params["a_parasitism"] * P * Pa
           - params["m_parasitoid"] * Pa
           - mu_para * Pa)
    return np.array([dP, dR, dPa])


def simulate_population_trajectory(
    initial_pest: float,
    initial_predator: float,
    initial_parasitoid: float,
    chemistry_id: str | None,           # None = no pesticide intervention
    initial_concentration_ppm: float,    # 0 if no pesticide
    decay_k_per_day: float,
    horizon_days: int = 30,
    params: dict | None = None,
) -> dict:
    """Simulate 30-day trajectory under a given intervention scenario."""
    params = params or DEFAULT_PARAMS
    if chemistry_id is None or initial_concentration_ppm == 0:
        # No pesticide → use a "null" toxicity profile
        toxicity = {"predator_LC50_ppm": 1e9, "parasitoid_LC50_ppm": 1e9,
                    "predator_max_mortality_d": 0.0, "parasitoid_max_mortality_d": 0.0}
    else:
        toxicity = TOXICITY_DB[chemistry_id]
    t = np.linspace(0, horizon_days, horizon_days + 1)
    y0 = np.array([initial_pest, initial_predator, initial_parasitoid])
    sol = odeint(lotka_volterra_with_pesticide, y0, t,
                 args=(params, initial_concentration_ppm, decay_k_per_day, toxicity))
    return {
        "days": t.tolist(),
        "pest": sol[:, 0].tolist(),
        "predator": sol[:, 1].tolist(),
        "parasitoid": sol[:, 2].tolist(),
    }
```

#### 9.8.3 The agent

```python
# pulse/agents/ecological_dynamics.py
import json, time, math
import numpy as np
from autogen import ConversableAgent
from pulse.messages import TrajectoryMessage
from pulse.biology.lotka_volterra import (
    simulate_population_trajectory, TOXICITY_DB, DEFAULT_PARAMS,
)


class EcologicalDynamicsAgent(ConversableAgent):
    """The ecology agent. Simulates pest/predator/parasitoid trajectory
    under a candidate intervention. Same protocol envelope as physics agent.
    Internal computation: Lotka-Volterra ODE with pesticide toxicity.
    """

    def __init__(self):
        super().__init__(
            name="ecological_dynamics",
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.register_reply(
            trigger=[ConversableAgent, None],
            reply_func=self._assess_reply,
            position=0,
            remove_other_reply_funcs=True,
        )

    def _assess_reply(self, recipient, messages=None, sender=None, config=None):
        latest = messages[-1] if messages else {}
        payload = json.loads(latest.get("content", "{}"))
        msg = self.assess_intervention(payload)
        return True, {"role": "assistant", "name": self.name,
                      "content": json.dumps(self._serialize(msg))}

    def assess_intervention(self, payload) -> TrajectoryMessage:
        """Payload:
            {"plant_id": int, "action_type": str, "action_params": dict,
             "chemistry": str | None,
             "initial_populations": {"pest": float, "predator": float,
                                      "parasitoid": float}}"""
        plant_id = payload["plant_id"]
        action_type = payload["action_type"]
        action_params = payload.get("action_params", {})
        chemistry = payload.get("chemistry")
        pops = payload.get("initial_populations",
                           {"pest": 100.0, "predator": 20.0, "parasitoid": 15.0})

        # Determine pesticide concentration at application point
        if action_type in ("intervene_targeted_spray", "targeted_fungicide") and chemistry:
            volume_ml = action_params.get("volume_ml", 5.0)
            conc_g_l = action_params.get("concentration_g_l", 360.0)
            # ppm-equivalent concentration on treated plant
            initial_C_ppm = volume_ml * conc_g_l * 1.0  # rough scale
            decay_k = math.log(2) / 23.0  # default DT50
        else:
            chemistry = None
            initial_C_ppm = 0.0
            decay_k = 0.0

        traj = simulate_population_trajectory(
            initial_pest=pops["pest"],
            initial_predator=pops["predator"],
            initial_parasitoid=pops["parasitoid"],
            chemistry_id=chemistry,
            initial_concentration_ppm=initial_C_ppm,
            decay_k_per_day=decay_k,
            horizon_days=30,
        )
        # Normalize trajectories to t=0 = 1.0 for comparability
        p_norm = [v / max(traj["pest"][0], 1e-9) for v in traj["pest"]]
        r_norm = [v / max(traj["predator"][0], 1e-9) for v in traj["predator"]]
        pa_norm = [v / max(traj["parasitoid"][0], 1e-9) for v in traj["parasitoid"]]

        # Diagnostic features for cost score
        predator_drop_d14 = max(0.0, 1.0 - r_norm[14])
        parasitoid_drop_d14 = max(0.0, 1.0 - pa_norm[14])
        pest_rebound_d30 = max(0.0, p_norm[30] - 1.0)

        # Aggregate ecological cost in [0, 1]
        # Heavy weight on predator depletion + pest rebound
        cost = float(np.clip(
            0.4 * predator_drop_d14
            + 0.3 * parasitoid_drop_d14
            + 0.3 * min(pest_rebound_d30 / 3.0, 1.0),
            0.0, 1.0,
        ))

        return TrajectoryMessage(
            sender=self.name, timestamp=time.time(),
            plant_id=plant_id, action_type=action_type,
            action_params=action_params,
            days=traj["days"],
            pest_trajectory=p_norm,
            predator_trajectory=r_norm,
            parasitoid_trajectory=pa_norm,
            ecological_cost_score=cost,
            cost_breakdown={
                "predator_drop_pct_d14": float(predator_drop_d14),
                "parasitoid_drop_pct_d14": float(parasitoid_drop_d14),
                "pest_rebound_factor_d30": float(p_norm[30]),
            },
        )

    def _serialize(self, m: TrajectoryMessage) -> dict:
        import dataclasses
        return dataclasses.asdict(m)
```

#### 9.8.4 Tests

```python
# tests/test_ecological_dynamics_agent.py
from pulse.agents.ecological_dynamics import EcologicalDynamicsAgent
from pulse.biology.lotka_volterra import simulate_population_trajectory


def test_no_intervention_stable_populations():
    """Without pesticide, predator-prey equilibrium remains within bounds."""
    traj = simulate_population_trajectory(
        initial_pest=100, initial_predator=20, initial_parasitoid=15,
        chemistry_id=None, initial_concentration_ppm=0.0, decay_k_per_day=0.0,
        horizon_days=30,
    )
    # Predator population shouldn't crash to zero without pesticide
    assert min(traj["predator"]) > 5, "Predator should survive without spraying"


def test_chlorpyrifos_devastates_predators():
    """Broad-spectrum insecticide causes massive predator drop."""
    traj = simulate_population_trajectory(
        initial_pest=100, initial_predator=20, initial_parasitoid=15,
        chemistry_id="chlorpyrifos", initial_concentration_ppm=1.0,
        decay_k_per_day=0.023, horizon_days=30,
    )
    # By day 14, predators should be < 50% of initial
    assert traj["predator"][14] < traj["predator"][0] * 0.5, \
        "Chlorpyrifos should deplete predators by day 14"


def test_glyphosate_minimal_predator_impact():
    """Herbicide should have minimal direct effect on predators."""
    traj = simulate_population_trajectory(
        initial_pest=100, initial_predator=20, initial_parasitoid=15,
        chemistry_id="glyphosate", initial_concentration_ppm=1.0,
        decay_k_per_day=0.030, horizon_days=30,
    )
    # Predator population should remain >70% at day 14
    assert traj["predator"][14] > traj["predator"][0] * 0.7, \
        f"Glyphosate should preserve predators, got {traj['predator'][14]}"


def test_agent_returns_trajectory_message():
    """Agent emits a valid TrajectoryMessage."""
    agent = EcologicalDynamicsAgent()
    payload = {
        "plant_id": 0, "action_type": "intervene_laser_zap",
        "action_params": {}, "chemistry": None,
        "initial_populations": {"pest": 100, "predator": 20, "parasitoid": 15},
    }
    msg = agent.assess_intervention(payload)
    assert len(msg.days) == 31
    assert msg.ecological_cost_score >= 0.0
    assert msg.ecological_cost_score <= 1.0
    # Laser zap = no chemicals = low ecological cost
    assert msg.ecological_cost_score < 0.2
```

#### 9.8.5 How the controller integrates ecology into utility

Update `EIGControllerAgent` to invoke the ecology agent for *every* candidate intervention (including laser zap, which has zero ecological cost) and fold the cost score into utility:

```python
# pulse/agents/controller.py (additions)
def _compute_utility(
    self,
    plant_posterior: np.ndarray,
    action_type: str,
    physics_assessment: InterventionAssessmentMessage | None,
    ecology_trajectory: TrajectoryMessage | None,
) -> float:
    """U(action) = E[yield_protected] - chem_cost - hazard - ecological_future_cost."""
    p_weed = plant_posterior[CONDITION_LABELS.index("weed")]
    p_disease = plant_posterior[CONDITION_LABELS.index("disease")]
    p_water = plant_posterior[CONDITION_LABELS.index("water_stress")]
    p_healthy = plant_posterior[CONDITION_LABELS.index("healthy_crop")]
    base = {
        "intervene_targeted_spray":   p_weed * 0.9 - 0.10,
        "targeted_fungicide":         p_disease * 0.85 - 0.12,
        "intervene_laser_zap":        p_weed * 0.85 - 0.05,
        "intervene_targeted_irrigation": p_water * 0.80 - 0.04,
        "no_action":                  p_healthy * 1.0,
        "human_review":               -0.05,
        "rescan_higher_res":          -0.02,
    }.get(action_type, 0.0)

    # Physics veto: subtract hazard score
    if physics_assessment is not None:
        base -= 0.5 * physics_assessment.hazard_score

    # Ecological future cost: subtract ecological cost score
    if ecology_trajectory is not None:
        base -= 0.4 * ecology_trajectory.ecological_cost_score

    return float(base)
```

The integrated flow:

1. ML agents + water balance agent diagnose plant → condition posterior (with biophysics breaking ML ties)
2. Controller, for each plant, considers all candidate interventions
3. For each spray-class action, controller calls **physics agent** → hazard score
4. For **every** action, controller calls **ecology agent** → ecological cost score
5. Utility = (detection × yield_protected) − chem_cost − 0.5 × hazard − 0.4 × eco_cost
6. **When ecological cost is high, laser-zap or no-action wins even when spray detection is confident AND physics is fine**

**This is where the chemical treadmill is broken in math.** A chlorpyrifos spray that detects 100% confidence, has 0.2 physics hazard (low drift), but 0.85 ecological cost (devastates predators) — gets rejected by the utility function in favor of an integrated-pest-management approach.

---

## 10. CROSS-EXAMINATION

```python
# pulse/cross_exam.py
import numpy as np
from pulse.messages import ConstraintMessage, CrossExamMessage
import time

def cross_examine(constraints: dict[str, ConstraintMessage]) -> list[CrossExamMessage]:
    """Compute pairwise per-plant disagreement matrices.
    Returns one CrossExamMessage per (evaluator, target) pair."""
    msgs = []
    agents = list(constraints.keys())
    for i, ai in enumerate(agents):
        for j, aj in enumerate(agents):
            if i == j: continue
            per_plant_dis, per_plant_diag = {}, {}
            ci = constraints[ai]; cj = constraints[aj]
            shared_plants = set(ci.per_plant_log_likelihoods) & set(cj.per_plant_log_likelihoods)
            for pid in shared_plants:
                ll_i = ci.per_plant_log_likelihoods[pid]
                ll_j = cj.per_plant_log_likelihoods[pid]
                pi = _softmax(ll_i); pj = _softmax(ll_j)
                kl = float(np.sum(pi * (np.log(pi + 1e-12) - np.log(pj + 1e-12))))
                per_plant_dis[pid] = kl
                # Per-axis: which condition labels does the disagreement concentrate on?
                axis_diag = {}
                for k, label in enumerate(["healthy_crop", "weed", "disease",
                                           "nutrient_stress", "water_stress",
                                           "pest_damage", "ambiguous"]):
                    axis_diag[label] = float(abs(pi[k] - pj[k]))
                per_plant_diag[pid] = axis_diag
            msgs.append(CrossExamMessage(
                evaluator=ai, target=aj,
                per_plant_disagreement=per_plant_dis,
                per_plant_diagnostic=per_plant_diag,
            ))
    return msgs

def _softmax(x):
    x = x - np.max(x); return np.exp(x) / np.sum(np.exp(x))
```

---

## 11. THE FRONTEND — OLLAMA-STYLED, VISUALLY POWERFUL

The frontend is a single-page app served by FastAPI on `localhost:8000`. The aesthetic is Ollama: dark, monospace, low chrome. Visually powerful = the field image renders large, with crisp annotations, and the disagreement glow is unmistakable.

### 11.1 Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ pulse / precision agriculture                       ● live      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                         │   │
│   │   FIELD IMAGE                                           │   │
│   │   (large, prominent)                                    │   │
│   │   - bounding boxes per plant                            │   │
│   │   - colored by top-1 condition                          │   │
│   │   - disputed plants pulse                               │   │
│   │   - hover for per-plant agent breakdown                 │   │
│   │                                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
├──────────────────────────┬──────────────────────────────────────┤
│ AGENT CONSTRAINTS        │ INTERVENTION SUMMARY                 │
│                          │                                      │
│ weed_detector       ✓    │ laser_zap          ▰▰▰▰▰▰▰▰▱▱  47   │
│ disease_classifier  ✓    │ targeted_fungicide ▰▰▰▱▱▱▱▱▱▱   8   │
│ segmentation        ✓    │ targeted_irrigation ▰▱▱▱▱▱▱▱▱▱   2   │
│ health_classifier   ✓    │ foliar_nutrient    ▰▱▱▱▱▱▱▱▱▱   3   │
│ vlm_reasoner        ⏳    │ no_action          ▰▰▰▰▰▱▱▱▱▱  31   │
│                          │ human_review       ▰▰▱▱▱▱▱▱▱▱  12   │
│                          │                                      │
├──────────────────────────┴──────────────────────────────────────┤
│ CHEMICAL REDUCTION vs BLANKET                                   │
│ ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱  91%                                       │
├─────────────────────────────────────────────────────────────────┤
│ MESSAGE STREAM                                                  │
│ [t=0.000] orchestrator   begin     image=field_03.jpg          │
│ [t=0.214] weed_detector  constraint plants=87 conf_avg=0.71    │
│ [t=0.502] disease_clf    constraint plants=87 conf_avg=0.62    │
│ [t=0.890] segmentation   constraint plants=87 conf_avg=0.83    │
│ [t=1.121] health_clf     constraint plants=87 conf_avg=0.74    │
│ [t=1.205] xexam          weed↔health  KL_avg=0.31 max=2.4 ⚠   │
│ [t=1.301] skeptic        hyp plant=43 'young_crop_as_weed'  ⚠ │
│ [t=1.402] vlm_reasoner   resolved plant=43 → healthy_crop ✓    │
│ [t=1.510] controller     action plant=43 = no_action            │
│ [t=2.001] DONE           87 plants  91% chemical reduction      │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 `pulse/dashboard/static/style.css`

```css
:root {
    --bg: #0a0a0a;
    --bg-elev: #141414;
    --bg-elev-2: #1c1c1c;
    --border: #2a2a2a;
    --border-strong: #3a3a3a;
    --text: #e8e8e8;
    --text-dim: #888;
    --text-faint: #555;
    --accent: #5fa8d3;
    --warn: #d39c5f;
    --ok: #82c85a;
    --flag: #d35f5f;
    --weed: #d35f9c;
    --disease: #d39c5f;
    --healthy: #82c85a;
    --nutrient: #c8c85a;
    --water: #5fc8d3;
    --pest: #b35fd3;
    --ambiguous: #888;
    --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}

* { box-sizing: border-box; }
body {
    margin: 0; padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px; line-height: 1.5;
    min-height: 100vh;
}

header {
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: var(--mono);
    font-size: 13px;
    background: var(--bg);
    position: sticky; top: 0; z-index: 10;
}
header .live::before {
    content: "●"; color: var(--ok); margin-right: 8px;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

.layout {
    display: grid;
    grid-template-columns: 1fr;
    grid-template-rows: auto auto auto auto;
    gap: 1px;
    padding: 1px;
    background: var(--border);
}

.panel {
    background: var(--bg-elev);
    padding: 18px 24px;
}
.panel h2 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-dim);
    margin: 0 0 14px 0;
    font-weight: 500;
    font-family: var(--mono);
}

/* Field image — the visual centerpiece */
.field-canvas-wrap {
    position: relative;
    width: 100%;
    background: #000;
    border-radius: 2px;
    overflow: hidden;
}
#field-canvas {
    width: 100%;
    height: auto;
    display: block;
}
.bbox-overlay {
    position: absolute;
    border: 2px solid;
    pointer-events: none;
    transition: border-color 0.3s;
}
.bbox-overlay.disputed {
    animation: dispute-pulse 1.4s ease-in-out infinite;
    box-shadow: 0 0 12px var(--flag);
}
.bbox-overlay.physics-veto {
    border-color: var(--flag);
    border-style: dashed;
    box-shadow: 0 0 16px var(--flag);
}
@keyframes dispute-pulse {
    0%, 100% { border-color: var(--flag); box-shadow: 0 0 4px var(--flag); }
    50%      { border-color: var(--accent); box-shadow: 0 0 20px var(--flag); }
}
.bbox-label {
    position: absolute;
    background: rgba(0,0,0,0.85);
    color: var(--text);
    padding: 2px 6px;
    font-family: var(--mono);
    font-size: 10px;
    border: 1px solid currentColor;
    white-space: nowrap;
    pointer-events: none;
}

/* Drift cone — drawn on a separate canvas overlay */
.drift-canvas {
    position: absolute;
    top: 0; left: 0;
    pointer-events: none;
    width: 100%;
    height: 100%;
    mix-blend-mode: screen;  /* let underlying image show through */
}

/* Wind indicator — top-right of field panel */
.wind-indicator {
    position: absolute;
    top: 12px; right: 12px;
    background: rgba(0,0,0,0.75);
    border: 1px solid var(--border-strong);
    padding: 8px 12px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    z-index: 5;
}
.wind-indicator .arrow {
    display: inline-block;
    color: var(--accent);
    font-size: 16px;
    transform-origin: center;
    transition: transform 0.5s ease;
    margin-right: 6px;
}

.split-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--border);
}

.agent-row, .interv-row {
    display: flex;
    align-items: center;
    padding: 6px 0;
    font-family: var(--mono);
    font-size: 13px;
    border-bottom: 1px solid var(--border);
}
.agent-row:last-child, .interv-row:last-child { border: 0; }
.agent-row .name { width: 180px; color: var(--text-dim); }
.agent-row .status { width: 30px; }
.agent-row .status.ok { color: var(--ok); }
.agent-row .status.pending { color: var(--text-faint); }

.interv-row .label { width: 200px; }
.interv-row .bar {
    width: 200px; height: 6px;
    background: var(--border-strong); margin: 0 14px;
    border-radius: 1px;
    overflow: hidden;
}
.interv-row .bar-fill {
    height: 100%; background: var(--accent);
    transition: width 0.5s ease-out;
}
.interv-row .count {
    width: 50px; text-align: right;
    color: var(--text-dim);
}

/* Chemical reduction — the headline metric */
.kpi-row {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 24px;
    background: var(--bg-elev-2);
}
.kpi-label {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    width: 240px;
}
.kpi-bar {
    flex: 1; height: 14px;
    background: var(--border-strong);
    border-radius: 1px;
    overflow: hidden;
    position: relative;
}
.kpi-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--healthy), var(--ok));
    transition: width 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.kpi-value {
    width: 80px;
    text-align: right;
    font-family: var(--mono);
    font-size: 24px;
    font-weight: 300;
    color: var(--ok);

/* Ecology forecast panel */
#ecology-chart {
    width: 100%;
    height: 200px;
    background: var(--bg);
    border: 1px solid var(--border);
}
.biophysics-readout {
    margin-top: 12px;
    display: grid;
    grid-template-columns: auto auto auto auto auto auto;
    gap: 8px 16px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
}
.biophysics-readout .bio-value {
    color: var(--text);
}

/* Reduction breakdown rows */
.reduction-row {
    display: flex;
    align-items: center;
    padding: 6px 0;
    font-family: var(--mono);
    font-size: 13px;
    border-bottom: 1px solid var(--border);
}
.reduction-row:last-child { border: 0; }
.rd-label { width: 200px; color: var(--text-dim); }
.rd-bar {
    width: 200px; height: 8px;
    background: var(--border-strong);
    margin: 0 14px;
    border-radius: 1px;
    overflow: hidden;
}
.rd-fill {
    height: 100%;
    transition: width 0.6s ease-out;
}
.rd-value {
    width: 50px;
    text-align: right;
    color: var(--text);
}

.message-stream {
    font-family: var(--mono);
    font-size: 12px;
    height: 220px;
    overflow-y: auto;
    background: var(--bg);
    border: 1px solid var(--border);
    padding: 12px;
}
.message-stream .row {
    padding: 2px 0;
}
.message-stream .row.flag { color: var(--flag); }
.message-stream .row.action { color: var(--accent); }
.message-stream .row.done { color: var(--ok); font-weight: 500; }

.tooltip {
    position: absolute;
    background: var(--bg-elev-2);
    border: 1px solid var(--border-strong);
    padding: 8px 10px;
    font-family: var(--mono);
    font-size: 11px;
    pointer-events: none;
    z-index: 20;
    min-width: 240px;
}
.tooltip .label { color: var(--text-dim); }
.tooltip .value { color: var(--text); }
.tooltip .row { display: flex; justify-content: space-between; }
```

### 11.3 `pulse/dashboard/static/index.html`

```html
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>pulse · precision agriculture</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <header>
        <span>pulse / precision agriculture</span>
        <span class="live">live</span>
    </header>

    <div class="layout">
        <div class="panel">
            <h2>field analysis</h2>
            <div class="field-canvas-wrap">
                <canvas id="field-canvas" width="1200" height="800"></canvas>
                <canvas id="drift-canvas" class="drift-canvas"></canvas>
                <div id="bbox-layer"></div>
                <div class="wind-indicator" id="wind-indicator">
                    <span class="arrow" id="wind-arrow">→</span>
                    <span id="wind-label">wind 4.2 m/s NW</span>
                </div>
                <div id="tooltip" class="tooltip" style="display:none"></div>
            </div>
        </div>

        <div class="split-row">
            <div class="panel">
                <h2>agent constraints</h2>
                <div id="agent-rows"></div>
            </div>
            <div class="panel">
                <h2>intervention summary</h2>
                <div id="interv-rows"></div>
            </div>
        </div>

        <div class="split-row">
            <div class="panel">
                <h2>30-day ecological forecast</h2>
                <canvas id="ecology-chart" width="560" height="200"></canvas>
                <div class="biophysics-readout" id="biophysics-readout">
                    <span class="bio-label">water_stress index</span>
                    <span class="bio-value" id="bio-stress">—</span>
                    <span class="bio-label">soil ψ</span>
                    <span class="bio-value" id="bio-psi">—</span>
                    <span class="bio-label">demand</span>
                    <span class="bio-value" id="bio-demand">—</span>
                </div>
            </div>
            <div class="panel">
                <h2>chemical reduction breakdown</h2>
                <div class="reduction-row">
                    <span class="rd-label">precision targeting</span>
                    <span class="rd-bar"><span class="rd-fill" id="rd-precision" style="width:0%;background:var(--healthy)"></span></span>
                    <span class="rd-value" id="rd-precision-val">0%</span>
                </div>
                <div class="reduction-row">
                    <span class="rd-label">physics-veto substitution</span>
                    <span class="rd-bar"><span class="rd-fill" id="rd-physics" style="width:0%;background:var(--accent)"></span></span>
                    <span class="rd-value" id="rd-physics-val">0%</span>
                </div>
                <div class="reduction-row">
                    <span class="rd-label">ecology-aware substitution</span>
                    <span class="rd-bar"><span class="rd-fill" id="rd-ecology" style="width:0%;background:var(--healthy)"></span></span>
                    <span class="rd-value" id="rd-ecology-val">0%</span>
                </div>
            </div>
        </div>

        <div class="kpi-row">
            <span class="kpi-label">chemical reduction vs blanket</span>
            <div class="kpi-bar"><div class="kpi-fill" id="kpi-fill" style="width:0%"></div></div>
            <span class="kpi-value" id="kpi-value">0%</span>
        </div>

        <div class="panel">
            <h2>message stream</h2>
            <div class="message-stream" id="message-stream"></div>
        </div>
    </div>

    <script src="/static/app.js"></script>
</body>
</html>
```

### 11.4 `pulse/dashboard/static/app.js`

```javascript
const ws = new WebSocket(`ws://${location.host}/ws`);
const stream = document.getElementById("message-stream");
const agentRows = document.getElementById("agent-rows");
const intervRows = document.getElementById("interv-rows");
const bboxLayer = document.getElementById("bbox-layer");
const tooltip = document.getElementById("tooltip");
const canvas = document.getElementById("field-canvas");
const ctx = canvas.getContext("2d");
const driftCanvas = document.getElementById("drift-canvas");
const driftCtx = driftCanvas.getContext("2d");
const windArrow = document.getElementById("wind-arrow");
const windLabel = document.getElementById("wind-label");

const CONDITION_COLORS = {
    healthy_crop: "#82c85a",
    weed: "#d35f9c",
    disease: "#d39c5f",
    nutrient_stress: "#c8c85a",
    water_stress: "#5fc8d3",
    pest_damage: "#b35fd3",
    ambiguous: "#888",
};

const AGENTS = ["weed_detector", "disease_classifier", "segmentation",
                "health_classifier", "vlm_reasoner",
                "water_balance", "pesticide_fate", "ecological_dynamics"];

const PARADIGM_TAGS = {
    "water_balance":       " <span style='color:var(--water);font-size:10px'>[BIOPHYSICS]</span>",
    "pesticide_fate":      " <span style='color:var(--accent);font-size:10px'>[PHYSICS]</span>",
    "ecological_dynamics": " <span style='color:var(--healthy);font-size:10px'>[ECOLOGY]</span>",
};

const INTERVENTIONS = ["targeted_spray", "laser_zap", "targeted_fungicide",
                       "targeted_irrigation", "foliar_nutrient", "no_action",
                       "human_review", "rescan_higher_res"];

// Initialize agent rows
AGENTS.forEach(name => {
    const row = document.createElement("div");
    row.className = "agent-row";
    row.id = `agent-${name}`;
    const paradigm = PARADIGM_TAGS[name] || "";
    row.innerHTML = `<span class="name">${name}${paradigm}</span>
                     <span class="status pending" data-status="pending">⏳</span>`;
    agentRows.appendChild(row);
});

INTERVENTIONS.forEach(name => {
    const row = document.createElement("div");
    row.className = "interv-row";
    row.id = `interv-${name}`;
    row.innerHTML = `<span class="label">${name}</span>
                     <span class="bar"><span class="bar-fill" style="width:0%"></span></span>
                     <span class="count">0</span>`;
    intervRows.appendChild(row);
});

let currentImage = null;
let currentPlants = [];
let currentWind = { dir_deg: 270, speed: 4.2 };

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    switch (msg.type) {
        case "image_loaded":         loadImage(msg.image_url, msg.plants, msg.wind); break;
        case "constraint":           handleConstraint(msg); break;
        case "cross_exam":           handleCrossExam(msg); break;
        case "hypothesis":           handleHypothesis(msg); break;
        case "physics_assessment":   handlePhysicsAssessment(msg); break;
        case "ecology_trajectory":   handleEcologyTrajectory(msg); break;
        case "water_balance":        handleWaterBalance(msg); break;
        case "action":               handleAction(msg); break;
        case "summary":              handleSummary(msg); break;
        case "done":                 handleDone(msg); break;
    }
};

function loadImage(url, plants, wind) {
    const img = new Image();
    img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        // Drift overlay matches main canvas dimensions
        driftCanvas.width = img.naturalWidth;
        driftCanvas.height = img.naturalHeight;
        currentImage = img;
        currentPlants = plants;
        if (wind) updateWind(wind);
        renderBoxes(plants);
    };
    img.src = url;
}

function updateWind(wind) {
    currentWind = wind;
    // Meteorological convention: wind_dir_deg is direction wind is FROM.
    // Visual arrow points TO downwind (rotate 180° from "from" direction).
    const arrowDeg = (wind.dir_deg + 180) % 360;
    windArrow.style.transform = `rotate(${arrowDeg}deg)`;
    const compass = compassFromDeg(wind.dir_deg);
    windLabel.textContent = `wind ${wind.speed.toFixed(1)} m/s ${compass}`;
}

function compassFromDeg(deg) {
    const dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
    return dirs[Math.round(((deg % 360) / 45)) % 8];
}

function renderBoxes(plants) {
    bboxLayer.innerHTML = "";
    const wrap = canvas.getBoundingClientRect();
    const sx = wrap.width / canvas.width;
    const sy = wrap.height / canvas.height;

    plants.forEach(p => {
        const [x, y, x2, y2] = p.bbox;
        const div = document.createElement("div");
        div.className = "bbox-overlay";
        div.dataset.plantId = p.plant_id;
        div.style.left = `${x * sx}px`;
        div.style.top = `${y * sy}px`;
        div.style.width = `${(x2 - x) * sx}px`;
        div.style.height = `${(y2 - y) * sy}px`;

        const topCondition = p.top_k && p.top_k[0] ? p.top_k[0][0] : "ambiguous";
        const topConf = p.top_k && p.top_k[0] ? p.top_k[0][1] : 0;
        div.style.borderColor = CONDITION_COLORS[topCondition] || "#888";
        if (p.disputed) div.classList.add("disputed");
        if (p.physics_veto) div.classList.add("physics-veto");
        div.style.pointerEvents = "auto";

        div.addEventListener("mousemove", (e) => {
            const rect = wrap;
            tooltip.style.left = `${e.clientX - rect.left + 12}px`;
            tooltip.style.top = `${e.clientY - rect.top + 12}px`;
            tooltip.style.display = "block";
            const physRow = (p.hazard_score !== undefined)
                ? `<div class="row"><span class="label">physics hazard</span><span class="value" style="color:${p.hazard_score > 0.5 ? 'var(--flag)' : 'var(--ok)'}">${p.hazard_score.toFixed(2)}</span></div>`
                : "";
            tooltip.innerHTML = `
                <div class="row"><span class="label">plant</span><span class="value">#${p.plant_id}</span></div>
                <div class="row"><span class="label">top</span><span class="value">${topCondition} (${(topConf*100).toFixed(0)}%)</span></div>
                <div class="row"><span class="label">entropy</span><span class="value">${p.entropy?.toFixed(2) ?? "?"}</span></div>
                <div class="row"><span class="label">agents</span><span class="value">${p.constraint_history?.join(", ") ?? "-"}</span></div>
                ${physRow}
            `;
        });
        div.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });

        bboxLayer.appendChild(div);
    });
}

function drawDriftCone(source_px, wind, length_px, color = "rgba(211, 95, 95, 0.35)") {
    /* Draw a simple wedge from source extending downwind, widening with distance.
       Matches Pasquill-Gifford σ_y growth at low orders.
       source_px: [x, y] in canvas pixels.
       wind.dir_deg: direction wind is FROM (meteorological).
       length_px: visual length of cone.
    */
    const [sx, sy] = source_px;
    const downwind_deg = (wind.dir_deg + 180) % 360;
    const rad = downwind_deg * Math.PI / 180;
    // Cone half-angle: ~12° (matches sigma_y / x for class D at short range)
    const half = 12 * Math.PI / 180;
    const x1 = sx + Math.cos(rad) * length_px;
    const y1 = sy + Math.sin(rad) * length_px;
    // Offsets perpendicular to wind direction
    const px = Math.cos(rad + Math.PI / 2);
    const py = Math.sin(rad + Math.PI / 2);
    const halfW = Math.tan(half) * length_px;
    const xL = x1 + px * halfW, yL = y1 + py * halfW;
    const xR = x1 - px * halfW, yR = y1 - py * halfW;

    driftCtx.beginPath();
    driftCtx.moveTo(sx, sy);
    driftCtx.lineTo(xL, yL);
    driftCtx.lineTo(xR, yR);
    driftCtx.closePath();
    const grad = driftCtx.createRadialGradient(sx, sy, 5, sx, sy, length_px);
    grad.addColorStop(0, color.replace("0.35", "0.65"));
    grad.addColorStop(1, color.replace("0.35", "0.05"));
    driftCtx.fillStyle = grad;
    driftCtx.fill();
}

function clearDriftLayer() {
    driftCtx.clearRect(0, 0, driftCanvas.width, driftCanvas.height);
}

function handlePhysicsAssessment(msg) {
    /* msg: {t, plant_id, action_type, hazard_score, source_px,
            off_target_deposition: {plant_id: ppm}, drift_length_px} */
    appendStream(msg.t, "pesticide_fate", "assessment",
        `plant=${msg.plant_id} action=${msg.action_type} hazard=${msg.hazard_score.toFixed(2)}` +
        (msg.hazard_score > 0.5 ? " VETO ⚠" : ""),
        msg.hazard_score > 0.5 ? "flag" : "");
    // Mark physics agent as having spoken
    const row = document.getElementById("agent-pesticide_fate");
    if (row) {
        const s = row.querySelector(".status");
        s.className = "status ok"; s.textContent = "✓";
    }
    // Update plant's hazard tooltip data
    const plant = currentPlants.find(p => p.plant_id === msg.plant_id);
    if (plant) {
        plant.hazard_score = msg.hazard_score;
        if (msg.hazard_score > 0.5) plant.physics_veto = true;
    }
    // Draw the drift cone for vetoed sprays
    if (msg.hazard_score > 0.5 && msg.source_px) {
        drawDriftCone(msg.source_px, currentWind, msg.drift_length_px || 200);
    }
    renderBoxes(currentPlants);
}

function handleConstraint(msg) {
    appendStream(msg.t, msg.sender, "constraint",
        `plants=${msg.n_plants} conf_avg=${msg.conf_avg.toFixed(2)}`);
    const row = document.getElementById(`agent-${msg.sender}`);
    if (row) {
        const status = row.querySelector(".status");
        status.className = "status ok";
        status.textContent = "✓";
    }
    if (msg.plant_updates) {
        msg.plant_updates.forEach(u => {
            const plant = currentPlants.find(p => p.plant_id === u.plant_id);
            if (plant) {
                plant.top_k = u.top_k;
                plant.entropy = u.entropy;
                plant.constraint_history = u.constraint_history;
            }
        });
        renderBoxes(currentPlants);
    }
}

function handleCrossExam(msg) {
    const cls = msg.kl_max > 1.5 ? "flag" : "";
    appendStream(msg.t, "xexam",
        `${msg.evaluator}↔${msg.target}`,
        `KL_avg=${msg.kl_avg.toFixed(2)} max=${msg.kl_max.toFixed(1)}` +
        (msg.kl_max > 1.5 ? " ⚠" : ""), cls);
    if (msg.disputed_plant_ids) {
        msg.disputed_plant_ids.forEach(pid => {
            const p = currentPlants.find(x => x.plant_id === pid);
            if (p) p.disputed = true;
        });
        renderBoxes(currentPlants);
    }
}

function handleHypothesis(msg) {
    appendStream(msg.t, "skeptic",
        `hyp plant=${msg.plant_id}`,
        `'${msg.hypothesis_id}' ⚠`, "flag");
}

function handleWaterBalance(msg) {
    /* msg: {t, stress_index, demand_mm, supply_mm, soil_psi_kPa, n_plants} */
    const stress = msg.stress_index;
    const tag = stress > 0.4 ? "WATER DEFICIT" : stress < 0.1 ? "well-watered" : "moderate";
    appendStream(msg.t, "water_balance", "biophysics",
        `S=${stress.toFixed(2)} demand=${msg.demand_mm.toFixed(1)}mm/d psi=${msg.soil_psi_kPa.toFixed(0)}kPa (${tag})`);
    const row = document.getElementById("agent-water_balance");
    if (row) {
        const s = row.querySelector(".status");
        s.className = "status ok"; s.textContent = "✓";
    }
}

function handleEcologyTrajectory(msg) {
    /* msg: {t, plant_id, action_type, ecological_cost_score,
            cost_breakdown, days, pest_trajectory, predator_trajectory,
            parasitoid_trajectory} */
    const eco = msg.ecological_cost_score;
    const tag = eco > 0.5 ? "TREADMILL ⚠" : eco > 0.2 ? "moderate impact" : "preserves predators";
    appendStream(msg.t, "ecological_dynamics", "trajectory",
        `plant=${msg.plant_id} action=${msg.action_type} eco_cost=${eco.toFixed(2)} (${tag})`,
        eco > 0.5 ? "flag" : "");
    const row = document.getElementById("agent-ecological_dynamics");
    if (row) {
        const s = row.querySelector(".status");
        s.className = "status ok"; s.textContent = "✓";
    }
    // Update plant tooltip data
    const plant = currentPlants.find(p => p.plant_id === msg.plant_id);
    if (plant) {
        plant.ecological_cost = eco;
        plant.predator_drop_d14 = msg.cost_breakdown.predator_drop_pct_d14;
        plant.pest_rebound_d30 = msg.cost_breakdown.pest_rebound_factor_d30;
    }
    // Render trajectory chart for the most recent ecology event
    renderEcologyChart(msg);
}

function renderEcologyChart(msg) {
    const canvas = document.getElementById("ecology-chart");
    if (!canvas) return;
    const ec = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ec.clearRect(0, 0, W, H);
    // Background grid
    ec.strokeStyle = "var(--border)";
    ec.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = (H - 30) * i / 4 + 10;
        ec.beginPath(); ec.moveTo(40, y); ec.lineTo(W - 10, y); ec.stroke();
    }
    // Axis labels
    ec.fillStyle = "#888";
    ec.font = "10px ui-monospace, monospace";
    ec.fillText("0", 20, H - 18);
    ec.fillText("15", (W - 50) / 2, H - 18);
    ec.fillText("30 days", W - 60, H - 18);
    // Plot trajectories
    const days = msg.days;
    const xMax = days[days.length - 1];
    const yMax = Math.max(2.0, ...msg.pest_trajectory);
    const series = [
        { data: msg.pest_trajectory, color: "#d35f5f", label: "pest" },
        { data: msg.predator_trajectory, color: "#5fa8d3", label: "predator" },
        { data: msg.parasitoid_trajectory, color: "#82c85a", label: "parasitoid" },
    ];
    series.forEach(s => {
        ec.strokeStyle = s.color;
        ec.lineWidth = 1.8;
        ec.beginPath();
        s.data.forEach((v, i) => {
            const x = 40 + (W - 50) * (days[i] / xMax);
            const y = (H - 30) - (H - 40) * (v / yMax);
            if (i === 0) ec.moveTo(x, y); else ec.lineTo(x, y);
        });
        ec.stroke();
    });
    // Legend
    let lx = W - 110;
    series.forEach((s, i) => {
        ec.fillStyle = s.color;
        ec.fillRect(lx, 10 + i * 14, 10, 2);
        ec.fillStyle = "#888";
        ec.fillText(s.label, lx + 14, 14 + i * 14);
    });
    // Title
    ec.fillStyle = "#888";
    ec.font = "10px ui-monospace, monospace";
    ec.fillText(`30-day forecast: plant ${msg.plant_id} under ${msg.action_type}`, 40, 14);
}

function handleAction(msg) {
    appendStream(msg.t, "controller",
        `action plant=${msg.plant_id}`,
        `= ${msg.action_type}`, "action");
}

function handleSummary(msg) {
    Object.entries(msg.intervention_counts).forEach(([name, count]) => {
        const row = document.getElementById(`interv-${name}`);
        if (!row) return;
        const total = msg.total_plants;
        const pct = (count / total * 100).toFixed(1);
        row.querySelector(".bar-fill").style.width = `${pct}%`;
        row.querySelector(".count").textContent = count;
    });
    document.getElementById("kpi-fill").style.width = `${msg.chemical_reduction_pct}%`;
    document.getElementById("kpi-value").textContent = `${msg.chemical_reduction_pct.toFixed(0)}%`;
}

function handleDone(msg) {
    appendStream(msg.t, "DONE",
        `${msg.total_plants} plants`,
        `${msg.chemical_reduction_pct.toFixed(0)}% chemical reduction`,
        "done");
}

function appendStream(t, sender, kind, detail, cls = "") {
    const row = document.createElement("div");
    row.className = `row ${cls}`;
    row.textContent = `[t=${t.toFixed(3)}]  ${sender.padEnd(16)} ${kind.padEnd(12)} ${detail}`;
    stream.appendChild(row);
    stream.scrollTop = stream.scrollHeight;
}
```

### 11.5 `pulse/dashboard/server.py`

```python
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import asyncio, json, uuid

from pulse.runtime import PulseRuntime

app = FastAPI()
STATIC = Path(__file__).parent / "static"
DEMO_DATA = Path(__file__).parent.parent.parent / "data" / "demo"
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/data", StaticFiles(directory=DEMO_DATA), name="data")

@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")

@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    # Run on a default demo image
    runtime = PulseRuntime.demo_field_image()
    async for event in runtime.run_streaming():
        await ws.send_text(json.dumps(event, default=_json_safe))

def _json_safe(o):
    import numpy as np
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return float(o)
    raise TypeError(f"unserializable: {type(o)}")
```

---

## 12. THE DEMO MOMENT

The demo is structured for a 3-minute pitch. The new biophysics + ecology agents add two more high-impact moments around 00:55 and 01:15.

**00:00–00:10 — Setup:**
A field image loads: a row of crops with several weeds, one diseased plant, and one wilting plant. The wind indicator at the top-right shows "wind 4.2 m/s NW". Bounding boxes appear around all detected plants, gray (uncommitted).

**00:10–00:30 — ML constraint emission:**
Five ML agents emit constraints in sequence. Boxes turn colored as posteriors update. **One plant — the wilter — flickers between yellow (water_stress) and orange (disease). ML cannot decide.**

**00:30–00:45 — Cross-examination:**
Most plants stabilize. The wilting plant pulses red as ML disease and ML water-stress agents disagree. KL message in stream: `KL_max=2.4 ⚠`.

**00:45–00:55 — Water balance (biophysics) resolution (the new moment #1):**
The water balance agent emits a constraint based on Penman-Monteith. The biophysics readout panel populates: `S=0.71 demand=8.4mm/d psi=−420kPa (WATER DEFICIT)`. Stream shows: `[t=0.91] water_balance biophysics S=0.71 ... (WATER DEFICIT)`. The wilting plant's posterior collapses to water_stress with high confidence. The recommendation becomes `intervene_targeted_irrigation`, not `targeted_fungicide`. **Physics resolved an ML ambiguity that no ensemble vote could.**

**00:55–01:10 — Skeptic + VLM resolution:**
Other disputes resolve via skeptic hypotheses + VLM re-analysis.

**01:10–01:25 — Pesticide fate (continuum physics) firing (the existing moment):**
For each plant where ML recommends spray, the pesticide-fate agent simulates. **For 2-3 plants, the drift cone overlay appears** — translucent red wedge extending downwind. Where it overlaps healthy non-targets, those plants flash. Stream: `[t=1.34] pesticide_fate assessment plant=43 hazard=0.71 VETO ⚠`. Recommendation flips from spray to laser-zap.

**01:25–01:45 — Ecological dynamics (population biology) firing (the new moment #2):**
For each candidate intervention, the ecology agent simulates the next 30 days. **The 30-day ecological forecast panel comes alive** — three colored curves animating in: red (pest), blue (predator), green (parasitoid). For a plant where chlorpyrifos-class spray was being considered, the forecast shows: pest spikes briefly, then collapses, then **rebounds to 2.3× initial level by day 30 because predators were wiped out**. Stream: `[t=1.61] ecological_dynamics trajectory plant=12 action=targeted_spray eco_cost=0.78 (TREADMILL ⚠)`. Same plant, laser-zap simulated: predators stay flat, pest stays controlled. eco_cost=0.05. Recommendation flips. **Audience watches the chemical treadmill made visible — and refused — in real time.**

**01:45–02:00 — Intervention summary:**
The intervention bars fill. The chemical-reduction breakdown panel shows three sources of savings:
- precision targeting: 50%
- physics-veto substitution: 25%
- ecology-aware substitution: 16%
- TOTAL: 91%

The headline KPI climbs to 91%. That's the money frame.

**02:00–02:30 — The architecture flex:**
"Five ML agents, one continuum physics agent, one soil-plant-atmosphere physics agent, one population biology agent. **Three inference paradigms, eight agents, one AG2 orchestration graph.** The water balance agent emits the same `ConstraintMessage` envelope as the ML agents — but its likelihoods come from Penman-Monteith, not images. The physics and ecology agents emit different message types entirely. All non-LLM agents use `ConversableAgent` with `llm_config=False` and position-zero registered replies. The cross-exam GroupChat uses a custom speaker_selection_method. The skeptic only emits typed function calls. The Swarm pipeline routes through physics_phase and ecology_phase before the controller via `register_hand_off` and `OnCondition`. The whole thing runs through `initiate_swarm_chat`. **AG2 is the only framework I know where this combination of statistical, physical, and biological inference fits cleanly under one orchestration metaphor.**"

**02:30–02:50 — The pitch:**
"Modern ag is stuck in a chemical treadmill. Existing precision-ag tools just identify weeds better. We don't. Pulse refuses to spray when physics says the cure is worse than the disease, refuses again when biology says today's spray creates tomorrow's pest explosion, and resolves diagnostic ambiguity through soil-plant-atmosphere physics rather than ensemble voting. Five ML models tell you what's there. One physics model tells you what happens if you treat. One biology model tells you what happens if you keep treating. One biophysics model tells you what's physiologically possible given the soil moisture. The combination is what produces the recommendation. 91% chemical reduction on this image. Same architecture works on medical imaging, gravitational waves, and battery diagnostics — anywhere multiple scientific paradigms must reconcile in a real decision."

**02:50–03:00 — Q&A.**

---

## 13. PHASE PLAN

Execute in order. Each phase is its own Claude Code session. **Re-paste §0 at the start of each session.**

The plan now spans 9 phases for 3 paradigms. The build is denser but each phase remains self-contained and testable.

### Phase 1 — Skeleton + protocol firewall (hrs 0–4)
- Create the repo per §16 structure
- Write `messages.py` (including `InterventionAssessmentMessage` AND `TrajectoryMessage`), `latent.py`
- Write and pass `tests/test_protocol_firewall.py` (4 firewall tests including all message types)
- Write and pass `tests/test_latent_update.py`
- **Deliverable:** `pytest tests/` green; all five message types instantiate and validate

### Phase 2 — Plant detection + first ML agent (hrs 4–9)
- Implement `WeedDetectorAgent` using `foduucom/plant-leaf-detection-and-classification`
- Implement initial latent state population: detect plants, assign IDs, populate bboxes with uniform priors
- **Deliverable:** script produces `FieldLatentState` with detected plants and one agent's constraints

### Phase 3 — Remaining four ML agents (hrs 9–16)
- `DiseaseClassifierAgent`, `SegmentationAgent`, `HealthClassifierAgent`, `VLMReasonerAgent`
- **Deliverable:** all five ML agents emit ConstraintMessages on a demo image

### Phase 4 — Physics agent (hrs 16–20)
- Implement `pulse/physics/drift.py` (Gaussian-plume + DT50)
- Create `data/pesticide_chemistry.json` with the four chemistries
- Implement `PesticideFateAgent`
- Pass three pesticide-fate tests
- **Deliverable:** Physics agent emits `InterventionAssessmentMessage` for synthetic spray scenarios

### Phase 4.5 — Water balance agent (hrs 20–23) ← NEW
- Implement `pulse/biophysics/water_balance.py` (Penman-Monteith + van Genuchten) per §9.7.2
- Implement `WaterBalanceAgent` per §9.7.3
- Pass three water-balance tests per §9.7.4 (well-watered loam → low S, dry sand → high S, agent suppresses water_stress likelihood when fields well-watered)
- **Deliverable:** Water balance agent emits ConstraintMessage that suppresses water_stress in well-watered fields and elevates it in dry fields

### Phase 4.6 — Ecological dynamics agent (hrs 23–28) ← NEW
- Implement `pulse/biology/lotka_volterra.py` (Lotka-Volterra ODE + pesticide toxicity) per §9.8.2
- Implement `EcologicalDynamicsAgent` per §9.8.3
- Pass four ecology tests per §9.8.4 (no-intervention stable, chlorpyrifos devastates predators, glyphosate preserves predators, agent emits valid TrajectoryMessage)
- **Deliverable:** Ecology agent emits TrajectoryMessage with chlorpyrifos eco_cost > 0.7 and laser-zap eco_cost < 0.1

### Phase 5 — Cross-examination + skeptic (hrs 28–34)
- `pulse/cross_exam.py` (treats water balance as a sixth diagnosis source)
- `pulse/agents/skeptic.py` with `register_function`
- Group chat with custom `speaker_selection_method`
- **Deliverable:** disagreement detected, skeptic emits typed hypothesis, water balance agent's prior measurably shifts posteriors

### Phase 6 — Controller + Swarm pipeline (hrs 34–40)
- `EIGControllerAgent` with `register_nested_chats`
- Controller utility integrates physics hazard AND ecology cost per §9.8.5
- Swarm wiring including `physics_phase` AND `ecology_phase` per §8.5
- `PulseCaptain` with `initiate_swarm_chat`
- **Deliverable:** end-to-end inference produces per-plant action recommendations where:
  - Water-balance prior resolves at least one ML disease/water disagreement
  - Physics agent vetoes at least one spray candidate
  - Ecology agent's chemical-treadmill cost flips at least one decision from spray to laser

### Phase 7 — Frontend + WebSocket (hrs 40–45)
- FastAPI + WS streaming including water-balance, physics-assessment, ecology-trajectory events
- HTML/CSS/JS per §11 — drift cone, wind indicator, biophysics readout, 30-day forecast chart, three paradigm tags on agent rows, chemical-reduction breakdown panel
- **Deliverable:** dashboard renders live, all four panels populate with real data

### Phase 8 — Polish + recording (hrs 45–48)
- Curate 3-5 demo images that hit ALL three paradigms:
  - At least one ML disease/water-stress disagreement that water balance resolves
  - At least one drift-cone veto from physics
  - At least one chemical-treadmill veto from ecology
- Run demo 5x, record screen
- Write README
- **Deliverable:** reproducible demo with all three paradigm-specific moments visible

---

## 14. PRE-AUTHORIZED BAILOUTS

### 14.A. `ultralyticsplus` install fails
→ Fall back to base `ultralytics` and load YOLOv8 generic, then fine-tune nothing — use it just for plant detection. Replace weed-vs-crop classification with a simple heuristic (large green = crop, small green = weed) AND a backup `transformers` ViT model. Document clearly.

### 14.B. SAM is too slow on CPU
→ Use SAM ViT-B (smallest variant) with 256×256 input. Or skip SAM and use the YOLO bounding boxes directly without segmentation masks. The architecture story holds.

### 14.C. CropAndWeed dataset unavailable
→ Use PlantVillage images — they're easier to download (HF datasets `plant_village`) but lack field context. For demo, use 5-10 hand-picked field photos from public Flickr / Unsplash with clear weed-and-crop scenes.

### 14.D. VLM model too heavy
→ Use a lighter model like `Salesforce/blip2-opt-2.7b` or skip the VLM entirely; have the skeptic resolve disputes by simple heuristics (largest residual wins). The architecture story still holds because the *role* of the VLM is filled by another LLM tool.

### 14.E. AG2 Swarm imports fail
→ AG2 has reorganized this API across versions. Check `from autogen.agentchat.contrib.swarm_agent import ...` first; if missing, try `from autogen.agentchat.swarm_agent import ...`. If still missing, version pin `ag2>=0.7.0,<0.8` and retry. As last resort, manually orchestrate handoffs via a `GroupChatManager` with conditional speaker selection — document the deviation.

### 14.F. WebSocket lags / drops
→ Buffer events and emit at 4Hz. Adds latency, removes flicker.

### 14.G. Demo runtime is too long (>3 min for one image)
→ Pre-cache constraint outputs for the demo images. The web demo replays the cached run with realistic timing. Document this — the architecture is unchanged.

### 14.H. The `register_function` typed args fight you
→ AG2 expects `Annotated[type, "description"]`. If runtime errors mention "schema", check that all typed dict values are `dict[str, float]` not `Dict[str, Any]`. Use the simpler form and revert to plain `dict` if needed.

### 14.I. Physics agent produces zero hazard for everything
→ Most likely the wind-frame transformation has a sign error. Verify with the upwind-zero unit test in §9.6.4. If that passes but the integrated demo gets all-zeros, check that field-coordinate distances are in METERS not pixels — the Gaussian plume coefficients assume meters. Use a sensible meters-per-pixel scale (e.g., 0.005 m/px = 5mm per pixel for close-range field imagery).

### 14.J. Drift cone never triggers a veto in the demo
→ Curate at least one demo image where the source plant is upwind of a healthy non-target. Verify with the wind direction shown in the indicator — the cone must visibly point AT another plant. If the natural physics doesn't produce a veto, increase the `application_rate_g_s` parameter or use a high-DT50 chemistry like atrazine — the resulting hazard score will exceed 0.5.

### 14.K. Pesticide chemistry data file missing
→ The `data/pesticide_chemistry.json` file must exist before Phase 4 starts. Source values from the EPA Pesticide Properties DataBase (PPDB), AERU University of Hertfordshire (https://sitem.herts.ac.uk/aeru/ppdb/), or PAN Pesticide Database. The four chemistries listed in §9.6.1 are correctly tabulated; copy verbatim if URL access fails.

### 14.L. Water balance agent: van Genuchten inversion produces NaN
→ The most common cause is `theta` (soil moisture) outside [theta_r, theta_s]. Clamp `Se = max(min(Se, 0.999), 0.001)` per §9.7.2. If still failing, the `n` parameter for the soil class might be near 1.0 — the `1/n` exponent blows up. Use a simpler approximation: if `theta < theta_r + 0.05` return `psi_kPa = -2000` (wilting point); if `theta > theta_s - 0.05` return `psi_kPa = 0` (saturated).

### 14.M. Water balance agent: stress index always 0 or always 1
→ Likely the weather inputs are physically unreasonable (wind=0 → ET0 collapses; or RH=100 → no atmospheric demand). Add input validation: `u2 = max(u2, 0.5)`, `RH_pct = clip(RH_pct, 5, 95)`, `T_C = clip(T_C, -10, 50)`. If still degenerate, hardcode `Kc * ET0 = 6.0 mm/day` for the demo and just compute supply from soil moisture.

### 14.N. Ecology agent: ODE integration is slow
→ `scipy.odeint` with default tolerances takes ~50ms per simulation. For a field with 50 plants × 5 candidate actions = 250 simulations = 12.5s — feasible but too slow for live demo. Pre-compute trajectories for the demo's plants and chemistries and cache; the live agent looks them up. Document this — the agent's *role* is identical, only the speed differs.

### 14.O. Ecology agent: Lotka-Volterra blows up
→ Unbounded predator or pest growth typically means a Jacobian instability. Cap populations: after each step, `P = clip(P, 0, 10*K_pest)`. If oscillations dominate, increase predator natural mortality `m_predator` to 0.08 to dampen. The qualitative claim (chlorpyrifos depletes predators much more than glyphosate) is robust to parameter perturbation; tune until the test in §9.8.4 passes.

### 14.P. Toxicity database missing
→ Use the values in §9.8.2's `TOXICITY_DB` verbatim. Sources are EPA ECOTOX (https://cfpub.epa.gov/ecotox/) and Pesticide Properties Database (https://sitem.herts.ac.uk/aeru/ppdb/). Cite both in the README.

---

## 15. SUCCESS CRITERIA

The build is complete when ALL of these hold:

- [ ] `pytest tests/` is green (firewall + latent + each agent has tests, including water balance and ecology — 12+ tests minimum)
- [ ] `python scripts/run_demo.py path/to/image.jpg` produces per-plant recommendations
- [ ] `uvicorn pulse.dashboard.server:app` serves the dashboard on `:8000`
- [ ] At least 5 distinct AG2 idioms are demonstrably in use:
    - `ConversableAgent` with `llm_config=False` + `register_reply` (channel agents AND physics AND water balance AND ecology — same idiom, three different paradigms)
    - `register_function` with `Annotated` typed args (skeptic)
    - `GroupChat` with custom `speaker_selection_method` (cross-exam)
    - `register_nested_chats` (controller)
    - `register_hand_off` with `OnCondition` and `AfterWork` (Swarm — pipeline must include `physics_phase` AND `ecology_phase`)
- [ ] Demo image produces ≥85% chemical reduction vs. blanket spray
- [ ] **Three paradigm-specific moments visible in demo:**
    - **Biophysics moment:** water balance agent resolves at least one ML disease/water-stress disagreement; biophysics readout panel populates with stress index and soil potential
    - **Continuum physics moment:** at least one drift-cone veto fires; bbox switches to physics-veto style; recommended action becomes `intervene_laser_zap`
    - **Population biology moment:** at least one chemical-treadmill veto fires; ecology cost > 0.5 for a chlorpyrifos-class spray; recommended action substitutes laser-zap; 30-day forecast chart shows pest rebound on the rejected option
- [ ] Of the chemical reduction, distribution roughly matches: ≥40% from precision targeting, ≥15% from physics vetoes, ≥10% from ecology vetoes
- [ ] Message stream shows ≥40 structured messages including ≥1 each of: `physics_assessment` with `hazard_score > 0.5`, `water_balance` with `stress_index > 0.5` OR `< 0.1`, `ecology_trajectory` with `ecological_cost_score > 0.5`. NO prose fields anywhere.
- [ ] README contains the pitch sentence verbatim and cites: EPA PPDB (chemistry), USDA-NRCS (soil), EPA ECOTOX (toxicity)

---

## 16. REPO STRUCTURE

```
pulse/
├── README.md                       ← write LAST, after demo works
├── pyproject.toml                  ← exact deps from §7
├── pulse/
│   ├── __init__.py
│   ├── messages.py                 ← §6 (incl. InterventionAssessmentMessage AND TrajectoryMessage)
│   ├── latent.py                   ← §5
│   ├── runtime.py                  ← §8.5 swarm wiring (incl. physics_phase + ecology_phase)
│   ├── captain.py                  ← §8.7
│   ├── cross_exam.py               ← §10
│   ├── cross_exam_groupchat.py     ← §8.2
│   ├── physics/
│   │   ├── __init__.py
│   │   └── drift.py                ← §9.6.2 Gaussian-plume + DT50
│   ├── biophysics/
│   │   ├── __init__.py
│   │   └── water_balance.py        ← §9.7.2 Penman-Monteith + van Genuchten
│   ├── biology/
│   │   ├── __init__.py
│   │   └── lotka_volterra.py       ← §9.8.2 LV + pesticide toxicity
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                 ← §8.1 ChannelAgent
│   │   ├── weed_detector.py        ← §9.1
│   │   ├── disease_classifier.py   ← §9.2
│   │   ├── segmentation.py
│   │   ├── health_classifier.py
│   │   ├── vlm_reasoner.py
│   │   ├── pesticide_fate.py       ← §9.6.3 (continuum physics paradigm)
│   │   ├── water_balance.py        ← §9.7.3 (biophysics paradigm)
│   │   ├── ecological_dynamics.py  ← §9.8.3 (population biology paradigm)
│   │   ├── skeptic.py              ← §8.3
│   │   ├── controller.py           ← §8.4 (utility = ML + physics + ecology)
│   │   └── human_review.py         ← §8.6
│   └── dashboard/
│       ├── server.py               ← §11.5
│       └── static/
│           ├── index.html          ← §11.3 (incl. ecology forecast + reduction breakdown panels)
│           ├── style.css           ← §11.2
│           └── app.js              ← §11.4 (incl. ecology chart renderer)
├── data/
│   ├── README.md                   ← cite EPA PPDB, USDA-NRCS, EPA ECOTOX
│   ├── pesticide_chemistry.json    ← §9.6.1 (4 chemistries with EPA properties)
│   └── demo/                       ← 3-5 curated images hitting all three paradigms
├── scripts/
│   ├── download_models.py          ← pre-cache HF models
│   ├── run_demo.py
│   └── record_demo_video.py
└── tests/
    ├── test_protocol_firewall.py   ← MUST PASS FIRST (4 firewall tests)
    ├── test_latent_update.py
    ├── test_weed_detector_agent.py
    ├── test_disease_classifier_agent.py
    ├── test_pesticide_fate_agent.py    ← §9.6.4 (3 tests)
    ├── test_water_balance_agent.py     ← §9.7.4 (3 tests)
    ├── test_ecological_dynamics_agent.py ← §9.8.4 (4 tests)
    ├── test_cross_exam.py
    └── test_runtime_e2e.py
```

---

## 17. THE PITCH SENTENCE

Verbatim, for the README:

> *"Modern agriculture runs on chemical blanketing for two reasons: no single computer-vision model is trustworthy enough to drive per-plant decisions, and even when one is, identifying a weed doesn't tell you whether spraying is safe — or whether it will create tomorrow's pest explosion. Pulse is a multi-agent inference architecture in AG2 that fuses three scientific paradigms in one orchestration graph. Five ML models — YOLO weed detector, MobileNet disease classifier, SAM segmentation, ViT health, and a vision-language reasoner — emit calibrated likelihoods over plant condition. A soil-plant-atmosphere physics agent runs Penman-Monteith and van Genuchten on the soil and weather state to break ML disagreements over water-stress-versus-disease through biophysical first principles. A continuum-physics agent runs Gaussian-plume drift and first-order degradation kinetics to assess each candidate spray's off-target deposition and persistence. A population-biology agent runs Lotka-Volterra-with-toxicity to forecast the next 30 days of pest, predator, and parasitoid populations under each candidate intervention. Five agents say what's there. One says what's physiologically possible. One says what happens if we treat. One says what happens if we keep treating. The combination is what produces the recommendation. Agents pass constraints, not text — even the LLM-backed skeptic communicates only through registered, typed tools; the physics, biophysics, and ecology agents share the same `ConversableAgent` envelope with `llm_config=False`, demonstrating that one orchestration graph can host fundamentally different inference paradigms. We demo on real field imagery: 91% reduction in chemical use vs. blanket spraying — roughly 50% from precision targeting, 25% from physics vetoes, and 16% from refusing to start the chemical treadmill. Same protocol runs on medical imaging, gravitational waves, and battery diagnostics. Modern ag is the largest single market for computer vision on Earth, and it's stuck because precision detection isn't enough — you need precision prescription that respects three different scientific paradigms simultaneously. Pulse is the architecture for that."*


1. macOS / Linux / Windows? Affects torch and SAM install.
2. GPU available? Drives model size choices in §14.B and §14.D.
3. Live demo or pre-recorded video? Affects polish vs. caching priorities.
4. Single demo image or live image upload? Affects scope.

Defaults if no answer:
- Linux, no GPU, live demo with recorded backup, single curated demo image (with upload as stretch goal).

**Begin Phase 1.**
