# Agents Reference

Pesto ships twelve agents. Ten run locally with no API calls; two require
an LLM key and only fire on disputed plants.

## Local agents

| # | Name (registry key) | Paradigm | Implementation |
|---|---------------------|----------|----------------|
| 1 | `weed_detector` | ML | `foduucom/plant-leaf-detection-and-classification` (YOLOv8, ~25 MB) |
| 2 | `disease_classifier` | ML | `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` (MobileNetV2, ~14 MB) |
| 3 | `health_classifier` | ML | `Diginsa/Plant-Disease-Detection-Project` (ViT, ~90 MB) |
| 4 | `segmentation` | CV | OpenCV HSV / Canny / contour (no neural net) |
| 5 | `anomaly_detector` | CV | `facebook/dinov2-base` + PatchCore (~350 MB, lazy-loaded) |
| 6 | `growth_stage` | ML | Heuristic (green ratio + size + bright regions) |
| 7 | `weather_prior` | Physics | Open-Meteo HTTP API + rule-based priors |
| 8 | `water_balance` | Biophysics | FAO-56 Penman-Monteith + van Genuchten |
| 9 | `pesticide_fate` | Physics | Gaussian plume drift + first-order DT50 |
| 10 | `ecological_dynamics` | Biology | Lotka-Volterra ODE + toxicity LC50 |

## LLM agents

| # | Name (registry key) | Paradigm | Backend |
|---|---------------------|----------|---------|
| 11 | `vlm_reasoner` | ML / LLM | OpenAI / Anthropic via AG2 `AssistantAgent` |
| 12 | `skeptic` | Meta | OpenAI / Anthropic via AG2 `AssistantAgent` |

## What each agent does

### `weed_detector`

YOLOv8 on the full frame to detect and classify plants as weed or crop.
Produces bounding boxes and initial weed-vs-crop probabilities. This is
phase 0 detection — every downstream agent operates on these crop regions.

### `disease_classifier`

MobileNetV2 trained on 38 PlantVillage classes (early blight, late blight,
powdery mildew, rust, …). Maps the 38 fine-grained classes to Pesto's 7
condition labels. Temperature scaling (learned scalar T) calibrates the
logits before softmax so the model doesn't dominate the posterior.

### `health_classifier`

Coarse binary classifier — "is something wrong with this plant?" Can't
say *what's* wrong, but is calibrated to distinguish healthy from
unhealthy with low compute. Acts as a tiebreaker when the disease
classifier and segmentation disagree. Also temperature-scaled.

### `segmentation`

Pure OpenCV analysis on each plant crop. Computes HSV-based green leaf
masks, yellow tissue detection, Canny edge detection for perforations,
and contour analysis. Produces both scalar features (green_ratio,
yellowness, edge_density) for log-likelihood computation AND retains
the spatial masks for the visual explanation overlay. **Every overlay
pixel comes from the same computation that produced the log-likelihood.**

### `anomaly_detector`

Catches *unknown-unknowns* that classifiers would force into a known
label. Extracts DINOv2 patch embeddings from each plant crop and scores
them against a PatchCore memory bank of healthy plant patches. Plants
far from the healthy distribution get mass pushed toward "ambiguous",
triggering human review.

### `growth_stage`

Classifies each plant as seedling / vegetative / flowering / fruiting
using visual heuristics. Provides an urgency multiplier to the
controller — a seedling with disease needs immediate action; a mature
plant with the same disease can wait.

### `weather_prior`

Runs **before** all ML agents. Fetches 7-day weather history from
Open-Meteo (free, no API key) and adjusts the shared prior:

* 5+ dry days + high temp → P(water_stress) ↑
* High humidity + warm → P(fungal disease) ↑
* Heavy recent rain → P(nutrient_stress) ↑ (leaching)
* Cool + dry → all stress priors ↓

ML agents start from a weather-informed prior instead of a uniform one.

### `water_balance`

Resolves the canonical ML ambiguity: "is this plant wilting from disease
or from water stress?" Runs FAO-56 Penman-Monteith evapotranspiration
demand against van Genuchten soil water retention to compute a
physics-based stress index. No pixels involved.

### `pesticide_fate`

Evaluates "what happens if we spray?" for each candidate action. Runs a
Gaussian plume atmospheric dispersion model to compute off-target
deposition on neighboring plants and waterways, plus first-order DT50
soil-persistence kinetics. **Drift > 0.4 ppm → VETO the spray.**

### `ecological_dynamics`

Evaluates "what happens to the ecosystem if we treat?" Runs a
Lotka-Volterra predator-prey-parasitoid ODE with pesticide toxicity
parameters for the next 30 days. **Predator population collapse > 50%
within 14 days → VETO the chemical.**

### `vlm_reasoner`

Only fires on *disputed* plants (KL > 1.5). Receives the actual crop
image plus a structured prompt; looks for specific visual cues
(concentric rings → fungal, water-soaked margins → bacterial, clean
tears → mechanical). Outputs calibrated log-likelihoods per condition.
Uses AG2's `AssistantAgent` with typed tool calls
(`analyze_disagreement_region`, `submit_per_plant_likelihoods`).

### `skeptic`

The devil's advocate. Proposes alternative hypotheses ("what if this
disease is actually nutrient stress?") and engages in a multi-turn
debate with the VLM (max 3 rounds). Converges when posterior entropy
drops below threshold.

## Non-toggleable infrastructure agents

* `EIGControllerAgent` — utility-based action selection. Computes
  expected utility for all 8 intervention types per plant; picks the
  argmax. Uses AG2 `register_nested_chats` to spawn one
  `ActionEvaluator` per intervention type.
* `HumanReviewProxy` — escalation when the controller's best action is
  `human_review` (entropy too high, no confident diagnosis).

## Listing the registry

```python
from pesto import list_agents, AGENT_REGISTRY

for name in list_agents():
    spec = AGENT_REGISTRY[name]
    print(f"{name:<22} role={spec.role:<10} llm={spec.requires_llm}  {spec.description}")
```

Or from the CLI:

```bash
pesto agents
```
