# Interventions

The eight built-in intervention types live in `pesto.INTERVENTION_TYPES`:

| Name | Meaning |
|------|---------|
| `no_action` | Plant looks healthy; do nothing |
| `laser_zap` | Mechanical weed removal (no chemistry) |
| `targeted_spray` | Spot-applied herbicide for a confirmed weed |
| `targeted_fungicide` | Spot-applied fungicide / insecticide for a confirmed disease or pest |
| `targeted_irrigation` | Local water application for water-stressed plant |
| `foliar_nutrient` | Foliar feed for nutrient-deficient plant |
| `human_review` | Posterior too uncertain — escalate to expert |
| `rescan_higher_res` | Re-image at higher resolution and try again |

The controller picks the argmax-utility action for each plant. Physics +
ecology can veto chemical actions by inflating their hazard / cost
penalty.

## Restricting the action space

```python
from pesto import Pipeline, PipelineConfig

# Organic mode — no chemical sprays.
pipe = Pipeline(PipelineConfig(
    interventions=[
        "no_action",
        "laser_zap",
        "targeted_irrigation",
        "foliar_nutrient",
        "human_review",
        "rescan_higher_res",
    ],
))
```

Or mutate after construction:

```python
pipe = Pipeline()
pipe.disable_intervention("targeted_spray")
pipe.disable_intervention("targeted_fungicide")
```

## Adding a new intervention

Three steps:

### 1. Register the name + default chemistry

```python
from pesto import register_intervention

register_intervention("targeted_uv_treatment", default_chemistry=None)
```

### 2. Add a utility branch

The controller's utility function is in
`pesto/agents/controller.py::EIGControllerAgent.compute_utility`. It
maps `(plant_posterior, action_type)` to a base utility, then subtracts
physics + ecology penalties. To add a branch, edit the `base = {…}`
literal:

```python
base = {
    ...
    "targeted_uv_treatment": p_disease * 0.80 + p_pest * 0.40 - 0.05,
    ...
}.get(action_type, 0.0)
```

The same pattern goes into `_heuristic_eig` so the EIG estimate has a
weight for your action.

### 3. Use it

```python
pipe = Pipeline(PipelineConfig(
    interventions=list(pesto.INTERVENTION_TYPES) + ["targeted_uv_treatment"],
))
```

## Default chemistry

Default chemistries per spray action are deliberately broad-spectrum so
the vetoes are visible in the demo:

| Action | Default chemistry | Why |
|--------|-------------------|-----|
| `targeted_spray` | `chlorpyrifos` | Crashes the predator population in the Lotka-Volterra simulation — ecology vetoes this even when ML is confident the plant is a weed |
| `targeted_fungicide` | `atrazine` | Long DT50 + high mobility → drift veto on most days the wind is moving |

Override via `PipelineConfig.chemistry_per_action`:

```python
PipelineConfig(chemistry_per_action={
    "targeted_spray": "spinosad",          # softer on predators
    "targeted_fungicide": "copper_hydroxide",
})
```

The chemistry id must exist in `data/pesticide_chemistry.json` (PPDB +
PAN). Unknown ids fall back to the agent's internal default
(`glyphosate`).

## Vetoes — how they work

Vetoes are not hard rules; they're soft penalties that make a chemical
action's utility drop below `laser_zap` or `human_review`. The total
utility:

```
U(action) = base(plant_posterior, action)
            - 0.5 × physics_hazard_score
            - 0.4 × ecological_cost_score
```

Hazard score thresholds that typically flip the argmax:

* Drift > 0.4 ppm at any neighbor → `physics_hazard_score` ≈ 0.6+
* Predator population drop > 50 % within 14 days → `ecological_cost_score` ≈ 0.7+

Either alone usually pushes a spray below the laser-zap utility on a
confirmed weed; both together also push it below `no_action` for any
non-weed.
