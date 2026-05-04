# Pesto Documentation

Pesto is a multi-agent inference SDK for precision agriculture. The package
ships twelve agents across four scientific paradigms (machine learning,
computer vision, continuum physics, population biology), wires them with
[AG2](https://github.com/ag2ai/ag2), and exposes a small SDK for assembling
custom pipelines.

## Where to start

| If you want to… | Read |
|-----------------|------|
| Run inference on an image in 5 lines | [SDK Quickstart](sdk-quickstart.md) |
| Understand the pipeline phases and agents | [Architecture](architecture.md) |
| Plug your own model into the pipeline | [Custom Agents](custom-agents.md) |
| Disable an agent or restrict the action space | [Configuration](configuration.md) |
| Look up a built-in agent's behavior | [Agents Reference](agents.md) |
| Add a new intervention type | [Interventions](interventions.md) |
| Run the live dashboard | [Dashboard](dashboard.md) |
| Run the test suite | [Testing](testing.md) |
| Use the `pesto` CLI | [CLI](cli.md) |

## Top-level surface

```python
import pesto

pesto.Pipeline                  # high-level orchestrator
pesto.PipelineConfig            # declarative config dataclass
pesto.PestoCaptain              # underlying captain (advanced use)
pesto.register_agent            # decorator: add a custom agent
pesto.register_intervention     # function: add a new action type
pesto.list_agents()             # introspect the registry
pesto.CONDITION_LABELS          # the 7 plant-condition labels
pesto.INTERVENTION_TYPES        # the 8 built-in actions
pesto.ConstraintMessage         # typed inter-agent payload
pesto.ActionMessage             # the controller's per-plant output
```

Everything else is internal — the modules under `pesto.agents`,
`pesto.physics`, `pesto.biology`, `pesto.biophysics` etc. are stable but
treat them as semi-private.
