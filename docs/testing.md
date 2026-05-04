# Testing

The test suite covers the inter-agent firewall, calibration, agent
behaviors, and end-to-end orchestration.

## Run the tests

```bash
# Core tests
.venv/bin/python -m pytest tests/ --ignore=tests/test_rag.py

# RAG tests (run separately due to FAISS+torch process interaction)
.venv/bin/python -m pytest tests/test_rag.py
```

## What's covered

* **Protocol firewall** (`tests/test_protocol_firewall.py`) — inter-agent
  messages contain no prose fields. If this fails, Pesto has degraded
  into a chatbot.
* **Temperature scaling** — calibration math and learned-T application.
* **Visual explanation** — overlay rendering correctness.
* **Weather prior** — Open-Meteo response → log-likelihood adjustments.
* **DINOv2 backbone** — patch embedding shape + dimensionality.
* **PatchCore anomaly scoring** — memory-bank distance score.
* **Growth stage classification** — heuristic boundaries.
* **Local model backend** — structured output parsing for Skeptic / VLM.
* **RAG retrieval** — FAISS index + utility adjustment.
* **Multi-turn debate** — Skeptic local inference and convergence check.
* **Temporal diff** — frame differencing + optical-flow change scores.
* **Active learning queue** — hard-case enqueueing rules.
* **Detection merging** — YOLO + Grounding DINO NMS dedup.
* **All agent unit tests** — segmentation, cross-exam, latent updates,
  controller utility, physics, ecology, weed detector, disease
  classifier, health classifier, water balance.

## Pre-requisites

A few tests require:

* `data/pesticide_chemistry.json` — chemistry parameters used by the
  pesticide-fate physics agent. The file is gitignored; create it from
  the PPDB / PAN dump or copy from a teammate's working install.
* `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` — only for tests that
  exercise the live LLM path. Local-mode tests stub the backend.
* HuggingFace cache populated (`python scripts/download_models.py`) —
  for tests that load YOLOv8 / MobileNetV2 / ViT weights.

## Writing a new test

Match the project's existing patterns:

* Import only from `pesto.…`.
* Build a tiny `FieldLatentState` directly — don't rely on YOLO
  detection in unit tests.
* Assert on log-likelihood vector signs and shapes, not on absolute
  magnitudes (the agents are calibrated, not deterministic).
* Use the `firewall` test as the canonical "no prose" assertion.

```python
from pesto.latent import FieldLatentState, PlantInstance
from pesto.messages import ConstraintMessage

def test_my_agent_emits_a_constraint():
    latent = FieldLatentState(image_shape=(640, 640))
    latent.plants.append(PlantInstance(plant_id=0, bbox=(0, 0, 640, 640)))

    msg = MyAgent().emit_constraint("path/to/test.jpg", latent)

    assert isinstance(msg, ConstraintMessage)
    assert msg.sender == "my_agent"
    assert 0 in msg.per_plant_log_likelihoods
```
