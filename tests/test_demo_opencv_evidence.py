"""Integration tests for the OpenCV Evidence Agent AG2 dispatch demo.

Covers the surfaces that matter for the demo path:
  • AG2 conversational dispatch — UserProxyAgent.initiate_chat into the
    SegmentationAgent's registered reply (ChannelAgent._emit_constraint_reply
    at position 0). Confirms the agent really is exercised through AG2 and
    not just by direct emit_constraint calls.
  • End-to-end script smoke — running scripts/demo_opencv_evidence.py on a
    fixture image produces a non-empty overlay PNG and exits 0.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from autogen import UserProxyAgent
from PIL import Image

from pulse.agents.segmentation import PlantEvidenceMaps, SegmentationAgent
from pulse.latent import FieldLatentState, PlantInstance


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_SCRIPT = REPO_ROOT / "scripts" / "demo_opencv_evidence.py"


def _make_yellow_image(path: Path, size: int = 96) -> None:
    Image.new("RGB", (size, size), color=(180, 180, 60)).save(path)


def _make_latent(size: int = 96) -> FieldLatentState:
    latent = FieldLatentState(image_shape=(size, size))
    latent.plants.append(PlantInstance(plant_id=0, bbox=(0, 0, size, size)))
    return latent


def test_ag2_dispatch_populates_evidence_maps(tmp_path):
    """The agent must answer through the registered AG2 reply chain AND
    retain its evidence_maps as a side effect.

    We deliberately do NOT call emit_constraint() directly — only
    initiate_chat. If the AG2 wiring breaks (e.g. someone removes
    register_reply, or changes the JSON envelope shape), this test
    fails first.
    """
    img_path = tmp_path / "yellow.jpg"
    _make_yellow_image(img_path)
    latent = _make_latent()
    seg_agent = SegmentationAgent()

    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_execution_config=False,
        llm_config=False,
        max_consecutive_auto_reply=0,
        default_auto_reply="",
    )
    envelope = json.dumps({
        "latent": latent.to_dict(),
        "image_path": str(img_path),
    })
    user.initiate_chat(
        seg_agent,
        message=envelope,
        max_turns=1,
        clear_history=True,
        silent=True,
    )

    history = user.chat_messages.get(seg_agent, [])
    assert history, "AG2 dispatch produced no reply"
    reply = json.loads(history[-1]["content"])

    assert reply["sender"] == "segmentation"
    assert "per_plant_log_likelihoods" in reply
    plant_keys = list(reply["per_plant_log_likelihoods"].keys())
    assert plant_keys, "expected at least one plant in the reply"

    # The evidence_maps side-channel must have been populated as a
    # side effect of dispatch — not a separate call.
    assert 0 in seg_agent.evidence_maps
    ev = seg_agent.evidence_maps[0]
    assert isinstance(ev, PlantEvidenceMaps)
    assert ev.leaf_mask.size > 0
    assert ev.yellow_mask.size > 0
    assert ev.edge_mask.size > 0
    # On a yellow swatch the yellow_mask should fire on most pixels.
    assert ev.yellow_mask.mean() > 0.5


def test_demo_script_smoke(tmp_path):
    """Run scripts/demo_opencv_evidence.py and confirm it produces an overlay PNG."""
    img_path = tmp_path / "yellow.jpg"
    _make_yellow_image(img_path, size=128)
    out_dir = tmp_path / "overlays"

    proc = subprocess.run(
        [
            sys.executable,
            str(DEMO_SCRIPT),
            "--image",
            str(img_path),
            "--out",
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"demo script exited {proc.returncode}\nSTDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    overlay = out_dir / (img_path.stem + ".overlay.png")
    assert overlay.exists(), f"missing overlay: {overlay}"
    assert overlay.stat().st_size > 0


def test_demo_script_synthetic_default_runs(tmp_path):
    """Without --image the script should generate a synthetic swatch and still emit a PNG."""
    out_dir = tmp_path / "overlays"
    proc = subprocess.run(
        [sys.executable, str(DEMO_SCRIPT), "--out", str(out_dir)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"demo script exited {proc.returncode}\nSTDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    pngs = list(out_dir.glob("*.overlay.png"))
    assert pngs, f"no overlay PNG written into {out_dir}"
    assert pngs[0].stat().st_size > 0
