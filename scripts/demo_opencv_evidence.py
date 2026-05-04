"""End-to-end demo of the OpenCV Evidence Agent via AG2 dispatch.

Unlike scripts/run_demo.py (which calls emit_constraint directly), this
script exercises the full AG2 conversational path:

    UserProxyAgent.initiate_chat(SegmentationAgent, message=<json>)
        -> ChannelAgent._emit_constraint_reply (registered at position 0)
        -> SegmentationAgent.emit_constraint(image_path, latent)
        -> ConstraintMessage as JSON reply

Then it pulls the spatial evidence the agent stored on
``agent.evidence_maps`` (PlantEvidenceMaps per plant) and saves an
overlay PNG via ``pesto.visual_explain.save_explanation``. Because those
masks are the same arrays whose pixel-counts produced the scalar
log-likelihoods, the overlay is bit-exactly the evidence that drove the
decision (Explanation === Evidence).

Usage:
    python scripts/demo_opencv_evidence.py
    python scripts/demo_opencv_evidence.py --image path/to/leaf.jpg
    python scripts/demo_opencv_evidence.py --image foo.jpg --out overlays/

If --image is not provided, a synthetic chlorotic-leaf swatch is
generated on the fly so the demo runs with no setup.

No LLM API key is required — the OpenCV Evidence Agent is local
code-only inference. (Skeptic + VLMReasoner elsewhere in Pesto do
require ANTHROPIC_API_KEY or OPENAI_API_KEY, loaded from .env.)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Make ``pesto`` importable when this script is run directly, regardless of
# whether the package was installed editable. ``python script.py`` puts the
# script's own directory on sys.path[0]; the package lives one level up.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from autogen import UserProxyAgent
from PIL import Image

from pesto.agents.segmentation import SegmentationAgent
from pesto.latent import CONDITION_LABELS, FieldLatentState, PlantInstance
from pesto.visual_explain import save_explanation


def _make_synthetic_image(out_path: Path, size: int = 256) -> None:
    """Generate a chlorotic-leaf-like swatch — mostly yellow with green
    fringes — so the segmentation agent has something meaningful to
    flag without needing real data on disk.
    """
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[..., 0] = 180  # R
    arr[..., 1] = 180  # G
    arr[..., 2] = 60   # B  -> overall yellow
    border = max(8, size // 16)
    arr[:border, :] = (60, 130, 40)        # green top
    arr[-border:, :] = (60, 130, 40)       # green bottom
    arr[:, :border] = (60, 130, 40)        # green left
    arr[:, -border:] = (60, 130, 40)       # green right
    Image.fromarray(arr, "RGB").save(out_path, "JPEG", quality=92)


def _build_latent(image_path: str) -> FieldLatentState:
    """Whole-image bbox so the agent inspects the entire frame.

    Production pipelines call ``pesto.detection.detect_plants_yolo`` to
    populate per-plant bboxes. This demo keeps the focus on the
    segmentation agent itself, not detection.
    """
    w, h = Image.open(image_path).size
    latent = FieldLatentState(image_shape=(h, w))
    latent.plants.append(PlantInstance(plant_id=0, bbox=(0, 0, w, h)))
    return latent


def _dispatch_via_ag2(
    seg_agent: SegmentationAgent,
    latent: FieldLatentState,
    image_path: str,
) -> dict:
    """Round-trip a JSON envelope through the AG2 reply machinery."""
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
        "image_path": image_path,
    })
    user.initiate_chat(
        seg_agent,
        message=envelope,
        max_turns=1,
        clear_history=True,
        silent=True,
    )
    history = user.chat_messages.get(seg_agent, [])
    if not history:
        raise RuntimeError("AG2 dispatch produced no reply from segmentation agent")
    return json.loads(history[-1]["content"])


def _print_constraint_summary(constraint: dict, evidence_maps: dict) -> None:
    print()
    print(f"  sender               = {constraint['sender']}")
    print(f"  iteration            = {constraint['iteration']}")
    print(f"  labels_discriminated = {constraint['labels_discriminated']}")
    print(f"  evidence_maps keys   = {sorted(evidence_maps.keys())}")
    print()
    per_ll = constraint["per_plant_log_likelihoods"]
    per_resid = constraint["per_plant_residual"]
    per_conf = constraint["per_plant_confidence"]
    for pid_str, ll in per_ll.items():
        ll = np.asarray(ll, dtype=float)
        order = np.argsort(ll)[::-1][:2]
        top = [(CONDITION_LABELS[i], round(float(ll[i]), 3)) for i in order]
        ev = evidence_maps.get(int(pid_str))
        feats = ev.features if ev else {}
        print(f"  plant #{pid_str}: top2={top}")
        print(
            f"           residual={per_resid[pid_str]:.3f}  "
            f"confidence={per_conf[pid_str]:.3f}  "
            f"green={feats.get('green_ratio', 0):.3f}  "
            f"yellow={feats.get('yellowness', 0):.3f}  "
            f"edge={feats.get('edge_density', 0):.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--image",
        default=None,
        help="Input image path. If omitted, a synthetic chlorotic swatch is generated.",
    )
    parser.add_argument("--out", default="overlays", help="Output directory for the overlay PNG.")
    args = parser.parse_args(argv)

    synthetic_dir: tempfile.TemporaryDirectory | None = None
    if args.image:
        image_path = str(Path(args.image).resolve())
        if not Path(image_path).exists():
            print(f"ERROR: image not found: {image_path}", file=sys.stderr)
            return 2
    else:
        synthetic_dir = tempfile.TemporaryDirectory(prefix="pesto_demo_")
        synthetic_path = Path(synthetic_dir.name) / "chlorotic_swatch.jpg"
        _make_synthetic_image(synthetic_path)
        image_path = str(synthetic_path)
        print(f"[i] No --image given. Using synthetic swatch: {image_path}")

    try:
        print(f"[1/4] Building latent for {image_path}")
        latent = _build_latent(image_path)
        print(f"      plants in latent  = {len(latent.plants)}")

        print("[2/4] Constructing AG2 agent (autogen.ConversableAgent subclass)")
        seg_agent = SegmentationAgent()
        print(f"      agent name        = {seg_agent.name}")
        print(f"      llm_config        = {seg_agent.llm_config}  (code-only)")

        print("[3/4] Dispatching via UserProxyAgent.initiate_chat -> ChannelAgent reply")
        constraint = _dispatch_via_ag2(seg_agent, latent, image_path)
        if not seg_agent.evidence_maps:
            print("ERROR: agent.evidence_maps is empty — segmentation did not retain masks", file=sys.stderr)
            return 1
        _print_constraint_summary(constraint, seg_agent.evidence_maps)

        print("[4/4] Rendering overlay PNG from stored evidence_maps")
        out_dir = Path(args.out).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (Path(image_path).stem + ".overlay.png")
        result_path = save_explanation(image_path, seg_agent.evidence_maps, out_path)
        print(f"      overlay PNG       = {result_path}")
        print()
        print("Done. Open the PNG to confirm:")
        print("  yellow tint  = chlorotic / yellow tissue (yellow_mask)")
        print("  magenta tint = high-gradient perforations (edge_mask)")
        print("  red lines    = Canny contour edges        (contour_mask)")
        print("  green tint   = healthy leaf area          (leaf_mask)")
        print("  green box    = plant bounding box")
        return 0
    finally:
        if synthetic_dir is not None:
            synthetic_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
