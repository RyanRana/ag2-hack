"""Pulse demo runner — Phase 2 minimum.

Detects plants in a field image, runs the WeedDetectorAgent, applies its
constraint to the FieldLatentState, and prints the resulting per-plant
posteriors. Later phases will append the four other ML agents, then physics,
biophysics, and ecology.

Usage:
    python scripts/run_demo.py path/to/field.jpg
"""

from __future__ import annotations

import argparse
import json
import sys

from pulse.agents.weed_detector import WeedDetectorAgent
from pulse.detection import detect_plants_yolo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Pulse precision-ag inference.")
    parser.add_argument("image", help="Path to a field image (JPG/PNG)")
    parser.add_argument(
        "--max-plants",
        type=int,
        default=64,
        help="Cap on number of plants to retain from initial detection.",
    )
    args = parser.parse_args(argv)

    agent = WeedDetectorAgent()
    yolo = agent._load_model()

    print(f"Detecting plants in {args.image} ...", flush=True)
    latent = detect_plants_yolo(args.image, yolo, max_plants=args.max_plants)
    print(
        f"Detected {len(latent.plants)} plants "
        f"in {latent.image_shape[1]}x{latent.image_shape[0]} image"
    )

    print("Running WeedDetectorAgent ...", flush=True)
    msg = agent.emit_constraint(args.image, latent)
    for plant in latent.plants:
        latent.update_plant(
            plant.plant_id,
            msg.per_plant_log_likelihoods[plant.plant_id],
            "weed_detector",
        )

    out = latent.to_dict()
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
