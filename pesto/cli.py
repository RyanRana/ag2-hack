"""Pesto CLI entry point — ``pesto`` console script.

Subcommands:

* ``pesto agents`` — list every agent in :data:`pesto.AGENT_REGISTRY`.
* ``pesto interventions`` — list every intervention name.
* ``pesto run IMAGE [--disable AGENT ...] [--only AGENT ...]`` — run the
  pipeline on one image and print the per-plant action recommendation.
* ``pesto serve`` — start the FastAPI dashboard (delegates to uvicorn).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence


def _cmd_agents(args: argparse.Namespace) -> int:
    from pesto.registry import AGENT_REGISTRY, ensure_builtins_registered

    ensure_builtins_registered()
    width = max(len(n) for n in AGENT_REGISTRY) if AGENT_REGISTRY else 0
    for name in sorted(AGENT_REGISTRY):
        spec = AGENT_REGISTRY[name]
        llm = "[llm]" if spec.requires_llm else "     "
        print(f"  {name:<{width}}  {llm}  {spec.role:<10}  {spec.description}")
    return 0


def _cmd_interventions(args: argparse.Namespace) -> int:
    from pesto.registry import INTERVENTION_REGISTRY

    for name in sorted(INTERVENTION_REGISTRY):
        meta = INTERVENTION_REGISTRY[name]
        chem = meta.get("default_chemistry") or "—"
        print(f"  {name:<24}  default_chem={chem}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from pesto import Pipeline, PipelineConfig

    cfg = PipelineConfig()
    if args.only:
        cfg = cfg.only(*args.only)
    if args.disable:
        cfg = cfg.disabled(*args.disable)
    if args.disable_intervention:
        kept = [i for i in (cfg.interventions or []) if i not in args.disable_intervention]
        cfg.interventions = kept or None

    pipe = Pipeline(cfg)
    field_state = json.loads(args.field_state) if args.field_state else {}
    result = pipe.run(args.image, field_state=field_state)
    out = {
        "actions": result.get("actions", []),
        "disputed_plants": result.get("disputed_plants", []),
        "active_agents": pipe.active_agents(),
        "active_interventions": pipe.active_interventions(),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "pesto.dashboard.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pesto")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("agents", help="List registered agents.").set_defaults(func=_cmd_agents)
    sub.add_parser("interventions", help="List registered interventions.").set_defaults(
        func=_cmd_interventions
    )

    p_run = sub.add_parser("run", help="Run the pipeline on one image.")
    p_run.add_argument("image", help="Path to an image file.")
    p_run.add_argument("--disable", nargs="+", default=[], help="Disable agents by name.")
    p_run.add_argument("--only", nargs="+", default=[], help="Enable only these agents.")
    p_run.add_argument(
        "--disable-intervention",
        nargs="+",
        default=[],
        help="Remove these interventions from the action space.",
    )
    p_run.add_argument(
        "--field-state",
        default=None,
        help="JSON string of field-state context (wind, soil, weather).",
    )
    p_run.set_defaults(func=_cmd_run)

    p_serve = sub.add_parser("serve", help="Run the dashboard with uvicorn.")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", default=8000, type=int)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
