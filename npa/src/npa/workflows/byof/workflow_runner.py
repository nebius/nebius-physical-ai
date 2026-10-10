"""Generic YAML workflow runner.

New workflows are YAML files, not CLI subcommands. This runner executes
the stages defined in a workflow YAML in order, passing a shared context.

Usage:
    python -m npa.workflows.byof.workflow_runner workflow.yaml [--param k=v]

Each stage declares:
    name:      Stage name.
    uses:      "module:function" implementing the stage.
    resources: (optional) Resource requirements, e.g. accelerators for GPUs.
               These are declarative for the scheduler; the local runner
               logs them and executes in-process.
    description: (optional) Human-readable description.

Stage functions receive a dict context and return a dict of outputs,
which are merged into the context for downstream stages.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

import yaml


def _load_stage(dotted: str):
    module_name, func_name = dotted.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def run_workflow(
    yaml_path: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Execute a workflow YAML and return the final context."""
    with open(yaml_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    context: dict[str, Any] = dict(spec.get("params", {}))
    if params:
        context.update(params)

    print(f"Workflow: {spec.get('name')} - {spec.get('description', '')}")
    for stage in spec.get("stages", []):
        name = stage["name"]
        uses = stage["uses"]
        resources = stage.get("resources", {})
        print(f"\n[stage: {name}] {stage.get('description', '')}")
        print(f"  uses: {uses}")
        if resources:
            print(f"  resources: {json.dumps(resources)}")
            if "accelerators" in resources:
                print(
                    "  -> GPU stage: submit to Nebius GPU job for real "
                    "acceleration; running in-process here."
                )
        func = _load_stage(uses)
        outputs = func(context)
        if isinstance(outputs, dict):
            context.update(outputs)
        print(f"  done: {name}")

    return context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a YAML workflow.")
    parser.add_argument("workflow", help="Path to workflow YAML.")
    parser.add_argument(
        "--param", action="append", default=[], help="Override param as k=v."
    )
    parser.add_argument("--output", default="", help="Write final context JSON here.")
    args = parser.parse_args(argv)

    params: dict[str, Any] = {}
    for p in args.param:
        k, _, v = p.partition("=")
        # try int/float, else string
        try:
            params[k] = int(v)
        except ValueError:
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v

    context = run_workflow(args.workflow, params)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2, default=str)
        print(f"\nContext written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
