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



def _submit_nebius_gpu_job(stage: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Submit a GPU stage to Nebius and wait for completion. Returns job logs."""
    import subprocess
    import time

    name = stage["name"]
    resources = stage.get("resources", {})
    accelerators = resources.get("accelerators", "")
    platform = "gpu-h100-sxm"
    if "rtxpro" in accelerators.lower() or "rtx" in accelerators.lower():
        platform = "gpu-l40s-d"

    image = stage.get("nebius_image", "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    files = stage.get("nebius_files", [])
    command = stage["nebius_command"]

    cmd = [
        "nebius", "--profile", "agent-sa", "ai", "job", "create",
        "--name", f"workflow-{name}",
        "--image", image,
        "--platform", platform,
        "--preset", "1gpu-16vcpu-200gb",
        "--subnet-id", "NEBIUS_SUBNET_ID_PLACEHOLDER",
    ]
    for f in files:
        cmd += ["--inject-file", f]
    cmd += ["--container-command", command, "--async"]

    print(f"  submitting Nebius GPU job for stage '{name}'...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Nebius job submit failed: {result.stderr}")
    job_id = result.stdout.strip().split(": ")[1]
    print(f"  job ID: {job_id}")

    for _ in range(60):
        time.sleep(30)
        r = subprocess.run(
            ["nebius", "--profile", "agent-sa", "ai", "job", "list"],
            capture_output=True, text=True, timeout=30,
        )
        for line in r.stdout.split("\n"):
            if job_id in line:
                if "COMPLETED" in line:
                    print(f"  job {job_id} COMPLETED")
                    lr = subprocess.run(
                        ["nebius", "--profile", "agent-sa", "ai", "job", "logs",
                         job_id, "--tail", "100"],
                        capture_output=True, text=True, timeout=30,
                    )
                    return {"job_id": job_id, "logs": lr.stdout, "platform": platform}
                elif "FAILED" in line:
                    raise RuntimeError(f"Nebius job {job_id} FAILED")
                break
    raise RuntimeError(f"Nebius job {job_id} timed out")


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
        if "accelerators" in resources and "nebius_script" in stage:
            outputs = _submit_nebius_gpu_job(stage, context)
        else:
            if "accelerators" in resources:
                print("  -> GPU requested but no nebius_script; running in-process.")
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
