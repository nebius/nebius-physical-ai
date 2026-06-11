#!/usr/bin/env python
"""Driver for container golden evals.

Modes:
  validate   Offline manifest completeness/consistency check (nightly CI gate).
  list       Print the golden-eval table (optionally JSON).
  run        Execute a single container's golden-eval command.

The actual GPU golden evals run on hosts that have the container runtime; this
driver is the single entrypoint used by both CI (validate) and operators (run).

Usage:
  npa/.venv/bin/python npa/scripts/run_golden_evals.py validate
  npa/.venv/bin/python npa/scripts/run_golden_evals.py list --json
  npa/.venv/bin/python npa/scripts/run_golden_evals.py run lerobot --execute
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys

from npa.deploy.images import CONTAINER_IMAGE_NAMES
from npa.smoke.manifest import container, load_manifest, validate_manifest


def _cmd_validate(_: argparse.Namespace) -> int:
    report = validate_manifest(expected_tools=set(CONTAINER_IMAGE_NAMES))
    if report.ok:
        print(f"OK: {len(load_manifest())} containers have valid golden-eval entries")
        return 0
    print("Golden-eval manifest validation failed:")
    for issue in report.issues:
        print(f"  - {issue}")
    return 1


def _cmd_list(args: argparse.Namespace) -> int:
    specs = load_manifest()
    if args.json:
        payload = {
            name: {
                "image": spec.image,
                "kind": spec.golden_eval.kind,
                "gpu": spec.golden_eval.gpu,
                "status": spec.golden_eval.status,
                "command": spec.golden_eval.command,
            }
            for name, spec in specs.items()
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    width = max(len(name) for name in specs)
    for name, spec in specs.items():
        ge = spec.golden_eval
        print(f"{name:<{width}}  {ge.kind:<16} gpu={ge.gpu:<8} {ge.status:<18} {ge.command}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        spec = container(args.container)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2
    ge = spec.golden_eval
    print(f"{spec.name} ({spec.image}) golden eval: {ge.kind}, gpu={ge.gpu}")
    print(f"  $ {ge.command}")

    if args.serverless:
        from npa.smoke.serverless_runner import submit_golden_eval

        result = submit_golden_eval(
            args.container,
            gpu_type=args.gpu or None,
            timeout=args.timeout,
            on_state_change=lambda job: print(f"  -> {getattr(job, 'status', '?')}"),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 1

    if not args.execute:
        return 0
    try:
        completed = subprocess.run(
            shlex.split(ge.command), timeout=ge.timeout_seconds, check=False
        )
    except FileNotFoundError as exc:
        print(f"command not runnable here: {exc}", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired:
        print(f"golden eval timed out after {ge.timeout_seconds}s", file=sys.stderr)
        return 124
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Offline manifest validation (CI gate).")
    p_validate.set_defaults(func=_cmd_validate)

    p_list = sub.add_parser("list", help="List golden evals.")
    p_list.add_argument("--json", action="store_true", help="Emit JSON.")
    p_list.set_defaults(func=_cmd_list)

    p_run = sub.add_parser("run", help="Run one container's golden eval.")
    p_run.add_argument("container", help="Container key, e.g. 'lerobot'.")
    p_run.add_argument(
        "--execute",
        action="store_true",
        help="Execute the command (requires the container runtime); default prints it.",
    )
    p_run.add_argument(
        "--serverless",
        action="store_true",
        help="Run the eval in its container image on a Nebius Serverless GPU.",
    )
    p_run.add_argument("--gpu", default="", help="Serverless GPU type override.")
    p_run.add_argument("--timeout", default="40m", help="Serverless job timeout.")
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
