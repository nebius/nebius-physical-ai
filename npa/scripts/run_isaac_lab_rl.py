#!/usr/bin/env python3
"""Submit or render Isaac Lab RSL-RL SkyPilot workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from npa.orchestration.skypilot import (
    WorkflowResult,  # noqa: F401 - kept for tests and downstream wrapper imports.
    cleanup_all_for_run,
    submit_workflow,
    workflow_status,
)
from npa.orchestration.skypilot.signal_teardown import (
    SignalTeardown,
    install_teardown_signal_handlers,
    restore_signal_handlers,
)
from npa.orchestration.skypilot._bin import (
    SkyPilotConfigError,
    SkyPilotNotInstalledError,
    SkyPilotVersionError,
    resolve_sky_bin,
)

DEFAULT_YAML = (
    Path(__file__).resolve().parents[1]
    / "workflows"
    / "workbench"
    / "skypilot"
    / "isaac-lab-rl-train.yaml"
)
DEFAULT_BUCKET = os.environ.get("NPA_S3_BUCKET", "your-bucket-name")
DEFAULT_OUTPUT_ROOT = f"s3://{DEFAULT_BUCKET}/isaac-lab-rl"
TERMINAL_STATUSES = {
    "SUCCEEDED",
    "CANCELLED",
    "FAILED",
    "FAILED_SETUP",
    "FAILED_PRECHECKS",
    "FAILED_NO_RESOURCE",
    "FAILED_CONTROLLER",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return _submit_and_wait(args)
    except (SkyPilotNotInstalledError, SkyPilotConfigError, SkyPilotVersionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("For a no-infrastructure check, rerun with --render-only.", file=sys.stderr)
        return 2


def render_workflow(
    yaml_path: Path,
    *,
    run_id: str,
    task: str,
    iterations: int,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    image: str = "",
) -> list[dict[str, Any]]:
    docs = _load_yaml_documents(yaml_path)
    train_docs = [doc for doc in docs[1:] if isinstance(doc.get("envs"), dict)]
    multiple = len(train_docs) > 1
    for doc in train_docs:
        envs = doc["envs"]
        envs["NPA_ISAAC_LAB_RUN_ID"] = run_id
        envs["ISAAC_LAB_TASK"] = task
        envs["ISAAC_LAB_ITERATIONS"] = str(iterations)
        variant = str(envs.get("RUN_VARIANT") or doc.get("name") or "").strip()
        prefix = output_root.rstrip("/") + f"/{run_id}/"
        if multiple and variant:
            prefix += f"{variant}/"
        envs["S3_OUTPUT_PREFIX"] = prefix
        if image:
            resources = doc.setdefault("resources", {})
            if isinstance(resources, dict):
                resources["image_id"] = f"docker:{image}" if not image.startswith("docker:") else image
    return docs


def output_paths(run_id: str, *, output_root: str = DEFAULT_OUTPUT_ROOT, variants: list[str] | None = None) -> dict[str, Any]:
    root = output_root.rstrip("/") + f"/{run_id}/"
    if not variants:
        return {
            "root": root,
            "checkpoint": root + "npa_isaac_lab_checkpoint.pt",
            "summary": root + "npa_isaac_lab_train_summary.json",
        }
    return {
        "root": root,
        "variants": {
            variant: {
                "checkpoint": root + f"{variant}/npa_isaac_lab_checkpoint.pt",
                "summary": root + f"{variant}/npa_isaac_lab_train_summary.json",
            }
            for variant in variants
        },
    }


def _submit_and_wait(args: argparse.Namespace) -> int:
    run_id = args.run_id or _default_run_id()
    docs = render_workflow(
        args.yaml_path,
        run_id=run_id,
        task=args.task,
        iterations=args.iterations,
        output_root=args.output_root,
        image=args.image,
    )
    variants = [
        str(doc.get("envs", {}).get("RUN_VARIANT"))
        for doc in docs[1:]
        if isinstance(doc.get("envs"), dict) and doc.get("envs", {}).get("RUN_VARIANT")
    ]
    outputs = output_paths(run_id, output_root=args.output_root, variants=variants)

    if args.render_only:
        render_dir = Path(tempfile.mkdtemp(prefix=f"npa-isaac-lab-rl-{run_id}-"))
        rendered_yaml = render_dir / "isaac-lab-rl.rendered.yaml"
        _write_yaml_documents(rendered_yaml, docs)
        print(json.dumps({"run_id": run_id, "rendered_yaml": str(rendered_yaml), "outputs": outputs}, indent=2))
        return 0

    with tempfile.TemporaryDirectory(prefix=f"npa-isaac-lab-rl-{run_id}-") as tmp:
        rendered_yaml = Path(tmp) / "isaac-lab-rl.rendered.yaml"
        _write_yaml_documents(rendered_yaml, docs)
        sky_bin = str(resolve_sky_bin(args.sky_bin or os.environ.get("NPA_SKYPILOT_BIN")))
        teardown_guard = SignalTeardown(
            run_id=run_id,
            isolated_config_dir=args.isolated_config_dir,
            sky_bin=sky_bin,
            poll_interval=max(float(args.poll_interval), 0.0),
        )
        # SIGTERM/SIGINT handlers call the same idempotent teardown path as normal exit.
        previous_handlers = install_teardown_signal_handlers(teardown_guard.teardown)
        summary: dict[str, Any] | None = None
        return_code = 1
        try:
            teardown_guard.mark_launched()
            result = submit_workflow(
                rendered_yaml,
                run_id,
                isolated_config_dir=args.isolated_config_dir,
                sky_bin=sky_bin,
                timeout=args.submit_timeout,
            )
            config_path = Path(result.log_paths["config"]) if result.log_paths.get("config") else None
            teardown_guard.mark_launched(config_path=config_path)
            summary = {
                "run_id": run_id,
                "submit": result.__dict__,
                "outputs": outputs,
            }
            if not result.ok or result.status != "SUBMITTED":
                return_code = result.returncode or 1
            else:
                deadline = time.monotonic() + args.wait_timeout
                final = result
                while time.monotonic() < deadline:
                    final = workflow_status(
                        result.job_id,
                        isolated_config_dir=args.isolated_config_dir,
                        config_path=config_path,
                        sky_bin=sky_bin,
                    )
                    if final.status in TERMINAL_STATUSES:
                        break
                    time.sleep(args.poll_interval)
                summary["final"] = final.__dict__
                return_code = 0 if final.status == "SUCCEEDED" else 1

            if args.cleanup:
                cleanup = cleanup_all_for_run(
                    run_id,
                    isolated_config_dir=args.isolated_config_dir,
                    config_path=config_path,
                    sky_bin=sky_bin,
                )
                summary["cleanup"] = cleanup.__dict__
        finally:
            teardown = teardown_guard.teardown()
            restore_signal_handlers(previous_handlers)

        if summary is not None:
            summary["teardown"] = teardown.__dict__
            print(json.dumps(summary, indent=2, sort_keys=True))
        return 1 if teardown.errors else return_code


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        docs = [doc for doc in yaml.safe_load_all(handle) if doc is not None]
    if not all(isinstance(doc, dict) for doc in docs):
        raise ValueError(f"SkyPilot YAML documents must be mappings: {path}")
    return docs


def _write_yaml_documents(path: Path, docs: list[dict[str, Any]]) -> None:
    path.write_text(yaml.safe_dump_all(docs, sort_keys=False), encoding="utf-8")


def _default_run_id() -> str:
    return "isaac-lab-rl-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", "--yaml-path", dest="yaml_path", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--task", default=os.environ.get("ISAAC_LAB_TASK", "Isaac-Cartpole-v0"))
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("ISAAC_LAB_ITERATIONS", "10")))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image", default="", help="Container image override, e.g. cr.../npa-isaac-lab:tag.")
    parser.add_argument("--sky-bin", default="")
    parser.add_argument("--isolated-config-dir", type=Path, default=None)
    parser.add_argument("--submit-timeout", type=int, default=1800)
    parser.add_argument("--wait-timeout", type=int, default=21600)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
