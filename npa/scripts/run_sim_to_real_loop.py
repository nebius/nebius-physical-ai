#!/usr/bin/env python3
"""Submit or render the Sereact sim-to-real SkyPilot controller workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from npa.orchestration.skypilot import WorkflowResult, submit_workflow
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
    / "sim-to-real-loop.yaml"
)
DEFAULT_BUCKET = os.environ.get("NPA_S3_BUCKET", "your-bucket-name")
DEFAULT_SOURCE = f"s3://{DEFAULT_BUCKET}/sereact/raw/"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = args.run_id or _default_run_id()
    docs = render_workflow(
        args.yaml_path,
        run_id=run_id,
        bucket=args.bucket,
        source_uri=args.source_uri,
        max_iterations=args.max_iterations,
        success_threshold=args.success_threshold,
        dry_run=args.controller_dry_run,
    )
    rendered = _write_rendered_yaml(docs, run_id=run_id)
    summary = {
        "run_id": run_id,
        "rendered_yaml": str(rendered),
        "outputs": output_paths(run_id, bucket=args.bucket),
    }
    if args.render_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    try:
        sky_bin = str(resolve_sky_bin(args.sky_bin or os.environ.get("NPA_SKYPILOT_BIN")))
        result = submit_workflow(
            rendered,
            run_id,
            isolated_config_dir=args.isolated_config_dir,
            sky_bin=sky_bin,
            timeout=args.submit_timeout,
        )
    except (SkyPilotNotInstalledError, SkyPilotConfigError, SkyPilotVersionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    summary["submit"] = result.__dict__ if isinstance(result, WorkflowResult) else result
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if getattr(result, "ok", False) else 1


def render_workflow(
    yaml_path: Path,
    *,
    run_id: str,
    bucket: str = DEFAULT_BUCKET,
    source_uri: str = DEFAULT_SOURCE,
    max_iterations: int = 3,
    success_threshold: float = 0.8,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Return SkyPilot YAML documents with run-scoped S3 paths injected."""
    docs = _load_yaml_documents(yaml_path)
    root_uri = f"s3://{bucket}/sereact-sim-to-real/{run_id}"
    for doc in docs:
        envs = doc.get("envs") if isinstance(doc, dict) else None
        if not isinstance(envs, dict):
            continue
        envs.update(
            {
                "NPA_PIPELINE_RUN_ID": run_id,
                "S3_BUCKET": bucket,
                "PIPELINE_ROOT_URI": root_uri,
                "SEREACT_SOURCE_URI": source_uri,
                "SEREACT_OUTPUT_URI": f"{root_uri}/",
                "SEREACT_MAX_ITERATIONS": str(max_iterations),
                "SEREACT_SUCCESS_THRESHOLD": str(success_threshold),
                "NPA_DRY_RUN": "1" if dry_run else "0",
            }
        )
    return docs


def output_paths(run_id: str, *, bucket: str = DEFAULT_BUCKET) -> dict[str, str]:
    root = f"s3://{bucket}/sereact-sim-to-real/{run_id}"
    return {
        "root": f"{root}/",
        "imported_data": f"{root}/iter-*/imported-data/",
        "cosmos_candidates": f"{root}/iter-*/cosmos-candidates/",
        "vlm_eval": f"{root}/iter-*/vlm-eval/",
    }


def _write_rendered_yaml(docs: list[dict[str, Any]], *, run_id: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix=f"npa-sereact-sim2real-{run_id}-"))
    path = tmp / "sim-to-real-loop.rendered.yaml"
    path.write_text(yaml.safe_dump_all(docs, sort_keys=False), encoding="utf-8")
    return path


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        docs = [doc for doc in yaml.safe_load_all(handle) if doc is not None]
    if not all(isinstance(doc, dict) for doc in docs):
        raise ValueError(f"SkyPilot YAML documents must be mappings: {path}")
    return docs


def _default_run_id() -> str:
    return "sereact-sim2real-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml-path", "--yaml", dest="yaml_path", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-uri", default=DEFAULT_SOURCE)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--controller-dry-run", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--sky-bin", default="")
    parser.add_argument("--isolated-config-dir", type=Path, default=None)
    parser.add_argument("--submit-timeout", type=int, default=1800)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
