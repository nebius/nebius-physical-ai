"""Warm and attest the Isaac cache inside the canonical SkyPilot workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.runtime_attestation import attest_warmed_isaac_cache
from npa.workflows.sim2real.workflow_io import source_sha, write_json


def warm_cache(*, root_uri: str, run_id: str) -> dict[str, Any]:
    """Run the real bootstrap, attest its content-addressed tree, and publish it."""

    root = str(root_uri or "").rstrip("/")
    if not root.startswith("s3://"):
        raise RuntimeError("root_uri must be an s3:// URI")
    if not str(run_id or "").strip():
        raise RuntimeError("run_id must not be empty")
    bootstrap = Path(
        os.environ.get("NPA_ISAAC_BOOTSTRAP_PATH", "/opt/npa/bin/isaac-bootstrap")
    )
    if not bootstrap.is_file() or not os.access(bootstrap, os.X_OK):
        raise RuntimeError(f"Isaac bootstrap is not executable: {bootstrap}")
    task_image = os.environ.get("NPA_TASK_IMAGE", "").strip()
    if "@sha256:" not in task_image:
        raise RuntimeError("Isaac cache warm state lacks an immutable NPA_TASK_IMAGE")
    completed = subprocess.run(
        [str(bootstrap), "warm"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "bootstrap failed").strip()
        raise RuntimeError(f"Isaac cache warm failed: {detail[-1200:]}")
    status = subprocess.run(
        [str(bootstrap), "status"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode or "ready=yes" not in status.stdout:
        detail = (status.stderr or status.stdout or "cache status failed").strip()
        raise RuntimeError(f"Isaac cache status is not ready: {detail[-1200:]}")
    payload = {
        "schema": "npa.sim2real.isaac_cache_warm.v1",
        "run_id": str(run_id),
        "source_sha": source_sha(),
        "runtime_image": task_image,
        "cache": attest_warmed_isaac_cache(),
        "status": "ready",
    }
    with tempfile.TemporaryDirectory(prefix="npa-s2r-cache-warm-") as directory:
        uri = f"{root}/cache/isaac-warm.json"
        write_json(uri, payload, directory=Path(directory))
    return {"uri": uri, **payload}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-uri", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            warm_cache(root_uri=args.root_uri, run_id=args.run_id), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
