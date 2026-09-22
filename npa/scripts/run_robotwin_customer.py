#!/usr/bin/env python3
"""Launch the fixed RoboTwin GPU profile after a customer-terminal decision.

The operator supplies an owner-private context and the customer's existing
receipt. This command never records acceptance. It runs the existing BYOF
launcher locally, so no remote NPA source installation is needed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from npa.clients.project_credentials import storage_env_for_project
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.robotwin_customer import (
    CustomerDecisionError,
    CustomerTerminalBoundary,
    DECISION_ENV,
    probe_runtime_inputs,
)
from npa.orchestration.npa_workflow.robotwin_preflight import (
    PUBLIC_CONTEXT_ENV,
    RobotwinPreflightError,
    RUNTIME_LOCK_SHA256,
    _materialize_authorization,
    read_owner_context,
    recognize_contract,
    require_runtime_lock_complete,
    validate_context_bytes,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_byof_repo as runner  # noqa: E402


WORKFLOW = Path(__file__).resolve().parents[2] / "workflows/testing/byof-robotwin.yaml"
RUNTIME_LOCK = (
    Path(__file__).resolve().parents[1] / "docker/workbench/robotwin/runtime-lock.json"
)
STORAGE_ENV = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ENDPOINT_URL",
    "NEBIUS_S3_ENDPOINT",
    "AWS_ENDPOINT_URL_S3",
    "NPA_STORAGE_ENDPOINT",
    "S3_ENDPOINT_URL",
)


@contextmanager
def _project_storage(project: str):
    """Bind only the selected project's credentials and restore the caller."""
    selected = storage_env_for_project(project, allow_host_creds=False)
    previous = {name: os.environ.get(name) for name in STORAGE_ENV}
    try:
        for name in STORAGE_ENV:
            os.environ.pop(name, None)
        for name in STORAGE_ENV:
            if selected.get(name):
                os.environ[name] = selected[name]
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run(context: Path, decision: Path) -> int:
    """Reuse the immutable workflow, image scan, submit, wait and cleanup path."""
    spec = load_spec(WORKFLOW)
    if not recognize_contract(spec):
        raise ValueError("RoboTwin workflow contract is unavailable")
    raw = read_owner_context({PUBLIC_CONTEXT_ENV: str(context)})
    authorization = require_runtime_lock_complete(
        validate_context_bytes(
            raw, customer_authorization_boundary=CustomerTerminalBoundary(decision)
        )
    )
    probe_runtime_inputs(RUNTIME_LOCK, expected_sha256=RUNTIME_LOCK_SHA256)
    plan = build_plan(spec, run_id=authorization.run_id)
    argv = plan.steps[0].argv
    if len(plan.steps) != 1 or argv[:4] != ["npa", "workbench", "byof", "run"]:
        raise ValueError("RoboTwin workflow invocation is unavailable")
    directory = Path(tempfile.mkdtemp(prefix="npa-robotwin-customer-"))
    # Retain exact config bytes after failure for ownership-safe recovery.
    # The existing launcher cancels its job and stops its owned API first.
    completed = False
    try:
        materialized = _materialize_authorization(authorization, directory)
        authorization = replace(
            authorization,
            kubeconfig_source=str(materialized.kubeconfig_path),
            skypilot_config_source=str(materialized.skypilot_config_path),
        )
        with _project_storage(authorization.project):
            code = runner._run_authorized_robotwin(
                argv[4:], authorization=authorization
            )
        completed = code == 0
        return code
    finally:
        if completed:
            shutil.rmtree(directory)
        else:
            print(
                "RoboTwin recovery configuration retained in an owner-private directory.",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context", type=Path, default=os.environ.get(PUBLIC_CONTEXT_ENV)
    )
    parser.add_argument(
        "--customer-decision", type=Path, default=os.environ.get(DECISION_ENV)
    )
    args = parser.parse_args(argv)
    if args.context is None or args.customer_decision is None:
        parser.error("--context and --customer-decision are required")
    try:
        return run(args.context, args.customer_decision)
    except (RobotwinPreflightError, CustomerDecisionError) as failure:
        print(json.dumps({"status": "refused", "error": str(failure)}))
        return 78
    except Exception:
        # Provider/config exceptions can contain credentials or private URIs.
        print(
            json.dumps({"status": "failed", "error": "RoboTwin customer launch failed"})
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
