#!/usr/bin/env python3
"""Zero-payload RoboTwin bootstrap with an intentionally closed Phase A gate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping


AUTH_ENV = "NPA_INTERNAL_BYOF_ROBOTWIN_RUNTIME_AUTH_V1"
AUTH_SCHEMA = "npa.byof.robotwin.runtime-authorization.v1"
LOCK_PATH = Path("/opt/npa/robotwin/runtime-lock.json")
SOURCE_REVISION = "96c1feab536306b50c26af200044fcdf126e8904"
CUROBO_REVISION = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
ASSET_REVISION = "785feb15aa4a4f532395ad2b1d2be5f28cb561ad"
WORKFLOW_SHA256 = "718bb6ae47c8e5e7e761303ebda9e962afa446a6b84030dade7c224cd255ece3"
RUNTIME_LOCK_SHA256 = "c42c4037392f51ad6c2473eb3f07843738a4c5147328ace1686ddb9cf553b4ef"
ACCELERATOR = "RTXPRO-6000-BLACKWELL-SERVER-EDITION"
REQUIRED_DECISIONS = frozenset(
    {
        "nvidia_cuda_eula",
        "nvidia_cudnn_sla",
        "curobo_noncommercial_research_or_evaluation",
        "robotwin2_aggregate_asset_and_output_terms",
    }
)
CONTEXT_FIELDS = frozenset(
    {
        "solution",
        "ownership_provenance",
        "workflow_sha256",
        "source_revision",
        "curobo_revision",
        "asset_revision",
        "runtime_lock_sha256",
        "bootstrap_image",
        "reservation",
        "license_acceptance",
        "project",
        "nebius_profile",
        "kubeconfig",
        "kubernetes_context",
        "skypilot_config_path",
        "bucket",
        "output_root",
        "run_id",
    }
)


class Refusal(RuntimeError):
    """Fixed-category refusal safe for logs."""


def _refuse(category: str) -> None:
    raise Refusal(f"ROBOTWIN_RUNTIME_REFUSED:{category}")


def _text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        _refuse(f"context-{name}-invalid")
    return value.strip()


def _decode_authorization(environ: Mapping[str, str]) -> tuple[dict[str, Any], bytes]:
    encoded = str(environ.get(AUTH_ENV) or "")
    if not encoded:
        _refuse("authorization-missing")
    try:
        envelope = json.loads(encoded)
    except (TypeError, json.JSONDecodeError):
        envelope = None
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema_version",
        "context_base64",
        "context_sha256",
    }:
        _refuse("authorization-envelope-invalid")
    if envelope.get("schema_version") != AUTH_SCHEMA:
        _refuse("authorization-version-mismatch")
    try:
        raw = base64.b64decode(envelope.get("context_base64"), validate=True)
    except (TypeError, ValueError):
        raw = b""
    if not raw or len(raw) > 64 * 1024:
        _refuse("authorization-context-invalid")
    if hashlib.sha256(raw).hexdigest() != envelope.get("context_sha256"):
        _refuse("authorization-digest-mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    if not isinstance(payload, dict) or set(payload) != CONTEXT_FIELDS:
        _refuse("authorization-context-schema-mismatch")
    return payload, raw


def _validate_authorization(environ: Mapping[str, str]) -> dict[str, Any]:
    payload, _ = _decode_authorization(environ)
    if payload.get("solution") != "robotwin":
        _refuse("wrong-solution")
    if payload.get("ownership_provenance") != "manager-issued":
        _refuse("ownership-provenance-invalid")
    if _text(payload, "source_revision") != SOURCE_REVISION:
        _refuse("source-revision-mismatch")
    if _text(payload, "curobo_revision") != CUROBO_REVISION:
        _refuse("curobo-revision-mismatch")
    if _text(payload, "asset_revision") != ASSET_REVISION:
        _refuse("asset-revision-mismatch")
    expected_digests = {
        "workflow_sha256": WORKFLOW_SHA256,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
    }
    for name, expected in expected_digests.items():
        if _text(payload, name) != expected:
            _refuse(f"{name.replace('_', '-')}-mismatch")
    image = _text(payload, "bootstrap_image")
    if re.fullmatch(r"[^\s]+/npa-robotwin@sha256:[0-9a-f]{64}", image) is None:
        _refuse("bootstrap-image-not-immutable")
    reservation = payload.get("reservation")
    if reservation != {"policy": "STRICT", "accelerator": ACCELERATOR, "count": 1}:
        _refuse("reservation-mismatch")
    decisions = payload.get("license_acceptance")
    if not isinstance(decisions, dict) or set(decisions) != REQUIRED_DECISIONS:
        _refuse("license-decision-schema-mismatch")
    if any(decisions[name] is not True for name in REQUIRED_DECISIONS):
        _refuse("license-decision-missing")
    for name in (
        "project",
        "nebius_profile",
        "kubeconfig",
        "kubernetes_context",
        "skypilot_config_path",
        "bucket",
        "output_root",
        "run_id",
    ):
        _text(payload, name)
    expected = {
        "BYOF_IMAGE": image,
        "NPA_BYOF_RUN_ID": _text(payload, "run_id"),
        "NPA_S3_BUCKET": _text(payload, "bucket"),
        "S3_OUTPUT_PREFIX": (
            f"{_text(payload, 'output_root').rstrip('/')}/"
            f"{_text(payload, 'run_id')}/"
        ),
    }
    if any(str(environ.get(name) or "") != value for name, value in expected.items()):
        _refuse("runtime-binding-mismatch")
    return payload


def _load_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError:
        _refuse("runtime-lock-unreadable")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        _refuse("runtime-lock-digest-mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict) or payload.get("schema_version") != "npa.robotwin.runtime-lock.v1":
        _refuse("runtime-lock-invalid")
    return payload


def run(*, lock_path: Path, environ: Mapping[str, str]) -> int:
    """Refuse before filesystem/network effects until the immutable lock is complete."""

    authorization = _validate_authorization(environ)
    lock = _load_lock(lock_path, _text(authorization, "runtime_lock_sha256"))
    if lock.get("status") != "complete":
        _refuse("runtime-lock-incomplete")
    _refuse("phase-a-runtime-disabled")


def assert_refusal() -> int:
    """Exercise the genuine first gate and prove no work directories are created."""

    with tempfile.TemporaryDirectory(prefix="robotwin-refusal-") as raw_root:
        root = Path(raw_root)
        watched = [root / name for name in ("source", "assets", "cache", "output")]
        env = dict(os.environ)
        env.pop(AUTH_ENV, None)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "run"],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 78 or "ROBOTWIN_RUNTIME_REFUSED:authorization-missing" not in completed.stderr:
            raise RuntimeError("RoboTwin missing-authorization refusal did not execute")
        if any(path.exists() for path in watched):
            raise RuntimeError("RoboTwin refusal created a runtime path")
    print("robotwin bootstrap refusal verified; no simulator capability claimed")
    return 0


def status(lock_path: Path) -> int:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        lock = {"status": "unreadable"}
    print(
        json.dumps(
            {
                "solution": "robotwin",
                "bootstrap": "zero-vendor-payload",
                "runtime_lock": lock.get("status", "invalid"),
                "capability_validated": False,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("health", "status", "assert-refusal", "run"))
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    args = parser.parse_args(argv)
    if args.command == "health":
        print("ok: robotwin zero-payload bootstrap")
        return 0
    if args.command == "status":
        return status(args.lock)
    if args.command == "assert-refusal":
        return assert_refusal()
    try:
        return run(lock_path=args.lock, environ=os.environ)
    except Refusal as exc:
        print(str(exc), file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
