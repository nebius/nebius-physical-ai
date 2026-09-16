#!/usr/bin/env python3
"""Zero-payload RoboTwin bootstrap with an intentionally closed runtime gate."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
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
AUTH_SCHEMA = "npa.byof.robotwin.runtime-authorization.v2"
CUSTOMER_ENTITLEMENT_SCHEMA = "npa.byof.robotwin.customer-runtime-entitlement.v1"
LOCK_PATH = Path("/opt/npa/robotwin/runtime-lock.json")
SOURCE_REVISION = "96c1feab536306b50c26af200044fcdf126e8904"
CUROBO_REVISION = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
ASSET_REVISION = "785feb15aa4a4f532395ad2b1d2be5f28cb561ad"
WORKFLOW_SHA256 = "718bb6ae47c8e5e7e761303ebda9e962afa446a6b84030dade7c224cd255ece3"
RUNTIME_LOCK_SHA256 = "d198a02d46dc2adc0dfbe33ff1a27d06f2525b9d05911c6b5552da1eb74d5b60"
ACCELERATOR = "RTXPRO-6000-BLACKWELL-SERVER-EDITION"
CUSTOMER_USE_SCOPE = (
    "noncommercial-containerization-and-technical-workload-"
    "validation-and-evaluation"
)
CUSTOMER_TERMS = (
    {
        "id": "nvidia-cuda-12.8.1-eula-2025-01-07",
        "url": "https://docs.nvidia.com/cuda/archive/12.8.1/eula/index.html",
    },
    {
        "id": "nvidia-cudnn-9.8.0-sla-2025-03-06",
        "url": (
            "https://docs.nvidia.com/deeplearning/cudnn/backend/"
            "v9.8.0/reference/eula.html"
        ),
    },
    {
        "id": "nvidia-curobo-v0.7.8-license-d64c4b005459",
        "url": (
            "https://github.com/NVlabs/curobo/blob/"
            f"{CUROBO_REVISION}/LICENSE"
        ),
    },
)
CONTEXT_FIELDS = frozenset(
    {
        "solution",
        "ownership_provenance",
        "customer_scope_id",
        "workflow_sha256",
        "source_revision",
        "curobo_revision",
        "asset_revision",
        "runtime_lock_sha256",
        "bootstrap_image",
        "reservation",
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
CUSTOMER_ENTITLEMENT_FIELDS = frozenset(
    {
        "schema_version",
        "provenance",
        "customer_scope_id",
        "run_id",
        "runtime_manifest_sha256",
        "expires_at",
        "decision",
        "intended_activity",
        "terms",
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


def _decode_authorization(
    environ: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        "customer_entitlement_base64",
        "customer_entitlement_sha256",
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
    try:
        entitlement_raw = base64.b64decode(
            envelope.get("customer_entitlement_base64"), validate=True
        )
    except (TypeError, ValueError):
        entitlement_raw = b""
    if not entitlement_raw or len(entitlement_raw) > 16 * 1024:
        _refuse("customer-entitlement-invalid")
    if (
        hashlib.sha256(entitlement_raw).hexdigest()
        != envelope.get("customer_entitlement_sha256")
    ):
        _refuse("customer-entitlement-digest-mismatch")
    try:
        entitlement = json.loads(entitlement_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        entitlement = None
    if (
        not isinstance(entitlement, dict)
        or set(entitlement) != CUSTOMER_ENTITLEMENT_FIELDS
    ):
        _refuse("customer-entitlement-schema-mismatch")
    return payload, entitlement


def _validate_customer_entitlement(
    payload: Mapping[str, Any], entitlement: Mapping[str, Any]
) -> None:
    """Revalidate customer/run/manifest/expiry binding inside the image."""

    if entitlement.get("schema_version") != CUSTOMER_ENTITLEMENT_SCHEMA:
        _refuse("customer-entitlement-version-mismatch")
    if entitlement.get("provenance") != "customer-issued":
        _refuse("customer-entitlement-provenance-invalid")
    if entitlement.get("decision") != "accepted":
        category = (
            "customer-entitlement-declined"
            if entitlement.get("decision") == "declined"
            else "customer-entitlement-decision-invalid"
        )
        _refuse(category)
    if entitlement.get("customer_scope_id") != _text(payload, "customer_scope_id"):
        _refuse("customer-entitlement-wrong-customer")
    if entitlement.get("run_id") != _text(payload, "run_id"):
        _refuse("customer-entitlement-wrong-run")
    if entitlement.get("runtime_manifest_sha256") != RUNTIME_LOCK_SHA256:
        _refuse("customer-entitlement-wrong-manifest")
    if entitlement.get("intended_activity") != CUSTOMER_USE_SCOPE:
        _refuse("customer-entitlement-use-scope-mismatch")
    if entitlement.get("terms") != list(CUSTOMER_TERMS):
        _refuse("customer-entitlement-terms-mismatch")
    expires_at = entitlement.get("expires_at")
    if not isinstance(expires_at, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", expires_at
    ) is None:
        _refuse("customer-entitlement-expiry-invalid")
    expiry: datetime | None = None
    try:
        expiry = datetime.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        pass
    if expiry is None:
        _refuse("customer-entitlement-expiry-invalid")
    if expiry <= datetime.now(timezone.utc):
        _refuse("customer-entitlement-stale")


def _validate_authorization(environ: Mapping[str, str]) -> dict[str, Any]:
    payload, entitlement = _decode_authorization(environ)
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
    if not isinstance(reservation, dict) or set(reservation) != {
        "policy",
        "accelerator",
        "count",
    }:
        _refuse("reservation-schema-mismatch")
    if reservation.get("policy") != "STRICT":
        _refuse("reservation-policy-not-strict")
    if reservation.get("accelerator") != ACCELERATOR:
        _refuse("reservation-accelerator-mismatch")
    if type(reservation.get("count")) is not int or reservation["count"] != 1:
        _refuse("reservation-count-not-one")
    _validate_customer_entitlement(payload, entitlement)
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
    """Refuse before effects until runtime delivery and authority are complete."""

    authorization = _validate_authorization(environ)
    lock = _load_lock(lock_path, _text(authorization, "runtime_lock_sha256"))
    bootstrap = lock.get("bootstrap")
    if not isinstance(bootstrap, dict) or bootstrap.get("status") != "complete":
        _refuse("runtime-lock-incomplete")
    delivery = lock.get("runtime_delivery")
    if not isinstance(delivery, dict) or delivery.get("status") != "complete":
        _refuse("runtime-delivery-disabled")
    _refuse("runtime-fetch-not-implemented")


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
