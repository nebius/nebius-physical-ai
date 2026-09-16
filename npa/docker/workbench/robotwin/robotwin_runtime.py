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
AUTH_SCHEMA = "npa.byof.robotwin.runtime-authorization.v3"
CUSTOMER_AUTHORIZATION_SCHEMA = (
    "npa.byof.robotwin.authenticated-customer-authorization.v1"
)
LOCK_PATH = Path("/opt/npa/robotwin/runtime-lock.json")
SOURCE_REVISION = "96c1feab536306b50c26af200044fcdf126e8904"
CUROBO_REVISION = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
ASSET_REVISION = "785feb15aa4a4f532395ad2b1d2be5f28cb561ad"
WORKFLOW_SHA256 = "718bb6ae47c8e5e7e761303ebda9e962afa446a6b84030dade7c224cd255ece3"
RUNTIME_LOCK_SHA256 = "dda9bfebe81250247d25259d655589f8f3b95af7d8629d31b49c59a6af3150ee"
ACCELERATOR = "RTXPRO-6000-BLACKWELL-SERVER-EDITION"
CUSTOMER_USE_SCOPE = (
    "noncommercial-containerization-and-technical-workload-validation-and-evaluation"
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
        "url": (f"https://github.com/NVlabs/curobo/blob/{CUROBO_REVISION}/LICENSE"),
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
CUSTOMER_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "issuer",
        "customer_scope_id",
        "run_id",
        "runtime_manifest_sha256",
        "issued_at",
        "expires_at",
        "decision",
        "intended_activity",
        "terms",
        "assertion_id",
        "nonce",
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
        "customer_authorization_base64",
        "customer_authorization_sha256",
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
        authorization_raw = base64.b64decode(
            envelope.get("customer_authorization_base64"), validate=True
        )
    except (TypeError, ValueError):
        authorization_raw = b""
    if not authorization_raw or len(authorization_raw) > 16 * 1024:
        _refuse("customer-authorization-invalid")
    if hashlib.sha256(authorization_raw).hexdigest() != envelope.get(
        "customer_authorization_sha256"
    ):
        _refuse("customer-authorization-digest-mismatch")
    try:
        customer_authorization = json.loads(authorization_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        customer_authorization = None
    if (
        not isinstance(customer_authorization, dict)
        or set(customer_authorization) != CUSTOMER_AUTHORIZATION_FIELDS
    ):
        _refuse("customer-authorization-schema-mismatch")
    return payload, customer_authorization


def _validate_customer_authorization(
    payload: Mapping[str, Any], customer_authorization: Mapping[str, Any]
) -> None:
    """Revalidate the authenticated-boundary receipt inside the image."""

    if customer_authorization.get("schema_version") != CUSTOMER_AUTHORIZATION_SCHEMA:
        _refuse("customer-authorization-version-mismatch")
    if customer_authorization.get("decision") != "accepted":
        category = (
            "customer-authorization-declined"
            if customer_authorization.get("decision") == "declined"
            else "customer-authorization-decision-invalid"
        )
        _refuse(category)
    if customer_authorization.get("customer_scope_id") != _text(
        payload, "customer_scope_id"
    ):
        _refuse("customer-authorization-wrong-customer")
    if customer_authorization.get("run_id") != _text(payload, "run_id"):
        _refuse("customer-authorization-wrong-run")
    if customer_authorization.get("runtime_manifest_sha256") != RUNTIME_LOCK_SHA256:
        _refuse("customer-authorization-wrong-manifest")
    if customer_authorization.get("intended_activity") != CUSTOMER_USE_SCOPE:
        _refuse("customer-authorization-activity-mismatch")
    if customer_authorization.get("terms") != list(CUSTOMER_TERMS):
        _refuse("customer-authorization-terms-mismatch")
    issuer = customer_authorization.get("issuer")
    assertion_id = customer_authorization.get("assertion_id")
    nonce = customer_authorization.get("nonce")
    if (
        not isinstance(issuer, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]{2,255}", issuer) is None
    ):
        _refuse("customer-authorization-issuer-invalid")
    if (
        not isinstance(assertion_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,255}", assertion_id) is None
    ):
        _refuse("customer-authorization-assertion-id-invalid")
    if (
        not isinstance(nonce, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,255}", nonce) is None
    ):
        _refuse("customer-authorization-nonce-invalid")
    timestamps: dict[str, datetime] = {}
    for name in ("issued_at", "expires_at"):
        label = "issuance" if name == "issued_at" else "expiry"
        value = customer_authorization.get(name)
        if (
            not isinstance(value, str)
            or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value) is None
        ):
            _refuse(f"customer-authorization-{label}-invalid")
        parsed: datetime | None = None
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
        if parsed is None:
            _refuse(f"customer-authorization-{label}-invalid")
        timestamps[name] = parsed
    current = datetime.now(timezone.utc)
    if timestamps["issued_at"] > current:
        _refuse("customer-authorization-not-yet-valid")
    if timestamps["expires_at"] <= current:
        _refuse("customer-authorization-stale")
    if timestamps["expires_at"] <= timestamps["issued_at"]:
        _refuse("customer-authorization-window-invalid")


def _validate_authorization(environ: Mapping[str, str]) -> dict[str, Any]:
    payload, customer_authorization = _decode_authorization(environ)
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
    _validate_customer_authorization(payload, customer_authorization)
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
            f"{_text(payload, 'output_root').rstrip('/')}/{_text(payload, 'run_id')}/"
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
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "npa.robotwin.runtime-lock.v1"
    ):
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
        _refuse("runtime-delivery-technical-gates-incomplete")
    _refuse("runtime-fetch-not-implemented")


def assert_refusal() -> int:
    """Exercise the genuine first gate and prove no work directories are created."""

    with tempfile.TemporaryDirectory(prefix="robotwin-refusal-") as raw_root:
        root = Path(raw_root)
        home = root / "home"
        cwd = root / "cwd"
        temporary = root / "tmp"
        home.mkdir(mode=0o700)
        cwd.mkdir(mode=0o700)
        temporary.mkdir(mode=0o700)
        runtime_paths = {
            "NPA_ROBOTWIN_SOURCE_DIR": root / "source",
            "NPA_ROBOTWIN_ASSETS_DIR": root / "assets",
            "NPA_ROBOTWIN_CACHE_DIR": root / "cache",
            "NPA_ROBOTWIN_OUTPUT_DIR": root / "output",
            "NPA_SMOKE_OUTPUT_DIR": root / "output" / "smoke",
            "XDG_CACHE_HOME": root / "cache" / "xdg",
            "XDG_CONFIG_HOME": root / "config",
            "XDG_DATA_HOME": root / "data",
            "HF_HOME": root / "cache" / "huggingface",
            "TORCH_HOME": root / "cache" / "torch",
            "WARP_CACHE_PATH": root / "cache" / "warp",
            "PIP_CACHE_DIR": root / "cache" / "pip",
            "CUDA_CACHE_PATH": root / "cache" / "cuda",
            "TMPDIR": temporary,
        }

        def tree() -> tuple[tuple[str, str], ...]:
            return tuple(
                sorted(
                    (
                        str(path.relative_to(root)),
                        "directory" if path.is_dir() else "entry",
                    )
                    for path in root.rglob("*")
                )
            )

        before = tree()
        env = dict(os.environ)
        env.pop(AUTH_ENV, None)
        env["HOME"] = str(home)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update({name: str(path) for name, path in runtime_paths.items()})
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "run"],
            env=env,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if (
            completed.returncode != 78
            or "ROBOTWIN_RUNTIME_REFUSED:authorization-missing" not in completed.stderr
        ):
            raise RuntimeError("RoboTwin missing-authorization refusal did not execute")
        if tree() != before:
            raise RuntimeError("RoboTwin refusal changed its isolated runtime tree")
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
    parser.add_argument(
        "command", choices=("health", "status", "assert-refusal", "run")
    )
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
