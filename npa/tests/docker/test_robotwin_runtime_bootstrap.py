"""Negative tests for the Phase A standard-library RoboTwin bootstrap."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from npa.orchestration.npa_workflow import robotwin_preflight


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "npa/docker/workbench/robotwin/robotwin_runtime.py"
LOCK = ROOT / "npa/docker/workbench/robotwin/runtime-lock.json"


def _load_runtime():
    spec = importlib.util.spec_from_file_location("robotwin_runtime", RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


runtime = _load_runtime()


def test_runtime_lock_digest_is_bound_to_both_validators() -> None:
    lock_sha256 = hashlib.sha256(LOCK.read_bytes()).hexdigest()

    assert lock_sha256 == runtime.RUNTIME_LOCK_SHA256
    assert lock_sha256 == robotwin_preflight.RUNTIME_LOCK_SHA256


def _context(**updates: object) -> dict[str, object]:
    image = "registry.example/private/npa-robotwin@sha256:" + "a" * 64
    payload: dict[str, object] = {
        "solution": "robotwin",
        "ownership_provenance": "manager-issued",
        "customer_scope_id": "customer-scope-canary",
        "workflow_sha256": runtime.WORKFLOW_SHA256,
        "source_revision": runtime.SOURCE_REVISION,
        "curobo_revision": runtime.CUROBO_REVISION,
        "asset_revision": runtime.ASSET_REVISION,
        "runtime_lock_sha256": runtime.RUNTIME_LOCK_SHA256,
        "bootstrap_image": image,
        "reservation": {
            "policy": "STRICT",
            "accelerator": runtime.ACCELERATOR,
            "count": 1,
        },
        "project": "manager-project-canary",
        "nebius_profile": "manager-profile-canary",
        "kubeconfig": "/owner-only/kubeconfig-canary",
        "kubernetes_context": "manager-context-canary",
        "skypilot_config_path": "/owner-only/skypilot-canary",
        "bucket": "manager-bucket-canary",
        "output_root": "s3://manager-bucket-canary/robotwin-output",
        "run_id": "robotwin-manager-canary",
    }
    payload.update(updates)
    return payload


def _entitlement(payload: dict[str, object], **updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": runtime.CUSTOMER_ENTITLEMENT_SCHEMA,
        "provenance": "customer-issued",
        "customer_scope_id": payload["customer_scope_id"],
        "run_id": payload["run_id"],
        "runtime_manifest_sha256": runtime.RUNTIME_LOCK_SHA256,
        "expires_at": "2099-01-01T00:00:00Z",
        "decision": "accepted",
        "intended_activity": runtime.CUSTOMER_USE_SCOPE,
        "terms": list(runtime.CUSTOMER_TERMS),
    }
    value.update(updates)
    return value


def _environment(**updates: object) -> dict[str, str]:
    payload = _context(**updates)
    raw = json.dumps(payload, sort_keys=True).encode()
    entitlement_raw = json.dumps(_entitlement(payload), sort_keys=True).encode()
    envelope = json.dumps(
        {
            "schema_version": runtime.AUTH_SCHEMA,
            "context_base64": base64.b64encode(raw).decode(),
            "context_sha256": hashlib.sha256(raw).hexdigest(),
            "customer_entitlement_base64": base64.b64encode(
                entitlement_raw
            ).decode(),
            "customer_entitlement_sha256": hashlib.sha256(
                entitlement_raw
            ).hexdigest(),
        },
        sort_keys=True,
    )
    return {
        runtime.AUTH_ENV: envelope,
        "BYOF_IMAGE": str(payload["bootstrap_image"]),
        "NPA_BYOF_RUN_ID": str(payload["run_id"]),
        "NPA_S3_BUCKET": str(payload["bucket"]),
        "S3_OUTPUT_PREFIX": (
            f"{str(payload['output_root']).rstrip('/')}/{payload['run_id']}/"
        ),
    }


def _environment_with_entitlement(**updates: object) -> dict[str, str]:
    payload = _context()
    raw = json.dumps(payload, sort_keys=True).encode()
    entitlement_raw = json.dumps(
        _entitlement(payload, **updates), sort_keys=True
    ).encode()
    envelope = json.dumps(
        {
            "schema_version": runtime.AUTH_SCHEMA,
            "context_base64": base64.b64encode(raw).decode(),
            "context_sha256": hashlib.sha256(raw).hexdigest(),
            "customer_entitlement_base64": base64.b64encode(
                entitlement_raw
            ).decode(),
            "customer_entitlement_sha256": hashlib.sha256(
                entitlement_raw
            ).hexdigest(),
        },
        sort_keys=True,
    )
    environment = _environment()
    environment[runtime.AUTH_ENV] = envelope
    return environment


def test_missing_authorization_refuses_before_lock_access(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime,
        "_load_lock",
        lambda *_args, **_kwargs: pytest.fail("lock read preceded authorization"),
    )
    with pytest.raises(runtime.Refusal, match="authorization-missing"):
        runtime.run(lock_path=LOCK, environ={})


@pytest.mark.parametrize(
    ("updates", "category"),
    [
        ({"source_revision": "0" * 40}, "source-revision-mismatch"),
        ({"curobo_revision": "main"}, "curobo-revision-mismatch"),
        ({"asset_revision": "latest"}, "asset-revision-mismatch"),
        ({"workflow_sha256": "0" * 64}, "workflow-sha256-mismatch"),
        ({"runtime_lock_sha256": "0" * 64}, "runtime-lock-sha256-mismatch"),
        (
            {"bootstrap_image": "registry.example/npa-robotwin:latest"},
            "bootstrap-image-not-immutable",
        ),
        (
            {
                "reservation": {
                    "policy": "STRICT",
                    "accelerator": "B200",
                    "count": 1,
                }
            },
            "reservation-accelerator-mismatch",
        ),
        (
            {
                "reservation": {
                    "policy": "STRICT",
                    "accelerator": runtime.ACCELERATOR,
                    "count": True,
                }
            },
            "reservation-count-not-one",
        ),
        (
            {
                "reservation": {
                    "policy": "STRICT",
                    "accelerator": runtime.ACCELERATOR,
                    "count": 1.0,
                }
            },
            "reservation-count-not-one",
        ),
        ({"run_id": "robotwin-other"}, "runtime-binding-mismatch"),
    ],
)
def test_mutated_authorization_refuses_before_lock_access(
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    category: str,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_load_lock",
        lambda *_args, **_kwargs: pytest.fail("lock read preceded authorization"),
    )
    environment = _environment(**updates)
    if "run_id" in updates:
        environment["NPA_BYOF_RUN_ID"] = "robotwin-divergent"
    with pytest.raises(runtime.Refusal, match=category):
        runtime.run(lock_path=LOCK, environ=environment)


@pytest.mark.parametrize(
    ("updates", "category"),
    [
        ({"provenance": "manager-issued"}, "customer-entitlement-provenance-invalid"),
        ({"decision": "declined"}, "customer-entitlement-declined"),
        ({"customer_scope_id": "other"}, "customer-entitlement-wrong-customer"),
        ({"run_id": "robotwin-other"}, "customer-entitlement-wrong-run"),
        ({"runtime_manifest_sha256": "0" * 64}, "customer-entitlement-wrong-manifest"),
        ({"expires_at": "not-a-date"}, "customer-entitlement-expiry-invalid"),
        ({"expires_at": "2000-01-01T00:00:00Z"}, "customer-entitlement-stale"),
        ({"terms": []}, "customer-entitlement-terms-mismatch"),
    ],
)
def test_customer_entitlement_refuses_before_lock_access(
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    category: str,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_load_lock",
        lambda *_args, **_kwargs: pytest.fail("lock read preceded authorization"),
    )
    with pytest.raises(runtime.Refusal, match=category):
        runtime.run(
            lock_path=LOCK,
            environ=_environment_with_entitlement(**updates),
        )


def test_runtime_expiry_refusal_discards_private_exception_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_load_lock",
        lambda *_args, **_kwargs: pytest.fail("lock read preceded authorization"),
    )
    with pytest.raises(runtime.Refusal) as caught:
        runtime.run(
            lock_path=LOCK,
            environ=_environment_with_entitlement(
                expires_at="private-expiry-canary"
            ),
        )
    assert "private-expiry-canary" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_valid_context_reaches_only_the_technical_delivery_refusal(
    tmp_path: Path,
) -> None:
    watched = [tmp_path / name for name in ("source", "assets", "cache", "output")]
    with pytest.raises(
        runtime.Refusal,
        match="runtime-delivery-technical-gates-incomplete",
    ):
        runtime.run(lock_path=LOCK, environ=_environment())
    assert not any(path.exists() for path in watched)


def test_real_cli_missing_access_refusal_is_numeric_and_nonzero() -> None:
    env = dict(os.environ)
    env.pop(runtime.AUTH_ENV, None)
    completed = subprocess.run(
        [sys.executable, str(RUNTIME), "run", "--lock", str(LOCK)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 78
    assert completed.stderr.strip() == "ROBOTWIN_RUNTIME_REFUSED:authorization-missing"


def test_assert_refusal_uses_the_real_run_gate() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNTIME), "assert-refusal"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "no simulator capability claimed" in completed.stdout
