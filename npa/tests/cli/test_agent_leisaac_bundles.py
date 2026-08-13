from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from npa.agent_backend.leisaac_bundles import (
    BUNDLE_SCHEMA,
    DEVICE_ACTION_ORDER,
    DEVICE_SCHEMA,
    BundleError,
    BundleStore,
    validate_declarative_python,
    validate_bundle,
)
from npa.agent_backend.leisaac_routes import LeIsaacDeps, register_leisaac_routes
from npa.agent_backend.leisaac_registry import REGISTRY_FINGERPRINT


def _file(path: str, content: bytes) -> dict[str, str]:
    return {
        "path": path,
        "content_base64": base64.b64encode(content).decode(),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _payload(*, kind: str = "robot") -> dict:
    if kind == "device":
        files = [
            _file(
                "mapping.json",
                json.dumps(
                    {
                        "schema": DEVICE_SCHEMA,
                        "driver": "custom-so101",
                        "action_order": DEVICE_ACTION_ORDER,
                        "rate_hz": 50,
                    },
                    separators=(",", ":"),
                ).encode(),
            ),
            _file(
                "device.py", b'"""Declarative device provenance."""\nDEVICE = "so101"\n'
            ),
        ]
        entrypoint = "mapping.json"
    else:
        files = [
            _file("robot.usda", b'#usda 1.0\ndef Xform "SO101" {}\n'),
            _file(
                "asset.py",
                b'from typing import Final\nROBOT_USD: str = "robot.usda"\n',
            ),
        ]
        entrypoint = "robot.usda"
    return {
        "schema": BUNDLE_SCHEMA,
        "name": f"custom-{kind}",
        "kind": kind,
        "entrypoint": entrypoint,
        "files": files,
    }


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}
        self.list_calls: list[dict] = []

    def put_object(self, *, Bucket, Key, Body, Metadata=None, IfNoneMatch=None):
        target = (Bucket, Key)
        if IfNoneMatch == "*" and target in self.objects:
            raise RuntimeError("precondition failed")
        content = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.objects[target] = (content, dict(Metadata or {}))

    def get_object(self, *, Bucket, Key):
        content, metadata = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(content), "Metadata": metadata}

    def list_objects_v2(self, **kwargs):
        self.list_calls.append(kwargs)
        items = [
            {"Key": key, "Size": len(value[0])}
            for (bucket, key), value in sorted(self.objects.items())
            if bucket == kwargs["Bucket"] and key.startswith(kwargs["Prefix"])
        ][: int(kwargs["MaxKeys"])]
        return {"Contents": items, "IsTruncated": False}


def test_validates_declarative_usd_and_device_bundles() -> None:
    robot, robot_files = validate_bundle(_payload())
    assert robot["kind"] == "robot"
    assert robot["entrypoint"] == "robot.usda"
    assert len(robot_files) == 2
    assert len(robot["bundle_sha256"]) == 64

    device, _device_files = validate_bundle(_payload(kind="device"))
    assert device["device_contract"] == "npa.leisaac.so101-action.v1"


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda payload: payload["files"][0].update(path="../robot.usda"), "unsafe"),
        (lambda payload: payload["files"][0].update(sha256="0" * 64), "checksum"),
        (
            lambda payload: payload["files"].append(
                _file("unsafe.py", b'import os\nos.system("id")\n')
            ),
            "declarative-only",
        ),
        (lambda payload: payload.update(entrypoint="asset.py"), "USD"),
    ],
)
def test_rejects_traversal_tampering_and_executable_python(mutate, match) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(BundleError, match=match):
        validate_bundle(payload)


@pytest.mark.parametrize(
    "reference,match",
    [
        ("https://attacker.invalid/robot.usda", "external or network"),
        ("file:///etc/passwd", "external or network"),
        ("omniverse://attacker.invalid/asset", "external or network"),
        ("/etc/passwd", "escapes the bundle root"),
        ("../outside.usda", "escapes the bundle root"),
        ("missing.usda", "not present in the bundle"),
    ],
)
def test_rejects_hostile_usd_asset_references(reference: str, match: str) -> None:
    payload = _payload()
    payload["files"][0] = _file(
        "robot.usda", f'#usda 1.0\ndef Xform "Robot" (references = @{reference}@) {{}}\n'.encode()
    )
    with pytest.raises(BundleError, match=match):
        validate_bundle(payload)


def test_rejects_opaque_usdc_and_accepts_bundle_local_usda_reference() -> None:
    opaque = _payload()
    opaque["entrypoint"] = "robot.usdc"
    opaque["files"][0] = _file("robot.usdc", b"PXR-USDC\x00opaque")
    with pytest.raises(BundleError, match="binary USDC"):
        validate_bundle(opaque)

    local = _payload()
    local["files"][0] = _file(
        "robot.usda", b'#usda 1.0\ndef Xform "Robot" (references = @parts/arm.usda@) {}\n'
    )
    local["files"].append(_file("parts/arm.usda", b'#usda 1.0\ndef Xform "Arm" {}\n'))
    manifest, _files = validate_bundle(local)
    assert manifest["entrypoint"] == "robot.usda"


def test_declarative_python_size_gate_runs_before_ast_parse(monkeypatch) -> None:
    parsed = False

    def forbidden_parse(_source):
        nonlocal parsed
        parsed = True
        raise AssertionError("oversized source reached ast.parse")

    monkeypatch.setattr("npa.agent_backend.leisaac_bundles.ast.parse", forbidden_parse)
    with pytest.raises(BundleError, match="too large"):
        validate_declarative_python(b"A = 1\n" + b"#" * (256 * 1024))
    assert parsed is False


def test_store_is_immutable_bounded_and_scoped_to_configured_s3() -> None:
    s3 = FakeS3()
    store = BundleStore(
        s3,
        "s3://bucket/datasets/leisaac",
        allowed_buckets=["bucket"],
    )
    published = store.publish(_payload())
    digest = published["bundle_sha256"]
    manifest = store.get(digest)
    assert manifest["bundle_sha256"] == digest
    assert manifest["files"][0]["sha256"]
    listing = store.list(kind="robot")
    assert listing["bundles"] == [published]
    # Browser retries are idempotent for an identical content-addressed bundle.
    assert store.publish(_payload()) == published
    assert s3.list_calls[-1]["MaxKeys"] == 50
    assert all(
        key.startswith("datasets/leisaac/bundles/") for _bucket, key in s3.objects
    )

    with pytest.raises(BundleError, match="outside configured"):
        BundleStore(
            s3,
            "s3://other/datasets/leisaac",
            allowed_buckets=["bucket"],
        )
    with pytest.raises(BundleError, match="checksum"):
        store.get("../manifest")

    object_key = (
        "bucket",
        f"datasets/leisaac/bundles/objects/{digest}/files/robot.usda",
    )
    s3.objects[object_key] = (b"conflicting bytes", {})
    with pytest.raises(BundleError, match="conflicts"):
        store.publish(_payload())


def test_store_materializes_only_checksum_verified_bundle_files(tmp_path) -> None:
    s3 = FakeS3()
    store = BundleStore(
        s3,
        "s3://bucket/datasets/leisaac",
        allowed_buckets=["bucket"],
    )
    published = store.publish(_payload())
    digest = published["bundle_sha256"]
    destination = tmp_path / digest

    materialized = store.materialize(digest, destination)

    assert materialized["entrypoint_path"] == str(destination / "robot.usda")
    assert (destination / "robot.usda").read_bytes().startswith(b"#usda")
    assert (destination / "asset.py").stat().st_mode & 0o777 == 0o600
    # Cache hits are revalidated and do not read an unbounded S3 body again.
    assert store.materialize(digest, destination)["bundle_sha256"] == digest
    (destination / "robot.usda").write_bytes(b"tampered")
    with pytest.raises(BundleError, match="cached bundle"):
        store.materialize(digest, destination)


def test_materialization_rejects_s3_file_checksum_mismatch(tmp_path) -> None:
    s3 = FakeS3()
    store = BundleStore(
        s3,
        "s3://bucket/datasets/leisaac",
        allowed_buckets=["bucket"],
    )
    published = store.publish(_payload())
    digest = published["bundle_sha256"]
    key = f"datasets/leisaac/bundles/objects/{digest}/files/robot.usda"
    s3.objects[("bucket", key)] = (b"tampered", {"sha256": "0" * 64})

    with pytest.raises(BundleError, match="integrity"):
        store.materialize(digest, tmp_path / digest)


def test_manifest_digest_does_not_depend_on_base64_encoding_details() -> None:
    manifest, _files = validate_bundle(_payload())
    canonical = dict(manifest)
    digest = canonical.pop("bundle_sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_bundle_routes_require_same_origin_upload_and_persist_exact_selection() -> None:
    nonce = "a" * 64
    raw_manifest = {
        "schema": "npa.leisaac.session.v2",
        "run_id": "bundle-route-run",
        "provider": "nebius-kubernetes",
        "task": "LeIsaac-SO101-PickOrange-v0",
        "teleop_device": "keyboard",
        "signal_host": "8.8.8.8",
        "signal_port": 49100,
        "media_host": "1.1.1.1",
        "media_server": "1.1.1.1",
        "media_port": 47998,
        "service_url": "http://8.8.8.8:8080",
        "session_nonce": nonce,
        "session_attestation": hashlib.sha256(
            f"npa-leisaac-session:{nonce}".encode()
        ).hexdigest(),
        "task_registry_fingerprint": REGISTRY_FINGERPRINT,
        "source_version": "0.4.0",
        "source_commit": "1" * 40,
        "isaac_sim_version": "5.1.0.0",
        "isaac_lab_version": "2.3.2.post1",
        "image": "registry.example/leisaac@sha256:" + "2" * 64,
        "environment": {
            "id": "operator-0",
            "index": 0,
            "seed": 42,
            "num_envs": 1,
        },
        "dataset": {
            "output_path": "s3://bucket/datasets/leisaac",
            "format": "LeRobotDataset",
            "lerobot_version": "0.5.1",
            "codebase_version": "v3.0",
        },
    }
    s3 = FakeS3()
    state = {"leisaac": {"run_id": "bundle-route-run"}}
    saved: list[dict] = []
    posts: list[tuple[str, dict]] = []

    def http_post(url, **kwargs):
        posts.append((url, kwargs))
        return SimpleNamespace(status_code=202, json=lambda: {"accepted": True})

    app = FastAPI()
    register_leisaac_routes(
        app,
        LeIsaacDeps(
            load_state=lambda: state,
            save_state=lambda value: saved.append(json.loads(json.dumps(value))),
            resolve_manifest=lambda run_id: (
                raw_manifest if run_id == "bundle-route-run" else None
            ),
            http_get=lambda *_args, **_kwargs: None,
            http_post=http_post,
            response=Response,
            websocket_connect=lambda *_args, **_kwargs: None,
            s3_client=lambda: (s3, {}),
            s3_buckets=lambda _s3, _settings: ["bucket"],
        ),
    )
    client = TestClient(app)
    url = "/leisaac/bundles?run_id=bundle-route-run"
    assert client.post(url, json=_payload()).status_code == 403
    headers = {
        "x-forwarded-proto": "https",
        "x-npa-leisaac-control": "1",
        "sec-fetch-site": "same-origin",
    }
    uploaded = client.post(url, headers=headers, json=_payload())
    assert uploaded.status_code == 201
    digest = uploaded.json()["bundle_sha256"]
    listed = client.get(url, headers={"x-forwarded-proto": "https"})
    assert listed.status_code == 200
    assert listed.json()["bundles"][0]["bundle_sha256"] == digest
    selected = client.post(
        "/leisaac/bundles/select?run_id=bundle-route-run",
        headers=headers,
        json={"kind": "robot", "bundle_sha256": digest},
    )
    assert selected.status_code == 202
    assert saved[-1]["leisaac"]["bundle_selection"]["robot"] == {
        "bundle_sha256": digest,
        "entrypoint": "robot.usda",
        "name": "custom-robot",
    }
    assert saved[-1]["leisaac"]["bundle_selection_scope"] == {
        "run_id": "bundle-route-run",
        "dataset_uri": "s3://bucket/datasets/leisaac",
        "task": "LeIsaac-SO101-PickOrange-v0",
        "task_registry_fingerprint": REGISTRY_FINGERPRINT,
    }
    assert posts[-1][0] == "http://8.8.8.8:8080/bundles/apply"
    assert posts[-1][1]["json"] == {"selection": {"robot": digest}}
    assert posts[-1][1]["headers"] == {"X-NPA-LeIsaac-Nonce": nonce}
    reset = client.post(
        "/leisaac/bundles/reset?run_id=bundle-route-run",
        headers=headers,
        json={},
    )
    assert reset.status_code == 202
    assert reset.json()["selected_bundles"] == {}
    assert reset.json()["configuration"]["scene"]["id"] == "kitchen_with_orange"
    assert posts[-1][1]["json"] == {"selection": {}}
    assert saved[-1]["leisaac"]["bundle_selection"] == {}
    assert saved[-1]["leisaac"]["bundle_selection_scope"]["dataset_uri"] == (
        "s3://bucket/datasets/leisaac"
    )
    cross_site = client.get(
        url,
        headers={
            "x-forwarded-proto": "https",
            "sec-fetch-site": "cross-site",
        },
    )
    assert cross_site.status_code == 403


def test_bundle_selection_prunes_state_from_a_previous_dataset_prefix() -> None:
    nonce = "b" * 64
    raw_manifest = {
        "schema": "npa.leisaac.session.v2",
        "run_id": "bundle-stale-run",
        "provider": "nebius-kubernetes",
        "task": "LeIsaac-SO101-PickOrange-v0",
        "teleop_device": "keyboard",
        "signal_host": "8.8.4.4",
        "signal_port": 49100,
        "media_host": "1.0.0.1",
        "media_server": "1.0.0.1",
        "media_port": 47998,
        "service_url": "http://8.8.4.4:8080",
        "session_nonce": nonce,
        "session_attestation": hashlib.sha256(
            f"npa-leisaac-session:{nonce}".encode()
        ).hexdigest(),
        "task_registry_fingerprint": REGISTRY_FINGERPRINT,
        "source_version": "0.4.0",
        "source_commit": "1" * 40,
        "isaac_sim_version": "5.1.0.0",
        "isaac_lab_version": "2.3.2.post1",
        "image": "registry.example/leisaac@sha256:" + "2" * 64,
        "environment": {"id": "operator-0", "index": 0, "seed": 42, "num_envs": 1},
        "dataset": {
            "output_path": "s3://bucket/current/leisaac",
            "format": "LeRobotDataset",
            "lerobot_version": "0.5.1",
            "codebase_version": "v3.0",
        },
    }
    s3 = FakeS3()
    state = {
        "leisaac": {
            "bundle_selection": {
                "scene": {
                    "bundle_sha256": "f" * 64,
                    "entrypoint": "scene.usda",
                    "name": "stale-scene",
                }
            }
        }
    }
    saved: list[dict] = []
    posts: list[dict] = []

    def http_post(_url, **kwargs):
        posts.append(kwargs)
        return SimpleNamespace(status_code=202, json=lambda: {"accepted": True})

    app = FastAPI()
    register_leisaac_routes(
        app,
        LeIsaacDeps(
            load_state=lambda: state,
            save_state=lambda value: saved.append(json.loads(json.dumps(value))),
            resolve_manifest=lambda _run_id: raw_manifest,
            http_get=lambda *_args, **_kwargs: None,
            http_post=http_post,
            response=Response,
            websocket_connect=lambda *_args, **_kwargs: None,
            s3_client=lambda: (s3, {}),
            s3_buckets=lambda _s3, _settings: ["bucket"],
        ),
    )
    client = TestClient(app)
    headers = {
        "x-forwarded-proto": "https",
        "x-npa-leisaac-control": "1",
        "sec-fetch-site": "same-origin",
    }
    uploaded = client.post(
        "/leisaac/bundles?run_id=bundle-stale-run",
        headers=headers,
        json=_payload(),
    )
    assert uploaded.status_code == 201, uploaded.text
    digest = uploaded.json()["bundle_sha256"]
    selected = client.post(
        "/leisaac/bundles/select?run_id=bundle-stale-run",
        headers=headers,
        json={"kind": "robot", "bundle_sha256": digest},
    )
    assert selected.status_code == 202
    assert posts[-1]["json"] == {"selection": {"robot": digest}}
    assert set(saved[-1]["leisaac"]["bundle_selection"]) == {"robot"}
    assert saved[-1]["leisaac"]["bundle_selection_scope"]["dataset_uri"] == (
        "s3://bucket/current/leisaac"
    )


def test_cumulative_bundle_selection_uses_atomic_backend_state_mutation() -> None:
    nonce = "c" * 64
    raw_manifest = {
        "schema": "npa.leisaac.session.v2",
        "run_id": "bundle-atomic-run",
        "provider": "nebius-kubernetes",
        "task": "LeIsaac-SO101-PickOrange-v0",
        "teleop_device": "keyboard",
        "signal_host": "8.8.8.8",
        "signal_port": 49100,
        "media_host": "1.0.0.1",
        "media_server": "1.0.0.1",
        "media_port": 47998,
        "service_url": "http://8.8.8.8:8080",
        "session_nonce": nonce,
        "session_attestation": hashlib.sha256(
            f"npa-leisaac-session:{nonce}".encode()
        ).hexdigest(),
        "task_registry_fingerprint": REGISTRY_FINGERPRINT,
        "source_version": "0.4.0",
        "source_commit": "1" * 40,
        "isaac_sim_version": "5.1.0.0",
        "isaac_lab_version": "2.3.2.post1",
        "image": "registry.example/leisaac@sha256:" + "2" * 64,
        "environment": {"id": "operator-0", "index": 0, "seed": 0, "num_envs": 1},
        "dataset": {
            "output_path": "s3://bucket/current/leisaac",
            "format": "LeRobotDataset",
            "lerobot_version": "0.5.1",
            "codebase_version": "v3.0",
        },
    }
    state: dict = {"leisaac": {}}
    s3 = FakeS3()
    applied: list[dict] = []
    runtime_selected: dict[str, dict[str, str]] = {}
    block_next_apply = False
    apply_entered = threading.Event()
    release_apply = threading.Event()

    def mutate_state(mutation):
        result = mutation(state)
        return result

    class Healthy:
        status_code = 200

        @staticmethod
        def json():
            return {
                "schema": "npa.leisaac.health.v1",
                "state": "ready",
                "webrtc_ready": True,
                "run_id": raw_manifest["run_id"],
                "task": raw_manifest["task"],
                "source_commit": raw_manifest["source_commit"],
                "session_nonce": raw_manifest["session_nonce"],
                "signal_port": 49100,
                "selected_bundles": json.loads(json.dumps(runtime_selected)),
            }

    def http_post(_url, **kwargs):
        nonlocal block_next_apply
        applied.append(kwargs["json"])
        runtime_selected.clear()
        for kind, digest in kwargs["json"]["selection"].items():
            manifest = s3.objects[
                (
                    "bucket",
                    f"current/leisaac/bundles/objects/{digest}/bundle.json",
                )
            ][0]
            bundle = json.loads(manifest)
            runtime_selected[kind] = {
                "bundle_sha256": digest,
                "name": bundle["name"],
                "entrypoint": bundle["entrypoint"],
            }
        if block_next_apply:
            block_next_apply = False
            apply_entered.set()
            assert release_apply.wait(5), "test did not release bundle apply"
        return SimpleNamespace(status_code=202, json=lambda: {"accepted": True})

    app = FastAPI()
    register_leisaac_routes(
        app,
        LeIsaacDeps(
            load_state=lambda: json.loads(json.dumps(state)),
            mutate_state=mutate_state,
            resolve_manifest=lambda _run_id: raw_manifest,
            http_get=lambda *_args, **_kwargs: Healthy(),
            http_post=http_post,
            response=Response,
            websocket_connect=lambda *_args, **_kwargs: None,
            s3_client=lambda: (s3, {}),
            s3_buckets=lambda _s3, _settings: ["bucket"],
        ),
    )
    client = TestClient(app)
    headers = {
        "x-forwarded-proto": "https",
        "x-npa-leisaac-control": "1",
        "sec-fetch-site": "same-origin",
    }
    for kind in ("scene", "device", "robot"):
        uploaded = client.post(
            "/leisaac/bundles?run_id=bundle-atomic-run",
            headers=headers,
            json=_payload(kind=kind),
        )
        assert uploaded.status_code == 201, uploaded.text
        selected = client.post(
            "/leisaac/bundles/select?run_id=bundle-atomic-run",
            headers=headers,
            json={
                "kind": kind,
                "bundle_sha256": uploaded.json()["bundle_sha256"],
            },
        )
        assert selected.status_code == 202, selected.text
        # The UI refreshes capability selection during each simulator restart.
        # That run-id write must not replace the cumulative bundle mapping.
        refreshed = client.post(
            "/leisaac/select",
            headers=headers,
            json={"run_id": "bundle-atomic-run"},
        )
        assert refreshed.status_code == 200, refreshed.text

    selection = state["leisaac"]["bundle_selection"]
    assert set(selection) == {"robot", "scene", "device"}
    assert state["leisaac"]["bundle_selection_scope"] == {
        "run_id": "bundle-atomic-run",
        "dataset_uri": "s3://bucket/current/leisaac",
        "task": "LeIsaac-SO101-PickOrange-v0",
        "task_registry_fingerprint": REGISTRY_FINGERPRINT,
    }
    assert [set(item["selection"]) for item in applied] == [
        {"scene"},
        {"scene", "device"},
        {"scene", "device", "robot"},
    ]

    # A fresh runtime after a Kubernetes rollout starts on built-ins. The
    # agent's scoped, atomically saved selection is reconciled exactly once.
    runtime_selected.clear()
    before_restore = len(applied)
    restoring = client.get(
        "/leisaac/status?run_id=bundle-atomic-run",
        headers={"x-forwarded-proto": "https"},
    )
    assert restoring.status_code == 200
    assert restoring.json()["available"] is False
    assert "Restoring persisted" in restoring.json()["reason"]
    assert len(applied) == before_restore + 1
    assert set(applied[-1]["selection"]) == {"robot", "scene", "device"}
    restored = client.get(
        "/leisaac/status?run_id=bundle-atomic-run",
        headers={"x-forwarded-proto": "https"},
    )
    assert restored.json()["available"] is True
    assert restored.json()["bundle_selection"] == selection
    assert restored.json()["configuration"]["custom_bundle_count"] == 3
    assert len(applied) == before_restore + 1

    # Slow runtime apply must not serialize independent status health/storage
    # discovery. The status snapshot may describe the last committed selection,
    # while the narrow mutation transaction prevents it from restoring over the
    # in-flight replacement.
    replacement = _payload(kind="scene")
    replacement["name"] = "replacement-scene"
    replacement["files"][0] = _file(
        "robot.usda", b'#usda 1.0\ndef Xform "ReplacementScene" {}\n'
    )
    uploaded = client.post(
        "/leisaac/bundles?run_id=bundle-atomic-run",
        headers=headers,
        json=replacement,
    )
    assert uploaded.status_code == 201
    block_next_apply = True
    before_replacement = len(applied)

    def select_replacement():
        # Starlette's TestClient owns an anyio portal and is not safe to share
        # across OS threads.  Give each concurrent request its own portal so
        # this test exercises the backend transaction instead of occasionally
        # deadlocking inside the test harness.
        with TestClient(app) as concurrent_client:
            return concurrent_client.post(
                "/leisaac/bundles/select?run_id=bundle-atomic-run",
                headers=headers,
                json={
                    "kind": "scene",
                    "bundle_sha256": uploaded.json()["bundle_sha256"],
                },
            )

    def fetch_concurrent_status():
        with TestClient(app) as concurrent_client:
            return concurrent_client.get(
                "/leisaac/status?run_id=bundle-atomic-run",
                headers={"x-forwarded-proto": "https"},
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        selection_future = executor.submit(select_replacement)
        assert apply_entered.wait(5), "bundle apply did not start"
        status_future = executor.submit(fetch_concurrent_status)
        concurrent_status = status_future.result(timeout=2)
        assert not selection_future.done(), "test apply was not held open"
        release_apply.set()
        selected = selection_future.result(timeout=5)
    assert selected.status_code == 202
    assert concurrent_status.status_code == 200
    assert concurrent_status.json()["available"] is False
    assert "Applying a checksum-verified" in concurrent_status.json()["reason"]
    assert len(applied) == before_replacement + 1
    assert runtime_selected["scene"]["name"] == "replacement-scene"
