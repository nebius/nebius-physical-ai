"""Tests for the RoboCasa shared implementation and service."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import types
import zipfile
from pathlib import Path
from zipfile import BadZipFile, ZIP_BZIP2, ZIP_LZMA, ZipFile, ZipInfo

import httpx
import numpy as np

import pytest
from fastapi.testclient import TestClient

from npa.sdk.workbench import robocasa as robocasa_sdk
from npa.workbench.robocasa import capabilities
from npa.workbench.robocasa.capabilities import (
    RoboCasaError,
    compute_manifest_sha256,
    kitchen_asset_availability,
    kitchen_task_registration,
    make_run_id,
    system_info,
)
from npa.workbench.robocasa.schemas import RoboCasaRunRequest
from npa.workbench.robocasa.schemas import RoboCasaStatusResponse
from npa.workbench.robocasa.service import RunRegistry, create_app


def _blocking_robocasa_worker(sender, request_payload) -> None:
    try:
        output_dir = Path(str(request_payload["_worker_output_dir"]))
        (output_dir / "partial.bin").write_bytes(b"x" * 1024 * 1024)
        asset_temp_root = str(request_payload.get("_worker_asset_temp_root") or "")
        if asset_temp_root:
            (Path(asset_temp_root) / "asset-partial.bin").write_bytes(
                b"y" * 1024 * 1024
            )
        time.sleep(30)
    finally:
        sender.close()


def _successful_robocasa_worker(sender, _request_payload) -> None:
    try:
        sender.send_bytes(b'{"kind":"result","result":{"ok":true}}')
        sender.recv_bytes(1)
    except (EOFError, OSError):
        pass
    finally:
        sender.close()


def _artifact_robocasa_worker(sender, request_payload) -> None:
    try:
        assert request_payload["output_uri"] == "s3://example/output"
        assert request_payload["_worker_defer_output_upload"] is True
        output_dir = Path(str(request_payload["_worker_output_dir"]))
        (output_dir / "artifact.bin").write_bytes(b"complete")
        sender.send_bytes(b'{"kind":"result","result":{"ok":true}}')
        sender.recv_bytes(1)
    except (EOFError, OSError):
        pass
    finally:
        sender.close()


def test_functional_registration_worker_emits_only_final_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.smoke.test_robocasa_functional import _registration_worker

    class RecordingSender:
        def __init__(self) -> None:
            self.messages: list[bytes] = []
            self.closed = False

        def send_bytes(self, payload: bytes) -> None:
            self.messages.append(payload)

        def recv_bytes(self, _size: int) -> bytes:
            return b"\0"

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        capabilities,
        "kitchen_task_registration",
        lambda **_kwargs: {"registered_env_count": 1},
    )
    sender = RecordingSender()

    _registration_worker(sender, {})

    assert sender.closed is True
    assert [json.loads(message) for message in sender.messages] == [
        {
            "kind": "result",
            "result": {"registered_env_count": 1},
        }
    ]


def _robocasa_worker_without_isolation_ack(sender, _request_payload) -> None:
    try:
        sender.send_bytes(b'{"kind":"result","result":{"ok":true}}')
        sender.recv_bytes(1)
    except (EOFError, OSError):
        pass
    finally:
        sender.close()


def _supervisor_without_containment_ack(
    sender, _request_payload, _worker_target
) -> None:
    try:
        sender.send_bytes(b'{"kind":"result","result":{"ok":true}}')
        sender.recv_bytes(1)
        sender.send_bytes(b'{"kind":"stopped","stopped":false}')
    except (EOFError, OSError):
        pass
    finally:
        sender.close()


def _robocasa_worker_with_term_ignoring_descendant(sender, _request_payload) -> None:
    try:
        descendant = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import signal,time;"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                    "print('ready', flush=True);"
                    "time.sleep(30)"
                ),
            ],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert descendant.stdout is not None
        assert descendant.stdout.readline().strip() == "ready"
        sender.send_bytes(
            json.dumps(
                {
                    "kind": "result",
                    "result": {"descendant_pid": descendant.pid},
                }
            ).encode("utf-8")
        )
        sender.recv_bytes(1)
    except (EOFError, OSError):
        pass
    finally:
        sender.close()


def _robocasa_worker_whose_leader_exits(sender, _request_payload) -> None:
    try:
        descendant = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import signal,time;"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                    "print('ready', flush=True);"
                    "time.sleep(30)"
                ),
            ],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert descendant.stdout is not None
        assert descendant.stdout.readline().strip() == "ready"
        sender.send_bytes(
            json.dumps(
                {
                    "kind": "result",
                    "result": {"descendant_pid": descendant.pid},
                }
            ).encode("utf-8")
        )
    finally:
        sender.close()


def _robocasa_worker_with_double_fork_daemon(sender, _request_payload) -> None:
    try:
        descendant = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import os,signal,time;"
                    "child=os.fork();"
                    "\nif child: os._exit(0);"
                    "\nos.setsid();"
                    "\nchild=os.fork();"
                    "\nif child: os._exit(0);"
                    "\nsignal.signal(signal.SIGTERM, signal.SIG_IGN);"
                    "\nprint(os.getpid(), flush=True);"
                    "\ntime.sleep(30)"
                ),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert descendant.stdout is not None
        daemon_pid = int(descendant.stdout.readline().strip())
        sender.send_bytes(
            json.dumps(
                {
                    "kind": "result",
                    "result": {"descendant_pid": daemon_pid},
                }
            ).encode("utf-8")
        )
        sender.recv_bytes(1)
    except (EOFError, OSError):
        pass
    finally:
        sender.close()


def _install_deployed_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, str]:
    source_sha = "a" * 40
    manifest_digest = "sha256:" + "b" * 64
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", source_sha)
    monkeypatch.setenv("ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA", "1")
    monkeypatch.setenv("ROBOCASA_DEPLOYED_IMAGE_SOURCE_SHA", source_sha)
    monkeypatch.setenv("ROBOCASA_DEPLOYED_IMAGE_MANIFEST_DIGEST", manifest_digest)
    return source_sha, manifest_digest


def _service_run_payload(source_sha: str, manifest_digest: str) -> dict[str, object]:
    return {
        "capability": "kitchen_task_registration",
        "env_id": "robocasa/PickPlaceCounterToCabinet",
        "output_uri": "s3://bucket/out",
        "download_assets": False,
        "expected_image_source_sha": source_sha,
        "expected_image_manifest_digest": manifest_digest,
    }


def _install_fake_robocasa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake robocasa + gymnasium module tree so capability tests run
    without the real simulation stack."""

    class FakeSpec:
        entry_point = "robocasa.envs:KitchenEnv"

    class FakeRegistry(dict):
        def __init__(self) -> None:
            super().__init__()
            self["robocasa/PickPlaceCounterToCabinet"] = FakeSpec()
            self["robocasa/StackHouseholdItems"] = FakeSpec()

    class FakeGym:
        envs = types.SimpleNamespace(registry=FakeRegistry())

    fake_robocasa = types.ModuleType("robocasa")
    fake_robocasa.__file__ = "/opt/robocasa/robocasa/__init__.py"
    monkeypatch.setitem(sys.modules, "robocasa", fake_robocasa)
    monkeypatch.setitem(sys.modules, "gymnasium", FakeGym())


def test_compute_manifest_sha256_is_deterministic() -> None:
    payload = {
        "env_id": "robocasa/PickPlaceCounterToCabinet",
        "capability": "kitchen_random_rollout",
    }
    a = compute_manifest_sha256("run", payload)
    b = compute_manifest_sha256("run", dict(payload))
    assert a == b
    assert len(a) == 64


def test_make_run_id_is_deterministic() -> None:
    a = make_run_id("kitchen_random_rollout", "abc")
    b = make_run_id("kitchen_random_rollout", "abc")
    assert a == b
    assert a.startswith("robocasa-kitchen_random_rollout-")


def test_system_info_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_IMAGE_SOURCE_SHA", raising=False)
    monkeypatch.delenv("ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA", raising=False)
    info = system_info()
    assert info.status == "ok"
    assert info.python
    assert info.source_identity == "local_unbound"
    assert info.image_source_sha == ""


def test_system_info_reports_policy_runtime_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "lerobot": "0.6.1+npa1",
        "torch": "2.13.0",
        "torchvision": "0.28.0",
    }
    monkeypatch.setattr(
        capabilities,
        "_package_version",
        lambda package: versions.get(package, ""),
    )
    monkeypatch.delenv("NPA_IMAGE_SOURCE_SHA", raising=False)
    monkeypatch.delenv("ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA", raising=False)

    info = system_info()

    assert info.lerobot_version == "0.6.1+npa1"
    assert info.torch_version == "2.13.0"
    assert info.torchvision_version == "0.28.0"


def test_system_info_reports_exact_image_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sha = "a" * 40
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", source_sha)
    monkeypatch.setenv("ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA", "1")

    info = system_info()

    assert info.source_identity == "container_image"
    assert info.image_source_sha == source_sha


def test_system_info_rejects_missing_required_image_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NPA_IMAGE_SOURCE_SHA", raising=False)
    monkeypatch.setenv("ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA", "1")

    with pytest.raises(RoboCasaError, match="missing required NPA_IMAGE_SOURCE_SHA"):
        system_info()


def test_system_info_rejects_malformed_image_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", "short")
    monkeypatch.delenv("ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA", raising=False)

    with pytest.raises(RoboCasaError, match="40-character git SHA"):
        system_info()


def test_kitchen_task_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_robocasa(monkeypatch)
    result = kitchen_task_registration()
    assert result["env_id"] == "robocasa/PickPlaceCounterToCabinet"
    assert result["registered_env_count"] == 2


def test_fresh_task_registration_imports_robocasa_before_checking_gym(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry: dict[str, object] = {}
    gym = types.SimpleNamespace(envs=types.SimpleNamespace(registry=registry))
    events: list[str] = []

    def import_robocasa() -> object:
        events.append("import-robocasa")
        registry["robocasa/PickPlaceCounterToCabinet"] = types.SimpleNamespace(
            entry_point="robocasa.envs:KitchenEnv"
        )
        return object()

    def import_gymnasium() -> object:
        events.append("import-gymnasium")
        assert "import-robocasa" in events
        return gym

    monkeypatch.setattr(
        capabilities, "_download_assets", lambda: events.append("assets")
    )
    monkeypatch.setattr(capabilities, "_import_robocasa", import_robocasa)
    monkeypatch.setattr(capabilities, "_import_gymnasium", import_gymnasium)

    result = kitchen_task_registration(download_assets=True)

    assert result["registered_env_count"] == 1
    assert events == ["assets", "import-robocasa", "import-gymnasium"]


def test_kitchen_task_registration_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_robocasa(monkeypatch)
    with pytest.raises(RoboCasaError):
        kitchen_task_registration(env_id="robocasa/DoesNotExist")


def test_kitchen_asset_availability_missing_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_robocasa(monkeypatch)
    with pytest.raises(RoboCasaError):
        kitchen_asset_availability()


def test_kitchen_asset_availability_rejects_empty_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    archive = capabilities._AssetArchive(
        "example/assets",
        "a" * 40,
        "assets.zip",
        ".",
        "assets",
        "assets",
    )
    monkeypatch.setattr(capabilities, "_assets_root", lambda: assets_root)
    monkeypatch.setattr(capabilities, "_asset_archives", lambda: (archive,))

    with pytest.raises(RoboCasaError, match="population is incomplete"):
        kitchen_asset_availability()


def test_kitchen_asset_availability_requires_valid_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = capabilities._AssetArchive(
        "example/assets",
        "a" * 40,
        "assets.zip",
        ".",
        "assets",
        "assets",
    )
    source_zip = tmp_path / "assets.zip"
    with ZipFile(source_zip, "w") as archive_zip:
        archive_zip.writestr("assets/complete.txt", "complete")
    assets_root = tmp_path / "published"
    state_root = assets_root / ".npa_asset_fetch"
    receipt = capabilities._asset_receipt_path(state_root, archive)
    capabilities._stage_publish_and_receipt(archive, source_zip, assets_root, receipt)
    monkeypatch.setattr(capabilities, "_assets_root", lambda: assets_root)
    monkeypatch.setattr(capabilities, "_asset_archives", lambda: (archive,))

    result = kitchen_asset_availability()

    assert result["archive_receipts_valid"] == 1
    assert result["subdirs"] == [".npa_asset_fetch", "assets"]


def test_run_capability_unsupported() -> None:
    # The schema rejects an unsupported capability before dispatch.
    with pytest.raises(ValueError):
        RoboCasaRunRequest(capability="bogus", output_uri="s3://bucket/out")


def test_run_request_validates_capability() -> None:
    with pytest.raises(ValueError):
        RoboCasaRunRequest(capability="bogus", output_uri="s3://bucket/out")


@pytest.mark.parametrize(
    "value", ["/tmp/out", "file:///tmp/out", "https://example.invalid/out"]
)
def test_run_request_rejects_non_s3_output(value: str) -> None:
    with pytest.raises(ValueError, match="expects an S3 URI"):
        RoboCasaRunRequest(
            capability="kitchen_task_registration",
            output_uri=value,
        )


def test_service_health() -> None:
    app = create_app(auth_mode="none")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["execution_available"] is True


def test_service_system_info() -> None:
    app = create_app(auth_mode="none")
    client = TestClient(app)
    response = client.get("/system-info")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_service_system_info_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_probe():
        started.set()
        assert release.wait(timeout=5)
        return system_info()

    monkeypatch.setattr("npa.workbench.robocasa.service.system_info", blocking_probe)
    app = create_app(auth_mode="none", runs=RunRegistry())

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            probe = asyncio.create_task(client.get("/system-info"))
            assert await asyncio.to_thread(started.wait, 2)
            health = await asyncio.wait_for(client.get("/health"), timeout=1)
            assert health.status_code == 200
            release.set()
            assert (await probe).status_code == 200

    asyncio.run(exercise())


def test_service_run_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_robocasa(monkeypatch)
    source_sha, manifest_digest = _install_deployed_runtime_identity(monkeypatch)
    app = create_app(
        auth_mode="none",
        capability_executor=lambda *_args, **_kwargs: {"registration_ok": True},
    )
    client = TestClient(app)
    response = client.post(
        "/run",
        json=_service_run_payload(source_sha, manifest_digest),
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    status_response = client.get("/status", params={"run_id": run_id})
    assert status_response.status_code == 200
    assert status_response.json()["status"] in {"running", "completed"}


def test_service_rejects_identity_mismatch_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sha, manifest_digest = _install_deployed_runtime_identity(monkeypatch)
    executed = False

    def unexpected_execution(*_args, **_kwargs):
        nonlocal executed
        executed = True

    monkeypatch.setattr(
        "npa.workbench.robocasa.service.run_capability_with_output",
        unexpected_execution,
    )
    runs = RunRegistry()
    client = TestClient(create_app(auth_mode="none", runs=runs))
    payload = _service_run_payload(source_sha, manifest_digest)
    payload["expected_image_source_sha"] = "c" * 40

    response = client.post("/run", json=payload)

    assert response.status_code == 409
    assert "does not match runtime" in response.json()["detail"]
    assert executed is False
    assert runs.values() == []


def test_duplicate_active_run_is_enqueued_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sha, manifest_digest = _install_deployed_runtime_identity(monkeypatch)
    scheduled: list[tuple[object, tuple[object, ...]]] = []

    def record_task(_self, function, *args, **_kwargs):
        scheduled.append((function, args))

    monkeypatch.setattr("starlette.background.BackgroundTasks.add_task", record_task)
    runs = RunRegistry()
    client = TestClient(create_app(auth_mode="none", runs=runs))
    payload = _service_run_payload(source_sha, manifest_digest)

    first = client.post("/run", json=payload)
    second = client.post("/run", json=payload)

    assert first.status_code == second.status_code == 200
    assert first.json()["run_id"] == second.json()["run_id"]
    assert first.json()["status"] == second.json()["status"] == "queued"
    assert len(scheduled) == 1
    assert len(runs.values()) == 1


def test_service_status_unknown_run() -> None:
    app = create_app(auth_mode="none")
    client = TestClient(app)
    response = client.get("/status", params={"run_id": "nope"})
    assert response.status_code == 404


def test_service_run_invalid_capability() -> None:
    app = create_app(auth_mode="none")
    client = TestClient(app)
    response = client.post(
        "/run",
        json={"capability": "bogus", "output_uri": "s3://bucket/out"},
    )
    assert response.status_code == 422


def test_sdk_service_mode_requires_exact_image_identity() -> None:
    with pytest.raises(
        robocasa_sdk.RoboCasaValidationError,
        match="expected_image_source_sha is required",
    ):
        robocasa_sdk.run(
            capability="kitchen_task_registration",
            output_path="s3://bucket/output",
            mode="service",
            endpoint="http://robocasa.invalid",
        )
    with pytest.raises(
        robocasa_sdk.RoboCasaValidationError,
        match="expected_image_manifest_digest is required",
    ):
        robocasa_sdk.run(
            capability="kitchen_task_registration",
            output_path="s3://bucket/output",
            mode="service",
            endpoint="http://robocasa.invalid",
            expected_image_source_sha="a" * 40,
        )


def test_sdk_service_mode_forwards_exact_image_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def request_json(method, endpoint, path, **kwargs):
        observed.update(
            method=method,
            endpoint=endpoint,
            path=path,
            payload=kwargs["payload"],
        )
        return {
            "run_id": "run-1",
            "status": "queued",
            "env_id": "robocasa/PickPlaceCounterToCabinet",
            "capability": "kitchen_task_registration",
            "output_uri": "s3://bucket/output",
            "manifest_sha256": "c" * 64,
        }

    monkeypatch.setattr(robocasa_sdk, "_request_json", request_json)
    source_sha = "a" * 40
    manifest_digest = "sha256:" + "b" * 64

    response = robocasa_sdk.run(
        capability="kitchen_task_registration",
        output_path="s3://bucket/output",
        mode="service",
        endpoint="http://robocasa.invalid",
        expected_image_source_sha=source_sha,
        expected_image_manifest_digest=manifest_digest,
    )

    assert response.run_id == "run-1"
    assert observed["payload"]["expected_image_source_sha"] == source_sha
    assert observed["payload"]["expected_image_manifest_digest"] == manifest_digest


def test_service_auth_token() -> None:
    app = create_app(auth_mode="token", token="secret")
    client = TestClient(app)
    # /health is intentionally unauthenticated so Kubernetes liveness/readiness
    # probes can reach it without a token; the protected surface is /system-info.
    assert client.get("/health").status_code == 200
    assert client.get("/system-info").status_code == 401
    assert (
        client.get(
            "/system-info", headers={"Authorization": "Bearer secret"}
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/system-info", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )


def test_service_list_runs() -> None:
    app = create_app(auth_mode="none")
    client = TestClient(app)
    response = client.get("/runs")
    assert response.status_code == 200
    assert "runs" in response.json()


def _status(run_id: str, status: str) -> RoboCasaStatusResponse:
    return RoboCasaStatusResponse(
        run_id=run_id,
        status=status,
        capability="kitchen_random_rollout",
        env_id="robocasa/PickPlaceCounterToCabinet",
        output_uri="s3://example/output",
    )


def test_run_registry_ttl_evicts_terminal_but_not_active_runs() -> None:
    now = [10.0]
    runs = RunRegistry(max_entries=2, ttl_seconds=5, clock=lambda: now[0])
    runs["done"] = _status("done", "completed")
    runs["active"] = _status("active", "running")
    now[0] = 16.0

    assert runs.get("done") is None
    assert runs.get("active") is not None


def test_run_registry_size_evicts_oldest_terminal_without_evicting_active() -> None:
    now = [1.0]
    runs = RunRegistry(max_entries=2, ttl_seconds=100, clock=lambda: now[0])
    runs["active"] = _status("active", "running")
    now[0] += 1
    runs["old"] = _status("old", "completed")
    now[0] += 1
    runs["new"] = _status("new", "completed")

    assert runs.get("active") is not None
    assert runs.get("old") is None
    assert runs.get("new") is not None


def test_run_registry_rejects_overflow_when_every_slot_is_active() -> None:
    runs = RunRegistry(max_entries=1, ttl_seconds=100, clock=lambda: 1.0)
    runs["one"] = _status("one", "running")

    with pytest.raises(RoboCasaError, match="registry is full"):
        runs["two"] = _status("two", "running")

    assert {run.run_id for run in runs.values()} == {"one"}


def test_service_returns_429_when_active_registry_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sha, manifest_digest = _install_deployed_runtime_identity(monkeypatch)
    runs = RunRegistry(max_entries=1)
    runs["existing"] = _status("existing", "running")
    client = TestClient(create_app(auth_mode="none", runs=runs))

    response = client.post(
        "/run", json=_service_run_payload(source_sha, manifest_digest)
    )

    assert response.status_code == 429
    assert "registry is full" in response.json()["detail"]
    assert {run.run_id for run in runs.values()} == {"existing"}


def test_poisoned_gpu_gate_wakes_waiters_and_rejects_new_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workbench.robocasa import service

    gate = service.GpuExecutionGate()
    assert gate.acquire() is True
    attempting = threading.Event()
    acquired: list[bool] = []

    def wait_for_gate() -> None:
        attempting.set()
        acquired.append(gate.acquire())

    waiter = threading.Thread(target=wait_for_gate)
    waiter.start()
    assert attempting.wait(timeout=1)
    gate.poison()
    waiter.join(timeout=1)

    assert acquired == [False]
    assert gate.available is False

    source_sha, manifest_digest = _install_deployed_runtime_identity(monkeypatch)
    runs = RunRegistry()
    client = TestClient(create_app(auth_mode="none", runs=runs, execution_lock=gate))
    health = client.get("/health")
    response = client.post(
        "/run", json=_service_run_payload(source_sha, manifest_digest)
    )

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["execution_available"] is False
    assert response.status_code == 503
    assert runs.values() == []


def test_run_registry_concurrent_updates_are_safe() -> None:
    runs = RunRegistry(max_entries=64, ttl_seconds=100, clock=lambda: 1.0)
    for index in range(32):
        run_id = f"run-{index}"
        runs[run_id] = _status(run_id, "running")

    threads = [
        threading.Thread(
            target=runs.update,
            kwargs={
                "run_id": f"run-{index}",
                "status": "completed",
                "result": {"index": index},
                "error": None,
            },
        )
        for index in range(32)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(runs.values()) == 32
    assert all(run.status == "completed" for run in runs.values())


def test_run_registry_concurrent_admission_never_exceeds_capacity() -> None:
    runs = RunRegistry(max_entries=3, ttl_seconds=100)
    barrier = threading.Barrier(10)
    accepted: list[str] = []
    rejected: list[str] = []
    result_lock = threading.Lock()

    def enqueue(index: int) -> None:
        run_id = f"run-{index}"
        barrier.wait()
        try:
            was_accepted, _status_record = runs.enqueue(
                run_id, _status(run_id, "queued")
            )
        except RoboCasaError:
            with result_lock:
                rejected.append(run_id)
        else:
            assert was_accepted
            with result_lock:
                accepted.append(run_id)

    threads = [threading.Thread(target=enqueue, args=(index,)) for index in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert len(accepted) == len(runs.values()) == 3
    assert len(rejected) == 7


def test_gpu_runs_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workbench.robocasa import service

    runs = RunRegistry()
    runs["one"] = _status("one", "queued")
    runs["two"] = _status("two", "queued")
    execution_lock = threading.Lock()
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    invocation_count = 0
    active = 0
    maximum_active = 0
    counter_lock = threading.Lock()

    def fake_run(*_args, **_kwargs):
        nonlocal invocation_count, active, maximum_active
        with counter_lock:
            invocation_count += 1
            invocation = invocation_count
            active += 1
            maximum_active = max(maximum_active, active)
        if invocation == 1:
            first_entered.set()
            assert release_first.wait(timeout=5)
        else:
            second_entered.set()
        with counter_lock:
            active -= 1
        return {"ok": True}

    monkeypatch.setattr(service, "run_capability_with_output", fake_run)
    request = RoboCasaRunRequest(
        capability="kitchen_random_rollout",
        output_uri="s3://example/output",
    )
    first = threading.Thread(
        target=service._run_capability,
        args=(request, "one", runs, execution_lock, fake_run),
    )
    second = threading.Thread(
        target=service._run_capability,
        args=(request, "two", runs, execution_lock, fake_run),
    )

    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert second_entered.is_set()
    assert maximum_active == 1
    assert runs.get("one").status == "completed"
    assert runs.get("two").status == "completed"


def test_capability_worker_timeout_stops_child_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from npa.workbench.robocasa import service

    output_root = tmp_path / "outputs"
    output_root.mkdir()
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    monkeypatch.setattr(service, "_assets_root", lambda: assets_root)
    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
        timeout_seconds=5,
    )
    started = time.monotonic()

    outcome = service._execute_capability_in_worker(
        request,
        worker_target=_blocking_robocasa_worker,
        process_context=multiprocessing.get_context("spawn"),
        output_root=output_root,
    )

    assert time.monotonic() - started < 10
    assert outcome.timed_out is True
    assert outcome.stopped is True
    assert outcome.result is None
    assert "timeout_seconds=5" in str(outcome.error)
    assert list(output_root.iterdir()) == []
    workers_root = assets_root / ".npa_asset_fetch" / "workers"
    assert list(workers_root.iterdir()) == []


def test_capability_worker_retains_result_before_group_cleanup() -> None:
    from npa.workbench.robocasa import service

    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
        timeout_seconds=15,
        download_assets=False,
    )

    outcome = service._execute_capability_in_worker(
        request,
        worker_target=_successful_robocasa_worker,
        process_context=multiprocessing.get_context("spawn"),
    )

    assert outcome == service._WorkerOutcome(result={"ok": True})


def test_capability_worker_retains_local_output_only_after_cleanup(
    tmp_path: Path,
) -> None:
    from npa.workbench.robocasa import service

    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
        timeout_seconds=15,
        download_assets=False,
    )

    outcome = service._execute_capability_in_worker(
        request,
        worker_target=_artifact_robocasa_worker,
        process_context=multiprocessing.get_context("spawn"),
        output_root=tmp_path,
        retain_output=True,
    )

    assert outcome.stopped is True
    assert outcome.result == {"ok": True}
    assert outcome.output_dir is not None
    assert (outcome.output_dir / "artifact.bin").read_bytes() == b"complete"
    shutil.rmtree(outcome.output_dir)


def test_production_worker_defers_upload_without_invalidating_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workbench.robocasa import service

    captured: dict[str, object] = {}

    def execute(body, *, output_dir, upload):
        captured.update(body=body, output_dir=output_dir, upload=upload)
        return {"ok": True}

    class Sender:
        def __init__(self) -> None:
            self.messages: list[bytes] = []

        def send_bytes(self, payload: bytes) -> None:
            self.messages.append(payload)

        def recv_bytes(self, _size: int) -> bytes:
            return b"\0"

        def close(self) -> None:
            return None

    output_dir = tmp_path / "worker"
    output_dir.mkdir()
    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
        download_assets=False,
    )
    payload = request.model_dump(mode="json")
    payload.update(
        _worker_output_dir=str(output_dir),
        _worker_asset_temp_root="",
        _worker_defer_output_upload=True,
    )
    monkeypatch.setattr(service, "run_capability_with_output", execute)
    sender = Sender()

    service._capability_worker_entry(sender, payload)

    body = captured["body"]
    assert isinstance(body, RoboCasaRunRequest)
    assert body.output_uri == "s3://example/output"
    assert captured["output_dir"] == output_dir
    assert captured["upload"] is False
    assert json.loads(sender.messages[0]) == {
        "kind": "result",
        "result": {"ok": True},
    }


def test_capability_worker_fails_closed_without_isolation_ack() -> None:
    from npa.workbench.robocasa import service

    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
        timeout_seconds=5,
        download_assets=False,
    )

    outcome = service._execute_capability_in_worker(
        request,
        worker_target=_robocasa_worker_without_isolation_ack,
        supervisor_target=_supervisor_without_containment_ack,
        process_context=multiprocessing.get_context("spawn"),
    )

    assert outcome.stopped is False
    assert outcome.result is None
    assert "containment was not acknowledged" in str(outcome.error)


def test_capability_worker_kills_term_ignoring_descendants() -> None:
    from npa.workbench.robocasa import service

    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
        timeout_seconds=15,
        download_assets=False,
    )

    outcome = service._execute_capability_in_worker(
        request,
        worker_target=_robocasa_worker_with_term_ignoring_descendant,
        process_context=multiprocessing.get_context("spawn"),
    )

    assert outcome.stopped is True
    assert outcome.error is None
    assert outcome.result is not None
    descendant_pid = int(outcome.result["descendant_pid"])
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)


def test_worker_supervisor_reaps_double_forked_session_daemon() -> None:
    from npa.workbench.robocasa import service

    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
        timeout_seconds=15,
        download_assets=False,
    )

    outcome = service._execute_capability_in_worker(
        request,
        worker_target=_robocasa_worker_with_double_fork_daemon,
        process_context=multiprocessing.get_context("spawn"),
    )

    assert outcome.stopped is True
    assert outcome.error is None
    assert outcome.result is not None
    descendant_pid = int(outcome.result["descendant_pid"])
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)


def test_worker_supervisor_preserves_unrelated_sibling_process() -> None:
    from npa.workbench.robocasa import service

    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        request = RoboCasaRunRequest(
            capability="kitchen_task_registration",
            output_uri="s3://example/output",
            timeout_seconds=15,
            download_assets=False,
        )
        outcome = service._execute_capability_in_worker(
            request,
            worker_target=_robocasa_worker_with_term_ignoring_descendant,
            process_context=multiprocessing.get_context("spawn"),
        )

        assert outcome.stopped is True
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_worker_supervisor_reaps_escaped_descendant_after_capability_exits() -> None:
    from npa.workbench.robocasa import service

    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
        timeout_seconds=15,
        download_assets=False,
    )

    outcome = service._execute_capability_in_worker(
        request,
        worker_target=_robocasa_worker_whose_leader_exits,
        process_context=multiprocessing.get_context("spawn"),
    )

    assert outcome.stopped is True
    assert outcome.error is None
    assert outcome.result is not None
    descendant_pid = int(outcome.result["descendant_pid"])
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)


def test_worker_supervisor_control_eof_still_reaps_descendants(
    tmp_path: Path,
) -> None:
    from npa.workbench.robocasa import service

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=True)
    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
        timeout_seconds=15,
        download_assets=False,
    )
    payload = request.model_dump(mode="json")
    worker_output = tmp_path / "worker"
    worker_output.mkdir()
    payload["_worker_output_dir"] = str(worker_output)
    payload["_worker_asset_temp_root"] = ""
    process = context.Process(
        target=service._supervisor_worker_entry,
        args=(sender, payload, _robocasa_worker_with_term_ignoring_descendant),
    )
    try:
        process.start()
        sender.close()
        ready = service._receive_worker_message(receiver)
        assert ready["kind"] == "ready"
        result = service._receive_worker_message(receiver)
        descendant_pid = int(result["result"]["descendant_pid"])
        receiver.close()
        process.join(timeout=15)
        assert process.exitcode == 0
        with pytest.raises(ProcessLookupError):
            os.kill(descendant_pid, 0)
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def test_pidfd_signal_rejects_reused_process_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workbench.robocasa import service

    identity = service._ProcessIdentity(
        pid=123,
        ppid=456,
        start_time=789,
        state="S",
    )
    descriptor, writer = os.pipe()
    signalled = []
    monkeypatch.setattr(service.os, "pidfd_open", lambda _pid: descriptor)
    monkeypatch.setattr(
        service,
        "_process_identity",
        lambda _pid: service._ProcessIdentity(
            pid=123,
            ppid=456,
            start_time=790,
            state="S",
        ),
    )
    monkeypatch.setattr(
        service.signal,
        "pidfd_send_signal",
        lambda *_args: signalled.append(True),
    )

    with pytest.raises(RoboCasaError, match="identity changed"):
        service._signal_process_identity(identity, signal.SIGKILL)

    os.close(writer)
    assert signalled == []


def test_worker_cleanup_exception_returns_fail_closed_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workbench.robocasa import service

    real_stop_worker = service._stop_worker

    def stop_then_raise(process, control, *, terminate, supervisor_pid=None):
        assert real_stop_worker(
            process,
            control,
            terminate=terminate,
            supervisor_pid=supervisor_pid,
        )
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(service, "_stop_worker", stop_then_raise)
    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
        timeout_seconds=15,
        download_assets=False,
    )

    outcome = service._execute_capability_in_worker(
        request,
        worker_target=_successful_robocasa_worker,
        process_context=multiprocessing.get_context("spawn"),
    )

    assert outcome.stopped is False
    assert outcome.result is None
    assert "injected cleanup failure" in str(outcome.error)


def test_worker_asset_cleanup_failure_returns_fail_closed_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workbench.robocasa import service

    assets = tmp_path / "assets"
    assets.mkdir()
    real_rmtree = service.shutil.rmtree

    def fail_asset_cleanup(path, *args, **kwargs):
        candidate = Path(path)
        if "workers" in candidate.parts:
            raise OSError("injected asset cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(service, "_assets_root", lambda: assets)
    monkeypatch.setattr(service.shutil, "rmtree", fail_asset_cleanup)
    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
        timeout_seconds=15,
        download_assets=False,
    )

    outcome = service._execute_capability_in_worker(
        request,
        worker_target=_successful_robocasa_worker,
        process_context=multiprocessing.get_context("spawn"),
    )

    assert outcome.stopped is False
    assert outcome.result is None
    assert "injected asset cleanup failure" in str(outcome.error)
    real_rmtree(assets)


def test_unstopped_worker_poisons_execution_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workbench.robocasa import service

    runs = RunRegistry()
    runs["run"] = _status("run", "queued")
    gate = service.GpuExecutionGate()
    monkeypatch.setattr(
        service,
        "_execute_capability_in_worker",
        lambda _body, **_kwargs: service._WorkerOutcome(
            error="cleanup failed", stopped=False
        ),
    )
    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
    )

    service._run_capability(request, "run", runs, gate)

    assert gate.available is False
    assert runs.get("run").status == "failed"
    assert runs.get("run").error == "cleanup failed"


def test_unstopped_worker_discards_successful_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workbench.robocasa import service

    runs = RunRegistry()
    runs["run"] = _status("run", "queued")
    gate = service.GpuExecutionGate()
    published: list[bool] = []
    monkeypatch.setattr(
        service, "upload_output", lambda *_args, **_kwargs: published.append(True)
    )
    monkeypatch.setattr(
        service,
        "_execute_capability_in_worker",
        lambda _body, **_kwargs: service._WorkerOutcome(
            result={"ok": True},
            stopped=False,
        ),
    )
    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/output",
    )

    service._run_capability(request, "run", runs, gate)

    status = runs.get("run")
    assert gate.available is False
    assert status.status == "failed"
    assert status.result is None
    assert published == []
    assert status.error == (
        "RoboCasa worker cleanup could not prove all descendants stopped"
    )


def test_service_publishes_retained_output_after_worker_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.workbench.robocasa import service

    output_dir = tmp_path / "retained"
    output_dir.mkdir()
    (output_dir / "result.json").write_text("{}\n", encoding="utf-8")
    runs = RunRegistry()
    runs["run"] = _status("run", "queued")
    published: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        service,
        "_execute_capability_in_worker",
        lambda _body, **_kwargs: service._WorkerOutcome(
            result={"ok": True},
            stopped=True,
            output_dir=output_dir,
        ),
    )
    monkeypatch.setattr(
        service,
        "upload_output",
        lambda path, uri, _result: published.append((path, uri)),
    )
    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/runs/exact",
    )

    service._run_capability(request, "run", runs, service.GpuExecutionGate())

    assert published == [(output_dir, "s3://example/runs/exact")]
    assert runs.get("run").status == "completed"
    assert not output_dir.exists()


def test_service_upload_failure_never_marks_run_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.workbench.robocasa import service

    output_dir = tmp_path / "retained"
    output_dir.mkdir()
    runs = RunRegistry()
    runs["run"] = _status("run", "queued")
    monkeypatch.setattr(
        service,
        "_execute_capability_in_worker",
        lambda _body, **_kwargs: service._WorkerOutcome(
            result={"ok": True},
            stopped=True,
            output_dir=output_dir,
        ),
    )

    def fail_upload(*_args, **_kwargs):
        raise OSError("injected upload failure")

    monkeypatch.setattr(service, "upload_output", fail_upload)
    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/runs/failure",
    )

    service._run_capability(request, "run", runs, service.GpuExecutionGate())

    status = runs.get("run")
    assert status.status == "failed"
    assert status.result is None
    assert "injected upload failure" in str(status.error)
    assert not output_dir.exists()


def test_service_retained_output_cleanup_failure_poisons_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workbench.robocasa import service

    output_dir = tmp_path / "retained"
    output_dir.mkdir()
    runs = RunRegistry()
    runs["run"] = _status("run", "queued")
    gate = service.GpuExecutionGate()
    monkeypatch.setattr(
        service,
        "_execute_capability_in_worker",
        lambda _body, **_kwargs: service._WorkerOutcome(
            result={"ok": True},
            stopped=True,
            output_dir=output_dir,
        ),
    )
    monkeypatch.setattr(service, "upload_output", lambda *_args, **_kwargs: None)
    real_rmtree = service.shutil.rmtree

    def fail_retained_cleanup(path, *args, **kwargs):
        if Path(path) == output_dir:
            raise OSError("injected retained cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(service.shutil, "rmtree", fail_retained_cleanup)
    request = RoboCasaRunRequest(
        capability="kitchen_task_registration",
        output_uri="s3://example/runs/failure",
    )

    service._run_capability(request, "run", runs, gate)

    status = runs.get("run")
    assert gate.available is False
    assert status.status == "failed"
    assert status.result is None
    assert "injected retained cleanup failure" in str(status.error)
    real_rmtree(output_dir)


class _FakeActionSpace:
    shape = (7,)

    def sample(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def seed(self, _seed: int) -> None:
        return None

    def contains(self, action) -> bool:
        return np.asarray(action).shape == self.shape


class _BoundedActionSpace(_FakeActionSpace):
    low = np.full(7, -1.0, dtype=np.float32)
    high = np.full(7, 1.0, dtype=np.float32)

    def contains(self, action) -> bool:
        values = np.asarray(action)
        return (
            values.shape == self.shape
            and bool(np.all(values >= self.low))
            and bool(np.all(values <= self.high))
        )


class _FakeEnv:
    action_space = _FakeActionSpace()

    def __init__(self) -> None:
        self._closed = False

    def reset(self, seed=None):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}

    @property
    def unwrapped(self):
        return self

    def _check_success(self) -> bool:
        return False

    def render(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self) -> None:
        self._closed = True

    @staticmethod
    def _obs() -> dict:
        return {
            "video.robot0_agentview_left": np.zeros((64, 64, 3), dtype=np.uint8),
            "video.robot0_eye_in_hand": np.zeros((64, 64, 3), dtype=np.uint8),
            "state.base_position": np.zeros(3, dtype=np.float32),
            "state.base_rotation": np.zeros(4, dtype=np.float32),
            "state.end_effector_position_relative": np.zeros(3, dtype=np.float32),
            "state.end_effector_rotation_relative": np.zeros(4, dtype=np.float32),
            "state.gripper_qpos": np.zeros(2, dtype=np.float32),
        }


def _install_fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake gymnasium whose make() returns a scripted RoboCasa env."""
    _install_fake_robocasa(monkeypatch)
    monkeypatch.setattr(capabilities, "_download_assets", lambda: None)

    class FakeGym:
        envs = types.SimpleNamespace(registry={})

        @staticmethod
        def make(env_id, **kwargs):
            return _FakeEnv()

    monkeypatch.setitem(sys.modules, "gymnasium", FakeGym())


def test_make_env_uses_nonempty_objaverse_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_robocasa(monkeypatch)
    observed: dict[str, object] = {}

    class FakeGym:
        @staticmethod
        def make(env_id, **kwargs):
            observed.update(kwargs)
            return _FakeEnv()

    monkeypatch.setitem(sys.modules, "gymnasium", FakeGym())
    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._download_assets", lambda: None
    )
    from npa.workbench.robocasa.capabilities import _make_env

    _make_env("robocasa/PickPlaceCounterToCabinet")

    assert observed == {"split": "all", "obj_registries": ("objaverse",)}


def test_assets_root_does_not_import_robocasa(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib.machinery import ModuleSpec

    imported = False

    def fail_import():
        nonlocal imported
        imported = True
        raise AssertionError("asset discovery must not import robocasa")

    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._import_robocasa", fail_import
    )
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: ModuleSpec(
            name, loader=None, origin="/opt/robocasa/source/robocasa/__init__.py"
        ),
    )
    from npa.workbench.robocasa.capabilities import _assets_root

    assert str(_assets_root()) == "/opt/robocasa/source/robocasa/models/assets"
    assert imported is False


def test_asset_catalog_uses_immutable_revisions() -> None:
    archives = capabilities._asset_archives()

    assert archives
    assert {(archive.repo_id, archive.revision) for archive in archives} == {
        (
            "robocasa/robocasa-assets",
            "1b92c3d02ca4354984fec961357db0bff7b32166",
        ),
        (
            "nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF",
            "420a04af939c34873e6839a586b70844baf28aab",
        ),
    }


def test_fixture_asset_publish_preserves_source_registry(tmp_path: Path) -> None:
    fixture_archive = next(
        archive
        for archive in capabilities._asset_archives()
        if archive.filename == "fixtures.zip"
    )
    assert fixture_archive.publish_path == "fixtures/accessories"
    assert fixture_archive.required_path == "fixtures/accessories"

    assets_root = tmp_path / "assets"
    fixture_registry = assets_root / "fixtures" / "fixture_registry"
    fixture_registry.mkdir(parents=True)
    wall = fixture_registry / "wall.yaml"
    cabinet = fixture_registry / "cabinet.yaml"
    wall.write_text("wall: pinned-source\n", encoding="utf-8")
    cabinet.write_text("cabinet: pinned-source\n", encoding="utf-8")
    stale_accessory = assets_root / "fixtures" / "accessories" / "stale.txt"
    stale_accessory.parent.mkdir(parents=True)
    stale_accessory.write_text("stale", encoding="utf-8")

    source_zip = tmp_path / "fixtures.zip"
    with ZipFile(source_zip, "w") as archive_zip:
        archive_zip.writestr("fixtures/accessories/stool/model.xml", "<mujoco/>")
    receipt = capabilities._asset_receipt_path(
        assets_root / ".npa_asset_fetch", fixture_archive
    )
    capabilities._stage_publish_and_receipt(
        fixture_archive, source_zip, assets_root, receipt
    )

    assert wall.read_text(encoding="utf-8") == "wall: pinned-source\n"
    assert cabinet.read_text(encoding="utf-8") == "cabinet: pinned-source\n"
    assert not stale_accessory.exists()
    assert (assets_root / "fixtures" / "accessories" / "stool" / "model.xml").read_text(
        encoding="utf-8"
    ) == "<mujoco/>"
    assert capabilities._asset_receipt_is_valid(receipt, fixture_archive, assets_root)


def test_asset_partial_fetch_retries_without_completion_receipt(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    state_root = assets_root / ".npa_asset_fetch"
    partial = assets_root / "textures"
    partial.mkdir(parents=True)
    (partial / "partial.txt").write_text("partial", encoding="utf-8")
    invalid_zip = tmp_path / "invalid.zip"
    invalid_zip.write_bytes(b"not-a-zip")
    valid_zip = tmp_path / "valid.zip"
    with ZipFile(valid_zip, "w") as archive_zip:
        archive_zip.writestr("textures/complete.txt", "complete")
    archive = capabilities._AssetArchive(
        capabilities.ROBOCASA_ASSET_REPOSITORY,
        capabilities.ROBOCASA_ASSET_REVISION,
        "textures.zip",
        ".",
        "textures",
        "textures",
    )
    downloads = iter((invalid_zip, valid_zip))

    def downloader(**kwargs):
        assert kwargs["revision"] == capabilities.ROBOCASA_ASSET_REVISION
        return str(next(downloads))

    with pytest.raises(RoboCasaError, match="not a valid zip"):
        capabilities._fetch_asset_archive(
            archive,
            assets_root=assets_root,
            state_root=state_root,
            downloader=downloader,
        )
    receipt = capabilities._asset_receipt_path(state_root, archive)
    assert not receipt.exists()
    assert (partial / "partial.txt").exists()

    capabilities._fetch_asset_archive(
        archive,
        assets_root=assets_root,
        state_root=state_root,
        downloader=downloader,
    )

    assert (partial / "complete.txt").read_text(encoding="utf-8") == "complete"
    assert not (partial / "partial.txt").exists()
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["revision"] == capabilities.ROBOCASA_ASSET_REVISION
    assert receipt_payload["file_count"] == 1


def test_asset_receipt_rejects_installed_tree_mutation_and_symlink(
    tmp_path: Path,
) -> None:
    assets_root = tmp_path / "assets"
    state_root = assets_root / ".npa_asset_fetch"
    source_zip = tmp_path / "textures.zip"
    with ZipFile(source_zip, "w") as archive_zip:
        archive_zip.writestr("textures/complete.txt", "complete")
    archive = capabilities._AssetArchive(
        capabilities.ROBOCASA_ASSET_REPOSITORY,
        capabilities.ROBOCASA_ASSET_REVISION,
        "textures.zip",
        ".",
        "textures",
        "textures",
    )
    receipt = capabilities._asset_receipt_path(state_root, archive)
    capabilities._stage_publish_and_receipt(archive, source_zip, assets_root, receipt)

    assert capabilities._asset_receipt_is_valid(receipt, archive, assets_root)
    installed = assets_root / "textures" / "complete.txt"
    installed.write_text("mutated", encoding="utf-8")
    assert not capabilities._asset_receipt_is_valid(receipt, archive, assets_root)

    installed.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("complete", encoding="utf-8")
    installed.symlink_to(outside)
    assert not capabilities._asset_receipt_is_valid(receipt, archive, assets_root)


@pytest.mark.parametrize("payload", ["[]", "null", "1", '"receipt"'])
def test_non_object_asset_receipt_is_invalid(tmp_path: Path, payload: str) -> None:
    assets_root = tmp_path / "assets"
    state_root = assets_root / ".npa_asset_fetch"
    archive = capabilities._AssetArchive(
        "example/assets",
        "a" * 40,
        "assets.zip",
        ".",
        "assets",
        "assets",
    )
    receipt = capabilities._asset_receipt_path(state_root, archive)
    receipt.write_text(payload, encoding="utf-8")

    assert not capabilities._asset_receipt_is_valid(
        receipt,
        archive,
        assets_root,
    )


def test_recursive_asset_receipt_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets_root = tmp_path / "assets"
    state_root = assets_root / ".npa_asset_fetch"
    archive = capabilities._AssetArchive(
        "example/assets",
        "a" * 40,
        "assets.zip",
        ".",
        "assets",
        "assets",
    )
    receipt = capabilities._asset_receipt_path(state_root, archive)
    receipt.write_text("{}\n", encoding="utf-8")

    def reject_recursive_json(_payload: str):
        raise RecursionError("injected recursive JSON")

    monkeypatch.setattr(capabilities.json, "loads", reject_recursive_json)

    assert not capabilities._asset_receipt_is_valid(
        receipt,
        archive,
        assets_root,
    )


def test_parent_asset_receipt_ignores_separately_receipted_nested_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = capabilities._AssetArchive(
        "example/parent",
        "a" * 40,
        "fixtures.zip",
        ".",
        "fixtures",
        "fixtures/accessories",
    )
    child = capabilities._AssetArchive(
        "example/child",
        "b" * 40,
        "nested.zip",
        "fixtures",
        "fixtures/vendor",
        "fixtures/vendor",
    )
    monkeypatch.setattr(capabilities, "_asset_archives", lambda: (parent, child))
    source_zip = tmp_path / "fixtures.zip"
    with ZipFile(source_zip, "w") as archive_zip:
        archive_zip.writestr("fixtures/accessories/core.txt", "core")
    child_zip = tmp_path / "nested.zip"
    with ZipFile(child_zip, "w") as archive_zip:
        archive_zip.writestr("vendor/nested.txt", "nested")
    assets_root = tmp_path / "assets"
    state_root = assets_root / ".npa_asset_fetch"
    parent_receipt = capabilities._asset_receipt_path(state_root, parent)
    child_receipt = capabilities._asset_receipt_path(state_root, child)
    capabilities._stage_publish_and_receipt(
        parent, source_zip, assets_root, parent_receipt
    )
    capabilities._stage_publish_and_receipt(
        child, child_zip, assets_root, child_receipt
    )

    assert capabilities._asset_receipt_is_valid(parent_receipt, parent, assets_root)
    assert capabilities._asset_receipt_is_valid(child_receipt, child, assets_root)
    (assets_root / "fixtures" / "accessories" / "core.txt").write_text(
        "mutated", encoding="utf-8"
    )
    assert not capabilities._asset_receipt_is_valid(parent_receipt, parent, assets_root)

    downloads: list[str] = []

    def downloader(**kwargs):
        downloads.append(str(kwargs["filename"]))
        return str(source_zip if kwargs["filename"] == parent.filename else child_zip)

    capabilities._fetch_asset_archive(
        parent,
        assets_root=assets_root,
        state_root=state_root,
        downloader=downloader,
    )
    assert not (assets_root / "fixtures" / "vendor").exists()
    capabilities._fetch_asset_archive(
        child,
        assets_root=assets_root,
        state_root=state_root,
        downloader=downloader,
    )

    assert downloads == ["fixtures.zip", "nested.zip"]
    assert capabilities._asset_receipt_is_valid(parent_receipt, parent, assets_root)
    assert capabilities._asset_receipt_is_valid(child_receipt, child, assets_root)


def test_asset_zip_rejects_traversal_symlinks_and_declared_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()

    traversal = tmp_path / "traversal.zip"
    with ZipFile(traversal, "w") as archive:
        archive.writestr("../escape.txt", "escape")
    with pytest.raises(RoboCasaError, match="unsafe path"):
        capabilities._extract_validated_zip(traversal, destination)
    assert not (tmp_path / "escape.txt").exists()

    symlink = tmp_path / "symlink.zip"
    with ZipFile(symlink, "w") as archive:
        member = ZipInfo("unsafe-link")
        member.create_system = 3
        member.external_attr = (0o120777 << 16) | 0xA000
        archive.writestr(member, "target")
    with pytest.raises(RoboCasaError, match="unsafe symlink"):
        capabilities._extract_validated_zip(symlink, destination)

    oversized = tmp_path / "oversized.zip"
    with ZipFile(oversized, "w") as archive:
        archive.writestr("one.bin", b"1234")
    monkeypatch.setattr(capabilities, "_ASSET_ARCHIVE_MEMBER_SIZE_LIMIT", 3)
    with pytest.raises(RoboCasaError, match="member exceeds the size limit"):
        capabilities._extract_validated_zip(oversized, destination)


@pytest.mark.parametrize("compression", [ZIP_LZMA, ZIP_BZIP2])
def test_asset_zip_rejects_unbounded_decoder_methods(
    tmp_path: Path, compression: int
) -> None:
    source_zip = tmp_path / "archive.zip"
    with ZipFile(source_zip, "w", compression=compression) as archive:
        archive.writestr("file.bin", b"payload")
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(RoboCasaError, match="unsupported compression"):
        capabilities._extract_validated_zip(source_zip, destination)

    assert list(destination.iterdir()) == []


def test_asset_zip_bounds_central_directory_before_member_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_zip = tmp_path / "many.zip"
    with ZipFile(source_zip, "w") as archive:
        archive.writestr("one.txt", "one")
        archive.writestr("two.txt", "two")
    destination = tmp_path / "destination"
    destination.mkdir()

    monkeypatch.setattr(capabilities, "_ASSET_ARCHIVE_MEMBER_LIMIT", 1)
    with pytest.raises(RoboCasaError, match="exceeds the member limit"):
        capabilities._extract_validated_zip(source_zip, destination)
    assert list(destination.iterdir()) == []

    monkeypatch.setattr(capabilities, "_ASSET_ARCHIVE_MEMBER_LIMIT", 100_000)
    monkeypatch.setattr(capabilities, "_ASSET_ARCHIVE_CENTRAL_DIRECTORY_LIMIT", 1)
    with pytest.raises(RoboCasaError, match="central-directory size limit"):
        capabilities._extract_validated_zip(source_zip, destination)
    assert list(destination.iterdir()) == []


def _write_zip64_member_count_archive(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        for index in range(zipfile.ZIP_FILECOUNT_LIMIT + 1):
            archive.writestr(f"{index:05x}", b"")


def test_asset_zip_accepts_zip64_central_directory(tmp_path: Path) -> None:
    source_zip = tmp_path / "zip64.zip"
    _write_zip64_member_count_archive(source_zip)

    with ZipFile(source_zip) as archive:
        assert len(archive.infolist()) == zipfile.ZIP_FILECOUNT_LIMIT + 1
    with source_zip.open("rb") as archive_file:
        capabilities._preflight_asset_zip(archive_file, source_zip)
        assert archive_file.tell() == 0


def test_asset_zip_rejects_malformed_zip64_central_directory(tmp_path: Path) -> None:
    source_zip = tmp_path / "malformed-zip64.zip"
    _write_zip64_member_count_archive(source_zip)
    with source_zip.open("r+b") as archive_file:
        end_record = zipfile._EndRecData(archive_file)
        assert end_record is not None
        start = int(end_record[zipfile._ECD_LOCATION]) - int(
            end_record[zipfile._ECD_SIZE]
        )
        archive_file.seek(start)
        archive_file.write(b"FAIL")

    with source_zip.open("rb") as archive_file:
        with pytest.raises(BadZipFile, match="central-directory record signature"):
            capabilities._preflight_asset_zip(archive_file, source_zip)


def test_asset_zip_rejects_forged_eocd_member_count(
    tmp_path: Path,
) -> None:
    source_zip = tmp_path / "forged-count.zip"
    with ZipFile(source_zip, "w") as archive:
        archive.writestr("one.txt", "one")
        archive.writestr("two.txt", "two")
    payload = bytearray(source_zip.read_bytes())
    eocd = payload.rfind(b"PK\x05\x06")
    assert eocd >= 0
    payload[eocd + 8 : eocd + 12] = struct.pack("<HH", 1, 1)
    source_zip.write_bytes(payload)
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(RoboCasaError, match="not a valid zip"):
        capabilities._extract_validated_zip(source_zip, destination)

    assert list(destination.iterdir()) == []


def test_asset_zip_bounds_implicit_directories_and_prefix_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory_bomb = tmp_path / "directories.zip"
    with ZipFile(directory_bomb, "w") as archive:
        archive.writestr("one/two/three/file.txt", "payload")
    destination = tmp_path / "destination"
    destination.mkdir()
    monkeypatch.setattr(capabilities, "_ASSET_ARCHIVE_DIRECTORY_LIMIT", 2)

    with pytest.raises(RoboCasaError, match="directory limit"):
        capabilities._extract_validated_zip(directory_bomb, destination)
    assert list(destination.iterdir()) == []

    monkeypatch.setattr(capabilities, "_ASSET_ARCHIVE_DIRECTORY_LIMIT", 100_000)
    collision = tmp_path / "collision.zip"
    with ZipFile(collision, "w") as archive:
        archive.writestr("parent", "file")
        archive.writestr("parent/child.txt", "child")

    with pytest.raises(RoboCasaError, match="prefix collision"):
        capabilities._extract_validated_zip(collision, destination)
    assert list(destination.iterdir()) == []


def test_asset_archive_path_replacement_cannot_change_opened_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_zip = tmp_path / "source.zip"
    with ZipFile(source_zip, "w") as archive_zip:
        archive_zip.writestr("textures/complete.txt", "complete")
    expected_archive_sha = hashlib.sha256(source_zip.read_bytes()).hexdigest()
    replacement_source = tmp_path / "replacement.zip"
    with ZipFile(replacement_source, "w") as archive_zip:
        archive_zip.writestr("textures/replaced.txt", "replaced")
    archive = capabilities._AssetArchive(
        capabilities.ROBOCASA_ASSET_REPOSITORY,
        capabilities.ROBOCASA_ASSET_REVISION,
        "textures.zip",
        ".",
        "textures",
        "textures",
    )
    assets_root = tmp_path / "assets"
    state_root = assets_root / ".npa_asset_fetch"
    receipt = capabilities._asset_receipt_path(state_root, archive)
    original_preflight = capabilities._preflight_asset_zip
    swapped = False

    def preflight_then_swap(archive_file, display_path):
        nonlocal swapped
        original_preflight(archive_file, display_path)
        if not swapped:
            swapped = True
            source_zip.replace(tmp_path / "original.zip")
            replacement_source.replace(source_zip)

    monkeypatch.setattr(capabilities, "_preflight_asset_zip", preflight_then_swap)

    capabilities._stage_publish_and_receipt(archive, source_zip, assets_root, receipt)

    assert (assets_root / "textures" / "complete.txt").is_file()
    assert not (assets_root / "textures" / "replaced.txt").exists()
    assert (
        json.loads(receipt.read_text(encoding="utf-8"))["archive_sha256"]
        == expected_archive_sha
    )


def test_replace_asset_tree_repairs_symlinked_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    target = tmp_path / "assets"
    target.symlink_to(outside, target_is_directory=True)
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "safe.txt").write_text("safe", encoding="utf-8")

    capabilities._replace_asset_tree(staged, target)

    assert target.is_dir()
    assert not target.is_symlink()
    assert (target / "safe.txt").read_text(encoding="utf-8") == "safe"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_kitchen_trajectory_export(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _install_fake_env(monkeypatch)
    _install_fake_video_writer(monkeypatch)
    from npa.workbench.robocasa.capabilities import kitchen_trajectory_export

    result = kitchen_trajectory_export(
        env_id="robocasa/PickPlaceCounterToCabinet",
        iterations=3,
        num_envs=2,
        seed=1,
        output_dir=tmp_path,
    )
    assert result["trajectory_export_ok"] is True
    assert result["schema"] == "npa.robocasa.trajectory_export.v1"
    assert result["temporal_alignment"] == "observation_before_action"
    assert result["policy"] == "random_action_baseline"
    assert result["num_episodes"] == 2
    for ep in range(2):
        ep_dir = tmp_path / f"episode_{ep:04d}"
        assert (ep_dir / "obs_workspace.npy").exists()
        assert (ep_dir / "obs_wrist.npy").exists()
        assert (ep_dir / "state.npy").exists()
        assert (ep_dir / "actions.npy").exists()
        ws = np.load(ep_dir / "obs_workspace.npy")
        assert ws.shape == (3, 64, 64, 3)
        assert ws.dtype == np.uint8
        st = np.load(ep_dir / "state.npy")
        assert st.shape == (3, 16)
        assert (ep_dir / "rollout.mp4").read_bytes() == b"video:4"
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "metrics.json").exists()


def _install_fake_video_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_write_video(frames, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video:" + str(len(frames)).encode())
        return path

    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._write_video", fake_write_video
    )


class _TemporalActionSpace(_FakeActionSpace):
    def __init__(self, env) -> None:
        self.env = env

    def sample(self) -> np.ndarray:
        return np.full(7, self.env.state + 10, dtype=np.float32)


class _TemporalEnv(_FakeEnv):
    def __init__(self) -> None:
        super().__init__()
        self.state = 0
        self.action_space = _TemporalActionSpace(self)

    def reset(self, seed=None):
        self.state = 0
        return self._obs(), {}

    def step(self, action):
        assert float(np.asarray(action)[0]) == self.state + 10
        self.state += 1
        return self._obs(), 0.0, False, False, {}

    def _obs(self) -> dict:
        observation = super()._obs()
        observation["video.robot0_agentview_left"].fill(self.state)
        observation["video.robot0_eye_in_hand"].fill(self.state)
        observation["state.base_position"] = np.full(3, self.state, dtype=np.float32)
        return observation


def test_trajectory_rows_store_observation_before_same_index_action() -> None:
    from npa.workbench.robocasa.capabilities import _collect_trajectory_episode

    arrays, episode, video_frames = _collect_trajectory_episode(
        _TemporalEnv(), iterations=3, seed=7
    )

    assert [float(state[0]) for state in arrays["state"]] == [0.0, 1.0, 2.0]
    assert [float(action[0]) for action in arrays["actions"]] == [10.0, 11.0, 12.0]
    assert [int(frame[0, 0, 0]) for frame in video_frames] == [0, 1, 2, 3]
    assert episode["seed"] == 7
    assert episode["length"] == 3
    assert episode["video_frames"] == 4
    assert episode["success"] is False


def test_native_task_success_rejects_disagreeing_signals() -> None:
    from npa.workbench.robocasa.capabilities import _native_task_success

    with pytest.raises(RoboCasaError, match="signals disagree"):
        _native_task_success(_FakeEnv(), {"success": True}, 0.0)


def test_native_task_success_does_not_round_dense_reward_to_success() -> None:
    from npa.workbench.robocasa.capabilities import _native_task_success

    success, sources = _native_task_success(_FakeEnv(), {}, 0.999999)

    assert success is False
    assert sources == ["environment._check_success"]


def test_policy_action_rejects_non_finite_values() -> None:
    from npa.workbench.robocasa.capabilities import _validated_action

    action = np.zeros(7, dtype=np.float32)
    action[0] = np.nan
    with pytest.raises(RoboCasaError, match="action contains non-finite"):
        _validated_action(action, _FakeActionSpace())


def test_policy_action_rejects_action_space_mismatch() -> None:
    from npa.workbench.robocasa.capabilities import _validated_action

    with pytest.raises(RoboCasaError, match="outside the RoboCasa action space"):
        _validated_action(np.zeros(6, dtype=np.float32), _FakeActionSpace())


def test_eval_clips_policy_action_and_records_raw_delta() -> None:
    from npa.workbench.robocasa.capabilities import _rollout_eval_episode

    env = _FakeEnv()
    env.action_space = _BoundedActionSpace()
    applied_actions: list[np.ndarray] = []

    def step(action):
        applied_actions.append(np.asarray(action))
        return env._obs(), 0.0, False, False, {}

    env.step = step
    raw_action = np.array([1.5, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _, result = _rollout_eval_episode(
        env,
        iterations=1,
        selector=lambda _env, _observation: raw_action,
        observation=env._obs(),
    )

    np.testing.assert_array_equal(applied_actions[0][:2], np.array([1.0, -1.0]))
    assert result["action_clipping_applied"] is True
    assert result["action_out_of_bounds_steps"] == 1
    assert result["max_action_bound_violation"] == pytest.approx(1.0)
    assert result["raw_action_sha256"] != result["action_sha256"]


def test_episode_outcome_rejects_non_finite_reward() -> None:
    from npa.workbench.robocasa.capabilities import (
        _empty_episode_outcome,
        _update_episode_outcome,
    )

    with pytest.raises(RoboCasaError, match="reward contains a non-finite"):
        _update_episode_outcome(
            _empty_episode_outcome(), _FakeEnv(), np.nan, False, False, {}
        )


def test_eval_rejects_non_finite_initial_state() -> None:
    from npa.workbench.robocasa.capabilities import _rollout_eval_episode

    observation = _FakeEnv()._obs()
    observation["state.base_position"][0] = np.inf
    with pytest.raises(RoboCasaError, match="robot state contains non-finite"):
        _rollout_eval_episode(
            _FakeEnv(),
            iterations=1,
            selector=lambda env, _observation: env.action_space.sample(),
            observation=observation,
        )


def test_robot_state_requires_the_pinned_panda_omron_layout() -> None:
    from npa.workbench.robocasa.capabilities import _obs_state

    observation = _FakeEnv()._obs()
    observation.pop("state.gripper_qpos")

    with pytest.raises(RoboCasaError, match="missing robot state keys"):
        _obs_state(observation)


@pytest.mark.parametrize(
    ("key", "wrong_width"),
    [
        ("state.base_position", 2),
        ("state.base_rotation", 3),
        ("state.end_effector_position_relative", 4),
        ("state.end_effector_rotation_relative", 3),
        ("state.gripper_qpos", 1),
    ],
)
def test_robot_state_rejects_wrong_panda_omron_component_width(
    key: str, wrong_width: int
) -> None:
    from npa.workbench.robocasa.capabilities import _obs_state

    observation = _FakeEnv()._obs()
    observation[key] = np.zeros(wrong_width, dtype=np.float32)

    with pytest.raises(RoboCasaError, match=rf"{key!r} has width"):
        _obs_state(observation)


def test_matched_eval_rejects_different_initial_workspace_frames() -> None:
    from npa.workbench.robocasa.capabilities import _require_matched_initial_state

    with pytest.raises(RoboCasaError, match="initial workspace frames do not match"):
        _require_matched_initial_state(
            {
                "initial_workspace_sha256": "policy",
                "initial_state_sha256": "same",
            },
            {
                "initial_workspace_sha256": "baseline",
                "initial_state_sha256": "same",
            },
        )


def test_matched_eval_rejects_different_initial_robot_states() -> None:
    from npa.workbench.robocasa.capabilities import _require_matched_initial_state

    with pytest.raises(RoboCasaError, match="initial robot states do not match"):
        _require_matched_initial_state(
            {
                "initial_workspace_sha256": "same",
                "initial_state_sha256": "policy",
            },
            {
                "initial_workspace_sha256": "same",
                "initial_state_sha256": "baseline",
            },
        )


def test_act_selector_maps_pixels_state_and_distinct_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workbench.robocasa.capabilities import _act_action_selector

    class FakeTensor:
        def __init__(self, value) -> None:
            self.value = np.asarray(value)

        @property
        def shape(self):
            return self.value.shape

        def permute(self, *axes):
            return FakeTensor(self.value.transpose(axes))

        def float(self):
            return FakeTensor(self.value.astype(np.float32))

        def div(self, value):
            return FakeTensor(self.value / value)

        def unsqueeze(self, axis):
            return FakeTensor(np.expand_dims(self.value, axis))

        def to(self, _device):
            return self

        def squeeze(self, axis):
            return FakeTensor(np.squeeze(self.value, axis=axis))

        def detach(self):
            return self

        def cpu(self):
            return self

        def __array__(self, dtype=None):
            return np.asarray(self.value, dtype=dtype)

    class FakeTorch:
        @staticmethod
        def from_numpy(value):
            return FakeTensor(value)

        @staticmethod
        def inference_mode():
            class Context:
                def __enter__(self):
                    return None

                def __exit__(self, *_args):
                    return False

            return Context()

    class FakePolicy:
        observation = None

        def select_action(self, observation):
            self.observation = observation
            return FakeTensor(np.full((1, 7), 0.25, dtype=np.float32))

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    policy = FakePolicy()
    selector = _act_action_selector(
        (policy, "cpu", lambda value: value, lambda value: value, FakeTorch)
    )

    observation = _FakeEnv()._obs()
    observation["video.robot0_agentview_left"].fill(255)
    observation["video.robot0_eye_in_hand"].fill(255)
    for index, key in enumerate(
        (
            "state.base_position",
            "state.base_rotation",
            "state.end_effector_position_relative",
            "state.end_effector_rotation_relative",
            "state.gripper_qpos",
        ),
        start=1,
    ):
        observation[key].fill(index)
    action = selector(_FakeEnv(), observation)

    assert set(policy.observation) == {
        "observation.images.workspace",
        "observation.images.wrist",
        "observation.state",
    }
    assert policy.observation["observation.images.workspace"].shape == (1, 3, 64, 64)
    assert policy.observation["observation.images.wrist"].shape == (1, 3, 64, 64)
    assert policy.observation["observation.state"].shape == (1, 16)
    assert np.all(policy.observation["observation.images.workspace"].value == 1.0)
    assert policy.observation["observation.state"].value.tolist() == [
        [1.0] * 3 + [2.0] * 4 + [3.0] * 3 + [4.0] * 4 + [5.0] * 2
    ]
    assert np.asarray(action).tolist() == pytest.approx([0.25] * 7)


def test_required_video_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.workbench.robocasa.capabilities import _write_required_video

    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._write_video",
        lambda _frames, _path: None,
    )
    with pytest.raises(RoboCasaError, match="video was not written"):
        _write_required_video(
            [np.zeros((4, 4, 3), dtype=np.uint8)], tmp_path / "missing.mp4"
        )


def test_random_rollout_fails_when_mp4_is_empty_or_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(capabilities, "_make_env", lambda *_args, **_kwargs: _FakeEnv())

    def write_empty_video(_frames, path):
        path.touch()
        return path

    monkeypatch.setattr(capabilities, "_write_video", write_empty_video)

    with pytest.raises(RoboCasaError, match="video was not written"):
        capabilities.kitchen_random_rollout(
            iterations=1,
            output_dir=tmp_path,
            download_assets=False,
        )


def test_rollout_output_has_machine_readable_execution_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_env(monkeypatch)
    source_sha = "b" * 40
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", source_sha)
    monkeypatch.setenv("ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA", "1")

    def fake_write_video(frames, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated-rollout")
        return path

    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._write_video", fake_write_video
    )
    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities.upload_output", lambda *args: None
    )
    from npa.workbench.robocasa.capabilities import run_capability_with_output

    result = run_capability_with_output(
        RoboCasaRunRequest(
            capability="kitchen_random_rollout",
            output_uri="s3://example/output",
            iterations=1,
        ),
        output_dir=tmp_path,
    )
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert result["execution_provenance"] == provenance
    assert provenance["schema"] == "npa.robocasa.execution_provenance.v2"
    assert provenance["generator"] == "robocasa"
    assert provenance["simulator"] == "mujoco"
    assert provenance["source_identity"] == "container_image"
    assert provenance["image_source_sha"] == source_sha
    assert provenance["stock_or_copied_fixture"] is False
    assert provenance["recording_formats"] == {
        "mp4": True,
        "rrd": False,
        "mcap": False,
    }
    assert provenance["mp4_artifacts"][0]["path"] == "rollout.mp4"
    assert len(provenance["mp4_artifacts"][0]["sha256"]) == 64


def test_kitchen_trajectory_export_records_panda_omron_multitask_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _install_fake_env(monkeypatch)
    _install_fake_video_writer(monkeypatch)
    from npa.workbench.robocasa.capabilities import kitchen_trajectory_export

    result = kitchen_trajectory_export(
        env_id="robocasa/TrainA,robocasa/TrainB",
        iterations=1,
        num_envs=4,
        output_dir=tmp_path,
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert result["embodiment"] == "PandaOmron"
    assert metadata["robot_type"] == "panda_omron"
    assert metadata["state_dim"] == 16
    assert len(metadata["state_keys"]) == 5
    assert metadata["task_env_ids"] == ["robocasa/TrainA", "robocasa/TrainB"]
    assert [episode["env_id"] for episode in metadata["episodes"]] == [
        "robocasa/TrainA",
        "robocasa/TrainB",
        "robocasa/TrainA",
        "robocasa/TrainB",
    ]
    assert metadata["temporal_alignment"] == "observation_before_action"


def test_kitchen_policy_eval_rejects_overlapping_tasks_before_loading_checkpoint(
    tmp_path,
) -> None:
    from npa.workbench.robocasa.capabilities import RoboCasaError, kitchen_policy_eval

    with pytest.raises(RoboCasaError, match="overlap"):
        kitchen_policy_eval(
            checkpoint_uri="s3://example/checkpoint/",
            train_env_ids="robocasa/TaskA,robocasa/TaskB",
            heldout_env_ids="robocasa/TaskB,robocasa/TaskC",
            iterations=1,
            num_envs=1,
            seed=0,
            output_dir=tmp_path,
        )


def test_checkpoint_identity_hashes_exact_pretrained_model_separately(tmp_path) -> None:
    from npa.workbench.robocasa.capabilities import _checkpoint_identity

    checkpoint = tmp_path / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"type":"act"}')
    (checkpoint / "model.safetensors").write_bytes(b"real-act-weights")
    (tmp_path / "config.json").write_text('{"type":"act"}')
    (tmp_path / "model.safetensors").write_bytes(b"flat-upload-copy")
    (tmp_path / "training.log").write_text("first log")

    resolved, checkpoint_sha, first_tree_sha = _checkpoint_identity(tmp_path)
    (tmp_path / "training.log").write_text("changed unrelated log")
    _, checkpoint_sha_after, second_tree_sha = _checkpoint_identity(tmp_path)

    assert resolved == checkpoint
    assert checkpoint_sha == checkpoint_sha_after
    assert first_tree_sha != second_tree_sha


def test_checkpoint_tree_hash_length_frames_path_and_content(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a").write_bytes(b"bc")
    (second / "ab").write_bytes(b"c")

    old_first = b"a" + b"bc"
    old_second = b"ab" + b"c"
    assert old_first == old_second
    assert capabilities._sha256_tree(first) != capabilities._sha256_tree(second)


def test_checkpoint_identity_rejects_ambiguous_numbered_checkpoints(tmp_path) -> None:
    from npa.workbench.robocasa.capabilities import _checkpoint_identity

    for step in ("000100", "000200"):
        checkpoint = tmp_path / "checkpoints" / step / "pretrained_model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text('{"type":"act"}')
        (checkpoint / "model.safetensors").write_bytes(step.encode())

    with pytest.raises(RoboCasaError, match="multiple loadable pretrained_model"):
        _checkpoint_identity(tmp_path)


def test_training_dataset_provenance_binds_content_and_exact_tasks(
    tmp_path: Path,
) -> None:
    from npa.workbench.lerobot.policy_container import (
        build_training_dataset_provenance,
        write_training_dataset_provenance,
    )

    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data").mkdir()
    task_rows = [
        {"task_index": 0, "task": "TrainA"},
        {"task_index": 1, "task": "TrainB"},
    ]
    (dataset / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in task_rows),
        encoding="utf-8",
    )
    data_path = dataset / "data" / "episode.bin"
    data_path.write_bytes(b"first-dataset")
    declared = "robocasa/TrainA,robocasa/TrainB"

    provenance = build_training_dataset_provenance(
        dataset,
        dataset_source="s3://example/dataset/",
        declared_training_env_ids=declared,
    )
    run_root = tmp_path / "training-run"
    write_training_dataset_provenance(run_root, provenance)
    verified = capabilities._verify_training_provenance(run_root, declared.split(","))

    assert verified["declared_training_tasks_verified"] is True
    assert verified["normalized_dataset_training_env_ids"] == declared.split(",")
    first_digest = provenance["dataset_tree_sha256"]
    data_path.write_bytes(b"changed-dataset")
    changed = build_training_dataset_provenance(
        dataset,
        dataset_source="s3://example/dataset/",
        declared_training_env_ids=declared,
    )
    assert changed["dataset_tree_sha256"] != first_digest
    with pytest.raises(RoboCasaError, match="do not exactly match"):
        capabilities._verify_training_provenance(
            run_root, ["robocasa/TrainB", "robocasa/TrainA"]
        )
    from npa.workbench.lerobot.policy_container import PolicyContainerError

    with pytest.raises(PolicyContainerError, match="undeclared RoboCasa tasks"):
        build_training_dataset_provenance(
            dataset,
            dataset_source="s3://example/dataset/",
            declared_training_env_ids="robocasa/TrainA",
        )


def test_kitchen_policy_eval_compares_matched_seed_random_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.workbench.robocasa.capabilities import kitchen_policy_eval

    class SuccessfulEnv(_TemporalEnv):
        def step(self, action):
            self.state += 1
            success = self.state >= 2
            return self._obs(), float(success), success, False, {}

        def _check_success(self) -> bool:
            return self.state >= 2

    class FakePolicy:
        def reset(self) -> None:
            return None

    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._make_env",
        lambda *_args, **_kwargs: SuccessfulEnv(),
    )
    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._download_s3_tree",
        lambda _uri, destination: destination,
    )
    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._checkpoint_identity",
        lambda root: (root, "a" * 64, "b" * 64),
    )
    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._verify_training_provenance",
        lambda _root, _ids: {
            "dataset_tree_sha256": "c" * 64,
            "artifact_sha256": "d" * 64,
            "artifact_path": "training_dataset_provenance.json",
        },
    )
    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._load_act_policy",
        lambda _path: (FakePolicy(),),
    )
    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._act_action_selector",
        lambda _runtime: lambda _env, _observation: np.full(7, 0.25, dtype=np.float32),
    )
    _install_fake_video_writer(monkeypatch)

    result = kitchen_policy_eval(
        checkpoint_uri="s3://example/checkpoint/",
        train_env_ids="robocasa/TrainA",
        heldout_env_ids="robocasa/HeldoutA",
        iterations=3,
        num_envs=2,
        seed=100,
        output_dir=tmp_path,
        download_assets=False,
    )

    assert result["success_rate"] == 1.0
    assert result["checkpoint_selection"] == "."
    assert result["baseline_success_rate"] == 1.0
    assert result["success_rate_delta"] == 0.0
    assert result["paired_outcomes"] == {
        "policy_wins": 0,
        "baseline_wins": 0,
        "ties": 2,
    }
    assert [pair["seed"] for pair in result["paired_episodes"]] == [100, 101]
    assert all(
        pair["policy"]["initial_workspace_sha256"]
        == pair["random_baseline"]["initial_workspace_sha256"]
        for pair in result["paired_episodes"]
    )
    assert all(
        pair["policy"]["initial_state_sha256"]
        == pair["random_baseline"]["initial_state_sha256"]
        for pair in result["paired_episodes"]
    )
    assert all(
        pair["policy"]["action_sha256"] != pair["random_baseline"]["action_sha256"]
        for pair in result["paired_episodes"]
    )
    assert all(
        not episode["action_clipping_applied"]
        and episode["raw_action_sha256"] == episode["action_sha256"]
        for pair in result["paired_episodes"]
        for episode in (pair["policy"], pair["random_baseline"])
    )
    assert result["split_proof"]["configured_task_sets_disjoint"] is True
    assert result["split_proof"]["checkpoint_training_tasks_verified"] is True
    assert result["split_proof"]["training_dataset_tree_sha256"] == "c" * 64
    assert all(
        episode["success_sources"] == ["binary_reward", "environment._check_success"]
        for episode in result["episodes"]
    )
    assert len(list(tmp_path.glob("episode_*/*.mp4"))) == 4
    assert (tmp_path / "eval_manifest.json").is_file()
    assert (
        result["split_proof"]["heldout_episode_manifest_sha256"]
        == hashlib.sha256((tmp_path / "eval_manifest.json").read_bytes()).hexdigest()
    )


def test_kitchen_trajectory_export_missing_image_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _install_fake_robocasa(monkeypatch)
    from npa.workbench.robocasa.capabilities import (
        RoboCasaError,
        kitchen_trajectory_export,
    )

    with pytest.raises(RoboCasaError):
        kitchen_trajectory_export(
            env_id="robocasa/PickPlaceCounterToCabinet",
            iterations=1,
            num_envs=1,
            output_dir=tmp_path,
        )


class _S3PreconditionFailed(Exception):
    def __init__(self) -> None:
        self.response = {
            "Error": {"Code": "PreconditionFailed"},
            "ResponseMetadata": {"HTTPStatusCode": 412},
        }
        super().__init__("precondition failed")


class _S3NotFound(Exception):
    def __init__(self) -> None:
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }
        super().__init__("not found")


class _TransactionalFakeS3:
    def __init__(
        self,
        *,
        objects: dict[str, tuple[bytes, dict[str, str]]] | None = None,
        fail_copy_number: int | None = None,
    ) -> None:
        self.objects = dict(objects or {})
        self.fail_copy_number = fail_copy_number
        self.copy_count = 0
        self.events: list[tuple[str, str]] = []

    def list_objects_v2(self, *, Prefix, MaxKeys, **_kwargs):
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        selected = keys[:MaxKeys]
        return {
            "KeyCount": len(selected),
            "Contents": [{"Key": key} for key in selected],
            "IsTruncated": len(keys) > MaxKeys,
        }

    def head_object(self, *, Key, **_kwargs):
        try:
            body, metadata = self.objects[Key]
        except KeyError as exc:
            raise _S3NotFound() from exc
        return {
            "ContentLength": len(body),
            "Metadata": dict(metadata),
        }

    def get_object(self, *, Key, **_kwargs):
        try:
            body, _metadata = self.objects[Key]
        except KeyError as exc:
            raise _S3NotFound() from exc
        return {"Body": io.BytesIO(body)}

    def upload_file(self, local_path, _bucket, key, *, ExtraArgs):
        self.objects[key] = (
            Path(local_path).read_bytes(),
            dict(ExtraArgs["Metadata"]),
        )
        self.events.append(("stage", key))

    def copy_object(self, *, Key, CopySource, Metadata, **_kwargs):
        self.copy_count += 1
        if self.copy_count == self.fail_copy_number:
            raise OSError("injected copy failure")
        body, _source_metadata = self.objects[CopySource["Key"]]
        self.objects[Key] = (body, dict(Metadata))
        self.events.append(("copy", Key))

    def put_object(self, *, Key, Body, Metadata, IfNoneMatch, **_kwargs):
        assert IfNoneMatch == "*"
        if Key in self.objects:
            raise _S3PreconditionFailed()
        self.objects[Key] = (bytes(Body), dict(Metadata))
        kind = "commit" if Key.endswith("_NPA_COMPLETE.json") else "claim"
        self.events.append((kind, Key))

    def delete_objects(self, *, Delete, **_kwargs):
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)
        self.events.append(("delete", "staging"))
        return {}


def test_upload_output_publishes_commit_marker_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "b.bin").write_bytes(b"bb")
    s3 = _TransactionalFakeS3()

    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)
    result = {"ok": True}

    capabilities.upload_output(tmp_path, "s3://bucket/runs/exact", result)

    assert result["output_uri"] == "s3://bucket/runs/exact"
    assert s3.events[-2] == ("commit", "runs/exact/_NPA_COMPLETE.json")
    assert s3.events[-1] == ("delete", "staging")
    assert all(
        event[0] != "commit"
        for event in s3.events[
            : next(i for i, event in enumerate(s3.events) if event[0] == "commit")
        ]
    )
    marker = json.loads(s3.objects["runs/exact/_NPA_COMPLETE.json"][0])
    assert marker["schema"] == "npa.robocasa.output-commit.v1"
    assert [item["path"] for item in marker["files"]] == ["a.bin", "b.bin"]


def test_upload_output_copy_failure_never_publishes_commit_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "b.bin").write_bytes(b"b")
    s3 = _TransactionalFakeS3(fail_copy_number=2)
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)

    with pytest.raises(OSError, match="injected copy failure"):
        capabilities.upload_output(tmp_path, "s3://bucket/runs/failure", {"ok": True})

    assert not any(kind == "commit" for kind, _key in s3.events)
    assert ("delete", "staging") in s3.events


def test_upload_output_resumes_same_commit_after_partial_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "b.bin").write_bytes(b"b")
    s3 = _TransactionalFakeS3(fail_copy_number=2)
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)
    result = {"ok": True}

    with pytest.raises(OSError, match="injected copy failure"):
        capabilities.upload_output(tmp_path, "s3://bucket/runs/resume", result)
    assert "runs/resume/a.bin" in s3.objects
    assert "runs/resume/_NPA_COMPLETE.json" not in s3.objects

    s3.fail_copy_number = None
    capabilities.upload_output(tmp_path, "s3://bucket/runs/resume", result)

    assert "runs/resume/b.bin" in s3.objects
    assert "runs/resume/_NPA_COMPLETE.json" in s3.objects
    assert sum(kind == "claim" for kind, _key in s3.events) == 1
    assert sum(kind == "commit" for kind, _key in s3.events) == 1


def test_upload_output_rejects_different_commit_after_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "b.bin").write_bytes(b"b")
    s3 = _TransactionalFakeS3(fail_copy_number=2)
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)

    with pytest.raises(OSError, match="injected copy failure"):
        capabilities.upload_output(
            tmp_path,
            "s3://bucket/runs/claimed",
            {"attempt": 1},
        )
    (tmp_path / "a.bin").write_bytes(b"different")
    s3.fail_copy_number = None

    with pytest.raises(RoboCasaError, match="object identity mismatch"):
        capabilities.upload_output(
            tmp_path,
            "s3://bucket/runs/claimed",
            {"attempt": 2},
        )


@pytest.mark.parametrize("mutation", ["foreign-object", "missing-claim"])
def test_upload_output_rechecks_complete_prefix_ownership(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    (tmp_path / "artifact.bin").write_bytes(b"artifact")
    s3 = _TransactionalFakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)
    result = {"ok": True}
    uri = "s3://bucket/runs/complete"
    capabilities.upload_output(tmp_path, uri, result)

    if mutation == "foreign-object":
        s3.objects["runs/complete/foreign.bin"] = (b"foreign", {})
        message = "foreign objects"
    else:
        del s3.objects["runs/complete/_NPA_CLAIM.json"]
        message = "not an owned resumable prefix"

    with pytest.raises(RoboCasaError, match=message):
        capabilities.upload_output(tmp_path, uri, result)


def test_upload_output_rechecks_committed_object_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "artifact.bin").write_bytes(b"artifact")
    s3 = _TransactionalFakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)
    result = {"ok": True}
    uri = "s3://bucket/runs/corrupt"
    capabilities.upload_output(tmp_path, uri, result)
    body, metadata = s3.objects["runs/corrupt/artifact.bin"]
    s3.objects["runs/corrupt/artifact.bin"] = (b"x" * len(body), metadata)

    with pytest.raises(RoboCasaError, match="object bytes changed"):
        capabilities.upload_output(tmp_path, uri, result)


def test_upload_output_rejects_nonempty_destination_before_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "result.json").write_text("{}\n", encoding="utf-8")
    s3 = _TransactionalFakeS3(objects={"runs/existing/result.json": (b"{}\n", {})})
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)

    with pytest.raises(RoboCasaError, match="not an owned resumable prefix"):
        capabilities.upload_output(tmp_path, "s3://bucket/runs/existing", {"ok": True})

    assert not any(kind == "stage" for kind, _key in s3.events)


def test_upload_output_rejects_symlink_before_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"outside")
    (tmp_path / "linked.bin").symlink_to(outside)
    staged: list[bool] = []

    class FakeS3:
        def upload_file(self, *_args, **_kwargs):
            staged.append(True)

    monkeypatch.setattr("boto3.client", lambda *a, **k: FakeS3())

    with pytest.raises(RoboCasaError, match="symbolic link"):
        capabilities.upload_output(tmp_path, "s3://bucket/runs/symlink", {"ok": True})

    assert staged == []


def test_upload_output_rejects_symlink_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    (real_root / "artifact.bin").write_bytes(b"artifact")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr("boto3.client", lambda *a, **k: _TransactionalFakeS3())

    with pytest.raises(RoboCasaError, match="root must be a real directory"):
        capabilities.upload_output(
            linked_root,
            "s3://bucket/runs/root-symlink",
            {"ok": True},
        )


def test_upload_output_rejects_result_symlink_before_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("do-not-overwrite\n", encoding="utf-8")
    (output_root / "result.json").symlink_to(outside)
    monkeypatch.setattr("boto3.client", lambda *a, **k: _TransactionalFakeS3())

    with pytest.raises(RoboCasaError, match="symbolic link"):
        capabilities.upload_output(
            output_root,
            "s3://bucket/runs/result-symlink",
            {"ok": True},
        )

    assert outside.read_text(encoding="utf-8") == "do-not-overwrite\n"


def test_result_rewrite_rejects_hardlink_before_truncate(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("do-not-truncate\n", encoding="utf-8")
    os.link(outside, output_root / "result.json")

    with pytest.raises(RoboCasaError, match="identity is unsafe"):
        capabilities._rewrite_result_file(output_root, b"replacement\n")

    assert outside.read_text(encoding="utf-8") == "do-not-truncate\n"


def test_upload_output_rejects_reserved_completion_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "_NPA_COMPLETE.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("boto3.client", lambda *a, **k: _TransactionalFakeS3())

    with pytest.raises(RoboCasaError, match="reserved marker"):
        capabilities.upload_output(
            tmp_path,
            "s3://bucket/runs/reserved",
            {"ok": True},
        )


# --------------------------------------------------------------------------- SDK local run output persistence
#
# The SDK local `run()` must persist and upload output exactly like a service
# run. Regression coverage for the review finding that local non-service
# capability execution dropped output because `run_capability()` received no
# output directory (and `kitchen_policy_eval` failed outright).


def test_sdk_local_run_uploads_produced_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Local SDK run() persists produced artifacts and uploads them to S3."""
    _install_fake_env(monkeypatch)
    s3 = _TransactionalFakeS3()
    monkeypatch.setattr("boto3.client", lambda *a, **k: s3)

    # imageio/ffmpeg is not installed in the unit-test venv, so _write_video
    # returns None and writes nothing. Stub it to write a real artifact so the
    # test proves the produced output is uploaded.
    def fake_write_video(frames, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-video-bytes")
        return path

    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._write_video", fake_write_video
    )

    from npa.sdk.workbench.robocasa import run

    response = run(
        capability="kitchen_random_rollout",
        output_path="s3://bucket/out",
        iterations=2,
        seed=1,
    )
    assert response.status == "completed"
    assert response.run_id == "local"
    assert response.output_uri == "s3://bucket/out"
    # The rollout produced a video artifact that was uploaded to S3.
    staged = [key for kind, key in s3.events if kind == "stage"]
    copied = [key for kind, key in s3.events if kind == "copy"]
    committed = [key for kind, key in s3.events if kind == "commit"]
    assert staged, "expected at least one uploaded artifact"
    assert all(key.startswith(".npa-staging/robocasa/") for key in staged)
    assert copied
    assert all(key.startswith("out/") for key in copied)
    assert committed == ["out/_NPA_COMPLETE.json"]


def test_sdk_local_run_passes_output_dir_to_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local SDK run() always supplies an output directory to the capability.

    Regression for the review finding that the SDK local path called
    ``run_capability()`` with no output directory, which made capabilities that
    require one (``kitchen_policy_eval``) fail and silently dropped produced
    artifacts for the others.
    """
    _install_fake_env(monkeypatch)

    captured: dict[str, object] = {}

    def fake_run_capability(request, *, output_dir=None):
        captured["output_dir"] = output_dir
        return {"ok": True}

    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities.run_capability", fake_run_capability
    )
    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities.upload_output", lambda *a, **k: None
    )

    from npa.sdk.workbench.robocasa import run

    response = run(
        capability="kitchen_random_rollout",
        output_path="s3://bucket/out",
        iterations=1,
        num_envs=1,
    )
    assert response.status == "completed"
    assert captured["output_dir"] is not None
    assert isinstance(captured["output_dir"], Path)
    assert not Path(captured["output_dir"]).exists()


def test_sdk_rejects_invalid_output_before_capability_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities.run_capability_with_output", unexpected
    )
    from npa.sdk.workbench.robocasa import RoboCasaValidationError, run

    with pytest.raises(RoboCasaValidationError, match="expects an S3 URI"):
        run(
            capability="kitchen_random_rollout",
            output_path="file:///tmp/output",
        )
    assert called is False
