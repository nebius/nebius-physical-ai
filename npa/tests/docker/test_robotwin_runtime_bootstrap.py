"""Negative tests for the Phase A standard-library RoboTwin bootstrap."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

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


def _environment(
    *,
    capability_updates: dict[str, object] | None = None,
    environment_updates: dict[str, str] | None = None,
) -> dict[str, str]:
    image = "registry.example/private/npa-robotwin@sha256:" + "a" * 64
    bucket = "manager-bucket-canary"
    run_id = "robotwin-manager-canary"
    output_prefix = f"s3://{bucket}/robotwin-output/{run_id}/"
    binding = {
        "bootstrap_image": image,
        "bucket": bucket,
        "output_prefix": output_prefix,
        "run_id": run_id,
    }
    capability: dict[str, object] = {
        "schema_version": runtime.AUTH_SCHEMA,
        "capability_id": "robotwin-inner-" + "1" * 20,
        "runtime_manifest_sha256": runtime.RUNTIME_LOCK_SHA256,
        "workflow_sha256": runtime.WORKFLOW_SHA256,
        "source_revision": runtime.SOURCE_REVISION,
        "curobo_revision": runtime.CUROBO_REVISION,
        "asset_revision": runtime.ASSET_REVISION,
        "bootstrap_image_sha256": hashlib.sha256(image.encode()).hexdigest(),
        "runtime_binding_sha256": hashlib.sha256(
            json.dumps(binding, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "expires_at": "2099-01-01T00:00:00Z",
    }
    capability.update(capability_updates or {})
    environment = {
        runtime.AUTH_ENV: json.dumps(capability, separators=(",", ":"), sort_keys=True),
        "BYOF_IMAGE": image,
        "NPA_BYOF_RUN_ID": run_id,
        "NPA_S3_BUCKET": bucket,
        "S3_OUTPUT_PREFIX": output_prefix,
    }
    environment.update(environment_updates or {})
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
        ({"runtime_manifest_sha256": "0" * 64}, "runtime-manifest-sha256-mismatch"),
        ({"bootstrap_image_sha256": "0" * 64}, "bootstrap-image-binding-mismatch"),
        ({"runtime_binding_sha256": "0" * 64}, "runtime-binding-mismatch"),
        ({"capability_id": "not-a-capability"}, "capability-id-invalid"),
        ({"expires_at": "not-a-date"}, "capability-expiry-invalid"),
        ({"expires_at": "2000-01-01T00:00:00Z"}, "capability-stale"),
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
    environment = _environment(capability_updates=updates)
    with pytest.raises(runtime.Refusal, match=category):
        runtime.run(lock_path=LOCK, environ=environment)


@pytest.mark.parametrize(
    ("updates", "category"),
    [
        (
            {"BYOF_IMAGE": "registry.example/npa-robotwin:latest"},
            "bootstrap-image-not-immutable",
        ),
        ({"NPA_BYOF_RUN_ID": "robotwin-divergent"}, "runtime-binding-mismatch"),
        ({"S3_OUTPUT_PREFIX": "s3://other/output/"}, "runtime-binding-mismatch"),
    ],
)
def test_mutated_runtime_binding_refuses_before_lock_access(
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, str],
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
            environ=_environment(environment_updates=updates),
        )


def test_old_raw_authority_envelope_and_extra_fields_refuse_before_lock_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_load_lock",
        lambda *_args, **_kwargs: pytest.fail("lock read preceded authorization"),
    )
    old = _environment()
    old[runtime.AUTH_ENV] = json.dumps(
        {
            "schema_version": runtime.AUTH_SCHEMA,
            "context_base64": "manager-context-canary",
            "customer_authorization_base64": "customer-assertion-canary",
        }
    )
    extra = _environment()
    payload = json.loads(extra[runtime.AUTH_ENV])
    payload["issuer"] = "customer-issuer-canary"
    extra[runtime.AUTH_ENV] = json.dumps(payload)

    for environment in (old, extra):
        with pytest.raises(runtime.Refusal, match="authorization-envelope-invalid"):
            runtime.run(lock_path=LOCK, environ=environment)


def test_capability_parse_refusal_discards_private_exception_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_load_lock",
        lambda *_args, **_kwargs: pytest.fail("lock read preceded authorization"),
    )
    private = "private-expiry-canary"
    with pytest.raises(runtime.Refusal) as caught:
        runtime.run(
            lock_path=LOCK,
            environ=_environment(capability_updates={"expires_at": private}),
        )
    assert private not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_exact_capability_contains_no_raw_authority_or_manager_context() -> None:
    encoded = _environment()[runtime.AUTH_ENV]
    assert set(json.loads(encoded)) == runtime.CAPABILITY_FIELDS
    for private in (
        "customer-assertion-canary",
        "customer-issuer-canary",
        "nonce-canary",
        "manager-project-canary",
        "manager-profile-canary",
        "manager-context-canary",
        "kubeconfig-canary",
        "skypilot-canary",
    ):
        assert private not in encoded


def test_capability_default_state_is_owner_controlled_and_not_temporary() -> None:
    assert runtime.CAPABILITY_STATE_DIR == Path(
        "/home/ubuntu/.local/state/npa/robotwin/capabilities"
    )
    assert Path(tempfile.gettempdir()) not in runtime.CAPABILITY_STATE_DIR.parents


def test_capability_state_directory_is_created_owner_only(tmp_path: Path) -> None:
    state_dir = tmp_path / "state" / "capabilities"
    capability = json.loads(_environment()[runtime.AUTH_ENV])

    runtime._consume_capability(capability, state_dir=state_dir)

    metadata = state_dir.stat()
    assert metadata.st_uid == os.geteuid()
    assert metadata.st_mode & 0o777 == 0o700


def test_capability_state_refuses_insecure_directory(tmp_path: Path) -> None:
    state_dir = tmp_path / "capabilities"
    state_dir.mkdir()
    state_dir.chmod(0o755)
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o755
    capability = json.loads(_environment()[runtime.AUTH_ENV])

    with pytest.raises(runtime.Refusal, match="capability-state-not-owner-only"):
        runtime._consume_capability(capability, state_dir=state_dir)


def test_capability_marker_does_not_follow_symlink(tmp_path: Path) -> None:
    state_dir = tmp_path / "capabilities"
    state_dir.mkdir(mode=0o700)
    capability = json.loads(_environment()[runtime.AUTH_ENV])
    marker_name = hashlib.sha256(capability["capability_id"].encode()).hexdigest()
    target = tmp_path / "target"
    target.write_bytes(b"unchanged\n")
    (state_dir / marker_name).symlink_to(target)

    with pytest.raises(runtime.Refusal, match="capability-replayed"):
        runtime._consume_capability(capability, state_dir=state_dir)

    assert target.read_bytes() == b"unchanged\n"


def test_capability_is_consumed_once_immediately_before_runtime_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runtime, "CAPABILITY_STATE_DIR", tmp_path / "capabilities")
    monkeypatch.setattr(
        runtime,
        "_load_lock",
        lambda *_args, **_kwargs: {
            "bootstrap": {"status": "complete"},
            "runtime_delivery": {"status": "complete"},
        },
    )
    environment = _environment()

    with pytest.raises(runtime.Refusal, match="runtime-fetch-not-implemented"):
        runtime.run(lock_path=LOCK, environ=environment)
    with pytest.raises(runtime.Refusal, match="capability-replayed") as caught:
        runtime.run(lock_path=LOCK, environ=environment)

    markers = list((tmp_path / "capabilities").iterdir())
    assert len(markers) == 1
    assert markers[0].read_bytes() == b"consumed\n"
    assert markers[0].stat().st_mode & 0o777 == 0o600
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_capability_write_failure_rolls_back_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "capabilities"
    capability = json.loads(_environment()[runtime.AUTH_ENV])
    original_write = runtime.os.write

    def fail_marker_write(descriptor: int, payload: bytes) -> int:
        if payload == b"consumed\n":
            raise OSError("synthetic marker write failure")
        return original_write(descriptor, payload)

    monkeypatch.setattr(runtime.os, "write", fail_marker_write)
    with pytest.raises(runtime.Refusal, match="capability-state-unavailable"):
        runtime._consume_capability(capability, state_dir=state_dir)
    assert list(state_dir.iterdir()) == []


def test_capability_file_fsync_failure_rolls_back_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "capabilities"
    capability = json.loads(_environment()[runtime.AUTH_ENV])
    original_fsync = runtime.os.fsync
    original_unlink = runtime.os.unlink
    marker_name = hashlib.sha256(capability["capability_id"].encode()).hexdigest()
    fsync_kinds: list[str] = []
    unlink_receipts: list[tuple[str, int | None]] = []

    def fail_file_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        fsync_kinds.append(kind)
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("synthetic marker fsync failure")
        original_fsync(descriptor)

    def record_unlink(name: str, *, dir_fd: int | None = None) -> None:
        unlink_receipts.append((str(name), dir_fd))
        original_unlink(name, dir_fd=dir_fd)

    monkeypatch.setattr(runtime.os, "fsync", fail_file_fsync)
    monkeypatch.setattr(runtime.os, "unlink", record_unlink)
    with pytest.raises(runtime.Refusal, match="capability-state-unavailable") as caught:
        runtime._consume_capability(capability, state_dir=state_dir)
    assert fsync_kinds == ["file", "directory"]
    assert [name for name, _dir_fd in unlink_receipts] == [marker_name]
    assert all(dir_fd is not None for _name, dir_fd in unlink_receipts)
    assert list(state_dir.iterdir()) == []
    recovery = caught.value.recovery_context
    assert recovery.cleanup_outcomes == ((marker_name, "removed"),)
    assert recovery.residual_names == ()
    assert recovery.directory_fsync == "synced"


def test_capability_directory_fsync_failure_rolls_back_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "capabilities"
    capability = json.loads(_environment()[runtime.AUTH_ENV])
    original_fsync = runtime.os.fsync
    original_unlink = runtime.os.unlink
    marker_name = hashlib.sha256(capability["capability_id"].encode()).hexdigest()
    fsync_kinds: list[str] = []
    unlink_receipts: list[tuple[str, int | None]] = []

    def fail_directory_commit(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        fsync_kinds.append(kind)
        if kind == "directory" and fsync_kinds.count("directory") == 1:
            raise OSError("synthetic directory fsync failure")
        original_fsync(descriptor)

    def record_unlink(name: str, *, dir_fd: int | None = None) -> None:
        unlink_receipts.append((str(name), dir_fd))
        original_unlink(name, dir_fd=dir_fd)

    monkeypatch.setattr(runtime.os, "fsync", fail_directory_commit)
    monkeypatch.setattr(runtime.os, "unlink", record_unlink)
    with pytest.raises(runtime.Refusal, match="capability-state-unavailable") as caught:
        runtime._consume_capability(capability, state_dir=state_dir)
    assert fsync_kinds == ["file", "directory", "directory"]
    assert [name for name, _dir_fd in unlink_receipts] == [marker_name]
    assert all(dir_fd is not None for _name, dir_fd in unlink_receipts)
    assert list(state_dir.iterdir()) == []
    recovery = caught.value.recovery_context
    assert recovery.cleanup_outcomes == ((marker_name, "removed"),)
    assert recovery.residual_names == ()
    assert recovery.directory_fsync == "synced"


def test_capability_rollback_fsync_failure_preserves_exact_recovery_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_dir = tmp_path / "capabilities"
    capability = json.loads(_environment()[runtime.AUTH_ENV])
    original_fsync = runtime.os.fsync
    original_unlink = runtime.os.unlink
    marker_name = hashlib.sha256(capability["capability_id"].encode()).hexdigest()
    fsync_kinds: list[str] = []
    unlink_receipts: list[tuple[str, int | None]] = []

    def fail_rollback_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        fsync_kinds.append(kind)
        if kind == "directory":
            raise OSError("synthetic rollback directory fsync failure")
        original_fsync(descriptor)

    def record_unlink(name: str, *, dir_fd: int | None = None) -> None:
        unlink_receipts.append((str(name), dir_fd))
        original_unlink(name, dir_fd=dir_fd)

    monkeypatch.setattr(runtime.os, "fsync", fail_rollback_fsync)
    monkeypatch.setattr(runtime.os, "unlink", record_unlink)
    with pytest.raises(runtime.Refusal, match="capability-state-unavailable"):
        runtime._consume_capability(capability, state_dir=state_dir)
    # The commit and rollback attempts are both visible, while the marker is
    # removed and no private contents enter the receipt.
    assert fsync_kinds == ["file", "directory", "directory"]
    assert [name for name, _dir_fd in unlink_receipts] == [marker_name]
    assert all(dir_fd is not None for _name, dir_fd in unlink_receipts)
    assert list(state_dir.iterdir()) == []


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


def test_assert_refusal_passes_only_isolated_allowlisted_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinels = {
        runtime.AUTH_ENV: "authorization-secret-canary",
        "AWS_SECRET_ACCESS_KEY": "storage-secret-canary",
        "GH_TOKEN": "source-token-canary",
        "HF_TOKEN": "model-token-canary",
        "NEBIUS_IAM_TOKEN": "provider-token-canary",
        "NGC_API_KEY": "runtime-token-canary",
        "NPA_MANAGER_CONTEXT": "manager-context-canary",
        "NPA_REGISTRY": "registry-context-canary",
        "NPA_ROBOTWIN_RUNTIME_CONTEXT": "runtime-context-canary",
        "NPA_S3_BUCKET": "storage-context-canary",
    }
    for name, value in sentinels.items():
        monkeypatch.setenv(name, value)
    observed: dict[str, str] = {}

    def capture_refusal(command, **kwargs):
        observed.update(kwargs["env"])
        return subprocess.CompletedProcess(
            command,
            78,
            stdout="",
            stderr="ROBOTWIN_RUNTIME_REFUSED:authorization-missing\n",
        )

    monkeypatch.setattr(runtime.subprocess, "run", capture_refusal)

    assert runtime.assert_refusal() == 0
    assert set(observed) == {
        "HOME",
        "PYTHONDONTWRITEBYTECODE",
        *runtime.REFUSAL_PATH_ENV_KEYS,
    }
    assert not sentinels.keys() & observed.keys()
    assert observed["PYTHONDONTWRITEBYTECODE"] == "1"


def test_assert_refusal_uses_the_real_run_gate_and_isolated_paths() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNTIME), "assert-refusal"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "no simulator capability claimed" in completed.stdout
