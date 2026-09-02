"""Tests for the RoboCasa shared implementation and service."""

from __future__ import annotations

import sys
import types
import json
from pathlib import Path

import numpy as np

import pytest
from fastapi.testclient import TestClient

from npa.workbench.robocasa.capabilities import (
    RoboCasaError,
    compute_manifest_sha256,
    kitchen_asset_availability,
    kitchen_task_registration,
    make_run_id,
    system_info,
)
from npa.workbench.robocasa.schemas import RoboCasaRunRequest
from npa.workbench.robocasa.service import create_app


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
    payload = {"env_id": "robocasa/PickPlaceCounterToCabinet", "capability": "kitchen_random_rollout"}
    a = compute_manifest_sha256("run", payload)
    b = compute_manifest_sha256("run", dict(payload))
    assert a == b
    assert len(a) == 64


def test_make_run_id_is_deterministic() -> None:
    a = make_run_id("kitchen_random_rollout", "abc")
    b = make_run_id("kitchen_random_rollout", "abc")
    assert a == b
    assert a.startswith("robocasa-kitchen_random_rollout-")


def test_system_info_returns_payload() -> None:
    info = system_info()
    assert info.status == "ok"
    assert info.python


def test_kitchen_task_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_robocasa(monkeypatch)
    result = kitchen_task_registration()
    assert result["env_id"] == "robocasa/PickPlaceCounterToCabinet"
    assert result["registered_env_count"] == 2


def test_kitchen_task_registration_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_robocasa(monkeypatch)
    with pytest.raises(RoboCasaError):
        kitchen_task_registration(env_id="robocasa/DoesNotExist")


def test_kitchen_asset_availability_missing_root(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_robocasa(monkeypatch)
    with pytest.raises(RoboCasaError):
        kitchen_asset_availability()


def test_run_capability_unsupported() -> None:
    # The schema rejects an unsupported capability before dispatch.
    with pytest.raises(ValueError):
        RoboCasaRunRequest(capability="bogus", output_uri="s3://bucket/out")


def test_run_request_validates_capability() -> None:
    with pytest.raises(ValueError):
        RoboCasaRunRequest(capability="bogus", output_uri="s3://bucket/out")


def test_service_health() -> None:
    app = create_app(auth_mode="none")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_service_system_info() -> None:
    app = create_app(auth_mode="none")
    client = TestClient(app)
    response = client.get("/system-info")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_service_run_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_robocasa(monkeypatch)
    app = create_app(auth_mode="none")
    client = TestClient(app)
    response = client.post(
        "/run",
        json={
            "capability": "kitchen_task_registration",
            "env_id": "robocasa/PickPlaceCounterToCabinet",
            "output_uri": "s3://bucket/out",
        },
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    status_response = client.get("/status", params={"run_id": run_id})
    assert status_response.status_code == 200
    assert status_response.json()["status"] in {"running", "completed"}


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


def test_service_auth_token() -> None:
    app = create_app(auth_mode="token", token="secret")
    client = TestClient(app)
    # /health is intentionally unauthenticated so Kubernetes liveness/readiness
    # probes can reach it without a token; the protected surface is /system-info.
    assert client.get("/health").status_code == 200
    assert client.get("/system-info").status_code == 401
    assert (
        client.get("/system-info", headers={"Authorization": "Bearer secret"}).status_code
        == 200
    )
    assert (
        client.get("/system-info", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )


def test_service_list_runs() -> None:
    app = create_app(auth_mode="none")
    client = TestClient(app)
    response = client.get("/runs")
    assert response.status_code == 200
    assert "runs" in response.json()


class _FakeActionSpace:
    def sample(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)


class _FakeEnv:
    action_space = _FakeActionSpace()

    def __init__(self) -> None:
        self._closed = False

    def reset(self, seed=None):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}

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

    class FakeGym:
        envs = types.SimpleNamespace(registry={})
        @staticmethod
        def make(env_id, **kwargs):
            return _FakeEnv()

    monkeypatch.setitem(sys.modules, "gymnasium", FakeGym())


def test_make_env_uses_nonempty_objaverse_registry(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_kitchen_trajectory_export(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _install_fake_env(monkeypatch)
    from npa.workbench.robocasa.capabilities import kitchen_trajectory_export

    result = kitchen_trajectory_export(
        env_id="robocasa/PickPlaceCounterToCabinet",
        iterations=3,
        num_envs=2,
        seed=1,
        output_dir=tmp_path,
    )
    assert result["trajectory_export_ok"] is True
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
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "metrics.json").exists()


def test_kitchen_trajectory_export_records_panda_omron_multitask_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _install_fake_env(monkeypatch)
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
    assert metadata["task_env_ids"] == ["robocasa/TrainA", "robocasa/TrainB"]
    assert [episode["env_id"] for episode in metadata["episodes"]] == [
        "robocasa/TrainA",
        "robocasa/TrainB",
        "robocasa/TrainA",
        "robocasa/TrainB",
    ]


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
    (tmp_path / "training.log").write_text("first log")

    resolved, checkpoint_sha, first_tree_sha = _checkpoint_identity(tmp_path)
    (tmp_path / "training.log").write_text("changed unrelated log")
    _, checkpoint_sha_after, second_tree_sha = _checkpoint_identity(tmp_path)

    assert resolved == checkpoint
    assert checkpoint_sha == checkpoint_sha_after
    assert first_tree_sha != second_tree_sha


def test_kitchen_trajectory_export_missing_image_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _install_fake_robocasa(monkeypatch)
    from npa.workbench.robocasa.capabilities import RoboCasaError, kitchen_trajectory_export

    with pytest.raises(RoboCasaError):
        kitchen_trajectory_export(
            env_id="robocasa/PickPlaceCounterToCabinet",
            iterations=1,
            num_envs=1,
            output_dir=tmp_path,
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

    uploaded: list[tuple[str, str, str]] = []

    class FakeS3:
        def upload_file(self, local_path, bucket, key):
            uploaded.append((str(local_path), bucket, key))

    monkeypatch.setattr("boto3.client", lambda *a, **k: FakeS3())

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
        output_uri="s3://bucket/out",
        iterations=2,
        seed=1,
    )
    assert response.status == "completed"
    assert response.run_id == "local"
    assert response.output_uri == "s3://bucket/out"
    # The rollout produced a video artifact that was uploaded to S3.
    assert uploaded, "expected at least one uploaded artifact"
    assert all(bucket == "bucket" for _, bucket, _ in uploaded)
    assert all(key.startswith("out/") for _, _, key in uploaded)


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
        output_uri="s3://bucket/out",
        iterations=1,
        num_envs=1,
    )
    assert response.status == "completed"
    assert captured["output_dir"] is not None
    assert Path(captured["output_dir"]).is_dir()
