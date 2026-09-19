"""Tests for the RoboCasa shared implementation and service."""

from __future__ import annotations

import asyncio
import hashlib
import sys
import threading
import types
import json
from pathlib import Path

import httpx
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
from npa.workbench.robocasa.schemas import RoboCasaStatusResponse
from npa.workbench.robocasa.service import RunRegistry, create_app


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


def test_kitchen_asset_availability_missing_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities.upload_output", lambda *args: None
    )
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


def test_run_registry_allows_temporary_overflow_when_every_run_is_active() -> None:
    runs = RunRegistry(max_entries=1, ttl_seconds=100, clock=lambda: 1.0)
    runs["one"] = _status("one", "running")
    runs["two"] = _status("two", "running")

    assert {run.run_id for run in runs.values()} == {"one", "two"}


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


class _FakeActionSpace:
    shape = (7,)

    def sample(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def seed(self, _seed: int) -> None:
        return None

    def contains(self, action) -> bool:
        return np.asarray(action).shape == self.shape


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
        assert (ep_dir / "rollout.mp4").stat().st_size > 0
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

    arrays, episode = _collect_trajectory_episode(_TemporalEnv(), iterations=3, seed=7)

    assert [float(state[0]) for state in arrays["state"]] == [0.0, 1.0, 2.0]
    assert [float(action[0]) for action in arrays["actions"]] == [10.0, 11.0, 12.0]
    assert episode["seed"] == 7
    assert episode["length"] == 3
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


def test_matched_eval_rejects_different_initial_workspace_frames() -> None:
    from npa.workbench.robocasa.capabilities import _require_matched_initial_state

    with pytest.raises(RoboCasaError, match="initial workspace frames do not match"):
        _require_matched_initial_state(
            {"initial_workspace_sha256": "policy"},
            {"initial_workspace_sha256": "baseline"},
        )


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


def test_rollout_output_has_machine_readable_execution_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_env(monkeypatch)

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
    assert provenance["generator"] == "robocasa"
    assert provenance["simulator"] == "mujoco"
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
    (tmp_path / "training.log").write_text("first log")

    resolved, checkpoint_sha, first_tree_sha = _checkpoint_identity(tmp_path)
    (tmp_path / "training.log").write_text("changed unrelated log")
    _, checkpoint_sha_after, second_tree_sha = _checkpoint_identity(tmp_path)

    assert resolved == checkpoint
    assert checkpoint_sha == checkpoint_sha_after
    assert first_tree_sha != second_tree_sha


def test_kitchen_policy_eval_compares_matched_seed_random_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.workbench.robocasa.capabilities import (
        _random_action_selector,
        kitchen_policy_eval,
    )

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
        "npa.workbench.robocasa.capabilities._load_act_policy",
        lambda _path: (FakePolicy(),),
    )
    monkeypatch.setattr(
        "npa.workbench.robocasa.capabilities._act_action_selector",
        lambda _runtime: _random_action_selector,
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
        output_path="s3://bucket/out",
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
