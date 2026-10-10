from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from npa.adapter.sim_to_lerobot import _episode_timestamps
from npa.workflows.byof.openpi_pipeline import (
    OpenPIPipelineError,
    _load_dataset,
    _single_host_runtime_evidence,
)
from npa.workflows.byof import openpi_service_lifecycle
from npa.workflows.sim2real import (
    pi05_closed_loop,
    pi05_closed_loop_entry,
    pi05_data,
    pi05_isaac,
)
from npa.workflows.byof.openpi_service_lifecycle import (
    _inject_checkpoint_probe,
    _patch_private_s3_checkpoint,
)
from npa.workflows.sim2real.pi05_closed_loop import (
    _configure_eval_object,
    _policy_phase,
    _sampler_noise_pairing,
    _settle_policy_reset,
    _success,
    _validate_eval_geometry,
    _validate_eval_robot_plant,
    _validated_geometry_receipt,
    _validated_robot_plant,
    _wrist_variant,
)
from npa.workflows.sim2real.pi05_contract import EPISODE_SCHEMA, Pi05ContractError
from npa.workflows.sim2real.pi05_collect_all import SPLITS, _collector_argv
from npa.workflows.sim2real.pi05_data import OPENPI_SCHEMA, export_local
from npa.workflows.sim2real.pi05_isaac import (
    _collection_report,
    _new_attempt_buffers,
    _publish_episode,
    _save_partial_attempt,
    configure_surface_goal,
    support_surface_contract,
)


TARGET = np.asarray([0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.7, 0.37], dtype=np.float32)
SUCCESS_EVENTS = [
    "contact",
    "grasp",
    "lift",
    "transport",
    "release",
    "support_contact",
    "stable",
]


def _robot_plant() -> dict:
    return {
        "schema": "npa.sim2real.pi05.robot_plant.v1",
        "source": "isaaclab_assets.FRANKA_PANDA_HIGH_PD_CFG",
        "configured_disable_robot_gravity": True,
        "observed_disable_gravity": {
            "shape": [1, 9],
            "unique": [1],
            "all_bodies_disabled": True,
        },
        "actuators": {"panda_shoulder": {"stiffness": [400.0]}},
    }


class _FakeJaxTopology:
    def __init__(self, processes: int = 1, local: int = 8, global_: int = 8):
        self._processes = processes
        self._local = local
        self._global = global_

    def process_count(self) -> int:
        return self._processes

    def process_index(self) -> int:
        return 0

    def local_devices(self) -> list[object]:
        return [object()] * self._local

    def devices(self) -> list[object]:
        return [object()] * self._global


class _LifecycleApp:
    def __init__(self, lifecycle: dict[str, bool]):
        self.lifecycle = lifecycle

    def close(self) -> None:
        self.lifecycle["app_closed"] = True


class _TrackingTemporaryDirectory:
    def __init__(self, path: Path, lifecycle: dict[str, bool]):
        self.name = str(path)
        self.lifecycle = lifecycle

    def __enter__(self) -> str:
        Path(self.name).mkdir()
        return self.name

    def __exit__(self, *_args) -> None:
        self.lifecycle["temp_exited"] = True


class _JointTensor:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)

    def __getitem__(self, item: object) -> _JointTensor:
        return _JointTensor(self.value[item])

    def detach(self) -> _JointTensor:
        return self

    def cpu(self) -> _JointTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _CudaActionTensor:
    def __init__(self, value: object, *, on_host: bool = False) -> None:
        self.value = np.asarray(value, dtype=np.float32)
        self.on_host = on_host

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def __getitem__(self, item: object) -> _CudaActionTensor:
        return _CudaActionTensor(self.value[item], on_host=self.on_host)

    def detach(self) -> _CudaActionTensor:
        return self

    def cpu(self) -> _CudaActionTensor:
        return _CudaActionTensor(self.value, on_host=True)

    def numpy(self) -> np.ndarray:
        if not self.on_host:
            raise TypeError("can't convert cuda:0 device type tensor to numpy")
        return self.value


class _ActionProxy:
    def __init__(self, value: object) -> None:
        self.torch = _CudaActionTensor(value)


def _policy_action_native(offset: object, scale: object) -> Namespace:
    term = Namespace(_offset=offset, _scale=scale, cfg=Namespace(clip=None))
    manager = Namespace(
        active_terms=["arm_action", "gripper_action"],
        action_term_dim=[7, 1],
        get_term=lambda name: term if name == "arm_action" else Namespace(),
    )
    return Namespace(action_manager=manager)


def test_joint_bridge_moves_proxy_action_parameters_to_host() -> None:
    native = _policy_action_native(_ActionProxy([[0.1] * 7]), _ActionProxy([[0.5] * 7]))

    defaults, scales, total, arm_start, grip_start = pi05_closed_loop._joint_bridge(
        native
    )

    assert np.allclose(defaults, [0.1] * 7)
    assert np.allclose(scales, [0.5] * 7)
    assert (total, arm_start, grip_start) == (8, 0, 7)


def test_joint_bridge_expands_pinned_uniform_scalar_scale() -> None:
    native = _policy_action_native(_ActionProxy([[0.1] * 7]), 0.5)

    defaults, scales, total, arm_start, grip_start = pi05_closed_loop._joint_bridge(
        native
    )

    assert np.allclose(defaults, [0.1] * 7)
    assert scales == [0.5] * 7
    assert (total, arm_start, grip_start) == (8, 0, 7)


def test_joint_bridge_rejects_broadcastable_action_parameter_shape() -> None:
    native = _policy_action_native(_ActionProxy([0.1] * 7), _ActionProxy([[0.5] * 7]))

    with pytest.raises(RuntimeError, match=r"shape \(1, 7\), got \(7,\)"):
        pi05_closed_loop._joint_bridge(native)


@pytest.mark.parametrize("scale", [0.0, float("nan"), float("inf")])
def test_joint_bridge_rejects_invalid_uniform_scale(scale: float) -> None:
    native = _policy_action_native(_ActionProxy([[0.1] * 7]), scale)

    with pytest.raises(RuntimeError, match="finite and nonzero"):
        pi05_closed_loop._joint_bridge(native)


def test_joint_bridge_rejects_zero_per_joint_scale() -> None:
    native = _policy_action_native(
        _ActionProxy([[0.1] * 7]), _ActionProxy([[0.5] * 6 + [0.0]])
    )

    with pytest.raises(RuntimeError, match="finite and nonzero"):
        pi05_closed_loop._joint_bridge(native)


def test_joint_bridge_rejects_configured_sdk_action_clipping() -> None:
    native = _policy_action_native(_ActionProxy([[0.1] * 7]), 0.5)
    native.action_manager.get_term("arm_action").cfg.clip = {".*": (-1.0, 1.0)}

    with pytest.raises(RuntimeError, match="unclipped Isaac affine"):
        pi05_closed_loop._joint_bridge(native)


def test_joint_bridge_rejects_broadcastable_scale_shape() -> None:
    native = _policy_action_native(_ActionProxy([[0.1] * 7]), _ActionProxy([0.5] * 7))

    with pytest.raises(RuntimeError, match=r"scalar or shape \(1, 7\), got \(7,\)"):
        pi05_closed_loop._joint_bridge(native)


def test_closed_loop_synchronizes_wrist_before_reading_rgb(monkeypatch) -> None:
    events: list[str] = []
    robot = Namespace(
        joint_names=list(pi05_closed_loop.JOINT_NAMES),
        data=Namespace(joint_pos=_JointTensor([np.arange(7, dtype=np.float32)])),
    )
    native = Namespace(scene={"robot": robot})

    monkeypatch.setattr(
        pi05_closed_loop,
        "_synchronize_wrist_before_capture",
        lambda _native: events.append("synchronized"),
    )

    def rgb(_native: object, name: str) -> np.ndarray:
        assert events[0] == "synchronized"
        events.append(name)
        return np.zeros((224, 224, 3), dtype=np.uint8)

    monkeypatch.setattr(pi05_closed_loop, "_rgb", rgb)
    monkeypatch.setattr(pi05_closed_loop, "_gripper_width", lambda _native: 0.08)

    _exterior, _wrist, joint, gripper = pi05_closed_loop._policy_observation(native)

    assert events == ["synchronized", "pi05_exterior", "pi05_wrist"]
    assert joint.tolist() == list(range(7))
    assert gripper.tolist() == [0.0]


def test_closed_loop_wrist_ablation_is_deterministic() -> None:
    first = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    second = np.flip(first, axis=0).copy()
    frozen = {"wrist_mode": "frozen", "ablation_seed": 7, "frozen_wrist": None}
    initial, _ = _wrist_variant(frozen, first, 0)
    later, receipt = _wrist_variant(frozen, second, 1)
    assert np.array_equal(initial, first)
    assert np.array_equal(later, first)
    assert receipt["source_sha256"] != receipt["policy_sha256"]

    left = {
        "wrist_mode": "pixel_channel_shuffle",
        "ablation_seed": 11,
        "frozen_wrist": None,
    }
    right = {
        "wrist_mode": "pixel_channel_shuffle",
        "ablation_seed": 11,
        "frozen_wrist": None,
    }
    assert np.array_equal(
        _wrist_variant(left, first, 3)[0], _wrist_variant(right, first, 3)[0]
    )


class _FailSecondControllerStep:
    calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("controller rejected gs://private/location")
        return {
            "terminated": False,
            "truncated": False,
            "command_application_time_s": 0.0,
            "resulting_state_time_s": 1.0 / 15.0,
        }


def _partial_execution_context(record: dict) -> dict:
    return {
        "record": record,
        "bridge": ([0.0] * 7, [0.5] * 7, 8, 0, 7),
        "env": object(),
        "native": object(),
        "initial_z": 0.024,
        "target_xy": np.asarray([0.5, 0.18]),
        "object_width_m": 0.048,
    }


def test_policy_query_receipt_survives_partial_controller_failure(monkeypatch) -> None:
    actions = np.zeros((15, 8), dtype=np.float64)
    actions[:, 7] = -0.2
    record = {"queries": [], "physics_trace": []}
    context = _partial_execution_context(record)
    monkeypatch.setattr(
        pi05_closed_loop,
        "_query_policy",
        lambda _context, _decision: (
            actions,
            {"sampler_noise_sha256": "a" * 64},
            {"observation_time_s": 0.0},
        ),
    )
    monkeypatch.setattr(pi05_closed_loop, "_apply_target", _FailSecondControllerStep())
    monkeypatch.setattr(pi05_closed_loop, "_physics_sample", lambda *_args: {})

    with pytest.raises(RuntimeError, match="controller rejected"):
        pi05_closed_loop._decision_step(context, 0)

    query = record["queries"][0]
    assert query["action_receipt"]["shape"] == [15, 8]
    assert query["action_receipt"]["gripper_scores"] == [-0.2] * 15
    assert query["action_receipt"]["thresholded_gripper_commands"] == [0] * 15
    assert query["execution_status"] == "failed"
    assert query["attempted_prefix"] == 2
    assert query["executed_prefix"] == 1
    assert len(query["controller_proofs"]) == 1
    assert query["controller_proofs"][0]["physics_observation_status"] == "recorded"
    assert record["physics_trace"][0]["simulator_time_s"] == 1.0 / 15.0
    assert (
        query["execution_error"] == "RuntimeError: controller rejected <redacted-uri>"
    )


def test_policy_query_records_applied_target_with_failed_physics_observation(
    monkeypatch,
) -> None:
    record = {"queries": [], "physics_trace": []}
    context = _partial_execution_context(record)
    actions = np.zeros((15, 8), dtype=np.float64)
    monkeypatch.setattr(
        pi05_closed_loop,
        "_query_policy",
        lambda *_args: (actions, {}, {"observation_time_s": 0.0}),
    )
    monkeypatch.setattr(
        pi05_closed_loop,
        "_apply_target",
        lambda *_args, **_kwargs: {
            "terminated": False,
            "truncated": False,
            "command_application_time_s": 0.0,
            "resulting_state_time_s": 1.0 / 15.0,
        },
    )
    monkeypatch.setattr(
        pi05_closed_loop,
        "_physics_sample",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("observer failed")),
    )

    with pytest.raises(RuntimeError, match="observer failed"):
        pi05_closed_loop._decision_step(context, 0)

    proof = record["queries"][0]["controller_proofs"][0]
    assert proof["physics_observation_status"] == "failed"
    assert proof["physics_observation_error"] == "RuntimeError: observer failed"
    assert record["physics_trace"] == []


def test_effectiveness_rejects_absent_sampler_identity() -> None:
    def episode(mode: str, identity: str | None) -> dict:
        proofs = [{"terminated": False, "truncated": False}] * 8
        return {
            "scenario_seed": 9,
            "wrist_ablation_mode": mode,
            "queries": [
                {
                    "decision": 0,
                    "execution_status": "completed",
                    "attempted_prefix": 8,
                    "executed_prefix": 8,
                    "controller_proofs": proofs,
                    "inference": {"sampler_noise_sha256": identity},
                }
            ],
        }

    modes = ["fresh", "frozen", "pixel_channel_shuffle"]
    absent = _sampler_noise_pairing([episode(mode, None) for mode in modes], modes)
    assert absent["sampler_noise_identity_verified"] is False
    paired = _sampler_noise_pairing([episode(mode, "a" * 64) for mode in modes], modes)
    assert paired["sampler_noise_identity_verified"] is True

    failed = [episode(mode, "a" * 64) for mode in modes]
    failed[0]["queries"][0]["execution_status"] = "failed"
    assert (
        _sampler_noise_pairing(failed, modes)["sampler_noise_identity_verified"]
        is False
    )

    terminal = [episode(mode, "a" * 64) for mode in modes]
    terminal[0]["queries"][0]["controller_proofs"][0] = {
        "terminated": True,
        "truncated": False,
    }
    assert not _sampler_noise_pairing(terminal, modes)[
        "sampler_noise_identity_verified"
    ]


def test_episode_failure_redacts_query_and_top_level_uri(monkeypatch) -> None:
    def fail(*args):
        record = args[3]
        record["queries"].append({"execution_error": "already redacted"})
        raise RuntimeError("controller rejected gs://private/location")

    monkeypatch.setattr(pi05_closed_loop, "_episode", fail)
    args = Namespace(
        wrist_ablation_modes=["fresh"], episodes=1, seed=1, decisions=1, ablation_seed=2
    )
    env = Namespace(reset=lambda **_kwargs: None)
    protocol = {"scenarios": [{}], "object_width_m": 0.048}

    result = pi05_closed_loop._evaluate(env, object(), args, protocol)[0]

    assert result["failure"] == "RuntimeError: controller rejected <redacted-uri>"
    assert "private/location" not in json.dumps(result)


def test_partial_episode_receipt_retains_query_without_raw_frames(
    tmp_path: Path,
) -> None:
    episode = {
        "queries": [{"execution_status": "failed", "action_receipt": {"sha256": "a"}}],
        "_exterior_frames": [np.zeros((2, 2, 3), dtype=np.uint8)],
        "_wrist_frames": [np.zeros((2, 2, 3), dtype=np.uint8)],
    }

    pi05_closed_loop._persist_partial_episodes(tmp_path, [episode])

    receipt = json.loads((tmp_path / "partial-episodes.json").read_text())[0]
    assert receipt["queries"] == episode["queries"]
    assert "_exterior_frames" not in receipt


def test_pi05_mesh_is_single_host_and_rejects_uninitialized_second_node() -> None:
    evidence = _single_host_runtime_evidence(
        _FakeJaxTopology(), {"SKYPILOT_NUM_NODES": "1"}
    )
    assert evidence["local_device_count"] == 8
    with pytest.raises(OpenPIPipelineError, match="supports one host only"):
        _single_host_runtime_evidence(_FakeJaxTopology(), {"SKYPILOT_NUM_NODES": "2"})
    with pytest.raises(OpenPIPipelineError, match="supports one host only"):
        _single_host_runtime_evidence(_FakeJaxTopology(processes=2), {})


def test_surface_task_replaces_airborne_goal_and_exports_exact_scenario() -> None:
    ranges = Namespace(pos_x=(0.3, 0.7), pos_y=(-0.25, 0.25), pos_z=(0.20, 0.42))
    cfg = Namespace(
        commands=Namespace(
            object_pose=Namespace(ranges=ranges, resampling_time_range=(3.0, 5.0))
        )
    )
    configure_surface_goal(cfg, (0.50, 0.18, 0.025))
    assert ranges.pos_z == (0.025, 0.025)
    assert cfg.commands.object_pose.debug_vis is False
    attempt = _episode_metadata("cube", "scene", 16)
    attempt["scene_seed"] = 7
    attempt["support_surface"] = support_surface_contract()
    geometry = {
        "source_untransformed_extent_m": [0.06, 0.06, 0.06],
        "effective_local_extent_m": [0.06, 0.06, 0.06],
        "declared_spawn_scale": [1.0, 1.0, 1.0],
        "grasp_width_axis": "object_local_y",
        "grasp_width_m": 0.06,
        "support_height_m": 0.06,
        "measurement": "UsdGeom.BBoxCache composed runtime prim",
    }
    plant = _robot_plant()
    report = _collection_report(
        [attempt], "cube", "", (1.0, 1.0, 1.0), "scene", geometry, plant
    )
    assert report["robot_plant"] == plant
    assert report["successful_scenarios"] == [
        {
            "scene_seed": 7,
            "scene_configuration_digest": "scene",
            "initial_object_position_m": [0.45, -0.15, 0.025],
            "target_position_m": [0.50, 0.18, 0.025],
            "destination_marker": attempt["destination_marker"],
            "support_surface": support_surface_contract(),
        }
    ]


def test_policy_runtime_rejects_a_different_expert_robot_plant(monkeypatch) -> None:
    plant = _robot_plant()
    assert _validated_robot_plant(plant) is plant
    monkeypatch.setattr(
        pi05_closed_loop,
        "_robot_plant_receipt",
        lambda _native: {
            **plant,
            "observed_disable_gravity": {
                **plant["observed_disable_gravity"],
                "all_bodies_disabled": False,
            },
        },
    )

    with pytest.raises(RuntimeError, match="robot plant differs"):
        _validate_eval_robot_plant(object(), {"robot_plant": plant})


def test_failed_semantic_validation_preserves_only_diagnostic_rgb(tmp_path) -> None:
    buffers = _new_attempt_buffers()
    exterior = np.full((224, 224, 3), 17, dtype=np.uint8)
    wrist = np.full((224, 224, 3), 29, dtype=np.uint8)
    buffers["diagnostic_exterior"].append(exterior)
    buffers["diagnostic_wrist"].append(wrist)
    buffers["diagnostic_receipts"].append(
        {"observation_time_s": 1.0, "wrist_capture_time_s": 1.0}
    )
    buffers["wrist_mount_receipts"].append({"refresh_action": "failed_before_rgb"})

    report = _save_partial_attempt(
        tmp_path, 0, buffers, RuntimeError("semantic label missing")
    )

    failed = tmp_path / "episode-000000-failed"
    assert report["training_admitted"] is False
    assert report["diagnostic_camera_rows"] == 1
    assert report["wrist_mount_receipts"] == [{"refresh_action": "failed_before_rgb"}]
    assert np.array_equal(np.load(failed / "diagnostic_workspace.npy"), exterior[None])
    assert np.array_equal(np.load(failed / "diagnostic_wrist.npy"), wrist[None])
    assert not (failed / "obs_workspace.npy").exists()


def test_episode_publication_removes_partial_io_failure(tmp_path, monkeypatch) -> None:
    def fail_after_partial(path, _buffers):
        (path / "obs_workspace.npy").write_bytes(b"partial")
        raise OSError("simulated episode write failure")

    monkeypatch.setattr(pi05_isaac, "_write_episode_arrays", fail_after_partial)
    with pytest.raises(OSError, match="simulated episode write failure"):
        _publish_episode(tmp_path, 3, _new_attempt_buffers(), {"schema": "test"})

    assert not (tmp_path / "episode-000003").exists()
    assert list(tmp_path.iterdir()) == []


def test_lerobot_preserves_episode_relative_source_timestamps() -> None:
    arrays = {
        "state": np.zeros((3, 8), dtype=np.float32),
        "timestamps": np.asarray([41.2, 41.267, 41.335], dtype=np.float64),
    }
    assert np.allclose(_episode_timestamps(arrays, 15), [0.0, 0.067, 0.135])


def test_each_split_collector_runs_in_a_fresh_isaac_process() -> None:
    args = Namespace(
        output_root_uri="s3://example/experts",
        episodes_per_split=3,
        seed=41,
        object_usd="",
    )
    argv = _collector_argv(args, "gold")
    assert argv[:3] == [
        sys.executable,
        "-m",
        "npa.workflows.sim2real.pi05_isaac",
    ]
    assert argv[argv.index("--seed") + 1] == "20041"
    assert argv[argv.index("--object-scale") + 1] == "1.2,1.2,1.2"

    args.object_usd = "/runtime/operator-object.usd"
    custom = _collector_argv(args, "validation")
    assert custom[custom.index("--object-identity") + 1] == (
        "operator-custom-usd-scale-100"
    )


def test_gold_evaluator_reuses_exact_custom_usd_and_scale() -> None:
    spawn = Namespace(usd_path="stock.usd", scale=(1.0, 1.0, 1.0))
    cfg = Namespace(scene=Namespace(object=Namespace(spawn=spawn)))
    protocol = {
        "object_usd": "/runtime/operator-object.usd",
        "object_scale": [0.8, 0.9, 1.1],
    }

    _configure_eval_object(cfg, protocol)

    assert spawn.usd_path == "/runtime/operator-object.usd"
    assert spawn.scale == (0.8, 0.9, 1.1)
    assert spawn.semantic_tags == [("class", "pi05_target_object")]


def test_gold_evaluator_remeasures_custom_geometry(monkeypatch) -> None:
    geometry = {
        "source_untransformed_extent_m": [0.073, 0.073, 0.073],
        "effective_local_extent_m": [0.0584, 0.0657, 0.0803],
        "source_untransformed_center_m": [0.0, 0.0, 0.0],
        "declared_spawn_scale": [0.8, 0.9, 1.1],
        "grasp_width_m": 0.0657,
        "support_height_m": 0.0803,
        "grasp_width_axis": "object_local_y",
        "measurement": "UsdGeom.BBoxCache composed runtime prim",
    }
    monkeypatch.setattr(
        pi05_closed_loop, "_object_asset_geometry", lambda _native, _scale: geometry
    )
    protocol = {"object_scale": [0.8, 0.9, 1.1], "object_geometry": geometry}
    assert _validate_eval_geometry(object(), protocol) == geometry

    changed = {**geometry, "grasp_width_m": 0.06}
    monkeypatch.setattr(
        pi05_closed_loop, "_object_asset_geometry", lambda _native, _scale: changed
    )
    with pytest.raises(RuntimeError, match="dimensions differ"):
        _validate_eval_geometry(object(), protocol)

    invalid = {**geometry, "measurement": "unverified guess"}
    with pytest.raises(RuntimeError, match="measurement provenance"):
        _validated_geometry_receipt(invalid, [0.8, 0.9, 1.1])


def test_default_asset_split_is_size_only_on_one_scene_layout() -> None:
    assert {row["scene_id"] for row in SPLITS.values()} == {
        "pi05-single-body-support-layout-v1"
    }
    assert {row["object_scale"] for row in SPLITS.values()} == {
        (0.8, 0.8, 0.8),
        (1.0, 1.0, 1.0),
        (1.2, 1.2, 1.2),
    }
    assert {row["object_identity"] for row in SPLITS.values()} == {
        "isaac-multicolor-cube-048mm",
        "isaac-multicolor-cube-060mm",
        "isaac-multicolor-cube-072mm",
    }


def test_collector_uploads_fail_closed_startup_receipt(monkeypatch) -> None:
    uploaded: dict[str, object] = {}

    class Storage:
        def upload_directory(self, directory: str, output_uri: str) -> None:
            uploaded["uri"] = output_uri
            uploaded["failure"] = json.loads(
                (Path(directory) / "collection-failure.json").read_text()
            )

    def fail_environment(*_args) -> None:
        raise RuntimeError("contact initialization rejected")

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        classmethod(lambda _cls: Storage()),
    )
    monkeypatch.setattr(pi05_isaac, "_graphics_preflight", lambda: {"status": "ok"})
    monkeypatch.setattr(pi05_isaac, "_make_environment", fail_environment)
    with pytest.raises(RuntimeError, match="contact initialization rejected"):
        pi05_isaac.collect(
            output_uri="s3://example/failure/",
            episodes=1,
            seed=7,
            object_identity="cube",
            object_usd="",
            object_scale=(1.0, 1.0, 1.0),
            scene_id="pi05-single-body-support-layout-v1",
        )
    assert uploaded["uri"] == "s3://example/failure/"
    assert uploaded["failure"]["valid_collection"] is False
    assert uploaded["failure"]["training_admitted"] is False
    assert uploaded["failure"]["phase"] == "environment_startup"


def test_collector_persists_reset_traceback_before_kit_shutdown(monkeypatch) -> None:
    events: list[str] = []

    class App:
        def close(self) -> None:
            events.append("app.close")

    class Storage:
        def upload_directory(self, directory: str, _output_uri: str) -> None:
            report = json.loads(
                (Path(directory) / "collection-failure.json").read_text()
            )
            assert "first reset failed" in report["traceback"]
            assert "app.close" not in events
            events.append("upload")

    def fail_after_app(*_args):
        raise pi05_isaac._EnvironmentSetupError(
            RuntimeError("first reset failed"), App()
        )

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        classmethod(lambda _cls: Storage()),
    )
    monkeypatch.setattr(pi05_isaac, "_graphics_preflight", lambda: {"status": "ok"})
    monkeypatch.setattr(pi05_isaac, "_make_environment", fail_after_app)
    with pytest.raises(RuntimeError, match="first reset failed"):
        pi05_isaac.collect(
            output_uri="s3://example/failure/",
            episodes=1,
            seed=7,
            object_identity="cube",
            object_usd="",
            object_scale=(1.0, 1.0, 1.0),
            scene_id="pi05-single-body-support-layout-v1",
        )
    assert events == ["upload", "app.close"]


def test_graphics_preflight_rejects_vulkan_without_nvidia_device(monkeypatch) -> None:
    responses = [
        Namespace(returncode=0, stdout="", stderr=""),
        Namespace(returncode=0, stdout="Vulkan Instance Version: 1.3", stderr=""),
    ]
    monkeypatch.setattr(
        pi05_isaac.subprocess, "run", lambda *_a, **_k: responses.pop(0)
    )
    with pytest.raises(RuntimeError, match="NVIDIA physical device"):
        pi05_isaac._graphics_preflight()


def test_runtime_closes_app_even_when_environment_close_fails() -> None:
    events: list[str] = []

    class Env:
        def close(self) -> None:
            events.append("env.close")
            raise RuntimeError("environment close failed")

    class App:
        def close(self) -> None:
            events.append("app.close")

    with pytest.raises(RuntimeError, match="environment close failed"):
        pi05_isaac._close_runtime(Env(), App())
    assert events == ["env.close", "app.close"]


def test_collection_failure_receipt_redacts_private_uris() -> None:
    report = pi05_isaac._failure_report(
        RuntimeError("asset s3://private-bucket/owned/object failed"),
        "scene",
        "environment_startup",
    )
    assert "private-bucket" not in json.dumps(report)
    assert "<redacted-uri>" in report["failure"]


def test_closed_loop_persists_graphics_failure_before_kit(monkeypatch) -> None:
    uploaded = {}

    class Storage:
        def upload_directory(self, directory, uri, *, require_empty):
            uploaded.update(
                uri=uri,
                require_empty=require_empty,
                report=json.loads((Path(directory) / "report.json").read_text()),
            )

    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        classmethod(lambda _cls: Storage()),
    )
    monkeypatch.setattr(
        pi05_closed_loop,
        "_graphics_preflight",
        lambda: (_ for _ in ()).throw(RuntimeError("Vulkan unavailable")),
    )
    args = Namespace(artifact_root_uri="s3://example/closed-loop-failure/")
    with pytest.raises(RuntimeError, match="Vulkan unavailable"):
        pi05_closed_loop.run(args)
    assert uploaded["require_empty"] is True
    assert uploaded["report"]["phase"] == "graphics_preflight"


def test_collector_main_returns_nonzero_on_runtime_failure(monkeypatch, capsys) -> None:
    def fail_collection(**_kwargs) -> None:
        raise RuntimeError("sensor reset failed")

    monkeypatch.setattr(pi05_isaac, "collect", fail_collection)
    result = pi05_isaac.main(
        [
            "--output-uri",
            "s3://example/failure/",
            "--episodes",
            "1",
            "--seed",
            "7",
            "--object-identity",
            "cube",
            "--scene-id",
            "pi05-single-body-support-layout-v1",
        ]
    )
    assert result == 1
    failure = json.loads(capsys.readouterr().err)
    assert failure["valid_collection"] is False
    assert failure["training_admitted"] is False
    assert failure["phase"] == "collection_runtime"


def _dense_steps(length: int) -> list[dict]:
    steps: list[dict] = []
    for index in range(length):
        timestamp = (index + 1) / 15
        action = TARGET.copy()
        action[7] = 1.0 if 1 <= index < 12 else 0.0
        steps.append(
            {
                "step": index,
                "timestamp_s": timestamp,
                "exterior_timestamp_s": timestamp,
                "wrist_timestamp_s": timestamp,
                "command_application_timestamp_s": timestamp,
                "resulting_state_timestamp_s": timestamp + 1 / 15,
                "joint_position": TARGET[:7].tolist(),
                "gripper_position": [0.0],
                "action": action.tolist(),
                "exterior_image_1_left": {
                    "path": f"obs_workspace.npy#{index}",
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                },
                "wrist_image_left": {
                    "path": f"obs_wrist.npy#{index}",
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                },
            }
        )
    return steps


def _settle_receipt() -> dict:
    return {
        "control_steps": 2,
        "physics_steps": 8,
        "training_row": False,
        "samples": [
            {
                "command_time_s": 0.0,
                "result_time_s": 1.0 / 15.0,
                "object_speed_m_s": 0.08,
                "object_support_force_n": 0.0,
            },
            {
                "command_time_s": 1.0 / 15.0,
                "result_time_s": 2.0 / 15.0,
                "object_speed_m_s": 0.0,
                "object_support_force_n": 0.6,
            },
        ],
    }


def _marker_receipt() -> dict:
    return {
        "source": "isaaclab_noncolliding_cylinder_primitive",
        "center_xy_m": [0.50, 0.18],
        "support_z_m": 0.0,
        "radius_m": 0.05,
        "height_m": 0.001,
        "color_rgb": [0.0, 0.85, 0.85],
        "semantic_label": "pi05_destination_marker",
        "minimum_exterior_semantic_pixels": 16,
        "physics_collision": False,
    }


def _episode_metadata(identity: str, scene: str, length: int) -> dict:
    return {
        "schema": EPISODE_SCHEMA,
        "episode_id": f"episode-{identity}",
        "source_backend": "isaac",
        "object_motion": "physics_only",
        "action_semantics": "absolute_joint_position_plus_normalized_gripper",
        "object_identity": identity,
        "object_width_m": 0.05,
        "object_height_m": 0.05,
        "initial_object_position_m": [0.45, -0.15, 0.025],
        "target_position_m": [0.50, 0.18, 0.025],
        "active_surface_command_m": [0.50, 0.18, 0.025],
        "reset_settle": _settle_receipt(),
        "destination_marker": _marker_receipt(),
        "support_surface": support_surface_contract(),
        "scene_configuration_digest": scene,
        "scene_seed": 41,
        "instruction": "pick and place",
        "steps": _dense_steps(length),
        "success_evidence": {
            "ordered_events": SUCCESS_EVENTS,
            "released_supported_surface_placement": True,
            "stable_steps": 3,
        },
    }


def _write_episode(root: Path, identity: str, scene: str, length: int = 16) -> None:
    episode = root / "episode-000000"
    episode.mkdir(parents=True)
    metadata = _episode_metadata(identity, scene, length)
    frames = np.zeros((length, 224, 224, 3), dtype=np.uint8)
    frames[:, 80:120, 80:120] = sum(identity.encode()) % 255
    wrist_frames = frames[:, :, ::-1]
    _bind_frame_hashes(metadata["steps"], frames, wrist_frames)
    (episode / "episode.json").write_text(json.dumps(metadata))
    np.save(episode / "obs_workspace.npy", frames)
    np.save(episode / "obs_wrist.npy", wrist_frames)
    state = np.r_[TARGET[:7], np.float32(0.0)]
    np.save(episode / "state.npy", np.tile(state, (length, 1)).astype(np.float32))
    actions = np.asarray([row["action"] for row in metadata["steps"]], dtype=np.float32)
    np.save(episode / "actions.npy", actions)
    np.save(episode / "timestamps.npy", np.arange(1, length + 1) / 15)
    _write_physics_trace(episode, length)


def _bind_frame_hashes(
    steps: list[dict], exterior: np.ndarray, wrist: np.ndarray
) -> None:
    for index, row in enumerate(steps):
        exterior_hash = hashlib.sha256(exterior[index].tobytes()).hexdigest()
        wrist_hash = hashlib.sha256(wrist[index].tobytes()).hexdigest()
        sensor_time = np.float32(0.0)
        updates = round(row["timestamp_s"] * 60)
        for _ in range(updates):
            sensor_time = np.float32(sensor_time + np.float32(1.0 / 60.0))
        raw_age = row["timestamp_s"] - float(sensor_time)
        row["exterior_timestamp_s"] = float(sensor_time)
        row["wrist_timestamp_s"] = float(sensor_time)
        timing = {
            "capture_time_s": float(sensor_time),
            "observation_time_s": row["timestamp_s"],
            "age_s": raw_age,
            "sensor_causal_age_s": 0.0,
        }
        row["exterior_image_1_left"].update({"sha256": exterior_hash, **timing})
        row["wrist_image_left"].update({"sha256": wrist_hash, **timing})
        identity = {
            "step": index,
            "exterior": exterior_hash,
            "wrist": wrist_hash,
            "action": row["action"],
        }
        row["row_identity_sha256"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()


def _write_physics_trace(episode: Path, length: int) -> None:
    position = np.tile([0.45, -0.15, 0.025], (length, 1)).astype(np.float32)
    position[1] = [0.45, -0.15, 0.031]
    position[2:12] = [0.50, 0.18, 0.10]
    position[12:] = [0.50, 0.18, 0.025]
    bilateral = np.zeros(length, dtype=np.float32)
    bilateral[:12] = 2.0
    support = np.zeros(length, dtype=np.float32)
    support[12:] = 2.0
    width = np.full(length, 0.05, dtype=np.float32)
    width[0], width[12:] = 0.08, 0.08
    speed = np.full(length, 0.1, dtype=np.float32)
    speed[12:] = 0.0
    hand_distance = np.full(length, 0.04, dtype=np.float32)
    hand_distance[12:] = 0.12
    clearance = np.full(length, 0.008, dtype=np.float32)
    np.savez(
        episode / "physics_trace.npz",
        object_position_m=position,
        object_speed_m_s=speed,
        left_finger_object_force_n=bilateral,
        right_finger_object_force_n=bilateral,
        object_support_force_n=support,
        gripper_width_m=width,
        measured_proximal_finger_clearance_m=clearance,
        hand_object_distance_m=hand_distance,
    )


def test_export_accepts_bound_float32_sensor_age(tmp_path, monkeypatch) -> None:
    roots = {name: tmp_path / name for name in ("train", "validation", "gold")}
    for index, (name, root) in enumerate(roots.items()):
        _write_episode(root, f"cube-{name}", f"scene-{index}")
    row = json.loads((roots["train"] / "episode-000000" / "episode.json").read_text())
    observation = row["steps"][0]["timestamp_s"]
    sensor_time = np.float32(0.0)
    for _ in range(round(observation * 60)):
        sensor_time = np.float32(sensor_time + np.float32(1.0 / 60.0))
    for name in ("exterior_image_1_left", "wrist_image_left"):
        receipt = row["steps"][0][name]
        receipt["capture_time_s"] = float(sensor_time)
        receipt["observation_time_s"] = observation
        receipt["age_s"] = observation - float(sensor_time)
        receipt["sensor_causal_age_s"] = 0.0
    (roots["train"] / "episode-000000" / "episode.json").write_text(json.dumps(row))
    monkeypatch.setitem(
        sys.modules, "npa.adapter.sim_to_lerobot", _fake_lerobot_module()
    )

    report = export_local(
        train_root=roots["train"],
        validation_root=roots["validation"],
        gold_root=roots["gold"],
        output=tmp_path / "output",
    )

    assert report["splits"]["train"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sensor_causal_age_s", 0.02, "sensor age is inconsistent"),
        ("capture_time_s", -1.0, "not bound to row 0"),
    ],
)
def test_export_rejects_inconsistent_camera_clock(
    tmp_path, monkeypatch, field: str, value: float, message: str
) -> None:
    roots = {name: tmp_path / name for name in ("train", "validation", "gold")}
    for index, (name, root) in enumerate(roots.items()):
        _write_episode(root, f"cube-{name}", f"scene-{index}")
    path = roots["train"] / "episode-000000" / "episode.json"
    metadata = json.loads(path.read_text())
    metadata["steps"][0]["wrist_image_left"][field] = value
    path.write_text(json.dumps(metadata))
    monkeypatch.setitem(
        sys.modules, "npa.adapter.sim_to_lerobot", _fake_lerobot_module()
    )

    with pytest.raises(Pi05ContractError, match=message):
        export_local(
            train_root=roots["train"],
            validation_root=roots["validation"],
            gold_root=roots["gold"],
            output=tmp_path / "output",
        )


def test_camera_clock_rejects_bound_but_stale_capture() -> None:
    observation = 1.0 / 15.0
    sensor_time = np.float32(0.0)
    for _ in range(4):
        sensor_time = np.float32(sensor_time + np.float32(1.0 / 60.0))
    capture = float(sensor_time) - 0.1
    receipt = {
        "capture_time_s": capture,
        "observation_time_s": observation,
        "age_s": observation - capture,
        "sensor_causal_age_s": 0.1,
    }
    with pytest.raises(Pi05ContractError, match="causal age is invalid"):
        pi05_data._camera_causal_age(receipt, "wrist_image_left", 0)


def test_export_rejects_self_consistent_receipt_shifted_from_row(
    tmp_path, monkeypatch
) -> None:
    roots = {name: tmp_path / name for name in ("train", "validation", "gold")}
    for index, (name, root) in enumerate(roots.items()):
        _write_episode(root, f"cube-{name}", f"scene-{index}")
    path = roots["train"] / "episode-000000" / "episode.json"
    metadata = json.loads(path.read_text())
    receipt = metadata["steps"][0]["wrist_image_left"]
    shifted_observation = metadata["steps"][1]["timestamp_s"]
    shifted_capture = metadata["steps"][1]["wrist_timestamp_s"]
    receipt["observation_time_s"] = shifted_observation
    receipt["capture_time_s"] = shifted_capture
    receipt["age_s"] = shifted_observation - shifted_capture
    receipt["sensor_causal_age_s"] = 0.0
    path.write_text(json.dumps(metadata))
    monkeypatch.setitem(
        sys.modules, "npa.adapter.sim_to_lerobot", _fake_lerobot_module()
    )

    with pytest.raises(Pi05ContractError, match="not bound to row 0"):
        export_local(
            train_root=roots["train"],
            validation_root=roots["validation"],
            gold_root=roots["gold"],
            output=tmp_path / "output",
        )


def _fake_lerobot_module() -> types.ModuleType:
    module = types.ModuleType("npa.adapter.sim_to_lerobot")

    def convert(source: Path, destination: Path, **_: object) -> Path:
        episodes = sorted(source.glob("episode_*"))
        frames = sum(np.load(path / "state.npy").shape[0] for path in episodes)
        (destination / "meta").mkdir(parents=True)
        info = {
            "codebase_version": "v3.0",
            "total_episodes": len(episodes),
            "total_frames": frames,
            "timestamp_source": "input_episode_relative",
        }
        (destination / "meta" / "info.json").write_text(json.dumps(info))
        return destination

    module.convert = convert
    return module


def test_dense_export_is_split_safe_and_openpi_readable(tmp_path, monkeypatch) -> None:
    roots = {name: tmp_path / name for name in ("train", "validation", "gold")}
    for index, (name, root) in enumerate(roots.items()):
        _write_episode(root, f"cube-{name}", f"scene-{index}")
    monkeypatch.setitem(
        sys.modules, "npa.adapter.sim_to_lerobot", _fake_lerobot_module()
    )
    output = tmp_path / "output"
    report = export_local(
        train_root=roots["train"],
        validation_root=roots["validation"],
        gold_root=roots["gold"],
        output=output,
    )
    arrays, manifest = _load_dataset(
        str(output / "openpi-dataset.npz"), str(output / "openpi-manifest.json")
    )
    assert manifest["schema"] == OPENPI_SCHEMA
    assert arrays["train_actions"].shape == (2, 15, 8)
    assert report["normalization"]["source_split"] == "train"
    assert report["splits"]["gold"][0]["object_identity"] == "cube-gold"


def test_exporter_rejects_invalid_canonical_report_before_publication(
    tmp_path, monkeypatch
) -> None:
    roots = {name: tmp_path / name for name in ("train", "validation", "gold")}
    for index, (name, root) in enumerate(roots.items()):
        _write_episode(root, f"cube-{name}", f"scene-{index}")
    monkeypatch.setitem(
        sys.modules, "npa.adapter.sim_to_lerobot", _fake_lerobot_module()
    )
    original = pi05_data._dataset_report

    def invalid_report(*args, **kwargs):
        report = original(*args, **kwargs)
        report["normalization"]["source_split"] = "validation"
        return report

    monkeypatch.setattr(pi05_data, "_dataset_report", invalid_report)
    output = tmp_path / "output"
    with pytest.raises(Pi05ContractError, match="computed from train only"):
        export_local(
            train_root=roots["train"],
            validation_root=roots["validation"],
            gold_root=roots["gold"],
            output=output,
        )
    assert not (output / "dataset-report.json").exists()


def test_export_rejects_incompatible_seven_dimensional_targets(
    tmp_path, monkeypatch
) -> None:
    roots = {name: tmp_path / name for name in ("train", "validation", "gold")}
    for index, (name, root) in enumerate(roots.items()):
        _write_episode(root, f"cube-{name}", f"scene-{index}")
    actions = roots["train"] / "episode-000000" / "actions.npy"
    np.save(actions, np.zeros((16, 7), dtype=np.float32))
    monkeypatch.setitem(
        sys.modules, "npa.adapter.sim_to_lerobot", _fake_lerobot_module()
    )
    with pytest.raises(Pi05ContractError, match="dense episode arrays"):
        export_local(
            train_root=roots["train"],
            validation_root=roots["validation"],
            gold_root=roots["gold"],
            output=tmp_path / "output",
        )


def test_export_rejects_claimed_success_without_post_retreat_physics(
    tmp_path, monkeypatch
) -> None:
    roots = {name: tmp_path / name for name in ("train", "validation", "gold")}
    for index, (name, root) in enumerate(roots.items()):
        _write_episode(root, f"cube-{name}", f"scene-{index}")
    path = roots["train"] / "episode-000000" / "physics_trace.npz"
    with np.load(path, allow_pickle=False) as loaded:
        trace = {key: loaded[key] for key in loaded.files}
    trace["hand_object_distance_m"][:] = 0.04
    np.savez(path, **trace)
    monkeypatch.setitem(
        sys.modules, "npa.adapter.sim_to_lerobot", _fake_lerobot_module()
    )
    with pytest.raises(Pi05ContractError, match="post-retreat support"):
        export_local(
            train_root=roots["train"],
            validation_root=roots["validation"],
            gold_root=roots["gold"],
            output=tmp_path / "output",
        )


@pytest.mark.parametrize("mode", ["missing", "below_contact", "nonfinite"])
def test_export_requires_measured_proximal_clearance(
    tmp_path, monkeypatch, mode: str
) -> None:
    roots = {name: tmp_path / name for name in ("train", "validation", "gold")}
    for index, (name, root) in enumerate(roots.items()):
        _write_episode(root, f"cube-{name}", f"scene-{index}")
    path = roots["train"] / "episode-000000" / "physics_trace.npz"
    with np.load(path, allow_pickle=False) as loaded:
        trace = {key: loaded[key] for key in loaded.files}
    if mode == "missing":
        trace.pop("measured_proximal_finger_clearance_m")
    elif mode == "below_contact":
        trace["measured_proximal_finger_clearance_m"][0] = 0.0
    else:
        trace["measured_proximal_finger_clearance_m"][0] = np.nan
    np.savez(path, **trace)
    monkeypatch.setitem(
        sys.modules, "npa.adapter.sim_to_lerobot", _fake_lerobot_module()
    )

    with pytest.raises(Pi05ContractError, match="physics trace|proximal"):
        export_local(
            train_root=roots["train"],
            validation_root=roots["validation"],
            gold_root=roots["gold"],
            output=tmp_path / "output",
        )


def _ordered_surface_trace() -> tuple[list[dict], dict]:
    base = {
        "bilateral_finger_contact": False,
        "gripper_closing_intent": False,
        "object_width_m": 0.05,
        "object_lift_m": 0.0,
        "target_xy_distance_m": 0.3,
        "gripper_width_m": 0.07,
        "object_support_force_n": 0.0,
        "object_speed_m_s": 0.0,
        "hand_object_distance_m": 0.02,
    }
    gripping = {
        **base,
        "bilateral_finger_contact": True,
        "gripper_closing_intent": True,
        "gripper_width_m": 0.05,
        "object_speed_m_s": 0.01,
    }
    placed = {
        **base,
        "object_lift_m": 0.0,
        "target_xy_distance_m": 0.01,
        "object_support_force_n": 1.0,
        "hand_object_distance_m": 0.12,
    }
    return [
        {**gripping, "object_speed_m_s": 0.0},
        gripping,
        gripping,
        {**gripping, "object_lift_m": 0.06},
        {**gripping, "object_lift_m": 0.06, "target_xy_distance_m": 0.01},
        placed,
        placed,
    ], placed


def test_surface_success_requires_three_consecutive_supported_release_steps() -> None:
    trace, placed = _ordered_surface_trace()
    assert not _success(trace)["released_supported_surface_placement"]
    trace.append(placed)
    assert not _success(trace)["released_supported_surface_placement"]
    trace.append(placed)
    result = _success(trace)
    assert result["released_supported_surface_placement"]
    assert result["support_contact"] is True
    assert list(result["ordered_event_steps"]) == result["required_event_order"]
    trace[-1] = {**placed, "hand_object_distance_m": 0.05}
    assert not _success(trace)["released_supported_surface_placement"]


def _policy_settle_native():
    import torch

    return Namespace(
        step_dt=1.0 / 15.0,
        cfg=Namespace(decimation=4),
        episode_length_buf=torch.tensor([0]),
        scene={
            "robot": Namespace(
                joint_names=[f"panda_joint{i}" for i in range(1, 8)],
                data=Namespace(joint_pos=torch.tensor([[0.1] * 7])),
            ),
            "object": Namespace(
                data=Namespace(
                    root_pos_w=torch.tensor([[0.45, -0.15, 0.024]]),
                    root_lin_vel_w=torch.zeros((1, 3)),
                )
            ),
        },
    )


def test_policy_reset_settle_holds_measured_joints_with_open_droid_gripper(
    monkeypatch,
) -> None:
    native = _policy_settle_native()
    env = Namespace(unwrapped=native)
    captured = {}

    def apply(_env, target, **_kwargs):
        captured["target"] = target.copy()
        native.episode_length_buf += 1
        return {"terminated": False, "truncated": False}

    monkeypatch.setattr(pi05_closed_loop, "_apply_target", apply)
    monkeypatch.setattr(pi05_isaac, "_contact_forces", lambda _native: (0.0, 0.0, 0.6))
    receipt = _settle_policy_reset(env, ([0.0] * 7, [1.0] * 7, 8, 0, 7))

    assert np.allclose(captured["target"][:7], [0.1] * 7)
    assert captured["target"][7] == 0.0
    assert receipt["physics_steps"] == 8
    assert len(receipt["samples"]) == 2
    assert len(receipt["controller_proofs"]) == 2
    assert receipt["object_speed_m_s"] == 0.0
    assert receipt["training_or_evaluation_row"] is False


def test_policy_phase_requires_ordered_release_before_retreat() -> None:
    initial = {
        "bilateral_finger_contact": False,
        "gripper_closing_intent": False,
        "object_width_m": 0.05,
        "object_lift_m": 0.0,
        "target_xy_distance_m": 0.18,
        "gripper_width_m": 0.08,
        "object_support_force_n": 0.6,
        "object_speed_m_s": 0.0,
        "hand_object_distance_m": 0.2,
    }
    gripping = {
        **initial,
        "bilateral_finger_contact": True,
        "gripper_closing_intent": True,
        "gripper_width_m": 0.05,
        "object_support_force_n": 0.0,
        "object_speed_m_s": 0.01,
        "hand_object_distance_m": 0.02,
    }
    trace = [initial]
    assert _policy_phase({"physics_trace": trace}) == "approach"
    lifted = {**gripping, "object_lift_m": 0.06}
    trace.extend([gripping, gripping, gripping, lifted])
    assert _policy_phase({"physics_trace": trace}) == "transport"
    trace.append({**gripping, "object_lift_m": 0.06, "target_xy_distance_m": 0.01})
    assert _policy_phase({"physics_trace": trace}) == "lower"
    placed = {
        **initial,
        "target_xy_distance_m": 0.01,
        "hand_object_distance_m": 0.12,
    }
    trace.append(placed)
    assert _policy_phase({"physics_trace": trace}) == "release"
    trace.append(placed)
    assert _policy_phase({"physics_trace": trace}) == "retreat"


def _private_checkpoint_manifests() -> dict:
    return {
        "secret": {"metadata": {"name": "secret"}, "data": {}},
        "deployment": {
            "metadata": {"labels": {"app.kubernetes.io/managed-by": "npa-openpi"}},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "env": [],
                                "command": [
                                    "/bin/bash",
                                    "-lc",
                                    "checkpoint_dir=$(python -c 'print(os.environ[\"OPENPI_CHECKPOINT_URI\"])'); "
                                    "exec /opt/venv/bin/python -c 'wrapper'",
                                ],
                            }
                        ]
                    }
                }
            },
        },
    }


def test_service_manifest_adds_private_checkpoint_probe_without_client_job(
    monkeypatch,
) -> None:
    manifests = _private_checkpoint_manifests()
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.invalid")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "redacted")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "redacted")
    _patch_private_s3_checkpoint(
        manifests,
        "s3://private/checkpoint",
        run_id="private-checkpoint",
        namespace="default",
    )
    _inject_checkpoint_probe(manifests)
    shell = manifests["deployment"]["spec"]["template"]["spec"]["containers"][0][
        "command"
    ][2]
    assert "npa-openpi-loaded-checkpoint.json" in shell
    assert "NPA_RESOLVED_CHECKPOINT_DIR" in shell
    assert set(manifests) == {"secret", "deployment"}


def _cleanup_args() -> Namespace:
    values = [
        "cleanup",
        "--run-id=private-checkpoint",
        "--output-uri=file:///tmp/cleanup.json",
        "--component-root-uri=file:///tmp/components",
        "--runtime-image=registry.invalid/openpi@sha256:" + "0" * 64,
        "--checkpoint-uri=s3://private/checkpoint",
        "--checkpoint-role=adapted",
        "--auth-secret=operator-openpi-auth",
        "--expected-gpu-type=RTXPRO6000",
        "--expected-compute-capability=12.0",
        "--gpu-node-selector-value=RTXPRO6000",
    ]
    return openpi_service_lifecycle.build_parser().parse_args(values)


def test_cleanup_manifest_reconstruction_needs_no_s3_credentials(monkeypatch) -> None:
    for name in ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        openpi_service_lifecycle,
        "build_manifests",
        lambda **_kwargs: {
            "deployment": {
                "metadata": {"labels": {"app.kubernetes.io/managed-by": "npa-openpi"}},
                "spec": {"template": {"spec": {"containers": [{"env": []}]}}},
            }
        },
    )
    monkeypatch.setattr(
        openpi_service_lifecycle, "_inject_checkpoint_probe", lambda _manifests: None
    )

    manifests = openpi_service_lifecycle._manifests(
        _cleanup_args(), include_checkpoint_credentials=False
    )

    assert set(manifests) == {"deployment"}
    assert "AWS_ACCESS_KEY_ID" not in str(manifests)


def test_cleanup_reconstructs_owned_runtime_secret_without_credentials(
    monkeypatch,
) -> None:
    args = _cleanup_args()
    deployment = {
        "metadata": {"labels": {"app.kubernetes.io/managed-by": "npa-openpi"}},
        "spec": {"template": {"spec": {"containers": [{"env": []}]}}},
    }
    monkeypatch.setattr(
        openpi_service_lifecycle,
        "_manifests",
        lambda *_args, **_kwargs: {"deployment": deployment, "service": {}},
    )
    monkeypatch.setattr(openpi_service_lifecycle, "_clients", lambda: (1, 2, 3))
    captured = {}

    def delete(*_clients, **kwargs):
        captured.update(kwargs)
        return {name: True for name in kwargs["created"]}

    monkeypatch.setattr(openpi_service_lifecycle, "_delete_and_verify", delete)
    monkeypatch.setattr(
        openpi_service_lifecycle, "_publish_cleanup_result", lambda *_args: {}
    )
    assert openpi_service_lifecycle._cleanup(args) == 0
    assert "secret" in captured["created"]
    runtime_secret = captured["manifests"]["secret"]
    assert runtime_secret["metadata"]["name"].endswith("-runtime")
    assert "data" not in runtime_secret and "stringData" not in runtime_secret


def _install_closed_loop_test_modules(monkeypatch, lifecycle) -> None:
    app_module = types.SimpleNamespace(
        AppLauncher=lambda **_kwargs: types.SimpleNamespace(
            app=_LifecycleApp(lifecycle)
        )
    )
    monkeypatch.setitem(sys.modules, "isaaclab", types.SimpleNamespace(app=app_module))
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_tasks.utils",
        types.SimpleNamespace(parse_env_cfg=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "gymnasium", types.SimpleNamespace())


def test_closed_loop_closes_app_and_temporary_directory_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifecycle = {"app_closed": False, "temp_exited": False}
    _install_closed_loop_test_modules(monkeypatch, lifecycle)
    state = tmp_path / "kit-state"
    state.mkdir()
    monkeypatch.setenv("NPA_ISAAC_STATE_DIR", str(state))
    monkeypatch.setenv("NPA_ISAAC_KIT_ARGS", f"--portable-root {state / 'kit'}")
    monkeypatch.setattr(
        pi05_closed_loop.tempfile,
        "TemporaryDirectory",
        lambda **_kwargs: _TrackingTemporaryDirectory(
            tmp_path / "closed-loop", lifecycle
        ),
    )
    monkeypatch.setattr(
        pi05_closed_loop,
        "_run_from_receipt",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("receipt failed")),
    )
    monkeypatch.setattr(
        pi05_closed_loop,
        "_graphics_preflight",
        lambda: {"status": "passed"},
    )
    monkeypatch.setattr(
        pi05_closed_loop,
        "_persist_evaluation_failure",
        lambda *_args: None,
    )
    with pytest.raises(RuntimeError, match="receipt failed"):
        pi05_closed_loop.run(Namespace())
    assert lifecycle == {"app_closed": True, "temp_exited": True}


def _closed_loop_report_args() -> Namespace:
    return Namespace(
        output_uri="s3://fixture/gold/report.json",
        checkpoint_role="base",
        split="gold",
        episodes=2,
        wrist_ablation_modes=["fresh", "frozen", "pixel_channel_shuffle"],
    )


def _successful_closed_loop_report() -> dict:
    return {
        "schema": "npa.sim2real.pi05.closed_loop_eval.v1",
        "checkpoint_role": "base",
        "split": "gold",
        "episode_count": 6,
        "episodes": [{"scenario_seed": index} for index in range(6)],
        "review_artifacts": {
            "mp4": {"bytes": 10},
            "rrd": {"bytes": 20},
        },
    }


def _rrd_episode() -> dict:
    trace, queries = [], []
    for index, (observation, command, result) in enumerate(
        ((0.133333, 0.133333, 0.2), (0.666667, 0.666667, 0.733333))
    ):
        trace.append(
            {
                "object_position_m": [0.5, 0.1, 0.03],
                "object_speed_m_s": 0.0,
                "target_xy_distance_m": 0.01,
                "simulator_time_s": result,
            }
        )
        queries.append(
            {
                "observation_time_s": observation,
                "exterior_capture_time_s": observation - 2.0e-6,
                "wrist_capture_time_s": observation - 1.0e-6,
                "inference": {"wire_input": {"capture_age_s": 1.0e-6}},
                "controller_proofs": [
                    {
                        "command_application_time_s": command,
                        "resulting_state_time_s": result,
                        "physics_observation_status": "recorded",
                        "physics_trace_index": index,
                    }
                ],
            }
        )
    frames = [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(2)]
    return {
        "physics_trace": trace,
        "queries": queries,
        "_exterior_frames": frames,
        "_wrist_frames": frames,
    }


def test_closed_loop_rrd_is_finalized_and_causally_timed(tmp_path: Path) -> None:
    episodes = [_rrd_episode()]

    rrd, steps = pi05_closed_loop._write_rrd(tmp_path, episodes, "base")

    assert steps == 2
    subprocess.run(
        [sys.executable, "-m", "rerun", "rrd", "verify", str(rrd)],
        check=True,
    )
    decoded = subprocess.run(
        [sys.executable, "-m", "rerun", "rrd", "print", "-vvv", str(rrd)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert decoded.count("* entity_path: /episodes/0/physics/") == 6
    assert decoded.count("* entity_path: /episodes/0/queries/") == 6
    assert all(
        value in decoded
        for value in ("PT0.2S", "PT0.733333S", "PT0.133331S", "PT0.666666S")
    )
    assert not any(
        "physics_step" in line and "query" in line for line in decoded.splitlines()
    )


@pytest.mark.parametrize("mutation", ["legacy", "count", "sample_time"])
def test_closed_loop_rrd_rejects_unbound_physics_rows(
    tmp_path: Path, mutation: str
) -> None:
    episode = _rrd_episode()
    if mutation == "legacy":
        episode["queries"][0]["controller_proofs"][0].pop("physics_observation_status")
    elif mutation == "count":
        episode["physics_trace"].pop()
    else:
        episode["physics_trace"][0]["simulator_time_s"] = 0.3
    with pytest.raises(RuntimeError, match="Rerun"):
        pi05_closed_loop._write_rrd(tmp_path, [episode], "base")


def test_closed_loop_entry_rejects_green_kit_exit_without_success_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _closed_loop_report_args()
    monkeypatch.setattr(pi05_closed_loop_entry, "parse_args", lambda _argv: args)
    monkeypatch.setattr(
        pi05_closed_loop_entry.subprocess,
        "run",
        lambda *_args, **_kwargs: Namespace(returncode=0),
    )
    monkeypatch.setattr(
        pi05_closed_loop_entry,
        "read_json",
        lambda *_args, **_kwargs: {
            "schema": "npa.sim2real.pi05.closed_loop_failure.v1"
        },
    )
    with pytest.raises(RuntimeError, match="absent, failed, or incomplete"):
        pi05_closed_loop_entry.main(["--fixture"])


def test_closed_loop_entry_accepts_only_complete_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _closed_loop_report_args()
    captured = {}
    monkeypatch.setattr(pi05_closed_loop_entry, "parse_args", lambda _argv: args)

    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return Namespace(returncode=0)

    monkeypatch.setattr(pi05_closed_loop_entry.subprocess, "run", run)
    monkeypatch.setattr(
        pi05_closed_loop_entry,
        "read_json",
        lambda uri, **_kwargs: (
            _successful_closed_loop_report() if uri == args.output_uri else {}
        ),
    )
    assert pi05_closed_loop_entry.main(["--fixture"]) == 0
    assert captured["env"]["NPA_PI05_CLOSED_LOOP_CHILD"] == "1"
    assert captured["command"][2] == "npa.workflows.sim2real.pi05_closed_loop"
