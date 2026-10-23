from __future__ import annotations

import copy
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
from npa.workflows.byof.openpi_service import (
    build_controller_rbac_manifests,
    controller_service_account_name,
)
from npa.workflows.sim2real import (
    pi05_closed_loop,
    pi05_closed_loop_entry,
    pi05_collect_all,
    pi05_data,
    pi05_isaac,
    pi05_stage,
)
from npa.workflows.byof.openpi_service_lifecycle import (
    _inject_checkpoint_probe,
    _patch_private_s3_checkpoint,
)
from npa.workflows.sim2real.pi05_closed_loop import (
    _configure_eval_object,
    _legacy_collection_object_binding,
    _policy_phase,
    _resolve_interchange_object,
    _sampler_noise_pairing,
    _settle_policy_reset,
    _success,
    _validate_eval_geometry,
    _validate_eval_object_source,
    _validate_eval_robot_plant,
    _validated_geometry_receipt,
    _validated_robot_plant,
    _wrist_variant,
)
from npa.workflows.sim2real.pi05_contract import (
    EPISODE_SCHEMA,
    Pi05ContractError,
    build_contract,
)
from npa.workflows.sim2real.pi05_collect_all import (
    SPLITS,
    _collector_argv,
    _run_collector,
)
from npa.workflows.sim2real.pi05_data import (
    OPENPI_SCHEMA,
    export_local,
    install_policy_adapter,
)
from npa.workflows.sim2real.pi05_isaac import (
    _collection_report,
    _new_attempt_buffers,
    _publish_episode,
    _save_partial_attempt,
    configure_surface_goal,
    support_surface_contract,
)
from npa.workflows.sim2real.interchange_contract import (
    CameraView,
    InterchangeContractError,
    Transform,
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


def test_pi05_service_rbac_contract_uses_two_exact_owners() -> None:
    run_id = "pi05-rbac"
    args = Namespace(
        run_id=run_id,
        service_namespace="default",
        base_service_account=controller_service_account_name(f"{run_id}-base"),
        adapted_service_account=controller_service_account_name(f"{run_id}-adapted"),
    )
    contract = pi05_stage._service_rbac_contract(args)

    for role in ("base", "adapted"):
        arm = contract[role]
        manifests = build_controller_rbac_manifests(
            run_id=arm["run_id"],
            namespace=arm["namespace"],
            service_account=arm["service_account"],
        )
        assert manifests["role"]["metadata"]["annotations"] == {
            "npa.nebius.ai/cleanup-owner": f"{run_id}-{role}"
        }
        assert (
            manifests["role_binding"]["subjects"][0]["name"] == arm["service_account"]
        )
    assert contract["base"]["service_account"] != contract["adapted"]["service_account"]


def test_pi05_service_rbac_contract_rejects_parent_or_other_arm_account() -> None:
    run_id = "pi05-rbac"
    expected_base = controller_service_account_name(f"{run_id}-base")
    args = Namespace(
        run_id=run_id,
        service_namespace="default",
        base_service_account=expected_base,
        adapted_service_account=expected_base,
    )

    with pytest.raises(RuntimeError, match="adapted service account must be"):
        pi05_stage._service_rbac_contract(args)


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


@pytest.mark.parametrize(
    "width",
    [
        -0.0401,
        0.09502406418323517,
        0.10055141896009445,
        0.10171762108802795,
        0.12018878012895584,
    ],
)
def test_physical_gripper_width_preserves_finite_joint_measurement(
    width: float,
) -> None:
    robot = Namespace(
        joint_names=[
            *pi05_closed_loop.JOINT_NAMES,
            "panda_finger_joint1",
            "panda_finger_joint2",
        ],
        data=Namespace(joint_pos=_JointTensor([[0.0] * 7 + [width / 2] * 2])),
    )
    assert pi05_isaac._gripper_width(
        Namespace(scene={"robot": robot})
    ) == pytest.approx(width)


@pytest.mark.parametrize("width", [float("-inf"), float("inf"), float("nan")])
def test_physical_gripper_width_rejects_non_finite_joint_measurement(
    width: float,
) -> None:
    robot = Namespace(
        joint_names=[
            *pi05_closed_loop.JOINT_NAMES,
            "panda_finger_joint1",
            "panda_finger_joint2",
        ],
        data=Namespace(joint_pos=_JointTensor([[0.0] * 7 + [width / 2] * 2])),
    )
    with pytest.raises(RuntimeError, match="measured physical finger width"):
        pi05_isaac._gripper_width(Namespace(scene={"robot": robot}))


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
    shuffled, receipt = _wrist_variant(left, first, 3)
    repeated, _ = _wrist_variant(right, first, 3)
    rng = np.random.default_rng(14)
    expected = first.reshape(-1)[rng.permutation(first.size)].reshape(first.shape)
    assert np.array_equal(shuffled, repeated)
    assert np.array_equal(shuffled, expected)
    assert receipt["intervention_label"] == (pi05_closed_loop.PIXEL_VALUE_SHUFFLE_LABEL)


def test_terminal_witness_binds_final_state_without_extra_control(monkeypatch) -> None:
    _trace, placed = _ordered_surface_trace()
    rows = [
        {
            **placed,
            "left_finger_object_force_n": 0.0,
            "right_finger_object_force_n": 0.0,
            "simulator_time_s": (index + 1) / 15.0,
        }
        for index in range(pi05_closed_loop.TERMINAL_WITNESS_DWELL_STEPS)
    ]
    record = {"physics_trace": rows}
    frame = np.full((8, 8, 3), 7, dtype=np.uint8)
    native = Namespace(step_dt=1.0 / 60.0)
    final_time = rows[-1]["simulator_time_s"]
    monkeypatch.setattr(pi05_closed_loop, "_rgb", lambda *_: frame)
    monkeypatch.setattr(pi05_closed_loop, "_camera_time", lambda *_: final_time)
    monkeypatch.setattr(pi05_closed_loop, "_sim_time", lambda *_: final_time)
    monkeypatch.setattr(
        pi05_closed_loop,
        "_sensor_clock_reference",
        lambda *_: {
            "sensor_time_s": final_time,
            "expected_observation_time_s": final_time,
            "representation_offset_s": 0.0,
            "reset_epoch": 0,
        },
    )
    monkeypatch.setattr(pi05_closed_loop, "_validate_capture_ages", lambda *_: None)
    monkeypatch.setattr(pi05_closed_loop, "_bind_rgb_clock", lambda *_: None)
    monkeypatch.setattr(pi05_closed_loop, "_target_pixel_count", lambda *_: 64)
    monkeypatch.setattr(pi05_closed_loop, "_destination_pixel_count", lambda *_: 64)
    monkeypatch.setattr(
        pi05_closed_loop,
        "_finger_pixel_counts",
        lambda *_: {"left": 16, "right": 16},
    )

    pi05_closed_loop._capture_terminal_witness(native, record)

    witness = record["terminal_witness_dwell"]["post_control_witness"]
    assert witness["physics_trace_index"] == len(rows) - 1
    assert witness["observation_time_s"] == final_time
    assert witness["additional_action_steps"] == 0
    assert witness["additional_physics_steps"] == 0
    assert (
        record["terminal_witness_dwell"]["verified_terminal_release_and_stability"]
        is True
    )


def test_terminal_witness_rejects_state_time_mismatch(monkeypatch) -> None:
    record = {"physics_trace": [{"simulator_time_s": 1.0}]}
    monkeypatch.setattr(
        pi05_closed_loop, "_rgb", lambda *_: np.zeros((2, 2, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(pi05_closed_loop, "_camera_time", lambda *_: 1.0)
    monkeypatch.setattr(pi05_closed_loop, "_sim_time", lambda *_: 1.1)

    with pytest.raises(RuntimeError, match="final physics state"):
        pi05_closed_loop._capture_terminal_witness(
            Namespace(step_dt=1.0 / 60.0), record
        )


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


def test_policy_query_receipt_survives_controller_failure(monkeypatch) -> None:
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

    def fail_first_target(*_args, **_kwargs):
        raise RuntimeError("controller rejected gs://private/location")

    monkeypatch.setattr(pi05_closed_loop, "_apply_target", fail_first_target)
    monkeypatch.setattr(pi05_closed_loop, "_physics_sample", lambda *_args: {})

    with pytest.raises(RuntimeError, match="controller rejected"):
        pi05_closed_loop._decision_step(context, 0)

    query = record["queries"][0]
    assert query["action_receipt"]["shape"] == [15, 8]
    assert query["action_receipt"]["gripper_scores"] == [-0.2] * 15
    assert query["action_receipt"]["thresholded_gripper_commands"] == [0] * 15
    assert query["execution_status"] == "failed"
    assert query["attempted_prefix"] == 1
    assert query["executed_prefix"] == 0
    assert query["controller_proofs"] == []
    assert query["action_selection"]["contributor_count"] == 1
    assert query["action_selection"]["contributors"][0]["action_index"] == 0
    assert record["physics_trace"] == []
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


def test_temporal_action_ensemble_aligns_only_causal_chunk_targets() -> None:
    context: dict = {}
    selections = []
    for decision in range(3):
        actions = np.zeros((15, 8), dtype=np.float64)
        actions[:, :7] = np.arange(15, dtype=np.float64)[:, None] + decision * 10.0
        if decision == 0:
            actions[2, 7] = 1.0
        elif decision == 1:
            actions[1, 7] = 1.0
        else:
            actions[0, 7] = 1.0
        query = {
            "decision": decision,
            "action_receipt": pi05_closed_loop._action_receipt(actions),
        }
        target, receipt = pi05_closed_loop._temporally_ensembled_target(
            context, actions, query
        )
        selections.append((target, receipt))

    target, receipt = selections[-1]
    assert np.allclose(target[:7], [11.0] * 7)
    assert target[7] == 1.0
    assert receipt["applied_gripper_command"] == 1
    assert receipt["contributor_count"] == 3
    assert [row["origin_decision"] for row in receipt["contributors"]] == [0, 1, 2]
    assert [row["action_index"] for row in receipt["contributors"]] == [2, 1, 0]
    assert receipt["weight_per_contributor"] == 1.0 / 3.0
    assert receipt["fresh_observation_per_control"] is True


def test_temporal_action_ensemble_resets_history_at_episode_boundary() -> None:
    context: dict = {}
    first = np.zeros((15, 8), dtype=np.float64)
    second = np.ones((15, 8), dtype=np.float64)
    for actions in (first, second):
        query = {
            "decision": 0,
            "action_receipt": pi05_closed_loop._action_receipt(actions),
        }
        target, receipt = pi05_closed_loop._temporally_ensembled_target(
            context, actions, query
        )

    assert np.array_equal(target, second[0])
    assert receipt["contributor_count"] == 1
    assert receipt["contributors"] == [
        {
            "origin_decision": 0,
            "action_index": 0,
            "chunk_sha256": query["action_receipt"]["sha256"],
        }
    ]
    assert len(context["action_chunks"]) == 1


def test_temporal_action_ensemble_lineage_rejects_tampering() -> None:
    queries = []
    context: dict = {}
    for decision in range(3):
        actions = np.full((15, 8), decision, dtype=np.float64)
        action_receipt = pi05_closed_loop._action_receipt(actions)
        query = {"decision": decision, "action_receipt": action_receipt}
        _target, selection = pi05_closed_loop._temporally_ensembled_target(
            context, actions, query
        )
        queries.append({**query, "action_selection": selection})
    episode = {"queries": queries}
    pi05_closed_loop._validate_action_selection_history(episode)
    queries[-1]["action_selection"]["contributors"][0]["action_index"] = 0
    with pytest.raises(RuntimeError, match="ensemble lineage"):
        pi05_closed_loop._validate_action_selection_history(episode)


def _paired_noise_episode(mode: str, identity: str | None) -> dict:
    proofs = [
        {
            "terminated": False,
            "truncated": False,
            "physics_observation_status": "recorded",
        }
    ] * pi05_closed_loop.EXECUTION_PREFIX
    return {
        "scenario_seed": 9,
        "wrist_ablation_mode": mode,
        "queries": [
            {
                "decision": 0,
                "execution_status": "completed",
                "attempted_prefix": pi05_closed_loop.EXECUTION_PREFIX,
                "executed_prefix": pi05_closed_loop.EXECUTION_PREFIX,
                "controller_proofs": proofs,
                "inference": {"sampler_noise_sha256": identity},
            }
        ],
    }


def test_effectiveness_rejects_absent_sampler_identity() -> None:

    modes = ["fresh", "frozen", "pixel_channel_shuffle"]
    absent = _sampler_noise_pairing(
        [_paired_noise_episode(mode, None) for mode in modes], modes
    )
    assert absent["sampler_noise_identity_verified"] is False
    paired = _sampler_noise_pairing(
        [_paired_noise_episode(mode, "a" * 64) for mode in modes], modes
    )
    assert paired["sampler_noise_identity_verified"] is True

    failed = [_paired_noise_episode(mode, "a" * 64) for mode in modes]
    failed[0]["queries"][0]["execution_status"] = "failed"
    assert (
        _sampler_noise_pairing(failed, modes)["sampler_noise_identity_verified"]
        is False
    )

    terminal = [_paired_noise_episode(mode, "a" * 64) for mode in modes]
    terminal[0]["queries"][0]["controller_proofs"][0] = {
        "terminated": True,
        "truncated": False,
    }
    assert not _sampler_noise_pairing(terminal, modes)[
        "sampler_noise_identity_verified"
    ]


def _comparison_trace(success: bool) -> list[dict]:
    trace, placed = _ordered_surface_trace()
    trace.extend([placed, placed])
    trace.extend([placed] * (48 - len(trace)))
    if not success:
        trace = [{**row, "hand_object_distance_m": 0.05} for row in trace]
    return [
        {
            **row,
            "left_finger_object_force_n": row.get("left_finger_object_force_n", 0.0),
            "right_finger_object_force_n": row.get("right_finger_object_force_n", 0.0),
            "simulator_time_s": (index + 1) / 15.0,
        }
        for index, row in enumerate(trace)
    ]


def _comparison_proofs(decision: int) -> list[dict]:
    proofs = []
    for local_index in range(pi05_closed_loop.EXECUTION_PREFIX):
        index = decision * pi05_closed_loop.EXECUTION_PREFIX + local_index
        proofs.append(
            {
                "terminated": False,
                "truncated": False,
                "physics_observation_status": "recorded",
                "physics_trace_index": index,
                "command_application_time_s": index / 15.0,
                "resulting_state_time_s": (index + 1) / 15.0,
            }
        )
    return proofs


def _comparison_inference(
    mode: str, seed: int, decision: int, source: str, policy: str
):
    noise = hashlib.sha256(f"noise-{seed}-{decision}".encode()).hexdigest()
    transform_hash = hashlib.sha256(f"transform-{policy}".encode()).hexdigest()
    return {
        "shape": [15, 8],
        "dtype": "float64",
        "finite": True,
        "sampler_noise_sha256": noise,
        "sampler_noise_shape": [15, 32],
        "sampling_sequence": f"scene-{seed}-decision-{decision}",
        "wire_input": {
            "payload_sha256": "d" * 64,
            "exterior_rgb_sha256": "e" * 64,
            "wrist_rgb_sha256": policy,
            "wrist_source_hash_matched": True,
            "capture_age_s": 0.0,
        },
        "model_wrist_transform": {
            "schema": "npa.openpi.pi05-live-wrist-transform.v1",
            "source_wrist_sha256": policy,
            "left_wrist_0_rgb_sha256": transform_hash,
            "right_wrist_0_rgb_sha256": "0" * 64,
            "left_wrist_mask": True,
            "right_wrist_mask": False,
        },
        "wrist_intervention": {
            "mode": mode,
            "intervention_label": (
                pi05_closed_loop.PIXEL_VALUE_SHUFFLE_LABEL
                if mode == "pixel_channel_shuffle"
                else mode
            ),
            "decision": decision,
            "source_sha256": source,
            "policy_sha256": policy,
        },
    }


def _comparison_query(mode: str, seed: int, decision: int) -> dict:
    source = hashlib.sha256(f"source-{seed}-{decision}".encode()).hexdigest()
    frozen = hashlib.sha256(f"source-{seed}-0".encode()).hexdigest()
    policy = frozen if mode == "frozen" else source
    if mode == "pixel_channel_shuffle":
        policy = hashlib.sha256(f"shuffle-{seed}-{decision}".encode()).hexdigest()
    return {
        "decision": decision,
        "observation_time_s": (decision * pi05_closed_loop.EXECUTION_PREFIX / 15.0),
        "wrist_rgb_sha256": source,
        "exterior_rgb_sha256": "e" * 64,
        "witness_rgb_sha256": "f" * 64,
        "execution_status": "completed",
        "attempted_prefix": pi05_closed_loop.EXECUTION_PREFIX,
        "executed_prefix": pi05_closed_loop.EXECUTION_PREFIX,
        "controller_proofs": _comparison_proofs(decision),
        "action_receipt": _comparison_action_receipt(),
        "action_selection": _comparison_action_selection(decision),
        "inference": _comparison_inference(mode, seed, decision, source, policy),
    }


def _comparison_action_receipt() -> dict:
    return pi05_closed_loop._action_receipt(np.zeros((15, 8), dtype=np.float64))


def _comparison_action_selection(decision: int) -> dict:
    chunk_sha256 = _comparison_action_receipt()["sha256"]
    origins = range(max(0, decision - 14), decision + 1)
    contributors = [
        {
            "origin_decision": origin,
            "action_index": decision - origin,
            "chunk_sha256": chunk_sha256,
        }
        for origin in origins
    ]
    return {
        **pi05_closed_loop._action_selection_protocol(),
        "decision": decision,
        "contributor_count": len(contributors),
        "contributors": contributors,
        "weight_per_contributor": 1.0 / len(contributors),
        "applied_action_sha256": hashlib.sha256(
            np.zeros(8, dtype=np.float64).tobytes()
        ).hexdigest(),
        "applied_action_dtype": "float64",
        "applied_action_shape": [8],
        "applied_action": [0.0] * 8,
        "applied_gripper_score": 0.0,
        "applied_gripper_command": 0,
    }


def _comparison_episode(mode: str, seed: int, success: bool) -> dict:
    trace = _comparison_trace(success)
    visibility = {
        phase: {
            "samples": 8,
            "wrist_visible_samples": 8,
            "exterior_visible_samples": 8,
            "destination_exterior_visible_samples": 8,
            "both_fingers_rendered_samples": 8,
            "witness_visible_samples": 8,
            "destination_witness_visible_samples": 8,
            "both_fingers_witness_samples": 8,
        }
        for phase in ("approach", "grasp", "lift", "transport", "lower", "release")
    }
    camera = {
        "wrist_world_motion_m": 0.1,
        "hand_world_motion_m": 0.1,
        "wrist_renderer_world_motion_m": 0.1,
        "wrist_mount_distance_drift_m": 0.0,
        "wrist_mount_angular_drift_rad": 0.0,
        "exterior_world_drift_m": 0.0,
        "wrist_renderer_position_error_m": 0.0,
        "wrist_renderer_orientation_error_rad": 0.0,
        "phase_visibility": visibility,
    }
    episode = {
        "scenario_seed": seed,
        "wrist_ablation_mode": mode,
        "queries": [_comparison_query(mode, seed, decision) for decision in range(48)],
        "physics_trace": trace,
        "camera_evidence": camera,
        "success": _success(trace),
    }
    final_time = trace[-1]["simulator_time_s"]
    physics_updates = len(trace) * 4
    sensor_time = pi05_closed_loop._accumulated_sensor_time(physics_updates, 1.0 / 60.0)
    witness = {
        "sha256": "f" * 64,
        "capture_time_s": sensor_time,
        "observation_time_s": final_time,
        "representation_offset_s": sensor_time - final_time,
        "sensor_causal_age_s": 0.0,
        "sensor_clock": {
            "expected_sensor_time_s": sensor_time,
            "physics_update_count": physics_updates,
            "physics_period_s": 1.0 / 60.0,
            "physics_updates_per_control_step": 4,
            "control_step_count": len(trace),
            "control_period_s": 1.0 / 15.0,
            "expected_observation_time_s": final_time,
            "reset_epoch": 0,
        },
        "capture_semantics": "post_final_control_no_additional_action_or_physics",
        "additional_action_steps": 0,
        "additional_physics_steps": 0,
        "physics_trace_index": len(trace) - 1,
        "final_state_time_s": final_time,
        "target_pixels": 64,
        "destination_pixels": 64,
        "finger_pixels": {"left": 16, "right": 16},
    }
    episode["terminal_witness_dwell"] = pi05_closed_loop._terminal_dwell_receipt(
        episode, witness
    )
    return episode


def test_camera_evidence_accepts_declared_retreat_occlusion_only() -> None:
    episode = _comparison_episode("frozen", 101, False)
    visibility = episode["camera_evidence"]["phase_visibility"]
    retreat = visibility.pop("release")
    retreat.update(
        destination_exterior_visible_samples=0,
        occlusion_permitted=True,
    )
    visibility["retreat"] = retreat

    pi05_closed_loop._validate_camera_evidence(episode)
    retreat["occlusion_permitted"] = False
    with pytest.raises(RuntimeError, match="destination marker"):
        pi05_closed_loop._validate_camera_evidence(episode)


@pytest.mark.parametrize("mutation", ["extra_step", "future_capture", "wrong_row"])
def test_terminal_witness_rejects_unbound_evidence(mutation: str) -> None:
    episode = _comparison_episode("fresh", 101, True)
    witness = episode["terminal_witness_dwell"]["post_control_witness"]
    if mutation == "extra_step":
        witness["additional_physics_steps"] = 1
    elif mutation == "future_capture":
        witness["capture_time_s"] = (
            witness["sensor_clock"]["expected_sensor_time_s"] + 0.1
        )
    else:
        witness["physics_trace_index"] -= 1

    with pytest.raises(RuntimeError, match="terminal witness dwell evidence"):
        pi05_closed_loop._validate_terminal_witness(episode)


def _comparison_geometry() -> dict:
    return {
        "source_untransformed_extent_m": [0.06, 0.06, 0.06],
        "effective_local_extent_m": [0.072, 0.072, 0.072],
        "source_untransformed_center_m": [0.0, 0.0, 0.0],
        "declared_spawn_scale": [1.2, 1.2, 1.2],
        "grasp_width_axis": "object_local_y",
        "measurement": "UsdGeom.BBoxCache composed runtime prim",
        "grasp_width_m": 0.072,
        "support_height_m": 0.072,
    }


def test_legacy_collection_object_binding_is_exact_and_disclosed() -> None:
    collection = {"task_contract": build_contract()}
    asset = {
        "family": "isaac-multicolor-cube",
        "identity": "isaac-multicolor-cube-072mm",
        "usd": "task_default",
    }

    binding = _legacy_collection_object_binding(collection, asset)

    assert binding == {
        "authority": "sealed_legacy_collection",
        "runtime_asset": ("Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"),
        "collection_contract_sha256": collection["task_contract"]["contract_sha256"],
        "source_hashes_available": False,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("family", "another-family"),
        ("identity", "another-object"),
        ("usd", "s3://example/another.usda"),
    ],
)
def test_legacy_collection_object_binding_rejects_different_assets(
    key: str, value: str
) -> None:
    asset = {
        "family": "isaac-multicolor-cube",
        "identity": "isaac-multicolor-cube-072mm",
        "usd": "task_default",
        key: value,
    }

    with pytest.raises(RuntimeError, match="not the canonical cube"):
        _legacy_collection_object_binding({"task_contract": build_contract()}, asset)


def test_interchange_object_rejects_partial_collection_provenance(
    tmp_path: Path,
) -> None:
    asset = {
        "family": "isaac-multicolor-cube",
        "identity": "isaac-multicolor-cube-072mm",
        "usd": "task_default",
        "profile_identity": "partial-profile",
    }

    with pytest.raises(RuntimeError, match="partial object source provenance"):
        _resolve_interchange_object({}, asset, {}, {}, tmp_path)


def test_eval_object_source_requires_exact_legacy_runtime_asset() -> None:
    protocol = {
        "object_binding": {
            "authority": "sealed_legacy_collection",
            "runtime_asset": pi05_closed_loop.LEGACY_DEFAULT_OBJECT_ASSET,
        }
    }
    native = Namespace(
        scene={
            "object": Namespace(
                cfg=Namespace(
                    spawn=Namespace(
                        usd_path=(
                            "omniverse://assets/Isaac/Props/Blocks/"
                            "DexCube/dex_cube_instanceable.usd"
                        )
                    )
                )
            )
        }
    )

    assert _validate_eval_object_source(native, protocol).endswith(
        "Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"
    )
    native.scene["object"].cfg.spawn.usd_path = (
        "omniverse://assets/Isaac/Props/Blocks/"
        "MultiColorCube/multi_color_cube_instanceable.usd"
    )
    with pytest.raises(RuntimeError, match="legacy object asset"):
        _validate_eval_object_source(native, protocol)


def _comparison_scenarios() -> list[dict]:
    return [
        {
            "scene_seed": seed,
            "initial_object_position_m": [0.45, -0.15, 0.036],
            "target_position_m": [0.5, 0.18, 0.036],
            "destination_marker": _marker_receipt(),
            "support_surface": support_surface_contract(),
        }
        for seed in (20281007, 20281008)
    ]


def _comparison_protocol() -> dict:
    protocol = {
        "task": "released_supported_surface_placement_v1",
        "split": "gold",
        "seed": 20281007,
        "episodes": 2,
        "decisions": 48,
        "object_identity": "multicolor-cube-72mm",
        "object_usd": "task_default",
        "object_scale": [1.2, 1.2, 1.2],
        "object_width_m": 0.072,
        "object_height_m": 0.072,
        "object_geometry": _comparison_geometry(),
        "scene_id": "pi05-single-body-support-layout-v1",
        "destination_marker_contract": _marker_receipt(),
        "support_surface_contract": support_surface_contract(),
        "robot_plant": _robot_plant(),
        "control_hz": 15,
        "execution_prefix": pi05_closed_loop.EXECUTION_PREFIX,
        "action_selection": pi05_closed_loop._action_selection_protocol(),
        "scenarios": _comparison_scenarios(),
    }
    pi05_closed_loop._bind_protocol_hashes(protocol)
    return protocol


def _effectiveness_report(role: str, successes: dict[str, set[int]]) -> dict:
    modes = list(pi05_closed_loop.WRIST_ABLATION_MODES)
    seeds = (20281007, 20281008)
    episodes = [
        _comparison_episode(mode, seed, seed in successes.get(mode, set()))
        for mode in modes
        for seed in seeds
    ]
    args = Namespace(checkpoint_role=role, split="gold", wrist_ablation_modes=modes)
    artifacts = {
        "mp4": {"bytes": 10, "frames": 288, "ffprobe": [{"codec_name": "h264"}]},
        "witness_mp4": {
            "bytes": 10,
            "frames": 294,
            "ffprobe": [{"codec_name": "h264"}],
            "policy_input": False,
            "replay": False,
        },
        "rrd": {
            "bytes": 20,
            "timeline_rows": 288,
            "rerun_verify": True,
            "decoded_expected_entity": True,
        },
    }
    protocol = _comparison_protocol()
    receipt = {
        "source_commit": "c" * 40,
        "config_name": "pi05_droid",
        "checkpoint_role": role,
        "checkpoint_sha256": ("1" if role == "base" else "4") * 64,
        "checkpoint_manifest_sha256": ("2" if role == "base" else "5") * 64,
        "image_digest": "3" * 64,
    }
    return pi05_closed_loop._result(args, episodes, artifacts, protocol, receipt)


def test_smolvla_uses_current_native_action_without_fake_noise_pairing() -> None:
    actions = np.arange(8, dtype=np.float64).reshape(1, 8)
    action_receipt = pi05_closed_loop._action_receipt(actions)
    target, selection = pi05_closed_loop._current_action_target(
        actions, {"decision": 4, "action_receipt": action_receipt}
    )
    assert target.tolist() == list(range(8))
    assert selection["method"] == "native_policy_queue_current_action"
    assert selection["contributor_count"] == 1


def _clock_native(control_steps: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        episode_length_buf=np.asarray([control_steps], dtype=np.int64),
        cfg=types.SimpleNamespace(decimation=4),
        physics_dt=1.0 / 60.0,
        step_dt=1.0 / 15.0,
    )


def _typed_clock_receipt(
    native: types.SimpleNamespace, *, capture_offset_s: float = 0.0
) -> dict:
    clock = pi05_isaac._sensor_clock_reference(native)
    observation = pi05_isaac._sim_time(native)
    capture = clock["expected_sensor_time_s"] + capture_offset_s
    receipt = pi05_isaac._rgb_receipt(
        np.zeros((2, 2, 3), dtype=np.uint8), capture, observation
    )
    pi05_isaac._bind_rgb_clock(receipt, clock)
    return receipt


def _typed_test_camera(maximum_skew: float = 0.1) -> CameraView:
    return CameraView(
        key="observation.images.wrist",
        role="wrist",
        mount_link="panda_hand",
        mount_from_camera=Transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        intrinsics=(100.0, 100.0, 1.0, 1.0),
        image_size=(2, 2),
        layout="HWC",
        mask_key="observation.images.wrist_is_pad",
        capture_timestamp_key="observation.images.wrist_capture_timestamp_s",
        state_timestamp_key="observation.state_timestamp_s",
        maximum_state_skew_s=maximum_skew,
    )


def test_smolvla_wire_uses_validated_sensor_clock_and_preserves_raw_times() -> None:
    receipt = _typed_clock_receipt(_clock_native(2))
    camera = _typed_test_camera()

    payload, provenance = pi05_closed_loop._typed_camera_payload(
        camera, np.zeros((2, 2, 3), dtype=np.uint8), receipt
    )

    state = receipt["sensor_clock"]["expected_sensor_time_s"]
    observation = receipt["observation_time_s"]
    assert payload["capture_timestamp_s"] == state
    assert payload["state_timestamp_s"] == state
    assert provenance["raw_observation_timestamp_s"] == observation
    assert provenance["representation_offset_s"] == state - observation
    camera.validate_timestamps(
        capture_s=payload["capture_timestamp_s"],
        state_s=payload["state_timestamp_s"],
    )


@pytest.mark.parametrize(
    ("control_steps", "elapsed_s"), [(15, 1.0), (75, 5.0), (450, 30.0)]
)
def test_smolvla_wire_tracks_accumulated_clock_mapping_for_full_episode(
    control_steps: int, elapsed_s: float
) -> None:
    receipt = _typed_clock_receipt(_clock_native(control_steps))
    payload, provenance = pi05_closed_loop._typed_camera_payload(
        _typed_test_camera(), np.zeros((2, 2, 3), dtype=np.uint8), receipt
    )

    assert receipt["observation_time_s"] == elapsed_s
    assert payload["state_timestamp_s"] == receipt["capture_time_s"]
    assert provenance["representation_offset_s"] == (
        payload["state_timestamp_s"] - elapsed_s
    )
    if elapsed_s == 30.0:
        assert abs(provenance["representation_offset_s"]) > 0.0002


def test_smolvla_wire_tracks_sensor_clock_reset_epoch() -> None:
    native = _clock_native(450)
    before_reset = _typed_clock_receipt(native)
    native.episode_length_buf[0] = 0
    at_reset = _typed_clock_receipt(native)
    native.episode_length_buf[0] = 75
    after_reset = _typed_clock_receipt(native)

    assert before_reset["sensor_clock"]["reset_epoch"] == 0
    assert at_reset["sensor_clock"]["reset_epoch"] == 1
    assert after_reset["sensor_clock"]["reset_epoch"] == 1
    payload, provenance = pi05_closed_loop._typed_camera_payload(
        _typed_test_camera(), np.zeros((2, 2, 3), dtype=np.uint8), after_reset
    )
    assert payload["state_timestamp_s"] == after_reset["capture_time_s"]
    assert provenance["reset_epoch"] == 1


def test_smolvla_wire_rejects_camera_beyond_sensor_clock_tolerance() -> None:
    receipt = _typed_clock_receipt(_clock_native(2), capture_offset_s=2.0e-7)

    with pytest.raises(RuntimeError, match="not bound to the shared sensor clock"):
        pi05_closed_loop._typed_camera_payload(
            _typed_test_camera(), np.zeros((2, 2, 3), dtype=np.uint8), receipt
        )


def test_smolvla_wire_retains_typed_stale_camera_rejection() -> None:
    native = _clock_native(15)
    clock = pi05_isaac._sensor_clock_reference(native)
    with pytest.raises(RuntimeError, match="stale relative to the causal observation"):
        pi05_isaac._validate_capture_ages(
            (clock["expected_sensor_time_s"] - 0.2,) * 2,
            clock,
            native.step_dt,
        )
    receipt = _typed_clock_receipt(native, capture_offset_s=-0.2)
    camera = _typed_test_camera(maximum_skew=0.1)
    payload, _ = pi05_closed_loop._typed_camera_payload(
        camera, np.zeros((2, 2, 3), dtype=np.uint8), receipt
    )

    with pytest.raises(
        InterchangeContractError, match="stale relative to measured state"
    ):
        camera.validate_timestamps(
            capture_s=payload["capture_timestamp_s"],
            state_s=payload["state_timestamp_s"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.pop("sensor_clock"), "lacks validated"),
        (
            lambda receipt: receipt.update(sensor_causal_age_s="invalid"),
            "malformed sensor clock",
        ),
        (
            lambda receipt: receipt["sensor_clock"].update(physics_period_s=0.0),
            "clock periods must be positive",
        ),
        (
            lambda receipt: receipt["sensor_clock"].update(physics_update_count=61),
            "counters are inconsistent",
        ),
        (
            lambda receipt: receipt["sensor_clock"].update(
                expected_observation_time_s=2.0
            ),
            "control-clock mapping is inconsistent",
        ),
        (
            lambda receipt: receipt["sensor_clock"].update(expected_sensor_time_s=2.0),
            "sensor-clock accumulation is inconsistent",
        ),
        (
            lambda receipt: receipt.update(representation_offset_s=0.0),
            "sensor-clock evidence is inconsistent",
        ),
    ],
)
def test_smolvla_wire_rejects_missing_or_malformed_clock_evidence(
    mutation, message: str
) -> None:
    receipt = _typed_clock_receipt(_clock_native(15))
    mutation(receipt)

    with pytest.raises(RuntimeError, match=message):
        pi05_closed_loop._typed_camera_payload(
            _typed_test_camera(), np.zeros((2, 2, 3), dtype=np.uint8), receipt
        )


def test_smolvla_report_discloses_repeated_trials_not_noise_coupling() -> None:
    modes = list(pi05_closed_loop.WRIST_ABLATION_MODES)
    episodes = [
        _comparison_episode(mode, seed, False)
        for mode in modes
        for seed in (20281007, 20281008)
    ]
    args = Namespace(
        checkpoint_role="base",
        split="gold",
        wrist_ablation_modes=modes,
        policy_backend="lerobot_smolvla",
    )
    receipt = {
        "schema": "npa.sim2real.smolvla_service_start.v1",
        "source_commit": "c" * 40,
        "checkpoint_role": "base",
        "runtime_image": "registry.example/smolvla@sha256:" + "d" * 64,
        "interchange": {
            "policy_identity": "smolvla-franka",
            "checkpoint_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "normalization_identity": "e" * 64,
        },
    }
    result = pi05_closed_loop._result(
        args,
        episodes,
        {"mp4": {"bytes": 1}, "rrd": {"bytes": 1}},
        _comparison_protocol(),
        receipt,
    )
    assert result["schema"] == ("npa.sim2real.policy_interchange.closed_loop_eval.v1")
    ablation = result["wrist_ablation"]
    assert ablation["sampler_noise_identity_verified"] is False
    assert ablation["paired_trial_protocol"].startswith("independently_seeded")
    assert ablation["qualification"] == (
        "matched_repeated_trials_without_noise_coupling"
    )


def test_final_comparison_uses_validated_fresh_wrist_metric() -> None:
    baseline = _effectiveness_report("base", {"frozen": {20281007, 20281008}})
    adapted = _effectiveness_report("adapted", {"fresh": {20281007}})

    comparison = pi05_stage._comparison(baseline, adapted)

    assert baseline["success_rate"] > adapted["success_rate"]
    assert comparison["evaluation_contract_valid"] is True
    assert comparison["baseline_success_rate"] == 0.0
    assert comparison["adapted_success_rate"] == 0.5
    assert comparison["measured_improvement"] == 0.5
    assert comparison["improvement_claimed"] is True


@pytest.mark.parametrize("mutation", [None, "normalization", "base_family", "load"])
def test_final_comparison_binds_changed_lora_configuration_to_actual_load(
    mutation,
) -> None:
    baseline = _effectiveness_report("base", {})
    adapted = _effectiveness_report("adapted", {"fresh": {20281007}})
    provenance = adapted["checkpoint_provenance"]
    provenance.update(
        config_name="npa_pi05_franka_mini_lora",
        base_config_name="pi05_droid",
        selected_policy={
            "loaded": True,
            "saved_step": 1000,
            "config_name": "npa_pi05_franka_mini_lora",
            "dataset_archive_sha256": "a" * 64,
            "paligemma_variant": "gemma_2b_lora",
            "action_expert_variant": "gemma_300m_lora",
            "action_horizon": 15,
            "normalization_source": "selected_step_assets_verified_against_training_only",
        },
    )
    if mutation == "normalization":
        provenance["selected_policy"]["normalization_source"] = "base_asset"
    elif mutation == "base_family":
        provenance["base_config_name"] = "unrelated_policy"
    elif mutation == "load":
        provenance["selected_policy"]["loaded"] = False

    comparison = pi05_stage._comparison(baseline, adapted)

    assert comparison["comparison_protocol_identical"] is (mutation is None)
    assert comparison["improvement_claimed"] is (mutation is None)
    assert comparison["measured_improvement"] == (0.5 if mutation is None else None)
    assert (
        comparison["policy_configuration_names"]["adapted"] == provenance["config_name"]
    )


def _mutate_effectiveness_report(report: dict, mutation: str) -> dict:
    invalid = copy.deepcopy(report)
    if mutation == "execution":
        invalid["episodes"][0]["queries"][0]["executed_prefix"] = 7
    elif mutation == "noise":
        invalid["episodes"][0]["queries"][0]["inference"]["sampler_noise_sha256"] = (
            "e" * 64
        )
    elif mutation == "camera":
        invalid["episodes"][0].pop("camera_evidence")
    elif mutation == "artifact":
        invalid["review_artifacts"]["rrd"]["rerun_verify"] = False
    elif mutation == "intervention":
        invalid["episodes"][0]["queries"][0]["inference"]["wrist_intervention"][
            "mode"
        ] = "frozen"
    elif mutation == "summary":
        invalid["wrist_ablation"]["per_mode"]["fresh"]["success_rate"] = 1.0
    elif mutation == "protocol":
        invalid["protocol"]["scenario_sha256"] = []
    elif mutation == "physics":
        invalid["episodes"][0]["physics_trace"] = [
            {**row, "hand_object_distance_m": 0.05}
            for row in invalid["episodes"][0]["physics_trace"]
        ]
    elif mutation == "wire":
        invalid["episodes"][0]["queries"][0]["inference"]["wire_input"][
            "wrist_rgb_sha256"
        ] = "9" * 64
    else:
        invalid["episodes"][0]["queries"][0]["inference"]["model_wrist_transform"][
            "left_wrist_mask"
        ] = False
    return invalid


@pytest.mark.parametrize(
    "mutation",
    [
        "execution",
        "noise",
        "camera",
        "artifact",
        "intervention",
        "summary",
        "protocol",
        "physics",
        "wire",
        "transform",
    ],
)
def test_final_comparison_rejects_invalid_effectiveness_report(mutation: str) -> None:
    baseline = _effectiveness_report("base", {})
    adapted = _effectiveness_report("adapted", {"fresh": {20281007}})
    invalid = _mutate_effectiveness_report(adapted, mutation)

    comparison = pi05_stage._comparison(baseline, invalid)

    assert comparison["evaluation_contract_valid"] is False
    assert comparison["comparison_protocol_identical"] is False
    assert comparison["measured_improvement"] is None
    assert comparison["improvement_claimed"] is False
    assert comparison["evaluation_contract_rejections"]["adapted"]


def test_final_comparison_rejects_rehashed_different_task() -> None:
    baseline = _effectiveness_report("base", {})
    adapted = _effectiveness_report("adapted", {"fresh": {20281007}})
    adapted["protocol"].pop("sha256")
    adapted["protocol"].pop("scenario_sha256")
    adapted["protocol"]["task"] = "different_task"
    pi05_closed_loop._bind_protocol_hashes(adapted["protocol"])

    comparison = pi05_stage._comparison(baseline, adapted)

    assert comparison["evaluation_contract_valid"] is False
    assert comparison["improvement_claimed"] is False


@pytest.mark.parametrize("key", ["source_commit", "config_name", "image_digest"])
def test_final_comparison_rejects_different_evaluator_runtime(key: str) -> None:
    baseline = _effectiveness_report("base", {})
    adapted = _effectiveness_report("adapted", {"fresh": {20281007}})
    adapted["checkpoint_provenance"][key] = (
        "6" * 40 if key == "source_commit" else "6" * 64
    )

    comparison = pi05_stage._comparison(baseline, adapted)

    assert comparison["evaluation_contract_valid"] is True
    assert comparison["comparison_protocol_identical"] is False
    assert comparison["improvement_claimed"] is False


def test_final_comparison_rejects_same_checkpoint_with_new_manifest() -> None:
    baseline = _effectiveness_report("base", {})
    adapted = _effectiveness_report("adapted", {"fresh": {20281007}})
    adapted["checkpoint_provenance"]["checkpoint_sha256"] = baseline[
        "checkpoint_provenance"
    ]["checkpoint_sha256"]

    comparison = pi05_stage._comparison(baseline, adapted)

    assert comparison["evaluation_contract_valid"] is True
    assert comparison["comparison_protocol_identical"] is False
    assert comparison["improvement_claimed"] is False


@pytest.mark.parametrize("mutation", ["noise_shape", "frozen_hash", "camera_nan"])
def test_final_comparison_rejects_invalid_paired_evidence(mutation: str) -> None:
    baseline = _effectiveness_report("base", {})
    adapted = _effectiveness_report("adapted", {"fresh": {20281007}})
    episode = adapted["episodes"][2]
    if mutation == "noise_shape":
        episode["queries"][0]["inference"]["sampler_noise_shape"] = [1, 1]
    elif mutation == "frozen_hash":
        lineage = episode["queries"][1]["inference"]
        lineage["wrist_intervention"]["policy_sha256"] = "7" * 64
        lineage["wire_input"]["wrist_rgb_sha256"] = "7" * 64
        lineage["model_wrist_transform"]["source_wrist_sha256"] = "7" * 64
    else:
        episode["camera_evidence"]["wrist_world_motion_m"] = float("nan")

    comparison = pi05_stage._comparison(baseline, adapted)

    assert comparison["evaluation_contract_valid"] is False
    assert comparison["improvement_claimed"] is False


@pytest.mark.parametrize("mutation", ["zero_fingers", "missing_release"])
def test_final_comparison_requires_fresh_manipulation_visibility(mutation: str) -> None:
    baseline = _effectiveness_report("base", {})
    adapted = _effectiveness_report("adapted", {"fresh": {20281007}})
    visibility = adapted["episodes"][0]["camera_evidence"]["phase_visibility"]
    if mutation == "zero_fingers":
        visibility["grasp"]["both_fingers_rendered_samples"] = 0
    else:
        visibility.pop("release")

    comparison = pi05_stage._comparison(baseline, adapted)

    assert comparison["evaluation_contract_valid"] is False
    assert comparison["improvement_claimed"] is False


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
        "/isaac-sim/python.sh",
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


def test_interchange_collector_passes_verified_object_identity_to_isaac() -> None:
    args = Namespace(
        output_root_uri="s3://example/experts",
        episodes_per_split=3,
        seed=41,
        object_usd="",
        preparation_uri="s3://example/preparation.json",
    )
    receipt = {
        "profile_identity": "offset-spool-v1",
        "source_uri": "s3://example/assets/offset-spool.usda",
        "source_sha256": "a" * 64,
        "bundle_sha256": "b" * 64,
    }
    prepared = Namespace(
        local_path=Path("/task-owned/offset-spool.usda"),
        receipt=lambda: receipt,
    )

    argv = _collector_argv(args, "gold", prepared)

    assert argv[argv.index("--object-usd") + 1] == str(prepared.local_path)
    for key, value in receipt.items():
        flag = f"--object-{key.replace('_', '-')}"
        assert argv[argv.index(flag) + 1] == value


def test_collector_propagates_isaac_launcher_failure(monkeypatch) -> None:
    args = Namespace(
        output_root_uri="s3://example/experts",
        episodes_per_split=3,
        seed=41,
        object_usd="",
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return Namespace(returncode=69, stderr="no ready Isaac cache", stdout="")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="train collector failed with exit 69"):
        _run_collector(args, "train")
    assert len(commands) == 1
    assert commands[0][:3] == [
        "/isaac-sim/python.sh",
        "-m",
        "npa.workflows.sim2real.pi05_isaac",
    ]


def _collector_readback_fixture(tmp_path: Path) -> tuple[Namespace, dict]:
    config = SPLITS["train"]
    identity = config["object_identity"]
    _write_episode(tmp_path, identity, "scene")
    attempt = json.loads((tmp_path / "episode-000000/episode.json").read_text())
    args = Namespace(
        output_root_uri="s3://example/experts",
        episodes_per_split=1,
        seed=41,
        object_usd="",
    )
    report = _collection_report(
        [attempt], identity, "", config["object_scale"], config["scene_id"], {}, {}
    )
    return args, report


def test_collector_reads_uploaded_report_when_kit_exits_before_print(
    tmp_path, monkeypatch
) -> None:
    args, report = _collector_readback_fixture(tmp_path)
    commands = []
    downloads = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return Namespace(returncode=0, stdout="Kit shutdown; no JSON report")

    def read(uri, *, directory):
        assert directory.is_dir()
        downloads.append(uri)
        return report

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(pi05_collect_all, "read_json", read)
    assert _run_collector(args, "train") is report
    assert downloads == ["s3://example/experts/train/collection.json"]
    assert commands == [(_collector_argv(args, "train"), {"check": False})]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "npa.sim2real.pi05.collection_failure.v1"),
        ("attempt_count", 2),
        ("successful_episode_count", 0),
        ("scene_id", "other-scene"),
        ("attempts", []),
        ("attempts", ["not-an-episode"]),
    ],
)
def test_collector_rejects_invalid_durable_report(
    tmp_path, monkeypatch, field, value
) -> None:
    args, report = _collector_readback_fixture(tmp_path)
    report[field] = value
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs: Namespace(returncode=0)
    )
    monkeypatch.setattr(pi05_collect_all, "read_json", lambda *_args, **_kwargs: report)
    with pytest.raises(RuntimeError, match="train collection"):
        _run_collector(args, "train")


@pytest.mark.parametrize(
    "field,value",
    [("identity", "other-object"), ("usd", "other.usd"), ("scale", [1.2] * 3)],
)
def test_collector_rejects_durable_report_for_another_asset(
    tmp_path, monkeypatch, field, value
) -> None:
    args, report = _collector_readback_fixture(tmp_path)
    report["object_asset"][field] = value
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs: Namespace(returncode=0)
    )
    monkeypatch.setattr(pi05_collect_all, "read_json", lambda *_args, **_kwargs: report)
    with pytest.raises(RuntimeError, match="does not match its request"):
        _run_collector(args, "train")


@pytest.mark.parametrize(
    "field,value", [("scene_seed", 42), ("object_identity", "other")]
)
def test_collector_rejects_success_from_another_request(
    tmp_path, monkeypatch, field, value
) -> None:
    args, report = _collector_readback_fixture(tmp_path)
    report["attempts"][0][field] = value
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs: Namespace(returncode=0)
    )
    monkeypatch.setattr(pi05_collect_all, "read_json", lambda *_args, **_kwargs: report)
    with pytest.raises(RuntimeError, match="successful episode does not match"):
        _run_collector(args, "train")


def test_collector_rejects_success_with_invalid_causal_camera_rows(
    tmp_path, monkeypatch
) -> None:
    args, report = _collector_readback_fixture(tmp_path)
    report["attempts"][0]["steps"][0]["wrist_timestamp_s"] = 1.0
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs: Namespace(returncode=0)
    )
    monkeypatch.setattr(pi05_collect_all, "read_json", lambda *_args, **_kwargs: report)
    with pytest.raises(Pi05ContractError):
        _run_collector(args, "train")


def test_collector_requires_durable_report_after_zero_child_exit(monkeypatch) -> None:
    args = Namespace(
        output_root_uri="s3://example/experts",
        episodes_per_split=1,
        seed=41,
        object_usd="",
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *_args, **_kwargs: Namespace(returncode=0)
    )

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("declared collection report absent")

    monkeypatch.setattr(pi05_collect_all, "read_json", missing)
    with pytest.raises(FileNotFoundError, match="declared collection report absent"):
        _run_collector(args, "train")


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


def test_gold_evaluator_prefers_verified_runtime_materialization() -> None:
    spawn = Namespace(usd_path="stock.usd", scale=(1.0, 1.0, 1.0))
    cfg = Namespace(scene=Namespace(object=Namespace(spawn=spawn)))
    protocol = {
        "object_usd": "s3://private-run/assets/cube.usda",
        "_runtime_object_usd": "/workspace/task-owned/object.usda",
        "object_scale": [1.2, 1.2, 1.2],
    }

    _configure_eval_object(cfg, protocol)

    assert spawn.usd_path == "/workspace/task-owned/object.usda"
    assert spawn.scale == (1.2, 1.2, 1.2)


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


def test_policy_adapter_must_match_real_lerobot_feature_dimensions(tmp_path) -> None:
    root = tmp_path / "lerobot-v3"
    (root / "meta").mkdir(parents=True)
    info = {
        "features": {
            "observation.state": {"shape": [8]},
            "action": {"shape": [8]},
            "observation.images.workspace": {"shape": [224, 224, 3]},
            "observation.images.wrist": {"shape": [224, 224, 3]},
        }
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    adapter = {
        "schema": "npa.sim2real.lerobot_policy_adapter.v1",
        "backend": "lerobot_smolvla",
        "measured_state_names": [f"state_{index}" for index in range(8)],
        "measured_state_units": ["rad"] * 7 + ["normalized_closed"],
        "cameras": [
            {
                "feature_key": f"observation.images.{name}",
                "image_size_wh": [224, 224],
            }
            for name in ("workspace", "wrist")
        ],
        "action": {"names": [f"action_{index}" for index in range(8)]},
    }
    source = tmp_path / "adapter.json"
    source.write_text(json.dumps(adapter))
    target = install_policy_adapter(source, root)
    assert target == root / "meta" / "npa_policy_adapter.json"

    adapter["action"]["names"].pop()
    source.write_text(json.dumps(adapter))
    with pytest.raises(Pi05ContractError, match="state/action dimensions"):
        install_policy_adapter(source, root)


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


def _write_terminal_episode(root, identity, scene, phases) -> None:
    _write_episode(root, identity, scene, len(phases))
    episode = root / "episode-000000"
    metadata = json.loads((episode / "episode.json").read_text())
    actions = np.load(episode / "actions.npy")
    actions[:, 0] += np.arange(len(phases), dtype=np.float32) * 0.001
    for step, phase, action in zip(metadata["steps"], phases, actions, strict=True):
        step.update(phase=phase, action=action.tolist())
    _bind_frame_hashes(
        metadata["steps"],
        np.load(episode / "obs_workspace.npy"),
        np.load(episode / "obs_wrist.npy"),
    )
    (episode / "episode.json").write_text(json.dumps(metadata))
    np.save(episode / "actions.npy", actions)


def test_terminal_rows_preserve_release_and_retreat_as_current_inputs(
    tmp_path, monkeypatch
) -> None:
    roots = {name: tmp_path / name for name in ("train", "validation", "gold")}
    phases = ["approach"] * 15 + ["release"] * 12 + ["retreat"] * 18
    for index, (name, root) in enumerate(roots.items()):
        _write_terminal_episode(root, f"cube-{name}", f"scene-{index}", phases)
    monkeypatch.setitem(
        sys.modules, "npa.adapter.sim_to_lerobot", _fake_lerobot_module()
    )
    output = tmp_path / "output"
    export_local(
        train_root=roots["train"],
        validation_root=roots["validation"],
        gold_root=roots["gold"],
        output=output,
    )
    arrays, manifest = _load_dataset(
        str(output / "openpi-dataset.npz"), str(output / "openpi-manifest.json")
    )
    lineage = manifest["splits"]["train"]["lineage"]
    assert sum(row["source_phase"] == "release" for row in lineage) == 12
    assert sum(row["source_phase"] == "retreat" for row in lineage) == 4
    assert arrays["train_actions"].shape == (31, 15, 8)
    actions = np.load(roots["train"] / "episode-000000" / "actions.npy")
    for index, targets in enumerate(arrays["train_actions"]):
        np.testing.assert_array_equal(targets, actions[index : index + 15])


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
    manifests["deployment"]["spec"]["template"]["spec"]["containers"][0]["env"] = [
        {"name": "OPENPI_CHECKPOINT_URI", "value": "s3://private/checkpoint"}
    ]
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


def test_failed_camera_admission_retains_pixels_without_a_fictitious_query(
    tmp_path, monkeypatch
) -> None:
    exterior = np.full((224, 224, 3), 17, dtype=np.uint8)
    wrist = np.full((224, 224, 3), 29, dtype=np.uint8)
    witness = np.full((224, 224, 3), 41, dtype=np.uint8)
    record = pi05_closed_loop._new_episode_record()
    monkeypatch.setattr(
        pi05_closed_loop, "_policy_observation", lambda _: (exterior, wrist, [], [])
    )
    monkeypatch.setattr(pi05_closed_loop, "_rgb", lambda *_: witness)
    timing = {"observation_time_s": 0.5, "wrist_capture_time_s": 0.5}
    receipts = {
        "exterior": {"sha256": "a" * 64},
        "wrist": {"sha256": "b" * 64},
        "witness": {"sha256": "c" * 64},
    }
    monkeypatch.setattr(
        pi05_closed_loop, "_camera_query_receipts", lambda *_: (timing, receipts)
    )
    monkeypatch.setattr(
        pi05_closed_loop, "_infer_query", lambda *_: ([], {"wire_input": {}})
    )

    def fail_marker(*_):
        raise RuntimeError("cyan destination marker is not observable")

    monkeypatch.setattr(pi05_closed_loop, "_record_query_camera", fail_marker)
    with pytest.raises(RuntimeError, match="destination marker"):
        pi05_closed_loop._query_policy({"native": None, "record": record}, 0)
    assert not record["queries"] and not record["_wrist_frames"]
    assert record["failed_camera_observations"][0]["policy_query_recorded"] is False
    artifacts = pi05_closed_loop._write_failed_camera_evidence(tmp_path, [record])
    assert len(artifacts) == 3
    assert np.array_equal(np.load(tmp_path / artifacts[1]["path"]), wrist[None])
    assert np.array_equal(np.load(tmp_path / artifacts[2]["path"]), witness[None])
    assert not any(key.startswith("_failed") for key in record)


def test_destination_occlusion_is_allowed_only_during_retreat(monkeypatch) -> None:
    phase = "approach"
    monkeypatch.setattr(pi05_closed_loop, "_camera_pose_sample", lambda _native: {})
    monkeypatch.setattr(pi05_closed_loop, "_destination_pixel_count", lambda *_args: 0)
    monkeypatch.setattr(pi05_closed_loop, "_target_pixel_count", lambda *_args: 0)
    monkeypatch.setattr(pi05_closed_loop, "_finger_in_frustum", lambda _native: True)
    monkeypatch.setattr(
        pi05_closed_loop,
        "_finger_pixel_counts",
        lambda *_args: {"left": 0, "right": 0},
    )
    monkeypatch.setattr(
        pi05_closed_loop,
        "_semantic_encoding_receipt",
        lambda *_args: {"schema": "test"},
    )
    monkeypatch.setattr(pi05_closed_loop, "_policy_phase", lambda _record: phase)
    record = {"physics_trace": [], "camera_trace": []}
    receipts = {"exterior": {}, "wrist": {}, "witness": {}}
    with pytest.raises(RuntimeError, match="destination marker"):
        pi05_closed_loop._record_query_camera(None, record, receipts, {})

    phase = "retreat"
    pi05_closed_loop._record_query_camera(None, record, receipts, {})
    assert record["camera_trace"][0]["phase"] == "retreat"
    assert record["camera_trace"][0]["exterior_destination_pixels"] == 0


def test_admitted_camera_frames_match_completed_query_count(monkeypatch) -> None:
    exterior = np.zeros((224, 224, 3), dtype=np.uint8)
    wrist = np.ones_like(exterior)
    witness = np.full_like(exterior, 2)
    record = pi05_closed_loop._new_episode_record()
    monkeypatch.setattr(
        pi05_closed_loop, "_policy_observation", lambda _: (exterior, wrist, [], [])
    )
    monkeypatch.setattr(pi05_closed_loop, "_rgb", lambda *_: witness)
    receipts = {
        "exterior": {"sha256": "a" * 64},
        "wrist": {"sha256": "b" * 64},
        "witness": {"sha256": "c" * 64},
    }
    monkeypatch.setattr(
        pi05_closed_loop, "_camera_query_receipts", lambda *_: ({}, receipts)
    )
    monkeypatch.setattr(
        pi05_closed_loop, "_infer_query", lambda *_: ("actions", {"wire_input": {}})
    )
    monkeypatch.setattr(pi05_closed_loop, "_record_query_camera", lambda *_: None)
    actions, _, _ = pi05_closed_loop._query_policy(
        {"native": None, "record": record}, 0
    )
    assert actions == "actions"
    assert (
        len(record["_exterior_frames"])
        == len(record["_wrist_frames"])
        == len(record["_witness_frames"])
        == 1
    )
    assert "failed_camera_observations" not in record


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
            "witness_mp4": {"bytes": 10},
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
                "witness_capture_time_s": observation - 1.5e-6,
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
        "_witness_frames": frames,
        "_terminal_witness_frame": frames[-1],
        "terminal_witness_dwell": {
            "post_control_witness": {
                "capture_time_s": trace[-1]["simulator_time_s"],
            }
        },
    }


def test_closed_loop_rrd_is_finalized_and_causally_timed(tmp_path: Path) -> None:
    episodes = [_rrd_episode()]

    rrd, steps = pi05_closed_loop._write_rrd(
        tmp_path, episodes, "base", "sealed-workflow-run"
    )

    assert steps == 2
    from npa.viz.recordings import load_recording

    assert load_recording(rrd).recording_id() == "sealed-workflow-run"
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
    assert decoded.count("* entity_path: /episodes/0/queries/") == 8
    assert decoded.count("* entity_path: /episodes/0/terminal/witness") == 1
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
        pi05_closed_loop._write_rrd(tmp_path, [episode], "base", "sealed-workflow-run")


def test_closed_loop_rrd_rejects_invalid_workflow_identity(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="recording id"):
        pi05_closed_loop._write_rrd(tmp_path, [_rrd_episode()], "base", "../run")


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
    assert captured["command"] == [
        "/isaac-sim/python.sh",
        "-m",
        "npa.workflows.sim2real.pi05_closed_loop",
        "--fixture",
    ]


def test_closed_loop_entry_propagates_isaac_launcher_failure(monkeypatch) -> None:
    args = _closed_loop_report_args()
    commands = []
    monkeypatch.setattr(pi05_closed_loop_entry, "parse_args", lambda _argv: args)

    def run(command, **kwargs):
        commands.append(command)
        return Namespace(returncode=69)

    def read_report(*_args, **_kwargs):
        pytest.fail("failed Isaac bootstrap must not read a previous success report")

    monkeypatch.setattr(pi05_closed_loop_entry.subprocess, "run", run)
    monkeypatch.setattr(pi05_closed_loop_entry, "read_json", read_report)
    with pytest.raises(RuntimeError, match="Kit evaluator failed with exit 69"):
        pi05_closed_loop_entry.main(["--fixture"])
    assert len(commands) == 1
    assert commands[0][0] == "/isaac-sim/python.sh"
