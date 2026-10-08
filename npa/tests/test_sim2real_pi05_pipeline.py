from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from npa.adapter.sim_to_lerobot import _episode_timestamps
from npa.workflows.byof.openpi_pipeline import _load_dataset
from npa.workflows.byof.openpi_service_lifecycle import (
    _inject_checkpoint_probe,
    _patch_private_s3_checkpoint,
)
from npa.workflows.sim2real.pi05_closed_loop import _success
from npa.workflows.sim2real.pi05_contract import EPISODE_SCHEMA, Pi05ContractError
from npa.workflows.sim2real.pi05_collect_all import _collector_argv
from npa.workflows.sim2real.pi05_data import OPENPI_SCHEMA, export_local
from npa.workflows.sim2real.pi05_isaac import (
    _collection_report,
    configure_surface_goal,
)


TARGET = np.asarray([0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.7, 0.37], dtype=np.float32)


def test_surface_task_replaces_airborne_goal_and_exports_exact_scenario() -> None:
    ranges = Namespace(pos_x=(0.3, 0.7), pos_y=(-0.25, 0.25), pos_z=(0.20, 0.42))
    cfg = Namespace(
        commands=Namespace(
            object_pose=Namespace(ranges=ranges, resampling_time_range=(3.0, 5.0))
        )
    )
    configure_surface_goal(cfg, (0.50, 0.18, 0.025))
    assert ranges.pos_z == (0.025, 0.025)
    attempt = _episode_metadata("cube", "scene", 16)
    attempt["scene_seed"] = 7
    report = _collection_report([attempt], "cube", "", (1.0, 1.0, 1.0), "scene")
    assert report["successful_scenarios"] == [
        {
            "scene_seed": 7,
            "scene_configuration_digest": "scene",
            "initial_object_position_m": [0.45, -0.15, 0.025],
            "target_position_m": [0.50, 0.18, 0.025],
        }
    ]


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


def _episode_metadata(identity: str, scene: str, length: int) -> dict:
    steps = []
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
    return {
        "schema": EPISODE_SCHEMA,
        "episode_id": f"episode-{identity}",
        "source_backend": "isaac",
        "object_motion": "physics_only",
        "action_semantics": "absolute_joint_position_plus_normalized_gripper",
        "object_identity": identity,
        "object_width_m": 0.05,
        "initial_object_position_m": [0.45, -0.15, 0.025],
        "target_position_m": [0.50, 0.18, 0.025],
        "scene_configuration_digest": scene,
        "scene_seed": 41,
        "instruction": "pick and place",
        "steps": steps,
        "success_evidence": {
            "ordered_events": [
                "contact",
                "grasp",
                "lift",
                "transport",
                "release",
                "support_contact",
                "stable",
            ],
            "released_supported_surface_placement": True,
            "stable_steps": 3,
        },
    }


def _write_episode(root: Path, identity: str, scene: str, length: int = 16) -> None:
    episode = root / "episode-000000"
    episode.mkdir(parents=True)
    metadata = _episode_metadata(identity, scene, length)
    (episode / "episode.json").write_text(json.dumps(metadata))
    frames = np.zeros((length, 224, 224, 3), dtype=np.uint8)
    frames[:, 80:120, 80:120] = sum(identity.encode()) % 255
    np.save(episode / "obs_workspace.npy", frames)
    np.save(episode / "obs_wrist.npy", frames[:, :, ::-1])
    state = np.r_[TARGET[:7], np.float32(0.0)]
    np.save(episode / "state.npy", np.tile(state, (length, 1)).astype(np.float32))
    actions = np.asarray([row["action"] for row in metadata["steps"]], dtype=np.float32)
    np.save(episode / "actions.npy", actions)
    np.save(episode / "timestamps.npy", np.arange(1, length + 1) / 15)
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
    np.savez(
        episode / "physics_trace.npz",
        object_position_m=position,
        object_speed_m_s=speed,
        left_finger_object_force_n=bilateral,
        right_finger_object_force_n=bilateral,
        object_support_force_n=support,
        gripper_width_m=width,
        hand_object_distance_m=hand_distance,
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


def test_surface_success_requires_three_consecutive_supported_release_steps() -> None:
    base = {
        "bilateral_finger_contact": False,
        "object_lift_m": 0.06,
        "target_xy_distance_m": 0.01,
        "gripper_width_m": 0.07,
        "object_support_force_n": 1.0,
        "object_speed_m_s": 0.0,
        "hand_object_distance_m": 0.12,
    }
    trace = [
        {**base, "bilateral_finger_contact": True},
        base,
        {**base, "object_support_force_n": 0.0},
        base,
        base,
    ]
    assert not _success(trace)["released_supported_surface_placement"]
    trace.append(base)
    assert _success(trace)["released_supported_surface_placement"]
    trace[-1] = {**base, "hand_object_distance_m": 0.05}
    assert not _success(trace)["released_supported_surface_placement"]


def test_service_manifest_adds_private_checkpoint_probe_without_client_job(
    monkeypatch,
) -> None:
    manifests = {
        "secret": {"metadata": {"name": "secret"}, "data": {}},
        "deployment": {
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
                                    "exec /opt/venv/bin/python /opt/byof/scripts/serve_policy.py",
                                ],
                            }
                        ]
                    }
                }
            }
        },
    }
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.invalid")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "redacted")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "redacted")
    _patch_private_s3_checkpoint(manifests, "s3://private/checkpoint")
    _inject_checkpoint_probe(manifests)
    shell = manifests["deployment"]["spec"]["template"]["spec"]["containers"][0][
        "command"
    ][2]
    assert "npa-openpi-loaded-checkpoint.json" in shell
    assert "NPA_RESOLVED_CHECKPOINT_DIR" in shell
    assert set(manifests) == {"secret", "deployment"}
