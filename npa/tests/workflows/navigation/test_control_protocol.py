"""Reject substituted native controls before any final learning or evaluation."""

import copy
import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.navigation import control_protocol as controls
from npa.workflows.navigation.artifacts import file_sha256
from npa.workflows.navigation.native import source_bundle_digest
from npa.workflows.navigation.probe_evidence import save_trace


def _trace(recipe, arm):
    cases = controls.control_cases(recipe, arm)
    rows = []
    for step in range(len(recipe.probe.actions) + 1):
        position = np.array([case.position_m for case in cases])
        position[0, 0] += step * 0.1
        contact = np.zeros(recipe.num_envs)
        if arm == "obstacle" and step:
            contact[0] = 2.0
        rows.append(
            {
                "state": {
                    "position_m": position,
                    "goal_m": np.array([case.goal_m for case in cases]),
                    "heading_rad": np.array([case.heading_rad for case in cases]),
                    "obstacle_contact": contact,
                    "peer_contact": np.zeros(recipe.num_envs),
                    "physical_failure": np.zeros(recipe.num_envs),
                    "upright_cosine": np.ones(recipe.num_envs),
                    "ground_clearance_m": np.full(recipe.num_envs, 0.5),
                },
                "observations": {"policy/value": np.ones((recipe.num_envs, 3))},
                "native": {"joint_pos/value": np.zeros((recipe.num_envs, 12))},
            }
        )
    return rows


def _provenance(recipe, stage):
    return {
        "runtime": {
            "robot_population": recipe.num_envs,
            "scene_instances": 1,
            "scene_sha256": recipe.scene_sha256,
            "scene_prim": recipe.scene_prim,
        },
        "settings": {"seed": recipe.train_cases[0].seed},
        "initialization": {"mode": "fixture", "policy_state_sha256": "f" * 64},
        "mode": {
            "stage": stage,
            "training": stage == "train",
            "enable_cameras": stage != "train",
            "physics_dt": 0.005,
            "control_decimation": 40,
        },
    }


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def attempt(recipe, tmp_path, monkeypatch):
    recipe = recipe.model_copy(
        update={
            "adapter_module": controls.REFERENCE,
            "sensor_mode": "static_raycast",
            "source_bundle_sha256": source_bundle_digest(),
            "adapter_sha256": file_sha256(
                controls.Path(controls.__file__).with_name("reference.py")
            ),
        }
    )
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "recipe.json").write_text(recipe.model_dump_json())
    (source / recipe.scene_file).write_bytes(b"fixture-only")
    monkeypatch.setenv("NPA_TASK_IMAGE", recipe.image)
    stage = "evaluate-checkpoint"
    digest = controls.create_request(recipe, source, output, stage)
    request, root = controls.read_request(recipe, source, output, stage, digest)
    provenance = _provenance(recipe, stage)
    for index, arm in enumerate(controls.ARMS):
        _seal_fixture_arm(
            recipe, root, request, digest, provenance, index, arm, monkeypatch
        )
    monkeypatch.setattr(controls, "native_clock", lambda: {"step": 0, "seconds": 0.0})
    return SimpleNamespace(
        recipe=recipe,
        source=source,
        output=output,
        stage=stage,
        digest=digest,
        root=root,
        provenance=provenance,
    )


def _seal_fixture_arm(
    recipe, root, request, digest, provenance, index, arm, monkeypatch
):
    folder = root / arm
    folder.mkdir()
    (folder / "runtime.log").write_text("native fixture\n")
    save_trace(folder, arm, _trace(recipe, arm))
    clocks = {
        "before": {"step": index * 100, "seconds": float(index)},
        "after": {"step": index * 100 + 80, "seconds": float(index) + 0.4},
    }
    with monkeypatch.context() as patch:
        patch.setattr(controls.os, "getpid", lambda: 10000 + index)
        patch.setattr(controls.os, "getpgrp", lambda: 20000 + index)
        controls.save_control(
            recipe, folder, request, digest, arm, provenance, clocks, 40
        )
    _write(
        folder / "process.json",
        {
            "arm": arm,
            "request_sha256": digest,
            "wrapper_pid": 20000 + index,
            "owned_process_group": 20000 + index,
            "interruption": None,
            "started_monotonic_ns": index * 10 + 1,
            "finished_monotonic_ns": index * 10 + 9,
            "returncode": 0,
            "runtime_log_sha256": file_sha256(folder / "runtime.log"),
        },
    )


def _verify(a, current=None):
    return controls.verify_controls(
        a.recipe, a.source, a.output, a.stage, a.digest, current
    )


def _modify_receipt(a, arm, change):
    path = a.root / arm / "control.json"
    record = json.loads(path.read_text())
    change(record)
    _write(path, record)


def _modify_trace(a, arm, change):
    trace = _trace(a.recipe, arm)
    change(trace)
    save_trace(a.root / arm, arm, trace)
    _modify_receipt(
        a,
        arm,
        lambda row: row.update(
            files=controls._files(
                a.root / arm, ("runtime.log", "control.json", "process.json")
            )
        ),
    )


def test_parent_and_fresh_final_recompute_identical_raw_controls(attempt):
    a = attempt
    digest = controls.accept_controls(a.recipe, a.source, a.output, a.stage, a.digest)
    result = controls.consume_controls(
        a.recipe, a.source, a.output, a.stage, a.digest, digest, a.provenance
    )
    assert result["passed"] is True
    assert result["warm_reset_repeatability_verified"] is False
    assert result["camera_isolation_verified"] is False
    assert result["maximum_peer_delta"] == result["maximum_repeat_delta"] == 0
    assert result["free_motion_m"] > 0
    final = json.loads((a.output / "native-process.json").read_text())
    assert final["native_pid"] == os.getpid()
    assert final["provenance"] == a.provenance
    # Origins deliberately differ across controls; only each process's measured interval is checked.
    assert (
        json.loads((a.root / "obstacle/control.json").read_text())["clocks"]["before"][
            "step"
        ]
        == 300
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempt_id", "other-attempt"),
        ("invocation_id", "another-arm"),
        ("request_sha256", "0" * 64),
        ("arm", "solo"),
        ("native_pid", 10000),
        ("native_pid", True),
        ("native_process_group", 123),
        ("physics_steps_per_control", 4),
    ],
)
def test_cross_attempt_or_reused_native_control_rejected(attempt, field, value):
    _modify_receipt(attempt, "repeat", lambda row: row.update({field: value}))
    with pytest.raises(ValueError):
        _verify(attempt)


@pytest.mark.parametrize(
    "field,value", [("stage", "train"), ("enable_cameras", False), ("physics_dt", 0.01)]
)
def test_wrong_native_mode_rejected(attempt, field, value):
    _modify_receipt(
        attempt, "repeat", lambda row: row["provenance"]["mode"].update({field: value})
    )
    with pytest.raises(ValueError, match="provenance|elapsed time"):
        _verify(attempt)


@pytest.mark.parametrize(
    "kind",
    [
        "clock",
        "nan-time",
        "wrong-positive-duration",
        "failed-process",
        "interrupted-zero-exit",
        "overlap",
        "physx",
    ],
)
def test_native_clock_lifetime_and_error_checks_are_repeated(attempt, kind):
    a = attempt
    if kind == "clock":
        _modify_receipt(
            a, "repeat", lambda row: row["clocks"]["after"].update(step=179)
        )
    elif kind == "nan-time":
        _modify_receipt(
            a, "repeat", lambda row: row["clocks"]["after"].update(seconds=float("nan"))
        )
    elif kind == "wrong-positive-duration":
        _modify_receipt(
            a, "repeat", lambda row: row["clocks"]["after"].update(seconds=1.5)
        )
    else:
        folder = a.root / "repeat"
        record = json.loads((folder / "process.json").read_text())
        if kind == "failed-process":
            record["returncode"] = 1
        elif kind == "interrupted-zero-exit":
            record["interruption"] = "KeyboardInterrupt"
        elif kind == "overlap":
            record["started_monotonic_ns"] = 2
        else:
            (folder / "runtime.log").write_text("PhysX error: contacts dropped")
            record["runtime_log_sha256"] = file_sha256(folder / "runtime.log")
        _write(folder / "process.json", record)
    with pytest.raises((ValueError, RuntimeError)):
        _verify(a)


@pytest.mark.parametrize(
    "arm,key,index,value",
    [
        ("repeat", "position_m", (0, 0), 1.0),
        ("overlap", "position_m", (0, 0), 1.0),
        ("solo", "obstacle_contact", (1,), 1.0),
        ("solo", "physical_failure", (1,), 1.0),
        ("overlap", "peer_contact", (1,), 1.0),
    ],
)
def test_original_physical_gates_recomputed_even_with_resealed_trace(
    attempt, arm, key, index, value
):
    def change(trace):
        trace[1]["state"][key][index] += value

    _modify_trace(attempt, arm, change)
    with pytest.raises(ValueError):
        _verify(attempt)
    assert (attempt.output / "isolation-comparisons.json").is_file()


@pytest.mark.parametrize(
    "kind",
    [
        "no-motion",
        "no-obstacle",
        "bad-reset",
        "observations",
        "missing-step",
        "nan",
        "shape",
    ],
)
def test_incomplete_or_nonmeasuring_controls_rejected(attempt, kind):
    arm = "obstacle" if kind == "no-obstacle" else "solo"

    def change(trace):
        if kind == "missing-step":
            trace.pop()
        elif kind == "bad-reset":
            trace[0]["state"]["position_m"][1, 0] += 1
        elif kind == "observations":
            trace[1]["observations"].clear()
        elif kind == "shape":
            trace[1]["state"]["position_m"] = np.ones((2, 2))
        elif kind == "nan":
            trace[1]["observations"]["policy/value"][0, 0] = np.nan
        else:
            for row in trace:
                if kind == "no-motion":
                    row["state"]["position_m"][0] = attempt.recipe.probe.free.position_m
                else:
                    row["state"]["obstacle_contact"][:] = 0

    _modify_trace(attempt, arm, change)
    with pytest.raises(ValueError):
        _verify(attempt)


@pytest.mark.parametrize(
    "mutation",
    [
        "input",
        "raw-trace",
        "aggregate",
        "final-state",
        "final-pid",
        "prefix",
        "missing-arm",
    ],
)
def test_final_child_rejects_stale_or_modified_evidence(attempt, mutation, monkeypatch):
    a = attempt
    digest = controls.accept_controls(a.recipe, a.source, a.output, a.stage, a.digest)
    current = copy.deepcopy(a.provenance)
    if mutation == "input":
        (a.source / a.recipe.scene_file).write_bytes(b"changed")
    elif mutation == "raw-trace":
        _modify_trace(
            a, "solo", lambda trace: trace[1]["native"]["joint_pos/value"].fill(1)
        )
    elif mutation == "aggregate":
        (a.root / "accepted.json").write_text("{}")
    elif mutation == "final-state":
        current["initialization"]["policy_state_sha256"] = "a" * 64
    elif mutation == "final-pid":
        monkeypatch.setattr(controls.os, "getpid", lambda: 10000)
    elif mutation == "prefix":
        moved = a.output.with_name("different-attempt")
        shutil.copytree(a.output, moved)
        a.output = moved
    else:
        shutil.rmtree(a.root / "repeat")
    with pytest.raises((ValueError, FileNotFoundError)):
        controls.consume_controls(
            a.recipe, a.source, a.output, a.stage, a.digest, digest, current
        )
    assert not (a.output / "native-process.json").exists()


def test_reference_camera_controls_not_silently_qualified(attempt):
    recipe = attempt.recipe.model_copy(update={"sensor_mode": "rgbd"})
    with pytest.raises(ValueError, match="RGB-D remains unqualified"):
        controls.read_request(
            recipe, attempt.source, attempt.output, attempt.stage, attempt.digest
        )


def test_symlink_artifacts_are_rejected(attempt):
    (attempt.root / "solo/extra").symlink_to(attempt.source / "recipe.json")
    with pytest.raises(ValueError, match="symlink"):
        _verify(attempt)


@pytest.mark.parametrize(
    "before_step,before_time,after_step,after_time",
    [
        (2, 0.01, 42, 0.21),
        (1202, 6.01, 1242, 6.21),
        (2402, 12.01, 2442, 12.21),
        (3602, 18.01, 3642, 18.21),
        (2, 0.01, 1202, 6.01),
        (1202, 6.01, 2402, 12.01),
    ],
)
def test_native_manager_count_over_rate_clock_representation(
    before_step, before_time, after_step, after_time
):
    # Isaac Sim 6's native manager derives double(step_count)/integer_rate,
    # not repeated addition of the float32 PhysX dt. These pairs also occurred
    # in retained native diagnostics; source and readback pins stay private.
    assert before_time == before_step / 200
    assert after_time == after_step / 200
    controls._check_elapsed_time(
        {"before": {"seconds": before_time}, "after": {"seconds": after_time}},
        after_step - before_step,
        0.005,
    )


def test_float_dt_accumulator_cannot_impersonate_native_manager_clock():
    after = 0.01 + 1200 * float(np.float32(0.005))
    with pytest.raises(ValueError, match="elapsed time"):
        controls._check_elapsed_time(
            {"before": {"seconds": 0.01}, "after": {"seconds": after}}, 1200, 0.005
        )
