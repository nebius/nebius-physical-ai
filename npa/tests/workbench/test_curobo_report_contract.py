"""Reject plausible summaries that invent benchmark identities or planner facts."""

from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workbench.curobo import audit, runner
from npa.workbench.curobo.artifacts import (
    CuroboError,
    canonical,
    summarize,
    validate_report,
)
from npa.workbench.curobo.benchmark_inventory import benchmark_identities, DATASET_FILES
from npa.workbench.curobo.schemas import DATASET_REVISION, SOURCE_REVISION


def benchmark_rows(modes=("kinematic",)):
    return [
        {
            "mode": mode,
            "dataset": dataset,
            "problem_id": identity,
            "status": "invalid" if invalid else "failed",
            **(
                {"reason": "upstream collision_buffer_ik is negative"}
                if invalid
                else {
                    "query": {
                        "start": [0.0] * 7,
                        "goal_pose": {
                            "position_xyz": [0.1, 0.0, 0.0],
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                    },
                    "metrics": {"wall_plan_seconds": 0.01},
                }
            ),
        }
        for (mode, dataset, identity), invalid in benchmark_identities(
            list(modes)
        ).items()
    ]


def report_for(rows, kind="benchmark"):
    return {
        "schema_version": "npa.curobo.result.v1",
        "engine": "nvidia-curobo-v2",
        "source_revision": SOURCE_REVISION,
        "dataset_revision": DATASET_REVISION if kind == "benchmark" else None,
        "run_id": "report-test",
        "kind": kind,
        "requested_modes": sorted({r["mode"] for r in rows}),
        "summary": summarize(rows),
    }


@pytest.mark.parametrize(
    "modes", [("kinematic",), ("dynamics",), ("kinematic", "dynamics")]
)
def test_complete_pinned_population_retains_all_eligible_failures(modes):
    rows = benchmark_rows(modes)
    validate_report(report_for(rows), rows, run_id="report-test")
    assert len(rows) == 2600 * len(modes)
    assert sum(r["status"] == "invalid" for r in rows) == 10 * len(modes)


@pytest.mark.parametrize(
    "change",
    [
        "invent_id",
        "rename_group",
        "out_of_range",
        "swap_dataset",
        "invalid_solved",
        "eligible_excluded",
        "move_exclusion",
    ],
)
def test_same_population_and_recomputed_summary_cannot_hide_wrong_inputs(change):
    rows = benchmark_rows()
    valid = next(r for r in rows if r["status"] == "failed")
    invalid = next(r for r in rows if r["status"] == "invalid")
    if change == "invent_id":
        valid["problem_id"] = "invented/0"
    elif change == "rename_group":
        valid["problem_id"] = "bookshelf_small_panda_typo/0"
    elif change == "out_of_range":
        valid["problem_id"] = "bookshelf_small_panda/100"
    elif change == "swap_dataset":
        other = next(r for r in rows if r["dataset"] == "mpinets")
        valid["dataset"], other["dataset"] = other["dataset"], valid["dataset"]
    if change in {"invalid_solved", "move_exclusion"}:
        invalid.update(status="failed", metrics={"wall_plan_seconds": 0.01})
    if change in {"eligible_excluded", "move_exclusion"}:
        valid["status"] = "invalid"
        valid.pop("metrics")
    report = report_for(rows)
    assert report["summary"]["kinematic"]["input_count"] == 2600
    with pytest.raises(CuroboError, match="(identities|exclusions)"):
        validate_report(report, rows, run_id="report-test")


@pytest.fixture
def solved_row(monkeypatch):
    """Drive the actual runner formatter through mocked CUDA/planner boundaries."""

    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data)
            self.shape = self.data.shape

        def detach(self):
            return self

        def cpu(self):
            return self

        def reshape(self, *shape):
            return Tensor(self.data.reshape(*shape))

        def numpy(self):
            return self.data

        def item(self):
            return self.data.item()

    joint_state = SimpleNamespace(
        from_position=lambda data, **kwargs: SimpleNamespace(position=data, **kwargs)
    )
    monkeypatch.setitem(
        sys.modules,
        "curobo.types",
        SimpleNamespace(JointState=joint_state, GoalToolPose=lambda **kwargs: kwargs),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
    )
    path = SimpleNamespace(
        joint_names=[f"joint{i}" for i in range(7)],
        position=Tensor([[0.0] * 7, [0.1] * 7]),
        velocity=Tensor([[0.0] * 7, [0.1] * 7]),
        acceleration=Tensor([[0.0] * 7, [0.0] * 7]),
        jerk=Tensor([[0.0] * 7, [0.0] * 7]),
        dt=Tensor(0.1),
    )

    def reorder(names):
        indices = [path.joint_names.index(name) for name in names]
        return SimpleNamespace(
            joint_names=list(names),
            position=Tensor(path.position.data[:, indices]),
        )

    path.reorder = reorder

    def forward_kinematics(state):
        assert state.joint_names == [f"joint{i}" for i in range(7)]
        active = np.asarray(state.position.data)
        return SimpleNamespace(
            tool_poses=SimpleNamespace(
                get_link_pose=lambda _name: SimpleNamespace(
                    position=Tensor(
                        np.column_stack(
                            (active[:, 0], np.zeros((len(active), 2), dtype=active.dtype))
                        )
                    ),
                    quaternion=Tensor(
                        np.tile([1.0, 0.0, 0.0, 0.0], (len(active), 1))
                    ),
                )
            )
        )

    result = SimpleNamespace(
        success=Tensor(True),
        get_interpolated_plan=lambda: path,
        total_time=0.008,
        solve_time=0.006,
        position_error=Tensor(0.001),
        rotation_error=Tensor(0.002),
        js_solution=path,
    )
    planner = SimpleNamespace(
        device_cfg=SimpleNamespace(to_device=Tensor),
        joint_names=[f"joint{i}" for i in range(7)],
        tool_frames=["tool"],
        reset_seed=lambda: None,
        plan_pose=lambda *_args, **_kwargs: result,
        kinematics=SimpleNamespace(compute_kinematics=forward_kinematics),
    )
    problem = {
        "start": [0.0] * 7,
        "goal_pose": {
            "position_xyz": [0.1, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }

    def solve(benchmark=False, success=True, include_fingers=False, positions=None):
        if positions is not None:
            assert not include_fingers
            values = np.asarray(positions)
            assert values.ndim == 2 and values.shape[1] == 7
            path.joint_names = [f"joint{i}" for i in range(7)]
            path.position = Tensor(values)
            for field in ("velocity", "acceleration", "jerk"):
                setattr(path, field, Tensor(np.zeros_like(values)))
        if include_fingers:
            # Locked/mimic fingers are returned in the full interpolation. Put
            # them between active joints so slicing the first seven is invalid.
            names = [
                "left_finger",
                "joint6",
                "joint2",
                "joint0",
                "right_finger",
                "joint5",
                "joint1",
                "joint4",
                "joint3",
            ]
            for field in ("position", "velocity", "acceleration", "jerk"):
                source = getattr(path, field).data
                columns = [
                    source[:, path.joint_names.index(name)]
                    if name in path.joint_names
                    else np.full(2, 0.04 if field == "position" else 0.0)
                    for name in names
                ]
                setattr(path, field, Tensor(np.stack(columns, axis=-1)))
            path.joint_names = names
        result.success = Tensor(success)
        upstream = (
            SimpleNamespace(
                compute_trajectory_energy=lambda *_args: {
                    "energy": 0.07,
                    "max_torque": 1.0,
                    "torque_violation": False,
                    "torques": np.ones((2, 7)),
                }
            )
            if benchmark
            else None
        )
        dynamics_model = (
            (None, None, np.asarray([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]))
            if benchmark
            else None
        )
        return runner._solve(
            planner,
            problem,
            benchmark_module=upstream,
            dynamics_model=dynamics_model,
            attached_mass_kg=0.0 if benchmark else None,
        )

    return solve


@pytest.mark.parametrize("success", [True, False])
def test_actual_runner_plan_record_matches_strict_metrics_contract(solved_row, success):
    rows = [
        {
            "mode": "kinematic",
            "dataset": "operator",
            "problem_id": "pose",
            **solved_row(success=success),
        }
    ]
    validate_report(report_for(rows, "plan"), rows, run_id="report-test")


def test_actual_runner_benchmark_record_matches_strict_metrics_contract(solved_row):
    rows = benchmark_rows()
    rows[0].update(solved_row(benchmark=True))
    validate_report(report_for(rows), rows, run_id="report-test")


def test_full_interpolation_retains_fingers_and_orders_active_joints_for_fk(solved_row):
    row = solved_row(include_fingers=True)
    trajectory = row["trajectory"]
    assert trajectory["joint_names"] == [
        "left_finger",
        "joint6",
        "joint2",
        "joint0",
        "right_finger",
        "joint5",
        "joint1",
        "joint4",
        "joint3",
    ]
    for field in ("position", "velocity", "acceleration", "jerk"):
        assert np.asarray(trajectory[field]).shape == (2, 9)
    np.testing.assert_array_equal(np.asarray(trajectory["position"])[:, [0, 4]], 0.04)
    assert trajectory["tool_position"] == [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]


def test_actual_runner_metrics_bind_serialized_float32_trajectory(solved_row):
    steps = np.linspace(-1.3, 0.2, 4097, dtype=np.float32)
    factors = np.asarray([1.0, -0.7, 0.3, -0.2, 0.11, -0.05, 0.02], dtype=np.float32)
    positions = steps[:, None] * factors[None, :]
    row = {
        "mode": "kinematic",
        "dataset": "operator",
        "problem_id": "float32-serialization",
        **solved_row(positions=positions),
    }

    raw_float32_metric = float(
        np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    )
    durable_metric = float(
        np.linalg.norm(
            np.diff(np.asarray(row["trajectory"]["position"], dtype=float), axis=0),
            axis=1,
        ).sum()
    )
    assert abs(raw_float32_metric - durable_metric) > 1e-9
    assert row["metrics"]["joint_path_length_rad"] == durable_metric

    journal = canonical(row) + b"\n"
    report = report_for([row], "plan")
    report["journal_sha256"] = hashlib.sha256(journal).hexdigest()
    validation = audit.audit_bytes(canonical(report), journal, run_id="report-test")
    assert validation["problem_count"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "unknown",
        "negative",
        "nan",
        "duration",
        "zero_duration",
        "dt",
        "boolean_dt",
        "dataset",
        "mode",
        "requested_mode",
    ],
)
def test_plan_report_rejects_bad_status_metrics_timeline_or_scope(solved_row, change):
    rows = [
        {
            "mode": "kinematic",
            "dataset": "operator",
            "problem_id": "pose",
            **solved_row(),
        }
    ]
    record = rows[0]
    if change == "missing":
        record["metrics"].pop("solver_seconds")
    elif change == "unknown":
        record["metrics"]["fictional_measurement"] = 1.0
    elif change == "negative":
        record["metrics"]["position_error_m"] = -0.1
    elif change == "duration":
        record["metrics"]["trajectory_duration_seconds"] = 0.3
    elif change == "zero_duration":
        record["metrics"]["trajectory_duration_seconds"] = 0
    elif change == "dt":
        record["trajectory"]["dt"] = 0.2
    elif change == "dataset":
        record["dataset"] = "arbitrary"
    elif change == "mode":
        record["mode"] = "dynamics"
    report = report_for(rows, "plan")
    if change == "nan":
        record["metrics"]["solver_seconds"] = float("nan")
    elif change == "boolean_dt":
        record["trajectory"]["dt"] = True
    elif change == "requested_mode":
        report["requested_modes"] = ["dynamics"]
    with pytest.raises(CuroboError):
        validate_report(report, rows, run_id="report-test")


@pytest.mark.parametrize(
    "change",
    [
        "missing_dynamic",
        "negative_energy",
        "torque_indicator",
        "failed_missing_wall",
        "invalid_metrics",
        "torque_evidence",
        "torque_limits",
        "payload_mass",
        "missing_evidence",
    ],
)
def test_benchmark_report_requires_the_actual_metrics_for_each_status(
    solved_row, change
):
    rows = benchmark_rows()
    rows[0].update(solved_row(benchmark=True))
    if change == "missing_dynamic":
        rows[0]["metrics"].pop("energy_proxy_j")
    elif change == "negative_energy":
        rows[0]["metrics"]["energy_proxy_j"] = -1
    elif change == "torque_indicator":
        rows[0]["metrics"]["torque_violation"] = 2
    elif change == "failed_missing_wall":
        rows[1].pop("metrics")
    elif change == "invalid_metrics":
        next(r for r in rows if r["status"] == "invalid")["metrics"] = {
            "wall_plan_seconds": 0.1
        }
    elif change == "torque_evidence":
        rows[0]["dynamics_evidence"]["torques_nm"][1][0] = 2.0
    elif change == "torque_limits":
        rows[0]["dynamics_evidence"]["torque_limits_nm"][0] = 88.0
    elif change == "payload_mass":
        rows[0]["dynamics_evidence"]["attached_mass_kg"] = 3.0
    else:
        rows[0].pop("dynamics_evidence")
    with pytest.raises(
        CuroboError, match="(metrics|torque violation|dynamics|inverse)"
    ):
        validate_report(report_for(rows), rows, run_id="report-test")


def test_benchmark_marker_cannot_mask_changed_dataset_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_runtime_source", lambda: tmp_path)
    monkeypatch.setenv("NPA_CUROBO_DATASET_SOURCE", str(tmp_path))
    (tmp_path / "NPA_SOURCE_REVISION").write_text(DATASET_REVISION)
    data = tmp_path / "robometrics/content/dataset"
    data.mkdir(parents=True)
    for filename, _sha in DATASET_FILES.values():
        (data / filename).write_text("unexpected dataset bytes")
    with pytest.raises(CuroboError, match="pinned inventory"):
        runner._benchmark_module()
