"""Bind fresh native control processes to one stage attempt and its exact inputs."""

import json
import os
import math
from pathlib import Path
import stat
import uuid

import numpy as np

from npa.workflows.navigation.artifacts import file_sha256
from npa.workflows.navigation.native import source_bundle_digest
from npa.workflows.navigation.probe_evidence import load_trace

ARMS = ("solo", "repeat", "overlap", "obstacle")
REFERENCE = "npa.workflows.navigation.reference"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _files(folder, excluded=()):
    result = {}
    for path in sorted(folder.rglob("*")):
        _require(not path.is_symlink(), "control artifacts must not contain symlinks")
        if path.is_dir():
            continue
        _require(stat.S_ISREG(path.stat().st_mode), "nonregular control artifact")
        name = path.relative_to(folder).as_posix()
        if name not in excluded:
            result[name] = file_sha256(path)
    return result


def _binding(recipe, source, stage):
    _require(recipe.adapter_module == REFERENCE, "fresh controls require the reference")
    _require(
        recipe.sensor_mode == "static_raycast",
        "fresh reference controls support static_raycast only; RGB-D remains unqualified",
    )
    _require(os.environ.get("NPA_TASK_IMAGE") == recipe.image, "native image differs")
    _require(
        source_bundle_digest() == recipe.source_bundle_sha256,
        "navigation source differs",
    )
    _require(
        stage in {"train", "evaluate", "evaluate-checkpoint"}, "invalid native mode"
    )
    _require(
        file_sha256(Path(__file__).with_name("reference.py")) == recipe.adapter_sha256,
        "reference adapter differs",
    )
    training = stage == "train"
    return {
        "stage": stage,
        "image": recipe.image,
        "source_bundle_sha256": recipe.source_bundle_sha256,
        "adapter_sha256": recipe.adapter_sha256,
        "source_directory": str(source.resolve()),
        "input_sha256": _files(source),
        "population": recipe.num_envs,
        "training": training,
        "enable_cameras": not training,
        "environment_seed": (recipe.train_cases if training else recipe.eval_cases)[
            0
        ].seed,
        "runner_seed": recipe.train_cases[0].seed,
    }


def create_request(recipe, source, output, stage):
    """Create a unique attempt request before any native control is launched.

    Args:
        recipe: Validated built-in reference recipe.
        source: Materialized immutable input directory.
        output: Fresh stage output directory.
        stage: Original train/evaluate/evaluate-checkpoint mode.
    Returns:
        SHA-256 of the request pinned into every child invocation.
    Raises:
        ValueError: Source, mode or camera contract differs.
        OSError: Attempt output already exists or cannot be written.
    """
    root = output / "controls"
    root.mkdir()
    request = {
        "schema": "npa.navigation.control-request.v1",
        "attempt_id": uuid.uuid4().hex,
        "output_directory": str(output.resolve()),
        "binding": _binding(recipe, source, stage),
        "arms": {arm: uuid.uuid4().hex for arm in ARMS},
    }
    _write(root / "request.json", request)
    return file_sha256(root / "request.json")


def read_request(recipe, source, output, stage, request_sha256, arm=None):
    """Revalidate a same-attempt request against current source and input bytes.

    Args:
        recipe: Current native recipe.
        source: Current materialized inputs.
        output: Child output directory; controls use controls/<arm>.
        stage: Unchanged original stage mode.
        request_sha256: Digest supplied by the stage parent.
        arm: One control name or None for final training/evaluation.
    Returns:
        Request and its canonical control root.
    Raises:
        ValueError: Request, namespace, source or inputs differ.
    """
    _require(arm is None or arm in ARMS, "unknown control arm")
    root = output.parent if arm else output / "controls"
    path = root / "request.json"
    _require(
        not path.is_symlink() and file_sha256(path) == request_sha256,
        "control request changed",
    )
    request = json.loads(path.read_text())
    expected = Path(request["output_directory"]) / "controls"
    _require(
        root.resolve() == expected and (arm is None or output.name == arm),
        "control attempt/prefix differs",
    )
    _require(
        request["schema"] == "npa.navigation.control-request.v1"
        and set(request["arms"]) == set(ARMS),
        "invalid control request",
    )
    _require(
        request["binding"] == _binding(recipe, source, stage),
        "control request bindings differ",
    )
    return request, root


def control_cases(recipe, arm):
    """Select one unchanged physical control's complete robot reset population.

    Args:
        recipe: Sealed reference recipe.
        arm: solo, repeat, overlap or obstacle.
    Returns:
        Exactly recipe.num_envs original reset cases.
    Raises:
        ValueError: Control arm is unknown.
    """
    _require(arm in ARMS, "unknown control arm")
    if arm == "overlap":
        return [recipe.probe.free] * recipe.num_envs
    focal = recipe.probe.obstacle if arm == "obstacle" else recipe.probe.free
    return [focal] + [recipe.probe.parked] * (recipe.num_envs - 1)


def native_clock():
    """Read the existing native event clock without changing simulation state.

    Args:
        None.
    Returns:
        Actual step count and time local to the current native process.
    Raises:
        RuntimeError: Native simulation manager is unavailable.
    """
    from isaacsim.core.simulation_manager.impl.extension import (
        acquire_simulation_manager_interface,
    )

    clock = acquire_simulation_manager_interface()
    if clock is None:
        raise RuntimeError("native simulation clock unavailable")
    return {
        "step": int(clock.get_num_physics_steps()),
        "seconds": float(clock.get_simulation_time()),
    }


def save_control(
    recipe, output, request, request_sha256, arm, provenance, clocks, decimation
):
    """Seal a native control's trace and provenance after its original calls finish.

    Args:
        recipe: Original recipe.
        output: This arm's artifact directory.
        request: Validated attempt request.
        request_sha256: Parent's request digest.
        arm: Current control name.
        provenance: Actual inspected runtime, runner settings and initialization.
        clocks: Separate native before/after clock observations.
        decimation: Actual physics steps per control action.
    Returns:
        None.
    Raises:
        ValueError: The native step count is incomplete.
        OSError: Receipt cannot be written.
    """
    _check_clocks(
        clocks, decimation, len(recipe.probe.actions), provenance["mode"]["physics_dt"]
    )
    _write(
        output / "control.json",
        {
            "schema": "npa.navigation.fresh-control.v1",
            "attempt_id": request["attempt_id"],
            "request_sha256": request_sha256,
            "arm": arm,
            "invocation_id": request["arms"][arm],
            "native_pid": os.getpid(),
            "native_process_group": os.getpgrp(),
            "provenance": provenance,
            "clocks": clocks,
            "physics_steps_per_control": decimation,
            "files": _files(output, ("runtime.log",)),
            "clock_scope": "Native clock observations belong only to this process; absolute times are not compared across processes",
        },
    )


def _check_reset(trace, recipe, arm):
    cases = control_cases(recipe, arm)
    state = trace[0]["state"]
    for name, expected in (
        ("position_m", [case.position_m for case in cases]),
        ("goal_m", [case.goal_m for case in cases]),
    ):
        _require(
            np.allclose(state[name], expected, rtol=0, atol=recipe.probe.tolerance),
            "recorded control reset differs",
        )
    expected = np.asarray([case.heading_rad for case in cases])
    error = (state["heading_rad"] - expected + np.pi) % (2 * np.pi) - np.pi
    _require(
        np.max(np.abs(error)) <= recipe.probe.tolerance,
        "recorded control heading differs",
    )
    _require(not state["physical_failure"].any(), "invalid initial control pose")


def _check_clocks(clocks, decimation, actions, physics_dt):
    _require(type(decimation) is int and decimation > 0, "invalid native decimation")
    for clock in clocks.values():
        _require(
            type(clock["step"]) is int and clock["step"] >= 0, "invalid native step"
        )
        _require(
            math.isfinite(clock["seconds"]) and clock["seconds"] >= 0,
            "invalid native time",
        )
    _require(
        clocks["after"]["step"] - clocks["before"]["step"] == actions * decimation,
        "control native step count differs",
    )
    _require(
        clocks["after"]["seconds"] > clocks["before"]["seconds"],
        "native clock did not advance",
    )

    _check_elapsed_time(clocks, actions * decimation, physics_dt)


def _check_elapsed_time(clocks, steps, physics_dt):
    _require(math.isfinite(physics_dt) and physics_dt > 0, "invalid physics timestep")
    before, after = clocks["before"]["seconds"], clocks["after"]["seconds"]
    expected = steps * physics_dt
    # The pinned native manager reports double(step_count)/integer_rate, not
    # an accumulated float32 dt. Bound subtraction, timestep representation and
    # product rounding only.
    # This is independent of the recipe's physical/observation tolerance.
    epsilon = (
        math.ulp(before)
        + math.ulp(after)
        + steps * math.ulp(physics_dt)
        + math.ulp(expected)
    )
    _require(
        abs((after - before) - expected) <= epsilon,
        "control native elapsed time differs from steps and physics timestep",
    )


def _check_process(folder, arm, request_sha256):
    from npa.workflows.navigation.control_processes import check_native_log

    process = json.loads((folder / "process.json").read_text())
    _require(
        process["returncode"] == 0
        and process["interruption"] is None
        and process["owned_process_group"] == process["wrapper_pid"]
        and process["request_sha256"] == request_sha256
        and process["arm"] == arm,
        "native child did not complete",
    )
    _require(
        process["runtime_log_sha256"] == file_sha256(folder / "runtime.log"),
        "native child log changed",
    )
    _require(
        type(process["wrapper_pid"]) is int and process["wrapper_pid"] > 0,
        "invalid child process identity",
    )
    _require(
        type(process["started_monotonic_ns"]) is int
        and type(process["finished_monotonic_ns"]) is int
        and 0 < process["started_monotonic_ns"] < process["finished_monotonic_ns"],
        "invalid child lifetime",
    )
    check_native_log(folder / "runtime.log")
    return process


def _read_control(root, arm, request, request_sha256, recipe):
    folder = root / arm
    record = json.loads((folder / "control.json").read_text())
    _require(
        record["schema"] == "npa.navigation.fresh-control.v1",
        "invalid native control receipt",
    )
    _require(
        record["attempt_id"] == request["attempt_id"]
        and record["request_sha256"] == request_sha256
        and record["arm"] == arm
        and record["invocation_id"] == request["arms"][arm],
        "control belongs to another attempt or arm",
    )
    _require(
        record["files"]
        == _files(folder, ("runtime.log", "control.json", "process.json")),
        "native control files changed",
    )
    _require(
        type(record["native_pid"]) is int and record["native_pid"] > 0,
        "invalid native process identity",
    )
    _check_clocks(
        record["clocks"],
        record["physics_steps_per_control"],
        len(recipe.probe.actions),
        record["provenance"]["mode"]["physics_dt"],
    )
    process = _check_process(folder, arm, request_sha256)
    _require(
        record["native_process_group"] == process["owned_process_group"],
        "native child does not belong to its launched process group",
    )
    trace = load_trace(
        folder / f"probe-{arm}.npz", recipe.num_envs, len(recipe.probe.actions)
    )
    _check_reset(trace, recipe, arm)
    return record, trace, process


def _check_provenance(records, processes, current):
    _require(
        len({row["native_pid"] for row in records}) == len(ARMS),
        "controls reused a native process",
    )
    for left, right in zip(processes, processes[1:]):
        _require(
            left["finished_monotonic_ns"] <= right["started_monotonic_ns"],
            "native control processes overlapped",
        )
    _require(
        len({row["native_process_group"] for row in records}) == len(ARMS),
        "controls reused a native process group",
    )
    provenance = records[0]["provenance"]
    _require(
        all(row["provenance"] == provenance for row in records),
        "native control provenance or mode differs",
    )
    _require(
        all(
            row["physics_steps_per_control"] == provenance["mode"]["control_decimation"]
            for row in records
        ),
        "control decimation differs from native mode",
    )
    if current is not None:
        _require(
            current == provenance, "final native initialization differs from controls"
        )
        _require(
            os.getpid() not in {row["native_pid"] for row in records},
            "final stage reused a control process",
        )
        _require(
            os.getpgrp() not in {row["native_process_group"] for row in records},
            "final stage reused a control process group",
        )
    return provenance


def _check_native_binding(recipe, request, provenance):
    binding, mode, runtime = (
        request["binding"],
        provenance["mode"],
        provenance["runtime"],
    )
    _require(
        all(
            mode[key] == binding[key] for key in ("stage", "training", "enable_cameras")
        ),
        "native mode differs from request",
    )
    _require(
        math.isfinite(mode["physics_dt"]) and mode["physics_dt"] > 0,
        "invalid physics timestep",
    )
    expected = {
        "robot_population": recipe.num_envs,
        "scene_instances": 1,
        "scene_sha256": recipe.scene_sha256,
        "scene_prim": recipe.scene_prim,
    }
    _require(
        all(runtime.get(key) == value for key, value in expected.items()),
        "native scene/population differs from request",
    )
    _check_controller_and_policy(recipe, binding, runtime, provenance)


def _check_controller_and_policy(recipe, binding, runtime, provenance):
    if recipe.reference_controller_sha256 is not None:
        _require(
            runtime["reference_controller"]["sha256"]
            == recipe.reference_controller_sha256,
            "native controller differs from request",
        )
    _require(
        provenance["settings"]["seed"] == binding["runner_seed"],
        "native runner seed differs",
    )
    digest = provenance["initialization"]["policy_state_sha256"]
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest),
        "invalid initialized policy digest",
    )


def verify_controls(recipe, source, output, stage, request_sha256, current=None):
    """Recompute all gates and exact bindings before trusting native control results.

    Args:
        recipe: Current sealed recipe.
        source: Current materialized inputs.
        output: This attempt's final stage output directory.
        stage: Original stage mode.
        request_sha256: Parent's fresh request pin.
        current: Current native runtime/settings/initialization, or None in parent.
    Returns:
        Isolation results and complete same-attempt control file inventory.
    Raises:
        ValueError: Any input, provenance, trace or physical gate differs.
    """
    from npa.workflows.navigation.measure import assess_control_traces

    request, root = read_request(recipe, source, output, stage, request_sha256)
    pairs = {
        arm: _read_control(root, arm, request, request_sha256, recipe) for arm in ARMS
    }
    records = [value[0] for value in pairs.values()]
    provenance = _check_provenance(
        records, [value[2] for value in pairs.values()], current
    )
    _check_native_binding(recipe, request, provenance)
    result = assess_control_traces(
        recipe, {arm: value[1] for arm, value in pairs.items()}, output
    )
    return {
        "schema": "npa.navigation.fresh-controls.v1",
        "request_sha256": request_sha256,
        "attempt_id": request["attempt_id"],
        "isolation": result,
        "files": _files(root, ("accepted.json",)),
        "provenance": provenance,
    }


def accept_controls(recipe, source, output, stage, request_sha256):
    """Write parent acceptance only after raw controls pass recomputation.

    Args:
        recipe: Sealed recipe.
        source: Materialized input directory.
        output: Same-attempt stage output directory.
        stage: Native mode.
        request_sha256: Fresh request pin.
    Returns:
        SHA-256 supplied only to the final native child.
    Raises:
        ValueError: Control evidence fails verification.
    """
    result = verify_controls(recipe, source, output, stage, request_sha256)
    path = output / "controls/accepted.json"
    _write(path, result)
    return file_sha256(path)


def consume_controls(
    recipe, source, output, stage, request_sha256, acceptance_sha256, current
):
    """Independently verify parent acceptance inside a fresh final native process.

    Args:
        recipe: Current sealed recipe.
        source: Materialized inputs.
        output: Same-attempt output directory.
        stage: Original stage mode.
        request_sha256: Parent's request pin.
        acceptance_sha256: Parent's newly completed aggregate pin.
        current: Actual native runtime/settings/initialization.
    Returns:
        Independently recomputed isolation evidence.
    Raises:
        ValueError: Aggregate, raw evidence or current bindings differ.
    """
    path = output / "controls/accepted.json"
    _require(
        acceptance_sha256 and file_sha256(path) == acceptance_sha256,
        "missing or changed current-attempt controls",
    )
    expected = json.loads(path.read_text())
    actual = verify_controls(recipe, source, output, stage, request_sha256, current)
    _require(
        expected == actual, "parent control receipt differs from recomputed evidence"
    )
    final = {
        "schema": "npa.navigation.fresh-final-process.v1",
        "attempt_id": actual["attempt_id"],
        "native_pid": os.getpid(),
        "native_process_group": os.getpgrp(),
        "provenance": current,
        "native_clock_origin": native_clock(),
        "control_receipt_sha256": acceptance_sha256,
        "control_request_sha256": request_sha256,
    }
    _write(output / "native-process.json", final)
    return {
        **actual["isolation"],
        "control_receipt_sha256": acceptance_sha256,
        "control_request_sha256": request_sha256,
        "native_clock_origin": final["native_clock_origin"],
    }
