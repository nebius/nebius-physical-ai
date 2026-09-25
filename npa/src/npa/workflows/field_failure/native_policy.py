"""Fine-tune and compare exact native Isaac checkpoints using sealed navigation bundles."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

from npa.workflows.field_failure.native_artifacts import (
    _archive,
    _bundle,
    _download,
    _identity,
    _protocol,
    _recipe,
    _upload,
    _upload_json,
)


def train(request: dict) -> dict:
    """Continue native PPO from the exact baseline on reconstructed training scenes.

    Args:
        request: Sealed field-failure training request.
    Returns:
        Changed checkpoint identity and native learning evidence for every scene.
    Raises:
        ValueError: Scene, protocol or baseline integrity differs.
        RuntimeError: Native training or its physical validation fails.
        OSError: Input, runtime or artifact publication fails.
    """
    with tempfile.TemporaryDirectory(prefix="npa-field-training-") as temporary:
        root = Path(temporary)
        protocol = _protocol(request, root)
        checkpoint = root / "baseline.pt"
        _download(request["baseline"]["checkpoint"], checkpoint)
        reports = []
        for index, scene in enumerate(request["scenes"]):
            output, report = _train_scene(
                request, protocol, scene, checkpoint, root, index
            )
            checkpoint = output / "policy.pt"
            reports.append(report)
        return _training_record(request, root, checkpoint, reports)


def _train_scene(request, protocol, scene, checkpoint, root, index):
    source = _bundle(scene["asset"], root, f"scene-{index}")
    baseline_replay = _diagnostic_replay(
        request, source, protocol, checkpoint, root, f"baseline-{index}"
    )
    _initialize(source, protocol, checkpoint)
    output = root / f"trained-{index}"
    report = _execute("train", source, output, protocol)
    candidate_replay = _diagnostic_replay(
        request, source, protocol, output / "policy.pt", root, f"candidate-{index}"
    )
    report["training_exposed_replay"] = {
        "baseline": baseline_replay,
        "candidate": candidate_replay,
        "used_for_promotion": False,
    }
    archive = root / f"training-{index}.tar"
    _archive(output, archive)
    evidence = _upload(archive, request["output_prefix"] + f"training-{index}.tar")
    return output, {
        "scenario_id": scene["scenario_id"],
        "native": report,
        "artifacts": evidence,
    }


def _training_record(request, root, checkpoint, reports):
    prefix = request["output_prefix"]
    candidate = {
        "policy_id": "native-candidate-" + request["attempt_id"],
        "checkpoint": _upload(checkpoint, prefix + "candidate.pt"),
    }
    if candidate["checkpoint"]["sha256"] == request["baseline"]["checkpoint"]["sha256"]:
        raise ValueError("native fine-tuning returned the unchanged baseline")
    evidence = _upload_json(
        reports, root / "learning.json", prefix + "native-learning.json"
    )
    return {
        **_identity(request, "npa.field-failure.training.v1"),
        "reconstruction_sha256": request["reconstruction_sha256"],
        "initial_checkpoint_sha256": request["baseline"]["checkpoint"]["sha256"],
        "candidate": candidate,
        "training_scenario_ids": [s["scenario_id"] for s in request["scenes"]],
        "training_group_ids": list(
            dict.fromkeys(s["group_id"] for s in request["scenes"])
        ),
        "training_scene_sha256": [s["asset"]["sha256"] for s in request["scenes"]],
        "evidence": evidence,
    }


def _initialize(source, protocol, checkpoint):
    from npa.workflows.navigation.artifacts import file_sha256
    from npa.workflows.navigation.contract import read_recipe

    recipe = _recipe(source, protocol)
    shutil.copyfile(checkpoint, source / "baseline.pt")
    recipe["initial_checkpoint"] = {
        "file": "baseline.pt",
        "sha256": file_sha256(checkpoint),
    }
    (source / "recipe.json").write_text(json.dumps(recipe, allow_nan=False))
    read_recipe(source)


def _execute(stage, source, output, protocol):
    from npa.workflows.navigation.stages import prepare, run_stage

    image = protocol["navigation_image"]
    if os.environ.get("NPA_TASK_IMAGE") != image:
        raise ValueError("executing native image differs from the sealed protocol")
    prepared = source.parent / (output.name + "-prepared")
    prepare(str(source), str(prepared), image)
    return run_stage(stage, str(prepared), str(output))


def _diagnostic_replay(request, source, protocol, checkpoint, root, name):
    recipe = json.loads((source / "recipe.json").read_text())
    if len(recipe["train_cases"]) != recipe["num_envs"]:
        raise ValueError("native replay requires one declared training case per robot")
    replay = root / (name + "-replay-input")
    shutil.copytree(source, replay)
    recipe["initial_checkpoint"] = None
    recipe["train_cases"], recipe["eval_cases"] = (
        recipe["eval_cases"],
        recipe["train_cases"],
    )
    (replay / "recipe.json").write_text(json.dumps(recipe, allow_nan=False))
    _initialize(replay, protocol, checkpoint)
    output = root / (name + "-replay-output")
    report = _execute("evaluate-checkpoint", replay, output, protocol)
    from npa.workflows.navigation.artifacts import file_sha256

    if report.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("diagnostic replay loaded a different checkpoint")
    archive = root / (name + "-replay.tar")
    _archive(output, archive)
    evidence = _upload(archive, request["output_prefix"] + name + "-replay.tar")
    return {
        "scope": "training-exposed diagnostic",
        "checkpoint_sha256": report["checkpoint_sha256"],
        "success_rate": report["success_rate"],
        "episodes": report["episodes"],
        "artifacts": evidence,
    }


def evaluate(request: dict) -> dict:
    """Run the selected checkpoint on every sealed held-out scene and measured seed.

    Args:
        request: Sealed single-policy evaluation request with common protocol.
    Returns:
        Completed episodes with metrics recomputed from native trajectory arrays.
    Raises:
        ValueError: Protocol, checkpoint, seed coverage or measurements differ.
        RuntimeError: Native simulator execution or isolation validation fails.
        OSError: Required input, runtime or artifact publication fails.
    """
    with tempfile.TemporaryDirectory(prefix="npa-field-evaluation-") as temporary:
        root = Path(temporary)
        protocol = _protocol(request, root)
        checkpoint = root / "baseline.pt"
        _download(request["policy"]["checkpoint"], checkpoint)
        episodes = []
        for index, scene in enumerate(request["held_out"]):
            episodes.extend(
                _evaluate_scene(request, protocol, scene, checkpoint, root, index)
            )
        return {
            **_identity(request, "npa.field-failure.evaluation.v1"),
            "protocol_sha256": request["protocol"]["sha256"],
            "policy": request["policy"],
            "episodes": episodes,
        }


def _evaluate_scene(request, protocol, scene, checkpoint, root, index):
    from npa.workflows.navigation.contract import read_recipe

    source = _bundle(scene["asset"], root, f"held-out-{index}")
    _initialize(source, protocol, checkpoint)
    recipe = read_recipe(source)
    if sorted(c.seed for c in recipe.eval_cases) != sorted(scene["seeds"]):
        raise ValueError("native held-out reset seeds differ from sealed cohort")
    output = root / f"evaluation-{index}"
    report = _execute("evaluate-checkpoint", source, output, protocol)
    if report.get("checkpoint_sha256") != request["policy"]["checkpoint"]["sha256"]:
        raise ValueError("native evaluation loaded a different checkpoint")
    trajectories = json.loads((output / "trajectory.json").read_text())
    rows = _measurements(recipe, trajectories, report)
    return [
        _episode(request, scene, row, trajectories, root, index, i)
        for i, row in enumerate(rows)
    ]


def _measurements(recipe, trajectory, report):
    import numpy as np
    from npa.workflows.navigation.measure import episode_rows

    arrays = [
        {key: np.asarray(value) for key, value in row.items()} for row in trajectory
    ]
    rows = episode_rows(recipe.eval_cases, arrays, recipe.goal_tolerance_m)
    if rows != report["episodes"] or not report["policy_loaded"]:
        raise ValueError("native evaluation summary differs from measured trajectories")
    return rows


def _episode(request, scene, row, trajectories, root, scene_index, index):
    name = f"scene-{scene_index}-episode-{index}"
    trace = [
        {key: value[index] for key, value in frame.items()} for frame in trajectories
    ]
    trajectory = _upload_json(
        trace,
        root / (name + "-trajectory.json"),
        request["output_prefix"] + name + "-trajectory.json",
    )
    record = _episode_metrics(scene, row, request["metrics"])
    wrapper = {
        **record,
        "schema_version": "npa.field-failure.episode.v1",
        "bundle_sha256": request["bundle_sha256"],
        "checkpoint_sha256": request["policy"]["checkpoint"]["sha256"],
        "protocol_sha256": request["protocol"]["sha256"],
        "trajectory": trajectory,
    }
    record["evidence"] = _upload_json(
        wrapper, root / (name + ".json"), request["output_prefix"] + name + ".json"
    )
    return record


def _episode_metrics(scene, row, metrics):
    measured = {
        key: float(row[key])
        for key in (
            "success",
            "collision_steps",
            "peer_collision_steps",
            "physical_failure_steps",
            "path_length_m",
            "goal_distance_m",
        )
    }
    names = {metric["name"] for metric in metrics}
    if not names.issubset(measured):
        raise ValueError("native evaluator received an unsupported metric")
    failure = (
        row["collision_steps"]
        or row["peer_collision_steps"]
        or row["physical_failure_steps"]
    )
    if row["success"] and failure:
        raise ValueError("native episode reports success after physical failure")
    return {
        "scenario_id": scene["scenario_id"],
        "scene_sha256": scene["asset"]["sha256"],
        "seed": row["seed"],
        "status": "completed",
        "steps": row["steps"],
        "termination": "success"
        if row["success"]
        else "failure"
        if failure
        else "timeout",
        "metrics": {key: measured[key] for key in names},
    }
