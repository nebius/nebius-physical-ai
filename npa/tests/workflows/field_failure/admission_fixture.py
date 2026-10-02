"""Build explicitly synthetic native evidence for hostile admission-contract tests."""

import json
import shutil
from types import SimpleNamespace

import numpy as np

from npa.workflows.navigation import control_protocol as controls
from npa.workflows.navigation.artifacts import _files, file_sha256, write_json
from npa.workflows.navigation.contract import read_recipe
from npa.workflows.navigation.measure import episode_rows
from npa.workflows.navigation.probe_evidence import save_trace


def build_observation(tmp_path, monkeypatch):
    from npa.workflows.field_failure.reference_demo_inputs import prepare_warehouse

    root = tmp_path / "observation"
    prepare_warehouse(
        root,
        image="registry.example/native@sha256:" + "a" * 64,
        count=2,
        iterations=3,
        steps=3,
    )
    (root / "baseline.pt").write_bytes(b"synthetic checkpoint only")
    recipe = json.loads((root / "recipe.json").read_text())
    recipe["initial_checkpoint"] = {
        "file": "baseline.pt",
        "sha256": file_sha256(root / "baseline.pt"),
    }
    write_json(root / "recipe.json", recipe)
    recipe = read_recipe(root)
    cases = [case.model_dump() for case in recipe.eval_cases]
    office = {
        "count": 2,
        "case_ids": [c["id"] for c in cases],
        "seeds": [c["seed"] for c in cases],
    }
    from npa.workflows.field_failure.reference_demo_inputs import cohort_manifest

    office.update(cohort_manifest({"office": cases})["cohorts"]["office"])
    report = _native_report(root, recipe, office, monkeypatch)
    result = SimpleNamespace(
        root=root, recipe=recipe, plan=_plan(recipe, office), report=report
    )
    set_outcomes(result, success=False)
    return result


def _plan(recipe, office):
    from npa.workflows.field_failure.reference_demo_failure_evidence import (
        FAILURE_RULE,
        observation_recipe_digest,
    )

    return {
        "num_envs": 2,
        "baseline_iterations": 3,
        "candidate_iterations": 3,
        "failure_observation": {
            "rule": FAILURE_RULE,
            "episode_steps": 3,
            "recipe_sha256": observation_recipe_digest(recipe),
            "input": {"uri": "s3://example/input.tar", "sha256": "a" * 64},
        },
        "cohorts": {
            "regions": {"training": {"office": office}},
            "geometry": {
                "scene_sha256": recipe.scene_sha256,
                "capture_manifest_sha256": "c" * 64,
            },
        },
        "capture": {"uri": "s3://example/capture.tar", "sha256": "d" * 64},
        "adapters": {"evaluate": {"runtime_image": recipe.image}},
    }


def _state(cases):
    count = len(cases)
    return {
        "position_m": np.array([c.position_m for c in cases]),
        "goal_m": np.array([c.goal_m for c in cases]),
        "heading_rad": np.array([c.heading_rad for c in cases]),
        "obstacle_contact": np.zeros(count),
        "peer_contact": np.zeros(count),
        "physical_failure": np.zeros(count),
        "upright_cosine": np.ones(count),
        "ground_clearance_m": np.full(count, 0.5),
    }


def _control_trace(recipe, arm):
    rows = []
    for step in range(len(recipe.probe.actions) + 1):
        state = _state(controls.control_cases(recipe, arm))
        state["position_m"][0, 0] += step * 0.1
        if arm == "obstacle" and step:
            state["obstacle_contact"][0] = 2.0
        rows.append(
            {
                "state": state,
                "observations": {"policy/value": np.ones((recipe.num_envs, 3))},
            }
        )
    return rows


def _process(folder, arm, digest, index):
    (folder / "runtime.log").write_text("Synthetic unit evidence only\n")
    write_json(
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


def _control_outputs(root, recipe, runtime, monkeypatch):
    provenance = {
        "runtime": runtime,
        "settings": {"seed": recipe.train_cases[0].seed},
        "initialization": {
            "mode": "checkpoint_evaluation",
            "policy_state_sha256": "f" * 64,
            "checkpoint_sha256": recipe.initial_checkpoint.sha256,
        },
        "mode": {
            "stage": "evaluate-checkpoint",
            "training": False,
            "enable_cameras": True,
            "physics_dt": 0.005,
            "control_decimation": 40,
        },
    }
    monkeypatch.setenv("NPA_TASK_IMAGE", recipe.image)
    source = root.parent / "native-input"
    shutil.copytree(root, source)
    digest = controls.create_request(recipe, source, root, "evaluate-checkpoint")
    request = json.loads((root / "controls/request.json").read_text())
    for index, arm in enumerate(controls.ARMS):
        _control_arm(root, recipe, request, digest, provenance, index, arm, monkeypatch)
    accepted = controls.accept_controls(
        recipe, source, root, "evaluate-checkpoint", digest
    )
    with monkeypatch.context() as patch:
        patch.setattr(controls, "native_clock", lambda: {"step": 0, "seconds": 0.0})
        patch.setattr(controls.os, "getpgrp", lambda: 20010)
        isolation = controls.consume_controls(
            recipe, source, root, "evaluate-checkpoint", digest, accepted, provenance
        )
    _process(root, "main", digest, 10)
    return isolation


def _control_arm(root, recipe, request, digest, provenance, index, arm, monkeypatch):
    folder = root / "controls" / arm
    folder.mkdir()
    save_trace(folder, arm, _control_trace(recipe, arm))
    steps = len(recipe.probe.actions) * 40
    clocks = {
        "before": {"step": 0, "seconds": 0.0},
        "after": {"step": steps, "seconds": steps * 0.005},
    }
    with monkeypatch.context() as patch:
        patch.setattr(controls.os, "getpid", lambda: 10000 + index)
        patch.setattr(controls.os, "getpgrp", lambda: 20000 + index)
        controls.save_control(
            recipe, folder, request, digest, arm, provenance, clocks, 40
        )
    _process(folder, arm, digest, index)


def _native_report(root, recipe, office, monkeypatch):
    runtime = {
        "scene_sha256": recipe.scene_sha256,
        "robot_population": recipe.num_envs,
        "scene_instances": 1,
        "scene_prim": recipe.scene_prim,
        "reference_controller": {"sha256": recipe.reference_controller_sha256},
    }
    report = {
        "schema": "npa.navigation.evaluation.v1",
        "policy_loaded": True,
        "checkpoint_sha256": recipe.initial_checkpoint.sha256,
        "evaluation_inputs_sha256": office["sha256"],
        "recipe_sha256": file_sha256(root / "recipe.json"),
        "image": recipe.image,
        "task": recipe.task,
        "adapter_sha256": recipe.adapter_sha256,
        "source_bundle_sha256": recipe.source_bundle_sha256,
        "initialization": {
            "mode": "resume_native_checkpoint",
            "checkpoint_sha256": recipe.initial_checkpoint.sha256,
        },
        "runtime": runtime,
        "workbench_sources": {
            p.name: file_sha256(p)
            for p in controls.Path(controls.__file__).parent.glob("*.py")
        },
    }
    report["isolation"] = _control_outputs(root, recipe, runtime, monkeypatch)
    return report


def set_outcomes(fixture, *, success=False):
    trajectory = []
    for step in range(2 if success else fixture.recipe.episode_steps + 1):
        state = _state(fixture.recipe.eval_cases)
        if step and success:
            state["position_m"][:, :2] = state["goal_m"]
        state["active"] = np.ones(fixture.recipe.num_envs, dtype=bool)
        trajectory.append(state)
    rows = episode_rows(
        fixture.recipe.eval_cases, trajectory, fixture.recipe.goal_tolerance_m
    )
    fixture.report.update(episodes=rows, success_rate=float(success), passed=success)
    write_json(
        fixture.root / "trajectory.json",
        [{k: v.tolist() for k, v in row.items()} for row in trajectory],
    )
    write_json(fixture.root / "evaluation.json", fixture.report)
    reseal(fixture.root)


def reseal(root):
    write_json(root / "checksums.json", _files(root))
