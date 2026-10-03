"""Verify complete native simulation failures against frozen training routes and raw traces."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np

from npa.workflows.navigation.artifacts import _files, file_sha256
from npa.workflows.navigation.contract import read_recipe
from npa.workflows.navigation.measure import episode_rows, snapshot

FAILURE_RULE = {
    "schema": "npa.field-failure.simulation-admission-rule.v1",
    "cohort": "training.office",
    "predicate": "completed native episode with success=false",
    "minimum_observed_failures": 1,
    "capture_selection": "admit the complete bound public capture; retain all training routes",
    "physical_field_failure_claim": False,
}


def observation_recipe_digest(recipe):
    """Hash frozen observation settings while excluding only the later baseline binding.

    Args:
        recipe: Validated native observation recipe.
    Returns:
        SHA-256 of every setting with initial_checkpoint set to None.
    Raises:
        ValueError: Recipe serialization contains a nonfinite value.
    """
    value = recipe.model_dump()
    value["initial_checkpoint"] = None
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def verify_observation(root, plan, checkpoint_sha256):
    """Recompute every observation outcome before any public capture is admitted.

    Args:
        root: Materialized completed native evaluation with verified file seals.
        plan: Pre-learning reference plan containing the immutable admission rule.
        checkpoint_sha256: Exact checkpoint from the completed baseline publication.
    Returns:
        Ordered measured episodes and hashes of every retained observation file.
    Raises:
        ValueError: Completion, controls, checkpoint, cases or raw outcomes differ.
        OSError: Required complete evidence is unavailable.
    """
    files = _files(root)
    if files != json.loads((root / "checksums.json").read_text()):
        raise ValueError("observation files differ from the completed checksum seal")
    if "failure.json" in files or plan["failure_observation"]["rule"] != FAILURE_RULE:
        raise ValueError("observation failed or admission rule differs from the freeze")
    recipe = read_recipe(root)
    report = json.loads((root / "evaluation.json").read_text())
    _verify_identity(root, recipe, report, plan, checkpoint_sha256)
    _verify_controls(root, recipe, report)
    trajectory = _trajectory(root, recipe)
    rows = episode_rows(recipe.eval_cases, trajectory, recipe.goal_tolerance_m)
    rate = sum(row["success"] for row in rows) / len(rows)
    if rows != report["episodes"] or report["success_rate"] != rate:
        raise ValueError("observed failures differ from the actual native trajectories")
    if report["passed"] is not (rate >= recipe.minimum_success_rate):
        raise ValueError("observation quality summary differs from measured outcomes")
    return rows, files


def _verify_identity(root, recipe, report, plan, checkpoint):
    if (
        observation_recipe_digest(recipe)
        != plan["failure_observation"]["recipe_sha256"]
    ):
        raise ValueError("observation recipe differs from its pre-learning freeze")
    office = plan["cohorts"]["regions"]["training"]["office"]
    cases = [case.model_dump() for case in recipe.eval_cases]
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    if (
        digest != office["sha256"]
        or len(cases) != office["count"]
        or [case["id"] for case in cases] != office["case_ids"]
        or [case["seed"] for case in cases] != office["seeds"]
    ):
        raise ValueError(
            "observation differs from ordered frozen office training cases"
        )
    expected = {
        "schema": "npa.navigation.evaluation.v1",
        "policy_loaded": True,
        "checkpoint_sha256": checkpoint,
        "evaluation_inputs_sha256": digest,
        "recipe_sha256": file_sha256(root / "recipe.json"),
        "image": recipe.image,
        "task": recipe.task,
        "adapter_sha256": recipe.adapter_sha256,
        "source_bundle_sha256": recipe.source_bundle_sha256,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "observation does not bind the frozen native checkpoint and recipe"
        )
    _verify_runtime(recipe, report, plan, checkpoint)


def _verify_runtime(recipe, report, plan, checkpoint):
    runtime = report["runtime"]
    if (
        recipe.num_envs != plan["num_envs"]
        or runtime["robot_population"] != recipe.num_envs
        or recipe.episode_steps != plan["failure_observation"]["episode_steps"]
        or recipe.iterations != plan["baseline_iterations"]
        or recipe.scene_sha256 != plan["cohorts"]["geometry"]["scene_sha256"]
        or runtime["scene_sha256"] != recipe.scene_sha256
        or runtime["scene_instances"] != 1
        or recipe.image != plan["adapters"]["evaluate"]["runtime_image"]
    ):
        raise ValueError("observation population, budget, scene or runtime differs")
    initial = report["initialization"]
    if (
        recipe.initial_checkpoint is None
        or recipe.initial_checkpoint.sha256 != checkpoint
        or initial["checkpoint_sha256"] != checkpoint
        or initial["mode"] != "resume_native_checkpoint"
        or report["isolation"]["passed"] is not True
    ):
        raise ValueError(
            "observation requires the loaded baseline and successful controls"
        )
    sources = hashlib.sha256(
        json.dumps(report["workbench_sources"], sort_keys=True).encode()
    ).hexdigest()
    if sources != recipe.source_bundle_sha256:
        raise ValueError("observation source inventory differs from its sealed recipe")


def _verify_controls(root, recipe, report):
    from npa.workflows.navigation.control_protocol import (
        ARMS,
        _check_native_binding,
        _check_provenance,
        _read_control,
    )
    from npa.workflows.navigation.measure import assess_control_traces

    request = json.loads((root / "controls/request.json").read_text())
    digest = file_sha256(root / "controls/request.json")
    pairs = {
        arm: _read_control(root / "controls", arm, request, digest, recipe)
        for arm in ARMS
    }
    native = json.loads((root / "native-process.json").read_text())
    provenance = _check_provenance(
        [p[0] for p in pairs.values()],
        [p[2] for p in pairs.values()],
        native["provenance"],
    )
    _check_native_binding(recipe, request, provenance)
    _verify_control_inputs(root, recipe, request, native, report)
    _verify_final_process(root, native, report, digest)
    if native["native_pid"] in {p[0]["native_pid"] for p in pairs.values()} or native[
        "native_process_group"
    ] in {p[0]["native_process_group"] for p in pairs.values()}:
        raise ValueError("observation reused a control process")
    measured = assess_control_traces(
        recipe, {arm: p[1] for arm, p in pairs.items()}, None
    )
    if any(report["isolation"].get(key) != value for key, value in measured.items()):
        raise ValueError(
            "observation physical controls differ from retained measurements"
        )


def _verify_final_process(root, native, report, digest):
    from npa.workflows.navigation.control_protocol import _check_process

    process = _check_process(root, "main", digest)
    if (
        native["schema"] != "npa.navigation.fresh-final-process.v1"
        or native["native_process_group"] != process["owned_process_group"]
        or native["control_request_sha256"] != digest
        or native["control_receipt_sha256"]
        != file_sha256(root / "controls/accepted.json")
        or native["provenance"]["initialization"]["checkpoint_sha256"]
        != report["checkpoint_sha256"]
        or native["provenance"]["initialization"]["mode"] != "checkpoint_evaluation"
        or report["isolation"]["control_request_sha256"] != digest
        or report["isolation"]["control_receipt_sha256"]
        != native["control_receipt_sha256"]
    ):
        raise ValueError(
            "observation final process differs from its after-load controls"
        )


def _verify_control_inputs(root, recipe, request, native, report):
    from npa.workflows.navigation.control_protocol import _files as control_files

    accepted = json.loads((root / "controls/accepted.json").read_text())
    expected = {
        "stage": "evaluate-checkpoint",
        "training": False,
        "enable_cameras": True,
        "population": recipe.num_envs,
        "image": recipe.image,
        "source_bundle_sha256": recipe.source_bundle_sha256,
        "adapter_sha256": recipe.adapter_sha256,
        "environment_seed": recipe.eval_cases[0].seed,
        "runner_seed": recipe.train_cases[0].seed,
    }
    binding = request["binding"]
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("observation controls used different frozen native settings")
    for name in ("recipe.json", recipe.scene_file, recipe.initial_checkpoint.file):
        if binding["input_sha256"][name] != file_sha256(root / name):
            raise ValueError("observation controls used different input bytes")
    if (
        accepted["files"] != control_files(root / "controls", ("accepted.json",))
        or accepted["provenance"] != native["provenance"]
        or not (accepted["attempt_id"] == native["attempt_id"] == request["attempt_id"])
        or native["provenance"]["runtime"] != report["runtime"]
    ):
        raise ValueError(
            "observation controls and native completion have different provenance"
        )


def _trajectory(root, recipe):
    raw = json.loads((root / "trajectory.json").read_text())
    if not 1 < len(raw) <= recipe.episode_steps + 1:
        raise ValueError("observation trajectory is incomplete or exceeds its horizon")
    states, active = [], np.ones(recipe.num_envs, dtype=bool)
    goals = np.asarray([case.goal_m for case in recipe.eval_cases])
    for index, row in enumerate(raw):
        state = snapshot(SimpleNamespace(measure=lambda _: row), None, recipe.num_envs)
        mask = np.asarray(row["active"])
        if (
            mask.dtype != bool
            or mask.shape != active.shape
            or not np.array_equal(mask, active)
        ):
            raise ValueError(
                "observation active mask differs from measured termination"
            )
        if not np.allclose(state["goal_m"], goals, atol=recipe.probe.tolerance, rtol=0):
            raise ValueError("observation goal differs from the frozen training route")
        states.append({**state, "active": mask})
        if index:
            reached = np.linalg.norm(state["position_m"][:, :2] - goals, axis=1)
            active &= reached > recipe.goal_tolerance_m
            active &= (state["obstacle_contact"] == 0) & (state["peer_contact"] == 0)
            active &= state["physical_failure"] == 0
    if len(raw) < recipe.episode_steps + 1 and active.any():
        raise ValueError("observation stopped before all episodes terminated")
    _verify_initial(states[0], recipe)
    return states


def _verify_initial(state, recipe):
    positions = [case.position_m for case in recipe.eval_cases]
    headings = np.asarray([case.heading_rad for case in recipe.eval_cases])
    error = (state["heading_rad"] - headings + np.pi) % (2 * np.pi) - np.pi
    if (
        not np.allclose(
            state["position_m"], positions, atol=recipe.probe.tolerance, rtol=0
        )
        or np.max(np.abs(error)) > recipe.probe.tolerance
        or state["physical_failure"].any()
    ):
        raise ValueError("observation did not begin at the frozen physical reset")
