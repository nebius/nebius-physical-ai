"""Bind scan-demo summary measurements to sealed reconstruction and policy artifacts."""

from __future__ import annotations

import json
import math

from npa.workflows.navigation.artifacts import file_sha256
from npa.workflows.navigation.contract import read_recipe


def _record(root, name):
    return json.loads((root / name).read_text())


def _scan_binding(scan, training, recipe):
    lineage = _record(training, "scan-lineage.json")
    if lineage.get("schema") != "npa.navigation.scan_handoff.v1":
        raise ValueError("training omitted the scan handoff lineage")
    if lineage.get("recipe_sha256") != file_sha256(training / "recipe.json"):
        raise ValueError("scan handoff recipe differs from the trained policy")
    expected = lineage["source_sha256"]
    for name in (
        "capture.json",
        "reconstruction.json",
        "provenance.json",
        "physics_validation.json",
        "cases.json",
    ):
        if file_sha256(training / "scan" / name) != expected[name]:
            raise ValueError("training scan evidence differs from its handoff")
    for name in ("capture.json", "reconstruction.json"):
        if file_sha256(scan / name) != expected[name]:
            raise ValueError("report reconstruction differs from the trained scan")
    if expected["scene.usdz"] != recipe.scene_sha256:
        raise ValueError("report scan scene differs from the trained scene")
    cases = _record(training / "scan", "cases.json")
    for name in ("train_cases", "eval_cases"):
        if cases[name] != [case.model_dump() for case in getattr(recipe, name)]:
            raise ValueError("recipe cases differ from the measured scan handoff")


def _training_binding(training, recipe):
    from npa.workflows.navigation.runtime import _verify_training_binding

    learning = _record(training, "training.json")
    _verify_training_binding(learning, recipe, training)
    if file_sha256(training / "policy.pt") != learning["checkpoint_sha256"]:
        raise ValueError("report policy bytes differ from the training checkpoint")
    if learning.get("heldout_used_for_training") is not False:
        raise ValueError("training did not retain the held-out split boundary")
    if learning["iterations"] != recipe.iterations:
        raise ValueError("training iteration count differs from the sealed recipe")
    if learning.get("source_bundle_sha256") != recipe.source_bundle_sha256:
        raise ValueError("training sources differ from the sealed recipe")
    return learning


def _evaluation_binding(evaluation, training, recipe, learning):
    from npa.workflows.navigation.runtime import _case_digest

    if not (evaluation / "evaluation.json").is_file():
        if not (evaluation / "failure.json").is_file():
            raise ValueError("evaluation has neither completion nor failure evidence")
        return None
    result = _record(evaluation, "evaluation.json")
    expected = {
        "schema": "npa.navigation.evaluation.v1",
        "policy_loaded": True,
        "checkpoint_sha256": learning["checkpoint_sha256"],
        "evaluation_inputs_sha256": _case_digest(recipe),
        "source_bundle_sha256": recipe.source_bundle_sha256,
    }
    expected.update(
        {
            key: learning[key]
            for key in ("task", "image", "adapter_sha256", "recipe_sha256")
        }
    )
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("report evaluation differs from the trained experiment")
    if result.get("runtime", {}).get("scene_sha256") != recipe.scene_sha256:
        raise ValueError("report evaluation belongs to a different scene")
    if file_sha256(evaluation / "policy.pt") != learning["checkpoint_sha256"]:
        raise ValueError("evaluation checkpoint bytes differ from training")
    if file_sha256(evaluation / "recipe.json") != file_sha256(training / "recipe.json"):
        raise ValueError("evaluation recipe differs from training")
    _cohort(result, recipe)
    return result


def _cohort(result, recipe):
    rows = result["episodes"]
    actual = [(row["case_id"], row["seed"]) for row in rows]
    expected = [(case.id, case.seed) for case in recipe.eval_cases]
    if actual != expected or any(type(row.get("success")) is not bool for row in rows):
        raise ValueError("report episodes differ from the fixed held-out cohort")
    rate = sum(row["success"] for row in rows) / len(rows)
    if not math.isclose(result["success_rate"], rate, rel_tol=0, abs_tol=1e-12):
        raise ValueError("reported success differs from measured episode outcomes")
    if result.get("passed") is not (rate >= recipe.minimum_success_rate):
        raise ValueError("reported quality gate differs from the sealed recipe")


def measured_summary(scan, training, evaluation) -> tuple[dict, dict | None]:
    """Verify experiment identity before deriving a portable, measured report.

    Args:
        scan: Materialized, verified reconstruction publication.
        training: Materialized, checksum-verified native training publication.
        evaluation: Materialized, checksum-verified native evaluation publication.
    Returns:
        Safe summary and completed evaluation, or None for an incomplete run.
    Raises:
        ValueError: Reconstruction, checkpoint, recipe or cohort evidence differs.
        OSError: Required artifact bytes cannot be read.
    """
    recipe = read_recipe(training)
    learning = _training_binding(training, recipe)
    _scan_binding(scan, training, recipe)
    result = _evaluation_binding(evaluation, training, recipe, learning)
    reconstruction = _record(scan, "reconstruction.json")
    summary = _summary_values(reconstruction, learning, recipe, result)
    summary["source_seals"] = {
        "reconstruction": file_sha256(scan / ".npa-navigation-complete.json"),
        "training": file_sha256(training / "checksums.json"),
        "evaluation": file_sha256(evaluation / "checksums.json"),
    }
    return summary, result


def _summary_values(reconstruction, learning, recipe, result):
    from npa.workbench.nurec.navigation_sample import ATTRIBUTION

    return {
        "schema": "npa.navigation.sample_report.v1",
        "passed": bool(result and result["passed"]),
        "evaluation_complete": result is not None,
        "success_rate": result["success_rate"] if result else None,
        "episodes": len(result["episodes"]) if result else 0,
        "minimum_success_rate": recipe.minimum_success_rate,
        "iterations": learning["iterations"],
        "checkpoint_sha256": learning["checkpoint_sha256"],
        "recipe_sha256": learning["recipe_sha256"],
        "sensor_mode": recipe.sensor_mode,
        "integration_frames": reconstruction["integration_frames"],
        "validation_frames": reconstruction["validation_frames"],
        "attribution": ATTRIBUTION,
        "focal_scored_end_step": result["episodes"][0]["steps"] if result else None,
        "preview_scope": "Scored focal episode only. Complete original recordings may also contain later observer motion.",
    }
