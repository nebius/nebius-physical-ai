"""Evaluate actual pre-update and trained checkpoints on identical held-out cases."""

import json
from pathlib import Path
import shutil
import tempfile

from npa.workflows.navigation.artifacts import (
    file_sha256,
    materialize,
    publish,
    write_json,
)
from npa.workflows.navigation.stages import prepare, run_stage


def compare_reference(input_path, output_path):
    """Reload both real native checkpoints and retain paired measurements.

    Args:
        input_path: Completed sealed native training bundle.
        output_path: Fresh local/S3 result prefix for both arm artifacts and comparison.
    Returns:
        Measured held-out success difference and checkpoint identities.
    Raises:
        ValueError: Training bindings or checkpoint evidence do not match.
        RuntimeError: Native evaluation fails.
    """
    with tempfile.TemporaryDirectory(prefix="npa-navigation-compare-") as temporary:
        root = Path(temporary)
        source = materialize(input_path, root / "training")
        output = root / "result"
        output.mkdir()
        reports = {
            arm: _evaluate_arm(source, root, output, arm)
            for arm in ("reference", "trained")
        }
        result = _paired_result(reports)
        write_json(output / "comparison.json", result)
        publish(output, output_path)
        return result


def _paired_result(reports):
    reference, trained = reports["reference"], reports["trained"]
    if reference["evaluation_inputs_sha256"] != trained["evaluation_inputs_sha256"]:
        raise ValueError("comparison arms used different held-out inputs")
    gain = trained["success_rate"] - reference["success_rate"]
    return {
        "schema": "npa.navigation.comparison.v1",
        "reference": reference,
        "trained": trained,
        "success_rate_gain": gain,
        "learning_improved": gain > 0,
        "evaluation_inputs_sha256": reference["evaluation_inputs_sha256"],
    }


def _evaluate_arm(source, root, output, arm):
    recipe = json.loads((source / "recipe.json").read_text())
    training = json.loads((source / "training.json").read_text())
    from npa.workflows.navigation.contract import read_recipe
    from npa.workflows.navigation.runtime import _verify_training_binding

    _verify_training_binding(training, read_recipe(source), source)
    name = "reference_checkpoint.pt" if arm == "reference" else "policy.pt"
    expected = (
        training["performance"]["reference_checkpoint_sha256"]
        if arm == "reference"
        else training["checkpoint_sha256"]
    )
    if file_sha256(source / name) != expected:
        raise ValueError("paired checkpoint differs from native training evidence")
    raw = root / (arm + "-raw")
    raw.mkdir()
    scene = raw / recipe["scene_file"]
    scene.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / recipe["scene_file"], scene)
    shutil.copy2(source / name, raw / "baseline.pt")
    recipe["initial_checkpoint"] = {"file": "baseline.pt", "sha256": expected}
    write_json(raw / "recipe.json", recipe)
    prepared = root / (arm + "-prepared")
    prepare(str(raw), str(prepared), recipe["image"])
    return run_stage("evaluate-checkpoint", str(prepared), str(output / arm))
