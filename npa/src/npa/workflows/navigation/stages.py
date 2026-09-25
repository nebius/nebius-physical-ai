"""Run sealed navigation preparation and native Isaac stages in workflow-owned pods."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from npa.workflows.navigation.artifacts import (
    file_sha256,
    materialize,
    publish,
    write_json,
)
from npa.workflows.navigation.contract import read_recipe


def prepare(input_path: str, output_path: str, image: str) -> dict:
    """Seal operator task/reset inputs without claiming simulator readiness.

    Args:
        input_path: Raw operator bundle with recipe.json and self-contained USDZ.
        output_path: Run-scoped prepared artifact prefix.
        image: Exact configured BYOF image digest, matching the recipe.
    Returns:
        Input integrity record; native readiness remains unverified.
    Raises:
        ValueError: Required input, scene hash or image binding is invalid.
        OSError: Input or artifact transfer fails.
    """
    with tempfile.TemporaryDirectory(prefix="npa-navigation-") as temp:
        output = materialize(input_path, Path(temp) / "prepared", sealed=False)
        recipe = read_recipe(output)
        if image != recipe.image:
            raise ValueError("workflow image must match the recipe's exact BYOF digest")
        evidence = {
            "schema": "npa.navigation.prepared.v1",
            "recipe_sha256": file_sha256(output / "recipe.json"),
            "scene_sha256": recipe.scene_sha256,
            "task": recipe.task,
            "image": image,
            "native_runtime_verified": False,
        }
        write_json(output / "prepared.json", evidence)
        publish(output, output_path)
        return evidence


def run_stage(stage: str, input_path: str, output_path: str) -> dict:
    """Execute native Isaac through its installed interpreter and publish real evidence.

    Args:
        stage: train or evaluate; the workflow owns stage order.
        input_path: Sealed upstream S3 prefix or local bundle.
        output_path: Fresh run-scoped result prefix or local directory.
    Returns:
        Native training or held-out evaluation evidence.
    Raises:
        ValueError: Provenance, input or output evidence is invalid.
        RuntimeError: Native execution or the held-out quality gate fails.
        OSError: Interpreter, required files or storage are unavailable.
    """
    if stage not in {"train", "evaluate"}:
        raise ValueError("native stage must be train or evaluate")
    with tempfile.TemporaryDirectory(prefix="npa-navigation-") as temp:
        source = materialize(input_path, Path(temp) / "input")
        recipe = read_recipe(source)
        if os.environ.get("NPA_TASK_IMAGE") != recipe.image:
            raise ValueError("NPA_TASK_IMAGE must match the sealed BYOF image digest")
        output = Path(temp) / "output"
        output.mkdir()
        try:
            evidence = _native(stage, source, output)
            _carry_inputs(source, output, recipe, stage)
        except Exception:
            _record_failure(output, stage)
            publish(output, output_path)
            raise
        publish(output, output_path)
        if stage == "evaluate" and not evidence["passed"]:
            raise RuntimeError(
                "held-out navigation success is below minimum_success_rate; evidence published"
            )
        return evidence


def _native(stage, source, output):
    interpreter = os.environ.get("ISAAC_LAB_PYTHON", "/isaac-sim/python.sh")
    if not Path(interpreter).is_file():
        raise FileNotFoundError(
            "native Isaac interpreter missing; supply the operator's Isaac BYOF image"
        )
    argv = [
        interpreter,
        "-m",
        "npa.workflows.navigation.runtime",
        stage,
        "--input-path",
        str(source),
        "--output-path",
        str(output),
        "--visualizer",
        "none",
    ]
    with (output / "runtime.log").open("w") as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (output / "runtime.log").read_text()
    if "PhysX error:" in log or "simulation will miss interactions" in log:
        raise RuntimeError(
            "Isaac reported invalid physics; inspect private runtime.log"
        )
    name = "training.json" if stage == "train" else "evaluation.json"
    evidence = json.loads((output / name).read_text())
    expected = "training" if stage == "train" else "evaluation"
    if evidence.get("schema") != f"npa.navigation.{expected}.v1":
        raise ValueError("native runtime omitted the required completion record")
    if (
        stage == "train"
        and file_sha256(output / "policy.pt") != evidence["checkpoint_sha256"]
    ):
        raise ValueError("training checkpoint does not match native evidence")
    if stage == "evaluate" and evidence.get("policy_loaded") is not True:
        raise ValueError("evaluation did not load the trained policy")
    return evidence


def _carry_inputs(source, output, recipe, stage):
    for name in ("recipe.json", recipe.scene_file):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    if stage == "evaluate":
        for name in ("training.json", "policy.pt", "agent.json"):
            shutil.copy2(source / name, output / name)


def _record_failure(output, stage):
    for name in ("training.json", "evaluation.json"):
        if (output / name).exists():
            (output / name).rename(output / name.replace(".json", ".incomplete.json"))
    write_json(
        output / "failure.json",
        {"stage": stage, "status": "failed", "native_runtime_verified": False},
    )
