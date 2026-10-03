"""Create explicit public reference inputs without submitting or provisioning jobs."""

import argparse
import json
from pathlib import Path
import shutil

from npa.workflows.navigation.artifacts import file_sha256, write_json
from npa.workflows.navigation.contract import read_recipe
from npa.workflows.navigation.reference import TASK


def build_bundle(
    output,
    *,
    image,
    iterations,
    episode_steps,
    num_envs,
    scene_file=None,
    cases_file=None,
    checkpoint=None,
):
    """Write reproducible native reference inputs with optional baseline initialization.

    Args:
        output: Fresh local bundle directory.
        image: Exact native runtime image digest.
        iterations: Explicit PPO learning iterations, including for resumed training.
        episode_steps: Explicit held-out episode horizon in native control steps.
        num_envs: Concurrent robot count, such as 4000 for the full reference.
        scene_file: Optional real metric Z-up static triangle-mesh USDZ.
        cases_file: Explicit train/eval/probe JSON required with an external scene.
        checkpoint: Optional real RSL-RL checkpoint to initialize or evaluate.
    Returns:
        Validated sealed recipe contents; use stages.prepare before runtime execution.
    Raises:
        ValueError: Required scene/case pairs or checkpoint bindings are invalid.
        FileExistsError: Output is not fresh.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reset_cases = _scene_inputs(output, scene_file, cases_file, num_envs)
    recipe = _recipe(output, image, iterations, episode_steps, num_envs)
    recipe.update(reset_cases)
    if checkpoint is not None:
        recipe["initial_checkpoint"] = _checkpoint(output, checkpoint)
    write_json(output / "recipe.json", recipe)
    return read_recipe(output).model_dump()


def _checkpoint(output, checkpoint):
    shutil.copy2(checkpoint, output / "baseline.pt")
    return {"file": "baseline.pt", "sha256": file_sha256(output / "baseline.pt")}


def _scene_inputs(output, scene_file, cases_file, num_envs):
    from npa.workflows.navigation.reference_scene import cases, warehouse

    if (scene_file is None) != (cases_file is None):
        raise ValueError(
            "external scene requires explicit measured train/eval/probe cases"
        )
    if scene_file is None:
        warehouse(output / "scene.usdz")
        return cases(num_envs)
    shutil.copy2(scene_file, output / "scene.usdz")
    return json.loads(Path(cases_file).read_text())


def _recipe(output, image, iterations, episode_steps, num_envs):
    from npa.workflows.navigation.native import source_bundle_digest
    from npa.workflows.navigation.reference_controller import CONTROLLER_SHA256

    return {
        "schema_version": "npa.navigation.recipe.v1",
        "task": TASK,
        "adapter_module": "npa.workflows.navigation.reference",
        "adapter_sha256": file_sha256(Path(__file__).with_name("reference.py")),
        "source_bundle_sha256": source_bundle_digest(),
        "reference_controller_sha256": CONTROLLER_SHA256,
        "image": image,
        "scene_file": "scene.usdz",
        "scene_sha256": file_sha256(output / "scene.usdz"),
        "scene_prim": "/World/Warehouse",
        "sensor_mode": "static_raycast",
        "num_envs": num_envs,
        "iterations": iterations,
        "episode_steps": episode_steps,
        "goal_tolerance_m": 0.5,
        "minimum_success_rate": 0.8,
    }


def main(argv=None):
    """Build local public reference inputs using an explicit experiment recipe.

    Args:
        argv: Optional CLI argument list.
    Returns:
        Zero after writing and validating the bundle.
    Raises:
        ValueError: Inputs violate the reference contract.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--episode-steps", required=True, type=int)
    parser.add_argument("--num-envs", required=True, type=int)
    parser.add_argument("--scene-file", type=Path)
    parser.add_argument("--cases-file", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = vars(parser.parse_args(argv))
    args["output"] = args.pop("output_path")
    build_bundle(**args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
