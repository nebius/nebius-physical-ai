"""Build reproducible public replay inputs and freeze development and final cohorts."""

import copy
import json
from pathlib import Path
import shutil

from npa.workflows.navigation.artifacts import file_sha256, write_json
from npa.workflows.navigation.reference_bundle import _recipe
from npa.workflows.navigation.reference_scene import _case, cases, warehouse


def prepare_warehouse(
    output, *, image, iterations=1500, count=4000, steps=300, cohort_seed=71000000
):
    """Create a baseline bundle with frozen train, development and final resets.

    Args:
        output: Fresh local destination.
        image: Immutable native Isaac runtime image digest.
        iterations: Baseline PPO update count.
        count: Concurrent robots and size of each disjoint cohort.
        steps: Evaluation action horizon.
        cohort_seed: Predeclared experiment seed namespace, chosen before evaluation.
    Returns:
        Cohort manifest binding actual case bytes before learning starts.
    Raises:
        ValueError: Counts, recipes or cohort separation are invalid.
        OSError: Scene or manifest publication fails.
    """
    from npa.workflows.navigation.contract import read_recipe

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    warehouse(output / "scene.usdz")
    recipe = _recipe(output, image, iterations, steps, count)
    recipe.update(cases(count))
    recipe["train_cases"] = _cohort(count, cohort_seed)
    recipe["eval_cases"] = _cohort(count, cohort_seed + count)
    write_json(output / "recipe.json", recipe)
    final = _cohort(count, cohort_seed + 2 * count)
    write_json(output / "final-cases.json", final)
    cohorts = {
        "training": recipe["train_cases"],
        "development": recipe["eval_cases"],
        "final": final,
    }
    manifest = cohort_manifest(cohorts)
    manifest.update(
        scope="new routes in the public warehouse; not unseen-layout transfer",
        final_used_for_selection=False,
        cohort_seed=cohort_seed,
    )
    write_json(output / "cohorts.json", manifest)
    read_recipe(output)
    return manifest


def _cohort(count, offset):
    return [_case(index, offset) for index in range(count)]


def cohort_manifest(cohorts):
    """Verify separation and bind immutable reset cohorts before any policy evaluation.

    Args:
        cohorts: Named reset lists for training, development and final evaluation.
    Returns:
        Cohort counts, seeds and SHA-256 identities without evaluation outcomes.
    Raises:
        ValueError: Cohorts reuse case identifiers, seeds or physical resets.
    """
    import hashlib
    from npa.workflows.navigation.contract import Case

    identities, seeds, physical = set(), set(), set()
    result = {}
    for name, rows in cohorts.items():
        rows = [Case.model_validate(row).model_dump() for row in rows]
        for row in rows:
            reset = json.dumps(
                {key: row[key] for key in ("position_m", "goal_m", "heading_rad")},
                sort_keys=True,
            )
            if row["id"] in identities or row["seed"] in seeds or reset in physical:
                raise ValueError("reference cohorts overlap")
            identities.add(row["id"])
            seeds.add(row["seed"])
            physical.add(reset)
        data = json.dumps(rows, sort_keys=True, allow_nan=False).encode()
        result[name] = {"count": len(rows), "sha256": hashlib.sha256(data).hexdigest()}
    return {"schema": "npa.field-failure.reference-cohorts.v1", "cohorts": result}


def capture_recipe(capture, scan_cases, baseline, *, iterations=1500):
    """Attach measured scan routes and full baseline replay to a real capture bundle.

    Args:
        capture: Local prepared public RGB-D capture directory.
        scan_cases: Measured supported office training and diagnostic reset cases.
        baseline: Local baseline bundle with scene, recipe and frozen cohorts.
        iterations: Candidate PPO update count.
    Returns:
        Hash-bound common protocol for native reconstruction, training and evaluation.
    Raises:
        ValueError: Route populations differ or cohorts overlap.
        OSError: Required bytes cannot be copied or written.
    """
    baseline_recipe = json.loads((baseline / "recipe.json").read_text())
    count = baseline_recipe["num_envs"]
    if len(scan_cases["train_cases"]) != count:
        raise ValueError("office and baseline replay require equal route populations")
    manifest = json.loads((baseline / "cohorts.json").read_text())
    development = json.loads((baseline / "development-cases.json").read_text())
    recipe = copy.deepcopy(baseline_recipe)
    recipe.update(
        train_cases=scan_cases["train_cases"],
        eval_cases=development,
        iterations=iterations,
        initial_checkpoint=None,
    )
    write_json(capture / "recipe.json", recipe)
    shutil.copyfile(baseline / "scene.usdz", capture / "replay.usdz")
    replay = {
        "schema": "npa.navigation.replay.v2",
        "scene_sha256": baseline_recipe["scene_sha256"],
        "office_translation_m": [50.0, 0.0, 0.0],
        "train_cases": baseline_recipe["train_cases"],
        "development_cases": development,
        "probe": baseline_recipe["probe"],
        "frozen_geometry": manifest["geometry"],
        "frozen_routes": {
            name: manifest["cohorts"][name]["sha256"]
            for name in ("training", "development")
        },
    }
    write_json(capture / "replay.json", replay)
    return {
        "schema_version": "npa.field-failure.native-protocol.v1",
        "navigation_image": recipe["image"],
        **{
            key: recipe[key]
            for key in (
                "task",
                "adapter_module",
                "adapter_sha256",
                "source_bundle_sha256",
            )
        },
    }


def evaluation_bundle(baseline, output, *, final):
    """Create checkpoint-independent development or final evaluation inputs.

    Args:
        baseline: Local public baseline bundle containing precommitted cohorts.
        output: Fresh evaluation bundle directory.
        final: Select final cases only when constructing the sealed final artifact.
    Returns:
        Selected evaluation recipe with no policy checkpoint attached.
    Raises:
        ValueError: The resulting recipe is invalid.
        OSError: Required scene or cohort bytes are unavailable.
    """
    from npa.workflows.navigation.contract import read_recipe

    output.mkdir()
    shutil.copyfile(baseline / "composite.usdz", output / "scene.usdz")
    recipe = json.loads((baseline / "recipe.json").read_text())
    recipe["initial_checkpoint"] = None
    recipe["scene_sha256"] = file_sha256(output / "scene.usdz")
    label = "final" if final else "development"
    recipe["eval_cases"] = json.loads((baseline / (label + "-cases.json")).read_text())
    write_json(output / "recipe.json", recipe)
    read_recipe(output)
    return recipe


def adapter_identities(navigation_image, reconstruction_image):
    """Bind built-in adapters to installed source bytes and immutable runtime images.

    Args:
        navigation_image: Digest-pinned Isaac runtime.
        reconstruction_image: Digest-pinned Open3D runtime.
    Returns:
        The three configured field-failure adapter identities.
    Raises:
        OSError: Adapter source files are unavailable.
    """
    result = {}
    for stage, module in (
        ("reconstruct", "native_reconstruction"),
        ("train", "native_policy"),
        ("evaluate", "native_policy"),
    ):
        result[stage] = {
            "entrypoint": "npa.workflows.field_failure." + module + ":" + stage,
            "source_sha256": file_sha256(Path(__file__).with_name(module + ".py")),
            "runtime_image": reconstruction_image
            if stage == "reconstruct"
            else navigation_image,
        }
    return result


def reference_metrics(steps):
    """Keep the existing success and per-case safety regression gates unchanged.

    Args:
        steps: Declared evaluation horizon for step-count metric bounds.
    Returns:
        Primary success gain of one percentage point plus strict regression guards.
    Raises:
        None.
    """
    rows = [
        ("success", "higher", 0.01, 0.0, 1.0),
        ("collision_steps", "lower", 0.0, 0.0, float(steps)),
        ("physical_failure_steps", "lower", 0.0, 0.0, float(steps)),
        ("goal_distance_m", "lower", 0.0, 0.25, 1000.0),
    ]
    return [
        {
            "name": name,
            "direction": direction,
            "minimum_improvement": gain,
            "maximum_regression": regression,
            "minimum": 0.0,
            "maximum": maximum,
        }
        for name, direction, gain, regression, maximum in rows
    ]
