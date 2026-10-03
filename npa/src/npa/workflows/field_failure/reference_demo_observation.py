"""Observe baseline failures on frozen office training routes before capture admission."""

import json
from pathlib import Path
import shutil
import tempfile

from npa.workflows.field_failure.artifacts import _read
from npa.workflows.field_failure.native_artifacts import _bundle
from npa.workflows.navigation.artifacts import file_sha256, materialize, write_json


def observation_bundle(baseline, capture, output):
    """Prepare every office training route for pre-adaptation baseline observation.

    Args:
        baseline: Pre-learning warehouse bundle with the frozen composite scene.
        capture: Prepared public capture containing its office training recipe.
        output: Fresh observation input directory.
    Returns:
        The recipe with unchanged warehouse training and office observation cases.
    Raises:
        ValueError: Observation cases differ from the frozen office training cohort.
        OSError: Required scene or recipe bytes cannot be read or written.
    """
    from npa.workflows.navigation.reference_replay import translated_cases
    from npa.workflows.field_failure.reference_demo_cohorts import TRANSLATION
    from npa.workflows.field_failure.reference_demo_inputs import cohort_manifest
    from npa.workflows.navigation.contract import read_recipe

    recipe = json.loads((baseline / "recipe.json").read_text())
    source = json.loads((capture / "recipe.json").read_text())
    recipe["eval_cases"] = translated_cases(source["train_cases"], TRANSLATION)
    frozen = json.loads((baseline / "cohorts.json").read_text())
    actual = cohort_manifest({"office": recipe["eval_cases"]})["cohorts"]["office"]
    expected = frozen["regions"]["training"]["office"]
    if actual != {key: expected[key] for key in ("count", "sha256")}:
        raise ValueError("observation must use every frozen office training route")
    output.mkdir()
    shutil.copyfile(baseline / "composite.usdz", output / "scene.usdz")
    recipe["scene_sha256"] = file_sha256(output / "scene.usdz")
    write_json(output / "recipe.json", recipe)
    read_recipe(output)
    return recipe


def observe_failures(args):
    """Run complete native baseline evaluation on training routes without reading final.

    Args:
        args: Public demo artifact root and run identity.
    Returns:
        The actual native observation report, including unsuccessful episodes.
    Raises:
        ValueError: Baseline, frozen observation input or protocol differs.
        RuntimeError: Native controls or checkpoint evaluation fail.
        OSError: Required sealed inputs or output publication are unavailable.
    """
    from npa.workflows.field_failure.native_policy import _initialize
    from npa.workflows.field_failure.reference_demo_prepare import _verify_baseline
    from npa.workflows.navigation.stages import prepare, run_stage

    plan, _ = _read(args.output_root + "/reference-plan.json")
    protocol, _ = _read(plan["protocol"]["uri"], plan["protocol"]["sha256"])
    with tempfile.TemporaryDirectory(prefix="npa-failure-observation-") as temporary:
        root = Path(temporary)
        training = materialize(
            args.output_root + "/baseline-training", root / "baseline"
        )
        _verify_baseline(training, plan)
        retained = _completed_observation(args, plan, training, root)
        if retained is not None:
            return retained
        source = _bundle(plan["failure_observation"]["input"], root, "input")
        _initialize(source, protocol, training / "policy.pt")
        prepare(str(source), str(root / "prepared"), protocol["navigation_image"])
        return run_stage(
            "evaluate-checkpoint",
            str(root / "prepared"),
            args.output_root + "/failure-observation",
        )


def _completed_observation(args, plan, training, root):
    from npa.workflows.field_failure.artifacts import _optional
    from npa.workflows.field_failure.reference_demo_failure_evidence import (
        verify_observation,
    )

    prefix = args.output_root + "/failure-observation"
    if _optional(prefix + "/completion.json") is None:
        if _optional(prefix + "/claim.json") is not None:
            raise ValueError(
                "failure observation has an incomplete immutable attempt; refusing another GPU launch"
            )
        return None
    observed = materialize(prefix, root / "retained-observation")
    verify_observation(observed, plan, file_sha256(training / "policy.pt"))
    return json.loads((observed / "evaluation.json").read_text())
