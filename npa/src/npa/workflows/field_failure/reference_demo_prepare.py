"""Automatically publish public baseline, capture, replay and sealed evaluation inputs."""

import json
from pathlib import Path
import tempfile

from npa.workflows.field_failure.artifacts import _publish, _read
from npa.workflows.field_failure.native_artifacts import _archive, _upload, _upload_json
from npa.workflows.field_failure.reference_demo_inputs import (
    adapter_identities,
    capture_recipe,
    evaluation_bundle,
    prepare_warehouse,
    reference_metrics,
)
from npa.workflows.navigation.artifacts import materialize, publish
from npa.workflows.field_failure.reference_demo_cohorts import freeze_balanced_cohorts


def prepare_reference(args):
    """Prepare all public data and freeze evaluation before starting baseline training.

    Args:
        args: Validated stage arguments including sample, measured cases and runtime pins.
    Returns:
        Immutable reference plan with every dataset artifact identity.
    Raises:
        ValueError: Sample, measured cases or recipe verification fails.
        OSError: Data staging or conditional publication fails.
    """
    with tempfile.TemporaryDirectory(prefix="npa-rl-reference-") as temporary:
        root = Path(temporary)
        capture = materialize(args.sample_path, root / "capture")
        baseline, cohorts, protocol = _prepare_inputs(args, root, capture)
        plan = _publish_inputs(args, root, baseline, capture, protocol)
        plan.update(
            cohorts=cohorts,
            adapters=adapter_identities(
                args.navigation_image, args.reconstruction_image
            ),
            metrics=reference_metrics(args.episode_steps),
            baseline_iterations=args.baseline_iterations,
            candidate_iterations=args.candidate_iterations,
            num_envs=args.num_envs,
        )
        _publish(args.output_root + "/reference-plan.json", plan)
        return plan


def _prepare_inputs(args, root, capture):
    from npa.workflows.navigation.artifacts import write_json

    scan_cases, support = _measured_cases(args.cases_path, root / "cases", capture)
    office, source_identity = _measured_scene(args.scene_path, root / "scene", support)
    baseline = root / "baseline"
    prepare_warehouse(
        baseline,
        image=args.navigation_image,
        iterations=args.baseline_iterations,
        count=args.num_envs,
        steps=args.episode_steps,
        cohort_seed=args.cohort_seed,
    )
    cohorts = freeze_balanced_cohorts(baseline, scan_cases, office)
    cohorts["geometry"].update(source_identity)
    write_json(baseline / "cohorts.json", cohorts)
    protocol = capture_recipe(
        capture, scan_cases, baseline, iterations=args.candidate_iterations
    )
    return baseline, cohorts, protocol


def _measured_cases(source, target, capture):
    from npa.workbench.nurec.navigation_assets import materialize as scan_materialize
    from npa.workbench.nurec.navigation_publication import verify_publication

    scan_materialize(source, target)
    verify_publication(target)
    from npa.workflows.navigation.artifacts import file_sha256

    support = json.loads((target / "support.json").read_text())
    if support["capture_sha256"] != file_sha256(capture / "capture.json"):
        raise ValueError("measured reset cases belong to another capture")
    return json.loads((target / "cases.json").read_text()), support


def _measured_scene(source, target, support):
    from npa.workbench.nurec.navigation_assets import materialize as scan_materialize
    from npa.workbench.nurec.navigation_publication import verify_publication
    from npa.workflows.navigation.artifacts import file_sha256

    scan_materialize(source, target)
    verify_publication(target)
    assembly = json.loads((target / "provenance.json").read_text())
    reconstruction = json.loads((target / "reconstruction.json").read_text())
    if (
        assembly["capture_manifest_sha256"] != support["capture_sha256"]
        or reconstruction["surface_sha256"] != support["surface_sha256"]
        or file_sha256(target / "scene.usdz") != assembly["scene_sha256"]
    ):
        raise ValueError(
            "frozen office geometry and supported cases have different capture provenance"
        )
    return target / "scene.usdz", {
        "capture_manifest_sha256": support["capture_sha256"],
        "surface_sha256": support["surface_sha256"],
        "assembly_provenance_sha256": file_sha256(target / "provenance.json"),
    }


def _publish_inputs(args, root, baseline, capture, protocol):
    plan = {"schema": "npa.field-failure.reference-plan.v1"}
    prefix = args.output_root + "/inputs/"
    for label, final in (("development", False), ("final", True)):
        target = root / label
        recipe = evaluation_bundle(baseline, target, final=final)
        plan[label] = _archive_upload(target, prefix + label + ".tar")
        plan[label + "_seeds"] = [row["seed"] for row in recipe["eval_cases"]]
    plan["capture"] = _archive_upload(capture, prefix + "capture.tar")
    plan["protocol"] = _upload_json(
        protocol, root / "protocol.json", prefix + "protocol.json"
    )
    # Keep final cases outside the learner's input directory after freezing them.
    (baseline / "final-cases.json").unlink()
    publish(baseline, args.output_root + "/baseline-input")
    return plan


def _archive_upload(source, uri):
    archive = source.parent / (source.name + ".tar")
    _archive(source, archive)
    return _upload(archive, uri)


def seal_bundle(args):
    """Bind the freshly trained baseline to the precommitted public reference bundle.

    Args:
        args: Stage arguments identifying this run's artifact root.
    Returns:
        Hash-bound field-failure bundle artifact descriptor.
    Raises:
        ValueError: Baseline training is incomplete or differs from the frozen recipe.
        OSError: Baseline staging or conditional publication fails.
    """
    from npa.workflows.field_failure.contracts import _Bundle

    plan, _ = _read(args.output_root + "/reference-plan.json")
    with tempfile.TemporaryDirectory(prefix="npa-baseline-seal-") as temporary:
        root = Path(temporary)
        training = materialize(
            args.output_root + "/baseline-training", root / "training"
        )
        _verify_baseline(training, plan)
        checkpoint = _upload(
            training / "policy.pt", args.output_root + "/inputs/baseline.pt"
        )
        value = _bundle(plan, checkpoint)
        _Bundle.model_validate(value)
        artifact = _upload_json(
            value, root / "bundle.json", args.output_root + "/inputs/bundle.json"
        )
    _publish(args.output_root + "/bundle-reference.json", artifact)
    return artifact


def _verify_baseline(training, plan):
    report = json.loads((training / "training.json").read_text())
    from npa.workflows.navigation.artifacts import file_sha256

    if (
        report["iterations"] != plan["baseline_iterations"]
        or report["runtime"]["robot_population"] != plan["num_envs"]
        or report["policy_parameter_delta_l2"] <= 0
        or report["isolation"]["passed"] is not True
        or report["checkpoint_sha256"] != file_sha256(training / "policy.pt")
    ):
        raise ValueError("baseline is not the complete physically qualified native run")


def _bundle(plan, checkpoint):
    return {
        "schema_version": "npa.field-failure.bundle.v1",
        "task": "navigation",
        "baseline": {
            "policy_id": "public-warehouse-baseline",
            "checkpoint": checkpoint,
        },
        "baseline_training_groups": ["warehouse-training-routes"],
        "captures": [
            {
                "scenario_id": "office-with-warehouse-replay",
                "group_id": "office-and-warehouse-training-routes",
                "asset": plan["capture"],
            }
        ],
        "held_out": [
            {
                "scenario_id": "balanced-known-layout-final-routes",
                "group_id": "office-and-warehouse-final-routes",
                "asset": plan["final"],
                "seeds": plan["final_seeds"],
            }
        ],
        "protocol": plan["protocol"],
        "adapters": plan["adapters"],
        "metrics": plan["metrics"],
        "primary_metric": "success",
    }
