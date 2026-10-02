"""Execute one declarative public RL demo stage using the native workflow components."""

import argparse
from pathlib import Path
import tempfile

from npa.workflows.field_failure.artifacts import _read


def run_reference_stage(args):
    """Dispatch one prepared reference stage without hiding its GPU or quality outcome.

    Args:
        args: Parsed workflow stage arguments.
    Returns:
        The component's measured completion record when available.
    Raises:
        ValueError: Stage selection or sealed artifact identity is invalid.
        RuntimeError: Native execution or policy quality gates fail.
        OSError: Required local or remote artifacts are unavailable.
    """
    from npa.workflows.field_failure.reference_demo_evaluate import (
        evaluate_development,
        select_candidate,
    )
    from npa.workflows.field_failure.reference_demo_report import publish_report

    if args.stage in {
        "prepare",
        "baseline",
        "observe-failures",
        "admit-capture",
        "seal",
    }:
        return _preparation_stage(args)
    if args.stage.startswith("development-"):
        return evaluate_development(args, args.stage.removeprefix("development-"))
    if args.stage == "select":
        return select_candidate(args)
    if args.stage == "report":
        report = publish_report(args)
        if not report["quality_passed"]:
            raise RuntimeError(
                "final quality gates failed; baseline retained; see reports/index.html"
            )
        return report
    return _loop_stage(args)


def _preparation_stage(args):
    from npa.workflows.field_failure.reference_demo_prepare import (
        prepare_reference,
        seal_bundle,
    )
    from npa.workflows.field_failure.reference_demo_observation import observe_failures
    from npa.workflows.field_failure.reference_demo_admission import admit_capture

    handlers = {
        "prepare": prepare_reference,
        "baseline": _baseline,
        "observe-failures": observe_failures,
        "admit-capture": admit_capture,
        "seal": seal_bundle,
    }
    return handlers[args.stage](args)


def _baseline(args):
    from npa.workflows.navigation.artifacts import materialize
    from npa.workflows.navigation.stages import prepare, run_stage

    with tempfile.TemporaryDirectory(prefix="npa-reference-baseline-") as temporary:
        source = materialize(
            args.output_root + "/baseline-input", Path(temporary) / "input"
        )
        prepare(
            str(source),
            args.output_root + "/baseline-prepared",
            args.navigation_image,
        )
    return run_stage(
        "train",
        args.output_root + "/baseline-prepared",
        args.output_root + "/baseline-training",
    )


def _loop_stage(args):
    from npa.workflows.field_failure.stages import run_stage

    artifact, _ = _read(args.output_root + "/bundle-reference.json")
    bundle, _ = _read(artifact["uri"], artifact["sha256"])
    if args.stage in {"reconstruct", "train"}:
        from npa.workflows.field_failure.reference_demo_admission import (
            verify_admission,
        )

        verify_admission(args, descriptor=artifact["admission"], bundle=bundle)
    if args.stage in {"baseline-evaluate", "candidate-evaluate", "compare"}:
        selection, _ = _read(args.output_root + "/selection.json")
        training, _ = _read(
            args.output_root + "/loop/" + args.run_id + "/training.json"
        )
        if (
            not selection["eligible"]
            or selection["selected_checkpoint_sha256"]
            != training["candidate"]["checkpoint"]["sha256"]
        ):
            raise ValueError(
                "final evaluation requires the development-selected candidate"
            )
    key = "evaluate" if args.stage.endswith("-evaluate") else args.stage
    adapter = bundle["adapters"].get(key, {})
    return run_stage(
        args.stage,
        artifact["uri"],
        artifact["sha256"],
        args.output_root + "/loop/" + args.run_id,
        args.run_id,
        adapter.get("entrypoint", ""),
        adapter.get("runtime_image", ""),
    )


_STAGES = (
    "prepare",
    "baseline",
    "observe-failures",
    "admit-capture",
    "seal",
    "validate",
    "reconstruct",
    "train",
    "development-baseline",
    "development-candidate",
    "select",
    "baseline-evaluate",
    "candidate-evaluate",
    "compare",
    "report",
)


def main(argv=None):
    """Parse workflow-owned arguments for a single real public reference stage.

    Args:
        argv: Optional command-line arguments.
    Returns:
        Zero after successful native execution and applicable quality checks.
    Raises:
        ValueError: Stage inputs or artifact contracts are invalid.
        RuntimeError: Native execution or quality fails after evidence publication.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=_STAGES,
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-path")
    parser.add_argument("--cases-path")
    parser.add_argument("--scene-path")
    parser.add_argument("--navigation-image")
    parser.add_argument("--reconstruction-image")
    parser.add_argument("--baseline-iterations", type=int, default=1500)
    parser.add_argument("--candidate-iterations", type=int, default=1500)
    parser.add_argument("--baseline-anchor-coefficient", type=float)
    parser.add_argument("--num-envs", type=int, default=4000)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--cohort-seed", type=int, default=71000000)
    run_reference_stage(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
