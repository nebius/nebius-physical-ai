"""Publish an offline HTML view of the real scan reconstruction and policy evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from npa.workbench.nurec.navigation_assets import materialize
from npa.workbench.nurec.navigation_publication import verify_publication
from npa.workbench.nurec.navigation_sample_publication import (
    evaluation_exists,
    publish_report,
)
from npa.workflows.navigation.artifacts import materialize as sealed_input
from npa.workflows.navigation.artifacts import write_json


def _copy_evidence(source, output, names):
    for name in names:
        path = source / name
        if path.is_file():
            shutil.copyfile(path, output / name)


def _metrics(summary):
    status = "Passed" if summary["passed"] else "Did not pass"
    if not summary["evaluation_complete"]:
        status = "Incomplete evaluation"
    rate = summary["success_rate"]
    return {
        "Quality gate": status,
        "Held-out navigation success": "Unavailable" if rate is None else f"{rate:.1%}",
        "Held-out cases": summary["episodes"],
        "Required success": f"{summary['minimum_success_rate']:.0%}",
        "Training iterations": summary["iterations"],
        "Scan frames integrated": summary["integration_frames"],
        "Held-out scan frames": summary["validation_frames"],
        "Policy observations": summary["sensor_mode"],
    }


def _write_html(summary, evaluation, result, path):
    from npa.workflows.navigation.preview import scored_rollout_group
    from npa.workflows.preview_html import write_preview

    groups = (
        []
        if result is None
        else [
            scored_rollout_group(
                evaluation, result, title="Held-out navigation · scored focal episode"
            )
        ]
    )
    write_preview(
        path,
        title="Public scan → navigation policy",
        metrics=_metrics(summary),
        summary=(
            "Public RGB-D office capture, measured collision geometry and native Isaac training. "
            "Evaluation uses held-out goals in the reconstructed training scene. It does not "
            "establish transfer to unseen buildings or a physical robot. "
            + summary["attribution"]
        ),
        groups=groups,
        details=summary,
        allow_empty_media=result is None,
    )


def build_report(
    reconstruction_path: str, evaluation_path: str, training_path: str, output_path: str
) -> dict:
    """Render measured scan and checkpoint evidence into a self-contained report.

    Args:
        reconstruction_path: Completed measured TSDF publication.
        evaluation_path: Completed native evaluation publication, including failures.
        training_path: Completed native training publication.
        output_path: Fresh local directory or S3 report prefix.
    Returns:
        Summary derived from verified artifacts, never from expected results.
    Raises:
        ValueError: Required evidence or artifact integrity is invalid.
        OSError: Evidence staging or report publication fails.
    """
    summary, _ = _build_report(
        reconstruction_path, evaluation_path, training_path, output_path
    )
    return summary


def _build_report(
    reconstruction_path, evaluation_path, training_path, output_path, *, complete=False
):
    with tempfile.TemporaryDirectory(prefix="npa-scan-report-") as temporary:
        work = Path(temporary)
        scan = materialize(reconstruction_path, work / "scan")
        verify_publication(scan)
        training = sealed_input(training_path, work / "training")
        evaluation = sealed_input(evaluation_path, work / "evaluation")
        output = work / "report"
        output.mkdir()
        from npa.workbench.nurec.navigation_sample_evidence import measured_summary

        summary, result = measured_summary(scan, training, evaluation)
        if complete and (
            result is None
            or (evaluation / "failure.json").exists()
            or (evaluation / "evaluation.incomplete.json").exists()
        ):
            raise ValueError(
                "existing native evaluation is incomplete; use a fresh run"
            )
        _evidence_files(scan, training, evaluation, output)
        summary["files"] = sorted(path.name for path in output.iterdir())
        write_json(output / "summary.json", summary)
        _write_html(summary, evaluation, result, output / "index.html")
        publish_report(output, output_path)
        return summary, result


def _evidence_files(scan, training, evaluation, output):
    _copy_evidence(scan, output, ["reconstruction.json", "capture.json"])
    _copy_evidence(training, output, ["training.json"])
    _copy_evidence(
        evaluation,
        output,
        [
            "evaluation.json",
            "evaluation.incomplete.json",
            "failure.json",
            "rollout.mp4",
        ],
    )


def evaluate_report(
    input_path: str, output_path: str, reconstruction_path: str, report_path: str
) -> dict:
    """Evaluate the trained policy and retain HTML even when the quality gate fails.

    Args:
        input_path: Completed native training prefix.
        output_path: Fresh native evaluation prefix.
        reconstruction_path: Completed TSDF reconstruction prefix.
        report_path: Fresh offline HTML report prefix.
    Returns:
        Native evaluation report after the unchanged quality gate passes.
    Raises:
        RuntimeError: Native execution or the configured success gate fails.
        ValueError: Evidence or provenance is invalid.
        OSError: Artifact staging or publication fails.
    """
    from npa.workflows.navigation.stages import run_stage

    if evaluation_exists(output_path):
        _, result = _build_report(
            reconstruction_path, output_path, input_path, report_path, complete=True
        )
        if not result["passed"]:
            raise RuntimeError(
                "held-out navigation success is below minimum_success_rate; evidence published"
            )
        return result
    try:
        result = run_stage("evaluate", input_path, output_path)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        try:
            build_report(reconstruction_path, output_path, input_path, report_path)
        except (ValueError, OSError) as report_error:
            error.add_note(f"HTML report unavailable: {type(report_error).__name__}")
        raise
    build_report(reconstruction_path, output_path, input_path, report_path)
    return result


def main(argv=None):
    """Run the workflow's native evaluation and offline reporting stage.

    Args:
        argv: Optional command-line argument list.
    Returns:
        Zero only after native quality validation passes and report publication.
    Raises:
        RuntimeError: Native execution or quality gate fails.
        ValueError: Input evidence differs or is invalid.
        OSError: Artifact staging or publication fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("input-path", "output-path", "reconstruction-path", "report-path"):
        parser.add_argument("--" + option, required=True)
    evaluate_report(**vars(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
