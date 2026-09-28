"""Publish an offline HTML view of the real scan reconstruction and policy evaluation."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import shutil
import tempfile

from npa.workbench.nurec.navigation_assets import materialize, publish
from npa.workbench.nurec.navigation_publication import verify_publication
from npa.workflows.navigation.artifacts import materialize as sealed_input
from npa.workflows.navigation.artifacts import write_json


def _copy_evidence(source, output, names):
    for name in names:
        path = source / name
        if path.is_file():
            shutil.copyfile(path, output / name)


def _html(summary):
    result = "Passed" if summary["passed"] else "Did not pass"
    rate = summary.get("success_rate")
    rate_text = "Unavailable" if rate is None else f"{rate:.1%}"
    rows = {
        "Held-out navigation success": rate_text,
        "Held-out cases": summary.get("episodes", 0),
        "Required success": "80%",
        "Training iterations": summary["iterations"],
        "Scan frames integrated": summary["integration_frames"],
        "Excluded scan frames used for validation": summary["validation_frames"],
    }
    table = "".join(
        f"<tr><th>{escape(key)}</th><td>{escape(str(value))}</td></tr>"
        for key, value in rows.items()
    )
    links = "".join(
        f'<li><a href="{name}">{name}</a></li>' for name in summary["files"]
    )
    video = (
        '<video controls preload="metadata" src="rollout.mp4"></video>'
        if "rollout.mp4" in summary["files"]
        else ""
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Public scan to navigation</title><style>
body{{font:17px/1.55 system-ui;margin:40px auto;padding:0 20px;max-width:950px;color:#172b3a;background:#f8fafc}}
video{{width:100%;max-height:650px;background:#111;border-radius:12px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #cdd5df}}a{{color:#075bb5}}
</style><h1>Public scan → navigation policy</h1><p><strong>{result}</strong></p>
<p>Full public TUM RGB-D office capture, measured collision geometry, native Isaac navigation training and held-out resets.</p>
{video}<table>{table}</table><p>Evaluation uses held-out goals in the reconstructed training scene. It does not establish transfer to unseen buildings, camera-conditioned control, or a physical robot. The policy observes static range rays.</p>
<h2>Evidence</h2><ul>{links}</ul><p>{escape(summary["attribution"])}</p></html>"""


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
    with tempfile.TemporaryDirectory(prefix="npa-scan-report-") as temporary:
        work = Path(temporary)
        scan = materialize(reconstruction_path, work / "scan")
        verify_publication(scan)
        training = sealed_input(training_path, work / "training")
        evaluation = sealed_input(evaluation_path, work / "evaluation")
        output = work / "report"
        output.mkdir()
        summary = _summarize(scan, training, evaluation)
        _evidence_files(scan, training, evaluation, output)
        summary["files"] = sorted(path.name for path in output.iterdir())
        write_json(output / "summary.json", summary)
        (output / "index.html").write_text(_html(summary))
        publish(output, output_path)
        return summary


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


def _summarize(scan, training, evaluation):
    from npa.workbench.nurec.navigation_sample import ATTRIBUTION

    reconstruction = json.loads((scan / "reconstruction.json").read_text())
    learning = json.loads((training / "training.json").read_text())
    scored = evaluation / "evaluation.json"
    result = json.loads(scored.read_text()) if scored.is_file() else {}
    if result and result.get("checkpoint_sha256") != learning["checkpoint_sha256"]:
        raise ValueError("report evaluation checkpoint differs from training")
    return {
        "schema": "npa.navigation.sample_report.v1",
        "passed": result.get("passed") is True,
        "success_rate": result.get("success_rate"),
        "episodes": len(result.get("episodes", [])),
        "iterations": learning["iterations"],
        "checkpoint_sha256": learning["checkpoint_sha256"],
        "integration_frames": reconstruction["integration_frames"],
        "validation_frames": reconstruction["validation_frames"],
        "attribution": ATTRIBUTION,
    }


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
        RuntimeError: Native execution or the existing 80% success gate fails.
        ValueError: Evidence or provenance is invalid.
        OSError: Artifact staging or publication fails.
    """
    from npa.workflows.navigation.stages import run_stage

    try:
        result = run_stage("evaluate", input_path, output_path)
    except RuntimeError as error:
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
