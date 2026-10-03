"""Show observed simulation-failure admission and honest no-failure outcomes offline."""

import json
from pathlib import Path
import tempfile

from npa.workflows.field_failure.reference_demo_publication import (
    publish_html,
    publish_record,
)
from npa.workflows.preview_html import write_preview


def bound_admission_summary(args):
    """Read only the admission whose digest is bound by the public bundle reference.

    Args:
        args: Run-scoped artifact root for this public demo.
    Returns:
        Sanitized observed-failure admission, without final-cohort reads.
    Raises:
        ValueError: The bound admission is absent, changed, or not admitted.
        OSError: Immutable reference records are unavailable.
    """
    from npa.workflows.field_failure.artifacts import _read

    reference, _ = _read(args.output_root + "/bundle-reference.json")
    descriptor = reference["admission"]
    if descriptor["uri"] != args.output_root + "/observed-failures.json":
        raise ValueError("report admission belongs to another run")
    admission, _ = _read(descriptor["uri"], descriptor["sha256"])
    if (
        admission["admitted"] is not True
        or admission["observed_failures"] < 1
        or admission["observed_failures"] != len(admission["failures"])
    ):
        raise ValueError("report requires genuinely admitted observed failures")
    return admission_summary(admission)


def admission_summary(record):
    """Describe measured capture admission without exposing private storage locations.

    Args:
        record: Verified immutable public simulation-failure admission.
    Returns:
        Public admission counts, input hashes and exact scope.
    Raises:
        KeyError: Required measured fields are absent.
    """
    names = (
        "admitted",
        "episodes",
        "observed_failures",
        "checkpoint_sha256",
        "capture_manifest_sha256",
        "scene_sha256",
        "training_cases_sha256",
        "raw_outcomes_sha256",
        "evaluation_sha256",
        "native_after_load_sha256",
        "selection_scope",
        "final_cohort_consumed",
    )
    return {key: record[key] for key in names}


def no_failures_report(args, plan, admission, observed):
    """Publish truthful terminal HTML when completed observation found no failures.

    Args:
        args: Run-scoped public reference artifact root.
        plan: Immutable pre-learning plan used for sample attribution.
        admission: Verified record with zero measured unsuccessful episodes.
        observed: Materialized completed baseline observation and rendered frames.
    Returns:
        Structured result explaining why continuation and final did not run.
    Raises:
        ValueError: A supposed zero-failure result contains an admitted failure.
        OSError: Retained preview or report publication is unavailable.
    """
    from npa.workflows.field_failure.reference_demo_attribution import sample_credit
    from npa.workflows.navigation.preview import scored_rollout_group

    if admission["admitted"] or admission["observed_failures"] != 0:
        raise ValueError("no-failures report requires actual zero-failure observation")
    result = {
        "schema": "npa.field-failure.no-observed-failures.v1",
        "observation_completed": True,
        "runtime_completed": False,
        "quality_passed": False,
        "status": "no_observed_failures",
        "continuation_started": False,
        "final_evaluated": False,
        "deployment_authorized": False,
        "admission": admission_summary(admission),
        "public_sample": sample_credit(args.output_root, plan),
    }
    report = json.loads((observed / "evaluation.json").read_text())
    group = scored_rollout_group(
        observed, report, title="Baseline training-route observation"
    )
    _publish_no_failures(args, result, group)
    return result


def _publish_no_failures(args, result, group):
    with tempfile.TemporaryDirectory(prefix="npa-admission-html-") as temporary:
        path = Path(temporary) / "index.html"
        write_preview(
            path,
            title="Public RL: no observed simulation failures",
            summary=(
                "The baseline completed every frozen office training route successfully. "
                "No capture was admitted for adaptation; continuation and final evaluation did not run. "
                "This observation covers simulated routes in a public scan, not physical field logs."
            ),
            metrics={
                "Observed episodes": result["admission"]["episodes"],
                "Observed failures": 0,
                "Final evaluated": False,
            },
            groups=[group],
            details=result,
        )
        publish_record(args.output_root + "/reports/result.json", result)
        publish_html(args.output_root + "/reports/index.html", path.read_text())
