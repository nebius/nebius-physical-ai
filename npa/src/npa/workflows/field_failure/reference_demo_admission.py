"""Admit a public capture only through immutable, recomputed baseline simulation failures."""

from pathlib import Path
import tempfile

from npa.workflows.field_failure.artifacts import _digest, _encode, _read
from npa.workflows.field_failure.reference_demo_failure_evidence import (
    FAILURE_RULE,
    verify_observation,
)
from npa.workflows.field_failure.reference_demo_publication import publish_record
from npa.workflows.navigation.artifacts import materialize
from npa.workflows.navigation.publication import _validate_record


def admit_capture(args):
    """Seal actual observed failures, or stop honestly when none justify adaptation.

    Args:
        args: Run-scoped public reference artifact root and run identity.
    Returns:
        Immutable admission with selected measured failure cases and provenance.
    Raises:
        ValueError: Native outcomes or immutable evidence do not verify.
        RuntimeError: No genuine failure was observed; no continuation is admitted.
        OSError: Required completed evidence or publication is unavailable.
    """
    plan, plan_sha = _read(args.output_root + "/reference-plan.json")
    with tempfile.TemporaryDirectory(prefix="npa-capture-admission-") as temporary:
        record, observed = _measured_admission(args, plan, plan_sha, Path(temporary))
        publish_record(args.output_root + "/observed-failures.json", record)
        if not record["admitted"]:
            from npa.workflows.field_failure.reference_demo_admission_report import (
                no_failures_report,
            )

            no_failures_report(args, plan, record, observed)
            raise RuntimeError(
                "no simulation failures observed; capture not admitted; continuation and final remain untouched; see reports/index.html"
            )
        return record


def verify_admission(args, *, descriptor=None, bundle=None):
    """Recheck the exact admitted failures before sealing, reconstruction or training.

    Args:
        args: Run-scoped public reference artifact root and identity.
        descriptor: Immutable admission URI and SHA from the sealed public reference.
        bundle: Optional sealed generic bundle whose capture and baseline must match.
    Returns:
        Verified admission and its exact URI/SHA descriptor.
    Raises:
        ValueError: Admission changed, has no failures, or differs from native evidence.
        OSError: Required immutable records or raw evidence are unavailable.
    """
    uri = args.output_root + "/observed-failures.json"
    if descriptor is not None and descriptor["uri"] != uri:
        raise ValueError("admission belongs to another run")
    record, digest = _read(uri, None if descriptor is None else descriptor["sha256"])
    plan, plan_sha = _read(args.output_root + "/reference-plan.json")
    with tempfile.TemporaryDirectory(prefix="npa-admission-verify-") as temporary:
        measured, _ = _measured_admission(args, plan, plan_sha, Path(temporary))
    if record != measured or record["admitted"] is not True:
        raise ValueError(
            "capture admission differs from actual observed simulation failures"
        )
    if bundle is not None:
        _verify_bundle_admission(bundle, record)
    return record, {"uri": uri, "sha256": digest}


def _completion(prefix):
    value, digest = _read(prefix + "/completion.json")
    files = _validate_record(value)
    if "failure.json" in files or any(
        name.endswith(".incomplete.json") for name in files
    ):
        raise ValueError("incomplete native execution cannot admit a capture")
    if value != _read(prefix + "/claim.json")[0]:
        raise ValueError(
            "observation or baseline completion differs from its immutable claim"
        )
    return {"uri": prefix + "/completion.json", "sha256": digest}, files


def _measured_admission(args, plan, plan_sha, root):
    baseline, baseline_files = _completion(args.output_root + "/baseline-training")
    completion, observed_files = _completion(args.output_root + "/failure-observation")
    observed = materialize(
        args.output_root + "/failure-observation", root / "observation"
    )
    rows, files = verify_observation(observed, plan, baseline_files["policy.pt"])
    if files != observed_files:
        raise ValueError("observation files differ from the pinned completion")
    _read(completion["uri"], completion["sha256"])
    failures = [row for row in rows if row["success"] is False]
    record = {
        "schema": "npa.field-failure.observed-simulation-admission.v1",
        "rule": FAILURE_RULE,
        "plan": {"uri": args.output_root + "/reference-plan.json", "sha256": plan_sha},
        "observation_input": plan["failure_observation"]["input"],
        "baseline_training": baseline,
        "checkpoint_sha256": baseline_files["policy.pt"],
        "capture": plan["capture"],
        "capture_manifest_sha256": plan["cohorts"]["geometry"][
            "capture_manifest_sha256"
        ],
        "scene_sha256": plan["cohorts"]["geometry"]["scene_sha256"],
        "training_cases_sha256": plan["cohorts"]["regions"]["training"]["office"][
            "sha256"
        ],
        "observation_completion": completion,
        "observed_files_sha256": _digest(_encode(files)),
        "raw_outcomes_sha256": files["trajectory.json"],
        "evaluation_sha256": files["evaluation.json"],
        "native_after_load_sha256": files["native-process.json"],
        "episodes": len(rows),
        "observed_failures": len(failures),
        "failures": failures,
        "admitted": bool(failures),
        "selection_scope": "simulation failures on frozen office training routes; complete capture admission",
        "final_cohort_consumed": False,
    }
    return record, observed


def _verify_bundle_admission(bundle, record):
    captures = bundle["captures"]
    if (
        len(captures) != 1
        or captures[0]["asset"] != record["capture"]
        or captures[0]["scenario_id"] != "office-with-warehouse-replay"
        or bundle["baseline"]["checkpoint"]["sha256"] != record["checkpoint_sha256"]
    ):
        raise ValueError(
            "sealed capture or baseline differs from its bound failure admission"
        )
