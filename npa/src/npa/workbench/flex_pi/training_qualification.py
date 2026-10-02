"""Compare independent native-DDP fixture processes without restoring reducers."""

import hashlib
import json
import time

from npa.workbench.flex_pi.runtime import FlexPiError


def _phase_plan(plan, work, label, fill, checkpoint):
    phase = {
        **plan,
        "work_directory": str(work / label),
        "execution": {**plan["execution"], "mode": "qualify", "memory_fill": fill},
    }
    if checkpoint is not None:
        phase.update(
            resume_directory=str(checkpoint),
            normalization_file=str(checkpoint / "dataset_stats.json"),
        )
    return phase


def _comparable_workload(work):
    identity = json.loads((work / "workload.json").read_text())
    identity["configuration"].pop("npa_memory_fill", None)
    return identity


def _check_qualification(reference, candidate, reference_work, candidate_work):
    if _comparable_workload(reference_work) != _comparable_workload(candidate_work):
        raise FlexPiError("memory-fill qualification changed the model workload")
    for key in (
        "initial_model_sha256",
        "training_source_sha256",
        "normalization_sha256",
        "qualification",
    ):
        if candidate[key] != reference[key]:
            raise FlexPiError(f"memory-fill qualification differs in {key}")
    if not candidate["memory_fill"]["strict_deterministic_algorithms"]:
        raise FlexPiError("qualification relaxed deterministic algorithms")


def _check_policy(result, fill):
    count = result["memory_fill"]["observed_disabled_scopes"]
    if type(count) is not int or (count != 0 if fill == "on" else count <= 0):
        raise FlexPiError("qualification lacks observed allocation-scope evidence")
    expected = {
        "model_execution_fill": fill == "on",
        "observed_disabled_scopes": count,
        "initialization_and_loader_creation_fill": True,
        "loader_worker_fill": True,
        "allocation_flag_scope": "process_global_including_pin_thread",
        "restored_fill": True,
        "strict_deterministic_algorithms": True,
    }
    if result["memory_fill"] != expected:
        raise FlexPiError(
            "qualification did not execute the requested memory-fill policy"
        )


def qualify_memory_fill(plan, root, work, run_phase, *, checkpoint=None):
    """Require bitwise cold/replay/candidate agreement in isolated processes.

    Args:
        plan: Frozen real-data workload and execution request.
        root: Private parent request directory.
        work: Run-scoped worker output directory.
        run_phase: Maintained vendor-interpreter phase launcher.
        checkpoint: Verified original checkpoint for the second qualification.
    Returns:
        Hash-bound receipt covering real full-96 and final-36 fixtures.
    Raises:
        FlexPiError: Any input, state, loss, gradient or update differs.
    """
    stage = "checkpoint" if checkpoint is not None else "initial"
    started = time.perf_counter()
    reports = []
    for name, fill in (("capture", "on"), ("replay", "on"), ("candidate", "off")):
        try:
            result = _run_qualification_phase(
                plan, root, work, stage, name, fill, checkpoint, run_phase
            )
            reports.append(result)
            _check_policy(result, fill)
            if name != "capture":
                _check_qualification(
                    reports[0],
                    result,
                    work / f"fill-{stage}-capture",
                    work / f"fill-{stage}-{name}",
                )
        except (FlexPiError, OSError, ValueError, KeyError) as exc:
            _persist_failure(work, stage, name, reports, started, exc)
            raise
    receipt = _qualification_receipt(stage, reports)
    receipt["seconds"] = time.perf_counter() - started
    return receipt


def _run_qualification_phase(
    plan, root, work, stage, name, fill, checkpoint, run_phase
):
    label = f"fill-{stage}-{name}"
    request_root = root / label
    request_root.mkdir()
    phase = _phase_plan(plan, work, label, fill, checkpoint)
    result = run_phase(phase, request_root)
    path = work / f"{label}-result.json"
    path.write_text(json.dumps(result, sort_keys=True, allow_nan=False))
    path.chmod(0o400)
    return result


def _persist_failure(work, stage, phase, reports, started, error):
    receipt = {
        "passed": False,
        "stage": stage,
        "failed_phase": phase,
        "error_type": type(error).__name__,
        "reports": reports,
        "seconds": time.perf_counter() - started,
        "diagnostic_only": True,
    }
    path = work / f"fill-{stage}-rejection.json"
    path.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False))
    path.chmod(0o400)


def _qualification_receipt(stage, reports):
    payload = json.dumps(reports, sort_keys=True, allow_nan=False).encode()
    return {
        "stage": stage,
        "passed": True,
        "native_ddp": True,
        "independent_processes": 3,
        "fixture_samples": [96, 36],
        "bitwise_equal": True,
        "reports_sha256": hashlib.sha256(payload).hexdigest(),
        "reports": reports,
        "diagnostic_only": True,
    }
