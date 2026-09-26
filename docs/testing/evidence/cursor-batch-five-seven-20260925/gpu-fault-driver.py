"""Inject an owned host-driver exit after observed GPU output publication.

This private harness explicitly invokes production supervision on a real RUNNING
observation and injects only a supported transient reason into that observation.
Native polling normally invokes that path while pending/retrying. It never invents
provider status. Local fake controls are not GPU evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

CONFIG = {}
MODE = ""
ARMED = None
CANCELLATION = None
PROVENANCE = {}
runtime = None
ORIGINAL_OBSERVE = None
ORIGINAL_EVENT = None
ORIGINAL_RUNTIME = None
ORIGINAL_BACKEND = None
MODES = {"blocked-live", "after-cancellation-event", "after-runtime-marker"}
IDENTITY_FIELDS = (
    "runtime",
    "run_id",
    "attempt",
    "logical_attempt_id",
    "provider_job_id",
    "provider_job_name",
    "workflow_sha256",
    "source_sha256",
    "image_digest",
)


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value)


def _cli_run_ids(argv):
    """Read only explicit run selectors; the real CLI still parses all options."""
    result = []
    for index, token in enumerate(argv):
        if token in {"--run-id", "--resume-run"}:
            if index + 1 == len(argv) or argv[index + 1].startswith("-"):
                raise ValueError("proof CLI run selector is missing its value")
            result.append(argv[index + 1])
        elif token.startswith(("--run-id=", "--resume-run=")):
            result.append(token.split("=", 1)[1])
    return result


def _validate_config(config, argv):
    if config.get("mode") not in MODES:
        raise ValueError("unknown proof injection mode")
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("proof run ID must be an explicit safe identifier")
    command = argv[:3] == ["workbench", "workflow", "submit"]
    if not command and argv[:2] != ["workflow", "submit"]:
        raise ValueError("proof injection requires workflow submit")
    selected = _cli_run_ids(argv)
    if not selected or any(value != run_id for value in selected):
        raise ValueError("every explicit CLI run selector must equal proof run ID")
    for key, length in (
        ("source_sha", 40),
        ("payload_sha", 64),
        ("staged_source_sha256", 64),
    ):
        if not _hex(config.get(key), length):
            raise ValueError(f"proof {key} must be a lowercase full digest")
    parsed = urlsplit(str(config.get("run_prefix_uri", "")))
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.strip("/")
        or ".." in parsed.path.split("/")
    ):
        raise ValueError("proof run prefix must be an unsigned exact S3 prefix")
    if not Path(config.get("candidate_root", "")).is_absolute():
        raise ValueError("proof candidate root must be absolute")
    if not Path(config.get("receipt", "")).is_absolute():
        raise ValueError("proof receipt must be absolute")


def _verify_candidate(config, imported_runtime):
    root = Path(config["candidate_root"]).resolve(strict=True)
    relative = "npa/src/npa/orchestration/npa_workflow/runtime.py"
    module = Path(imported_runtime.__file__).resolve(strict=True)
    if module != root / relative:
        raise RuntimeError("proof runtime import is outside the exact candidate")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != config["source_sha"]:
        raise RuntimeError("proof candidate HEAD differs from configured commit")
    clean = subprocess.run(
        ["git", "-C", str(root), "diff", "HEAD", "--quiet", "--", "npa/src/npa"],
        check=False,
        capture_output=True,
    )
    if clean.returncode:
        raise RuntimeError("proof candidate Python source is changed or unreadable")
    return {
        "candidate_commit": revision,
        "runtime_module": str(module),
        "runtime_module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
    }


def _prepare_receipt(config):
    receipt = Path(config["receipt"])
    receipt.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    owner = receipt.parent.stat()
    if owner.st_uid != os.getuid() or owner.st_mode & 0o077:
        raise ValueError("proof receipt parent must be owned and mode 0700")
    if os.path.lexists(receipt):
        raise ValueError("proof receipt must be new before starting the CLI")


def _attempt_identity(run_id, attempt):
    return {
        "runtime": "skypilot",
        "run_id": run_id,
        "attempt": attempt.attempt,
        "logical_attempt_id": attempt.logical_launch_id,
        "provider_job_id": attempt.job_id,
        "provider_job_name": attempt.job_name,
        "workflow_sha256": attempt.workflow_sha256,
        "source_sha256": attempt.source_sha256,
        "image_digest": attempt.image_digest,
    }


def _matches_armed(identity):
    return ARMED is not None and all(
        type(identity.get(key)) is type(ARMED["identity"][key])
        and identity.get(key) == ARMED["identity"][key]
        for key in IDENTITY_FIELDS
    )


def _source_artifact(attempt):
    uri = os.environ.get("NPA_SRC_S3_URI", "").rstrip("/")
    parsed = urlsplit(uri)
    digest = parsed.path.rsplit("/", 1)[-1]
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.query
        or parsed.fragment
        or ".." in parsed.path.split("/")
        or not _hex(digest, 64)
    ):
        raise RuntimeError("proof requires a content-addressed staged NPA source")
    if digest != CONFIG["staged_source_sha256"] or digest != attempt.source_sha256:
        raise RuntimeError("proof staged source digest differs from runtime identity")
    if runtime._source_identity() != digest:
        raise RuntimeError(
            "proof runtime source identity differs from source environment"
        )
    return {"uri": uri, "sha256": digest}


def progress(executor, attempt):
    if executor.run_id != CONFIG["run_id"] or attempt.states != ["train"]:
        return None
    if executor.ledger.store.run_prefix_uri.rstrip("/") != CONFIG[
        "run_prefix_uri"
    ].rstrip("/"):
        raise RuntimeError(
            "proof run store prefix differs from configured owned prefix"
        )
    if not attempt.outputs or not executor._outputs_exist(attempt.outputs):
        return None
    try:
        value = json.loads(
            executor.ledger.store.read_artifact("training/post-output-progress.json")
        )
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise TypeError("post-output CUDA progress must be a JSON object")
    for key in ("run_id", "source_sha", "payload_sha"):
        if value.get(key) != CONFIG[key]:
            raise RuntimeError(f"post-output progress {key} identity differs")
    if (
        type(value.get("verification_round")) is not int
        or value["verification_round"] < 1
    ):
        raise RuntimeError("post-output CUDA work did not execute")
    loss = value.get("heldout_loss")
    if type(loss) not in {int, float} or not math.isfinite(loss) or not 0 < loss < 0.02:
        raise RuntimeError("post-output CUDA verification loss is invalid")
    return {
        key: value[key]
        for key in (
            "run_id",
            "source_sha",
            "payload_sha",
            "verification_round",
            "heldout_loss",
        )
    }


def _arm(executor, job_id, attempt, observed):
    global ARMED
    if job_id != attempt.job_id or not job_id or not attempt.job_name:
        raise RuntimeError("proof observation does not name the exact provider job")
    if (
        attempt.status != "running"
        or type(attempt.attempt) is not int
        or attempt.attempt < 1
    ):
        raise RuntimeError("proof requires a running numbered runtime attempt")
    identity = _attempt_identity(executor.run_id, attempt)
    logical_id = identity["logical_attempt_id"]
    if not isinstance(logical_id, str) or not re.fullmatch(
        r"npa-launch-[0-9a-f]{32}", logical_id
    ):
        raise RuntimeError("proof attempt lacks a production logical launch ID")
    for key in (
        "workflow_sha256",
        "source_sha256",
        "image_digest",
    ):
        if not _hex(identity[key], 64):
            raise RuntimeError(f"proof attempt lacks complete {key}")
    if ARMED is not None:
        raise RuntimeError("proof crash hook was already armed")
    ARMED = {
        "identity": identity,
        "progress": observed,
        "source_artifact": _source_artifact(attempt),
        "scheduler_observation": {"job_id": job_id, "status": "RUNNING"},
    }


def _attempt_receipt(attempt):
    """Keep only proof fields, never arbitrary provider logs or credential metadata."""
    return {
        "key": attempt.key,
        "states": list(attempt.states),
        "attempt": attempt.attempt,
        "job_id": attempt.job_id,
        "job_name": attempt.job_name,
        "logical_launch_id": attempt.logical_launch_id,
        "status": attempt.status,
        "sky_status": attempt.sky_status,
        "recovery_decision": attempt.recovery_decision,
        "cancellation": {
            "state": attempt.cancellation_state,
            "error": attempt.cancellation_error,
        },
        "immutable_identity": {
            key: getattr(attempt, key)
            for key in ("workflow_sha256", "source_sha256", "image_digest")
        },
    }


def crash(payload, code):
    if ARMED is None:
        raise RuntimeError("proof crash is not armed")
    receipt = {
        "schema": "npa-gpu-fault-receipt/v2",
        "mode": MODE,
        "exit_code": code,
        "run_id": CONFIG["run_id"],
        "candidate_source_sha": CONFIG["source_sha"],
        "proof_payload_sha256": CONFIG["payload_sha"],
        "run_prefix_uri": CONFIG["run_prefix_uri"],
        **PROVENANCE,
        **ARMED,
        **payload,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(CONFIG["receipt"], flags, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os._exit(code)


def observe(self, job_id, attempt, **kwargs):
    ORIGINAL_OBSERVE(self, job_id, attempt, **kwargs)
    if kwargs.get("scheduler_state") != "RUNNING":
        return
    observed = progress(self, attempt)
    if observed is None:
        return
    _arm(self, job_id, attempt, observed)
    if MODE == "blocked-live":
        attempt.status = "failed"
        attempt.recovery_decision = "block_relaunch"
        attempt.error_category = "controller"
        attempt.error = "Declared proof injection: host driver interrupted"
        self.ledger.record(attempt)
        crash({"attempt": _attempt_receipt(attempt)}, 93)
    # Deliberate injection on an exact observed live job, never synthetic PENDING.
    self._supervise_pending(attempt, scheduler_status="RUNNING")
    raise RuntimeError("Expected crash boundary was not reached")


def observe_backend(self, identity):
    from npa.orchestration.npa_workflow.supervisor import BackendState

    observed = ORIGINAL_BACKEND(self, identity)
    if MODE == "blocked-live" or not _matches_armed(identity.to_dict()):
        return observed
    if (
        observed.state is not BackendState.RUNNING
        or not observed.exact_identity
        or not observed.workload_observable
        or observed.reason_code
        or observed.evidence.get("status") != "RUNNING"
    ):
        raise RuntimeError(
            "reason injection requires an exact healthy RUNNING observation"
        )
    injected = replace(observed, reason_code="PROVIDER_INTERRUPTION")
    ARMED["injected_observation"] = {
        "classification": "Injected transient reason over observed healthy live job; no actual provider fault claim",
        "original_reason_code": observed.reason_code,
        "injected_reason_code": injected.reason_code,
        "original_state": observed.state.value,
        "injected_state": injected.state.value,
        "exact_identity": observed.exact_identity,
        "workload_observable": observed.workload_observable,
        "provider_status": observed.evidence["status"],
        "only_changed_field": "reason_code",
    }
    return injected


def _verified_cancellation(event):
    cancellation = event.get("cancellation") or {}
    if cancellation.get("exact") is not True:
        raise RuntimeError("proof cancellation did not assert exact ownership")
    if cancellation.get("provider_job_id") != ARMED["identity"]["provider_job_id"]:
        raise RuntimeError("proof cancellation provider ID differs from armed attempt")
    if cancellation.get("status") != "cancelled" or cancellation.get("error") not in (
        None,
        "",
    ):
        raise RuntimeError("proof exact cancellation did not complete")
    terminal = cancellation.get("provider_terminal_status")
    if not isinstance(terminal, str) or not runtime.is_terminal(terminal):
        raise RuntimeError("proof cancellation lacks factual terminal provider status")
    return {
        "provider_job_id": cancellation["provider_job_id"],
        "exact": True,
        "status": "cancelled",
        "error": "",
        "provider_terminal_status": terminal,
    }


def record_event(self, event):
    global CANCELLATION
    uri = ORIGINAL_EVENT(self, event)
    if not _matches_armed(event.get("attempt_identity") or {}):
        return uri
    if event.get("phase") != "cancellation":
        return uri
    if (event.get("recovery") or {}).get("action") != "reuse_completed_wave":
        return uri
    cancellation = _verified_cancellation(event)
    prefix = CONFIG["run_prefix_uri"].rstrip("/")
    logical_id = ARMED["identity"]["logical_attempt_id"]
    expected = f"{prefix}/npa-workflow/supervisor/attempts/{logical_id}/cancellation-"
    if not isinstance(uri, str) or not uri.startswith(expected):
        raise RuntimeError(
            "proof cancellation event is outside its exact attempt prefix"
        )
    suffix = uri.removeprefix(expected)
    if not suffix.endswith(".json") or not _hex(suffix.removesuffix(".json"), 64):
        raise RuntimeError(
            "proof cancellation event lacks its immutable content address"
        )
    CANCELLATION = {"event_uri": uri, "cancellation": cancellation}
    if MODE == "after-cancellation-event":
        crash(CANCELLATION, 91)
    return uri


def record_runtime(self, attempt):
    ORIGINAL_RUNTIME(self, attempt)
    if MODE != "after-runtime-marker" or attempt.states != ["train"]:
        return
    if not _matches_armed(_attempt_identity(self.state.run_id, attempt)):
        return
    if (
        attempt.recovery_decision != "reuse_completed_wave"
        or attempt.status != "running"
    ):
        return
    if CANCELLATION is None:
        raise RuntimeError(
            "proof runtime marker lacks the exact persisted cancellation event"
        )
    if attempt.cancellation_state != "verified" or attempt.cancellation_error:
        raise RuntimeError("proof runtime reuse lacks verified cancellation")
    terminal = CANCELLATION["cancellation"]["provider_terminal_status"]
    if not runtime.is_terminal(attempt.sky_status) or attempt.sky_status != terminal:
        raise RuntimeError(
            "proof runtime reuse lost factual cancellation terminal status"
        )
    crash({**CANCELLATION, "attempt": _attempt_receipt(attempt)}, 92)


def main():
    global CONFIG, MODE, PROVENANCE, runtime
    global ORIGINAL_OBSERVE, ORIGINAL_EVENT, ORIGINAL_RUNTIME, ORIGINAL_BACKEND
    config_path = Path(os.environ["NPA_PROOF_FAULT_CONFIG"])
    CONFIG = json.loads(config_path.read_text())
    _validate_config(CONFIG, sys.argv[1:])
    from npa.orchestration.npa_workflow import runtime as candidate_runtime
    from npa.orchestration.npa_workflow.supervisor import (
        SkyPilotSupervisorAdapter,
        SupervisorLedger,
    )

    runtime = candidate_runtime
    MODE = CONFIG["mode"]
    PROVENANCE = _verify_candidate(CONFIG, runtime)
    _prepare_receipt(CONFIG)
    ORIGINAL_OBSERVE = runtime.SkyPilotWaveExecutor._observe_concurrency
    ORIGINAL_EVENT = SupervisorLedger.record
    ORIGINAL_RUNTIME = runtime.RuntimeLedger.record
    ORIGINAL_BACKEND = SkyPilotSupervisorAdapter.observe
    runtime.SkyPilotWaveExecutor._observe_concurrency = observe
    runtime.RuntimeLedger.record = record_runtime
    SupervisorLedger.record = record_event
    SkyPilotSupervisorAdapter.observe = observe_backend
    from npa.cli.entry import main as cli_main

    sys.argv[0] = "npa-proof-fault"
    cli_main()


if __name__ == "__main__":
    main()
