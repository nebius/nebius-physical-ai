"""Explicit fake controls for the private hook only; not provider or GPU proof."""

import importlib.util
import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

DRIVER = Path(__file__).parents[1] / "gpu-fault-driver.py"


class FakeExit(BaseException):
    def __init__(self, code):
        self.code = code


@pytest.fixture
def harness(tmp_path):
    spec = importlib.util.spec_from_file_location("private_fault_controls", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tmp_path.chmod(0o700)
    module.CONFIG = {
        "mode": "after-cancellation-event",
        "run_id": "synthetic-run",
        "source_sha": "1" * 40,
        "payload_sha": "2" * 64,
        "staged_source_sha256": "3" * 64,
        "candidate_root": str(tmp_path),
        "run_prefix_uri": "s3://example-bucket/runs/synthetic-run/cuda-regression",
        "receipt": str(tmp_path / "receipt.json"),
    }
    module.MODE = module.CONFIG["mode"]
    sequence = []

    def fake_exit(code):
        sequence.append(("exit", code))
        raise FakeExit(code)

    def fsync(fd):
        os.fsync(fd)
        sequence.append(("fsync",))

    module.os = SimpleNamespace(
        environ={"NPA_SRC_S3_URI": "s3://example-bucket/source/" + "3" * 64},
        open=os.open,
        fdopen=os.fdopen,
        fsync=fsync,
        _exit=fake_exit,
        getuid=os.getuid,
        path=os.path,
        O_WRONLY=os.O_WRONLY,
        O_CREAT=os.O_CREAT,
        O_EXCL=os.O_EXCL,
        O_NOFOLLOW=os.O_NOFOLLOW,
    )
    module.runtime = SimpleNamespace(
        is_terminal=lambda status: status in {"CANCELLED", "FAILED", "SUCCEEDED"},
        _source_identity=lambda: "3" * 64,
    )
    module.ORIGINAL_OBSERVE = lambda *args, **kwargs: sequence.append(
        ("observe", kwargs)
    )

    def fake_event(self, event):
        sequence.append(("event", event))
        return (
            module.CONFIG["run_prefix_uri"]
            + "/npa-workflow/supervisor/attempts/"
            + "npa-launch-"
            + "4" * 32
            + "/cancellation-"
            + "7" * 64
            + ".json"
        )

    module.ORIGINAL_EVENT = fake_event
    module.ORIGINAL_RUNTIME = lambda self, attempt: sequence.append(
        ("runtime", module._attempt_receipt(attempt))
    )
    module.control_sequence = sequence
    return module


def attempt():
    return SimpleNamespace(
        states=["train"],
        outputs=[{"uri": "s3://example-bucket/training/manifest.json"}],
        attempt=1,
        key="synthetic-train",
        logical_launch_id="npa-launch-" + "4" * 32,
        job_id="synthetic-provider-id",
        job_name="synthetic-job-name",
        status="running",
        sky_status="SUBMITTED",
        workflow_sha256="5" * 64,
        source_sha256="3" * 64,
        image_digest="6" * 64,
        cancellation_state="",
        cancellation_error="",
        recovery_decision="",
        error_category="",
        error="",
    )


def executor(module, progress_changes=None):
    progress = {
        key: module.CONFIG[key] for key in ("run_id", "source_sha", "payload_sha")
    }
    progress.update(
        verification_round=1, heldout_loss=0.001, ignored_private_value="SYNTHETIC_ONLY"
    )
    progress.update(progress_changes or {})

    def read(key):
        module.control_sequence.append(("read", key))
        return json.dumps(progress).encode()

    store = SimpleNamespace(
        run_prefix_uri=module.CONFIG["run_prefix_uri"], read_artifact=read
    )
    ledger = SimpleNamespace(
        store=store, state=SimpleNamespace(run_id=module.CONFIG["run_id"])
    )
    ledger.record = lambda value: module.record_runtime(ledger, value)
    return SimpleNamespace(
        run_id=module.CONFIG["run_id"],
        ledger=ledger,
        _outputs_exist=lambda outputs: True,
    )


def arm(module, current=None):
    current = current or attempt()
    owner = executor(module)
    module._arm(owner, current.job_id, current, module.progress(owner, current))
    return current, owner


def event(module, current, terminal="CANCELLED"):
    return {
        "attempt_identity": module._attempt_identity(module.CONFIG["run_id"], current),
        "phase": "cancellation",
        "recovery": {"action": "reuse_completed_wave"},
        "cancellation": {
            "exact": True,
            "provider_job_id": current.job_id,
            "status": "cancelled",
            "error": "",
            "provider_terminal_status": terminal,
        },
    }


@pytest.mark.parametrize(
    "mode,code",
    [
        ("after-cancellation-event", 91),
        ("after-runtime-marker", 92),
        ("blocked-live", 93),
    ],
)
@pytest.mark.parametrize("terminal", ["CANCELLED", "FAILED", "SUCCEEDED"])
def test_owned_boundary_persists_before_exit_and_preserves_factual_status(
    harness, mode, code, terminal
):
    harness.MODE = mode
    current, owner = attempt(), executor(harness)

    def supervise(value, *, scheduler_status):
        assert scheduler_status == "RUNNING"
        value.cancellation_state, value.sky_status = "verified", terminal
        harness.record_event(None, event(harness, value, terminal))
        value.recovery_decision = "reuse_completed_wave"
        owner.ledger.record(value)

    owner._supervise_pending = supervise
    with pytest.raises(FakeExit) as interrupted:
        harness.observe(owner, current.job_id, current, scheduler_state="RUNNING")
    assert interrupted.value.code == code
    receipt = json.loads(Path(harness.CONFIG["receipt"]).read_text())
    assert receipt["run_id"] == "synthetic-run"
    assert receipt["identity"] == harness._attempt_identity("synthetic-run", current)
    assert receipt["progress"]["verification_round"] == 1
    assert "ignored_private_value" not in receipt["progress"]
    assert receipt["candidate_source_sha"] == "1" * 40
    assert receipt["source_artifact"]["sha256"] == "3" * 64
    assert receipt["proof_payload_sha256"] == "2" * 64
    assert stat.S_IMODE(Path(harness.CONFIG["receipt"]).stat().st_mode) == 0o600
    assert harness.control_sequence[-2:] == [("fsync",), ("exit", code)]
    kinds = [item[0] for item in harness.control_sequence]
    if mode == "after-cancellation-event":
        assert "event" in kinds and "runtime" not in kinds
    elif mode == "after-runtime-marker":
        assert kinds.index("event") < kinds.index("runtime") < kinds.index("exit")
        assert receipt["attempt"]["sky_status"] == terminal
    else:
        assert "event" not in kinds and "runtime" in kinds
        assert receipt["attempt"]["status"] == "failed"
        assert receipt["attempt"]["sky_status"] == "SUBMITTED"
        assert receipt["scheduler_observation"]["status"] == "RUNNING"


@pytest.mark.parametrize("state", ["PENDING", "STARTING", "SUCCEEDED", "UNKNOWN"])
def test_only_actual_running_observation_arms(harness, state):
    current, owner = attempt(), executor(harness)
    harness.observe(owner, current.job_id, current, scheduler_state=state)
    assert harness.ARMED is None
    assert [item[0] for item in harness.control_sequence] == ["observe"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("run_id", "other-run"),
        ("source_sha", "f" * 40),
        ("payload_sha", "f" * 64),
        ("verification_round", 0),
        ("verification_round", True),
        ("verification_round", "1"),
        ("heldout_loss", float("nan")),
        ("heldout_loss", float("inf")),
        ("heldout_loss", 0.5),
        ("heldout_loss", True),
    ],
)
def test_invalid_progress_cannot_arm(harness, key, value):
    current, owner = attempt(), executor(harness, {key: value})
    with pytest.raises(RuntimeError):
        harness.observe(owner, current.job_id, current, scheduler_state="RUNNING")
    assert harness.ARMED is None
    assert not Path(harness.CONFIG["receipt"]).exists()


def test_owned_prefix_and_progress_relative_path(harness):
    current, owner = attempt(), executor(harness)
    harness.progress(owner, current)
    assert ("read", "training/post-output-progress.json") in harness.control_sequence
    owner.ledger.store.run_prefix_uri += "-different"
    with pytest.raises(RuntimeError, match="prefix differs"):
        harness.progress(owner, current)


@pytest.mark.parametrize(
    "condition",
    ["other-run", "other-stage", "missing-output", "empty-output", "missing-progress"],
)
def test_unready_or_unrelated_work_does_not_arm(harness, condition):
    current, owner = attempt(), executor(harness)
    if condition == "other-run":
        owner.run_id = "other"
    elif condition == "other-stage":
        current.states = ["verify"]
    elif condition == "missing-output":
        owner._outputs_exist = lambda outputs: False
    elif condition == "empty-output":
        current.outputs = []
    else:

        def missing(key):
            raise FileNotFoundError(key)

        owner.ledger.store.read_artifact = missing
    harness.observe(owner, current.job_id, current, scheduler_state="RUNNING")
    assert harness.ARMED is None


@pytest.mark.parametrize(
    "key",
    [
        "runtime",
        "run_id",
        "attempt",
        "logical_attempt_id",
        "provider_job_id",
        "provider_job_name",
        "workflow_sha256",
        "source_sha256",
        "image_digest",
    ],
)
def test_immutable_event_for_different_attempt_never_exits(harness, key):
    current, _ = arm(harness)
    document = event(harness, current)
    document["attempt_identity"][key] = 2 if key == "attempt" else "different"
    harness.record_event(None, document)
    assert harness.CANCELLATION is None
    assert not Path(harness.CONFIG["receipt"]).exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("exact", False),
        ("exact", "true"),
        ("provider_job_id", "different"),
        ("status", "requested"),
        ("error", "synthetic-error"),
        ("provider_terminal_status", "RUNNING"),
        ("provider_terminal_status", ""),
    ],
)
def test_cancellation_must_be_exact_successfully_verified_and_terminal(
    harness, key, value
):
    current, _ = arm(harness)
    document = event(harness, current)
    document["cancellation"][key] = value
    with pytest.raises(RuntimeError):
        harness.record_event(None, document)
    assert harness.CANCELLATION is None
    assert not Path(harness.CONFIG["receipt"]).exists()


@pytest.mark.parametrize("mode", ["after-cancellation-event", "after-runtime-marker"])
def test_unarmed_hooks_never_exit(harness, mode):
    harness.MODE = mode
    current, owner = attempt(), executor(harness)
    current.recovery_decision, current.cancellation_state = (
        "reuse_completed_wave",
        "verified",
    )
    current.sky_status = "CANCELLED"
    harness.record_event(None, event(harness, current))
    harness.record_runtime(owner.ledger, current)
    assert harness.ARMED is None
    assert not Path(harness.CONFIG["receipt"]).exists()


@pytest.mark.parametrize(
    "condition",
    [
        "missing-event",
        "cancellation-error",
        "unverified",
        "terminal-disagreement",
        "not-terminal",
    ],
)
def test_runtime_marker_requires_matching_durable_cancellation(harness, condition):
    harness.MODE = "after-runtime-marker"
    current, owner = arm(harness)
    if condition != "missing-event":
        harness.record_event(None, event(harness, current))
    current.recovery_decision, current.cancellation_state = (
        "reuse_completed_wave",
        "verified",
    )
    current.sky_status = "CANCELLED"
    if condition == "cancellation-error":
        current.cancellation_error = "synthetic-error"
    elif condition == "unverified":
        current.cancellation_state = "requested"
    elif condition == "terminal-disagreement":
        current.sky_status = "SUCCEEDED"
    elif condition == "not-terminal":
        current.sky_status = "RUNNING"
    with pytest.raises(RuntimeError):
        harness.record_runtime(owner.ledger, current)
    assert not Path(harness.CONFIG["receipt"]).exists()


@pytest.mark.parametrize(
    "condition", ["job", "source-env", "source-runtime", "bad-logical", "not-running"]
)
def test_arming_rejects_wrong_exact_job_or_source(harness, condition):
    current, owner = attempt(), executor(harness)
    job_id = current.job_id
    if condition == "job":
        job_id = "different"
    elif condition == "source-env":
        harness.os.environ["NPA_SRC_S3_URI"] = "s3://example-bucket/source/" + "f" * 64
    elif condition == "source-runtime":
        harness.runtime._source_identity = lambda: "f" * 64
    elif condition == "bad-logical":
        current.logical_launch_id = ""
    else:
        current.status = "failed"
    with pytest.raises(RuntimeError):
        harness._arm(owner, job_id, current, harness.progress(owner, current))
    assert harness.ARMED is None


@pytest.mark.parametrize(
    "identity",
    [
        "4" * 64,
        "npa-launch-" + "4" * 31,
        "npa-launch-" + "4" * 33,
        "npa-launch-" + "G" * 32,
        "npa-launch-" + "4" * 32 + "/other",
        None,
    ],
)
def test_arming_rejects_nonproduction_logical_identity(harness, identity):
    current, owner = attempt(), executor(harness)
    current.logical_launch_id = identity
    with pytest.raises(RuntimeError, match="production logical launch ID"):
        harness._arm(owner, current.job_id, current, harness.progress(owner, current))
    assert harness.ARMED is None


def test_production_logical_identity_arms_and_matches_exactly(harness):
    from npa.orchestration.skypilot.launch_transaction import logical_launch_identity

    current, owner = attempt(), executor(harness)
    current.logical_launch_id = logical_launch_identity("synthetic-run", "train", "1")
    harness._arm(owner, current.job_id, current, harness.progress(owner, current))
    identity = harness._attempt_identity(harness.CONFIG["run_id"], current)
    assert harness._matches_armed(identity)
    identity["logical_attempt_id"] = logical_launch_identity("other-run", "train", "1")
    assert not harness._matches_armed(identity)


def backend_fixture(harness, **changes):
    from npa.orchestration.npa_workflow.supervisor import (
        AttemptIdentity,
        BackendObservation,
        BackendState,
    )

    current, _owner = arm(harness)
    identity = AttemptIdentity(
        **harness._attempt_identity(harness.CONFIG["run_id"], current)
    )
    observed = BackendObservation(
        BackendState.RUNNING, evidence={"status": "RUNNING", "blockers": []}
    )
    observed = replace(observed, **changes)
    harness.ORIGINAL_BACKEND = lambda adapter, selected: observed
    return identity, observed


def test_reason_only_injection_preserves_observed_facts_and_changes_policy(harness):
    from npa.orchestration.npa_workflow.supervisor import (
        ArtifactValidation,
        RecoveryAction,
        RecoveryContext,
        decide_recovery,
    )

    identity, observed = backend_fixture(harness)
    injected = harness.observe_backend(None, identity)
    assert replace(injected, reason_code=observed.reason_code) == observed
    assert injected.evidence is observed.evidence
    assert identity.to_dict() == {**harness.ARMED["identity"], "checkpoint_prefix": ""}
    assert injected.reason_code == "PROVIDER_INTERRUPTION"
    assert harness.ARMED["injected_observation"]["only_changed_field"] == "reason_code"
    outputs = ArtifactValidation(
        "valid",
        declared=("s3://example-bucket/model",),
        valid=("s3://example-bucket/model",),
    )
    context = RecoveryContext(
        identity.workflow_sha256, identity.source_sha256, identity.image_digest, outputs
    )
    assert (
        decide_recovery(identity, observed, context).action
        is RecoveryAction.ADOPT_EXACT_ATTEMPT
    )
    assert (
        decide_recovery(identity, injected, context).action
        is RecoveryAction.REUSE_COMPLETED_WAVE
    )
    assert not Path(harness.CONFIG["receipt"]).exists()


@pytest.mark.parametrize("condition", ["blocked-mode", "unarmed", "other-id"])
def test_reason_injection_does_not_change_unselected_observations(harness, condition):
    identity, observed = backend_fixture(harness)
    if condition == "blocked-mode":
        harness.MODE = "blocked-live"
    elif condition == "unarmed":
        harness.ARMED = None
    else:
        identity = replace(identity, provider_job_id="different-exact-id")
    assert harness.observe_backend(None, identity) is observed


@pytest.mark.parametrize(
    "condition",
    [
        "queued",
        "failed",
        "ambiguous",
        "succeeded",
        "cancelled",
        "inexact",
        "unobservable",
        "real-reason",
        "wrong-provider-status",
    ],
)
def test_reason_injection_requires_original_healthy_running_evidence(
    harness, condition
):
    from npa.orchestration.npa_workflow.supervisor import BackendState

    changes = {}
    if condition in {"queued", "failed", "ambiguous", "succeeded", "cancelled"}:
        changes["state"] = BackendState(condition)
    elif condition == "inexact":
        changes["exact_identity"] = False
    elif condition == "unobservable":
        changes["workload_observable"] = False
    elif condition == "real-reason":
        changes["reason_code"] = "NODE_NOT_READY"
    else:
        changes["evidence"] = {"status": "FAILED"}
    identity, _observed = backend_fixture(harness, **changes)
    with pytest.raises(RuntimeError, match="exact healthy RUNNING"):
        harness.observe_backend(None, identity)
    assert "injected_observation" not in harness.ARMED
    assert not Path(harness.CONFIG["receipt"]).exists()


@pytest.mark.parametrize(
    "selectors",
    [
        ["--run-id", "synthetic-run"],
        ["--run-id=synthetic-run"],
        ["--resume-run", "synthetic-run"],
        ["--resume-run=synthetic-run"],
        ["--run-id", "synthetic-run", "--resume-run=synthetic-run"],
    ],
)
def test_cli_accepts_only_matching_explicit_run_selectors(harness, selectors):
    harness._validate_config(
        harness.CONFIG, ["workbench", "workflow", "submit", *selectors]
    )


@pytest.mark.parametrize(
    "selectors",
    [
        [],
        ["--run-id"],
        ["--run-id=other"],
        ["--resume-run", "other"],
        ["--run-id", "synthetic-run", "--resume-run", "other"],
        ["--run-id", "synthetic-run", "--run-id=other"],
        ["--run-id", "--runtime"],
    ],
)
def test_cli_rejects_missing_or_conflicting_run_before_installing_hooks(
    harness, selectors
):
    with pytest.raises(ValueError):
        harness._validate_config(
            harness.CONFIG, ["workbench", "workflow", "submit", *selectors]
        )
    assert harness.ARMED is None


def test_existing_receipt_rejected_before_cli_and_no_clobber(harness):
    target = Path(harness.CONFIG["receipt"])
    target.write_text("existing")
    with pytest.raises(ValueError, match="must be new"):
        harness._prepare_receipt(harness.CONFIG)
    assert target.read_text() == "existing"


def test_public_receipt_directory_rejected(harness):
    Path(harness.CONFIG["receipt"]).parent.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        harness._prepare_receipt(harness.CONFIG)


def test_candidate_verification_binds_module_checkout_commit_and_clean_code(harness):
    root = Path(harness.CONFIG["candidate_root"])
    module = root / "npa/src/npa/orchestration/npa_workflow/runtime.py"
    module.parent.mkdir(parents=True)
    module.write_text("# explicitly fake candidate source\n")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="1" * 40 + "\n", returncode=0)

    harness.subprocess = SimpleNamespace(run=fake_run)
    provenance = harness._verify_candidate(
        harness.CONFIG, SimpleNamespace(__file__=str(module))
    )
    assert provenance["candidate_commit"] == "1" * 40
    assert len(provenance["runtime_module_sha256"]) == 64
    assert commands[0][-2:] == ["rev-parse", "HEAD"]
    assert commands[1][-5:] == ["diff", "HEAD", "--quiet", "--", "npa/src/npa"]


@pytest.mark.parametrize("condition", ["wrong-import", "wrong-head", "dirty-source"])
def test_candidate_mismatch_fails_before_cloud(harness, condition):
    root = Path(harness.CONFIG["candidate_root"])
    module = root / "npa/src/npa/orchestration/npa_workflow/runtime.py"
    module.parent.mkdir(parents=True)
    module.write_text("# fake source\n")
    if condition == "wrong-import":
        module = root / "other.py"
        module.write_text("# different fake module\n")

    def fake_run(command, **kwargs):
        return SimpleNamespace(
            stdout=("f" if condition == "wrong-head" else "1") * 40,
            returncode=1 if condition == "dirty-source" else 0,
        )

    harness.subprocess = SimpleNamespace(run=fake_run)
    with pytest.raises(RuntimeError):
        harness._verify_candidate(harness.CONFIG, SimpleNamespace(__file__=str(module)))


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "attempt",
        "logical_launch_id",
        "job_id",
        "job_name",
        "workflow_sha256",
        "source_sha256",
        "image_digest",
        "states",
    ],
)
def test_runtime_marker_for_different_attempt_never_exits(harness, field):
    harness.MODE = "after-runtime-marker"
    current, owner = arm(harness)
    harness.record_event(None, event(harness, current))
    current.recovery_decision, current.cancellation_state = (
        "reuse_completed_wave",
        "verified",
    )
    current.sky_status = "CANCELLED"
    if field == "run_id":
        owner.ledger.state.run_id = "different"
    else:
        setattr(
            current,
            field,
            2
            if field == "attempt"
            else ["verify"]
            if field == "states"
            else "different",
        )
    harness.record_runtime(owner.ledger, current)
    assert not Path(harness.CONFIG["receipt"]).exists()


@pytest.mark.parametrize("condition", ["permission", "invalid-json", "non-object"])
def test_progress_read_failures_propagate_without_arming(harness, condition):
    current, owner = attempt(), executor(harness)

    def unreadable(key):
        if condition == "permission":
            raise PermissionError("synthetic-denial")
        return b"not-json" if condition == "invalid-json" else b"[]"

    owner.ledger.store.read_artifact = unreadable
    with pytest.raises((PermissionError, json.JSONDecodeError, TypeError)):
        harness.observe(owner, current.job_id, current, scheduler_state="RUNNING")
    assert harness.ARMED is None


@pytest.mark.parametrize(
    "uri",
    [
        "s3://example-bucket/other-event",
        "s3://example-bucket/runs/synthetic-run/cuda-regression/npa-workflow/supervisor/attempts/"
        + "4" * 64
        + "/cancellation-bad.json",
    ],
)
def test_cancellation_receipt_requires_exact_content_addressed_event_path(harness, uri):
    current, _ = arm(harness)
    harness.ORIGINAL_EVENT = lambda self, document: uri
    with pytest.raises(RuntimeError):
        harness.record_event(None, event(harness, current))
    assert harness.CANCELLATION is None
    assert not Path(harness.CONFIG["receipt"]).exists()


def test_receipt_never_follows_a_symlink(harness, tmp_path):
    arm(harness)
    target = tmp_path / "untouched.json"
    target.write_text("private target remains unchanged")
    Path(harness.CONFIG["receipt"]).symlink_to(target)
    with pytest.raises(FileExistsError):
        harness.crash({}, 91)
    assert target.read_text() == "private target remains unchanged"
    assert not any(item[0] == "exit" for item in harness.control_sequence)
