"""Verify specialist reporting from downloaded original campaign evidence."""

from copy import deepcopy
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import av
import numpy as np
import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.behavior_challenge import campaign_runner, rlc_policy
from npa.workflows.behavior_challenge import rlc_specialist_admission as admission
from npa.workflows.behavior_challenge.campaign import (
    canonical_digest,
    declare_panel,
    freeze_policy_identity,
    partition_panel,
)
from npa.workflows.behavior_challenge.case_store import CaseStore
from npa.workflows.behavior_challenge.evaluator_versions import UPSTREAM_COMMITS


class MemoryStorage:
    def __init__(self):
        self.objects = {}
        self.revision = 0

    def read_bytes_with_etag(self, uri):
        return self.objects.get(uri)

    def put_bytes_conditional(
        self, payload, uri, *, if_match="", if_none_match=False, **_
    ):
        current = self.objects.get(uri)
        if (if_none_match and current) or (
            if_match and (not current or current[1] != if_match)
        ):
            raise StoragePreconditionFailed("conditional conflict")
        self.revision += 1
        self.objects[uri] = (payload, str(self.revision))
        return str(self.revision)

    def download_file(self, uri, destination):
        Path(destination).write_bytes(self.objects[uri][0])


def _policy(name: str, marker: str) -> dict:
    return freeze_policy_identity(
        name,
        {
            "checkpoint": {"sha256": marker * 64, "bytes": 1},
            "serving": {"sha256": marker * 64, "bytes": 2},
        },
    )


def _panel(policy: dict, split: str, upstream_commit: str | None = None) -> dict:
    tasks = [f"task-{index}" for index in range(100)]
    tasks[1] = "picking_up_trash"
    kwargs = {} if upstream_commit is None else {"upstream_commit": upstream_commit}
    return declare_panel(policy, tasks, ["picking_up_trash"], split, **kwargs)


def _write_rollout(case: dict, output: Path, q_score: float) -> None:
    stem = f"{case['task']}_{case['instance_id']}_0"
    (output / "json").mkdir(exist_ok=True)
    (output / "videos").mkdir(exist_ok=True)
    distances = {key: 1.0 for key in ("base", "left", "right")}
    metrics = {key: case[key] for key in ("task", "instance_id", "rollout_id")}
    metrics.update(
        steps=2,
        success=q_score >= 0.5,
        q_score={"final": q_score},
        agent_distance=distances,
        normalized_agent_distance=distances,
        time={"simulator_steps": 2, "simulator_time": 2 / 30, "normalized_time": 10.0},
    )
    (output / f"json/{stem}.json").write_text(json.dumps(metrics) + "\n")
    with av.open(str(output / f"videos/{stem}.mp4"), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(
            np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24"
        )
        container.mux(stream.encode(frame))
        container.mux(stream.encode())


def _populate(storage, panel: dict, prefix: str, workspace: Path, score: float) -> None:
    store = CaseStore(storage, prefix, panel["panel_id"])
    campaign_runner.run_partition(
        panel,
        partition_panel(panel, 1),
        0,
        store,
        workspace,
        lambda case, output: _write_rollout(case, output, score),
    )


def _write(path: Path, value: dict) -> dict:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return {"path": str(path), "sha256": admission.file_digest(path)}


def _self_digest(value: dict, field: str) -> dict:
    return {**value, field: canonical_digest(value)}


def _seal(tmp_path: Path, arm: str, panel: dict) -> dict:
    value = {
        "schema": "npa.behavior.baseline-results-seal.v1",
        "status": "frozen_before_results_unseal",
        "arm": arm,
        "panel_id": panel["panel_id"],
        "policy_identity_sha256": panel["policy"]["identity_sha256"],
    }
    return _write(tmp_path / f"{arm}-seal.json", _self_digest(value, "seal_sha256"))


def _evidence(tmp_path, storage, name, panel, prefix, seal_sha):
    declared = {"panel": panel, "state_prefix": prefix, "seal_sha256": seal_sha}
    value = admission._actual_panel_evidence(
        storage, declared, tmp_path / f"verify-{name}"
    )
    return _write(tmp_path / f"{name}-evidence.json", value)


def _equivalence(runtime: dict, upstream_commit: str) -> dict:
    schema, status, changes, claims = admission._equivalence_contract(upstream_commit)
    value = {
        "schema": schema,
        "status": status,
        "legacy_policy": admission._LEGACY_POLICY,
        "legacy_authorization": admission._LEGACY_AUTHORIZATION,
        "runtime_identity": runtime,
        "control_plane_changes": list(changes),
        "claims": claims,
    }
    return _self_digest(value, "receipt_sha256")


def _selection(tmp_path: Path, panel: dict, evidence_sha: str) -> dict:
    value = {
        "schema": "npa.behavior.specialist-candidate-selection.v1",
        "status": "development_only_candidate_selected",
        "policy_identity_sha256": admission._LEGACY_POLICY["identity_sha256"],
        "development_panel_id": panel["panel_id"],
        "development_evidence_sha256": evidence_sha,
        "selection_inputs": ["development"],
    }
    return _write(tmp_path / "candidate.json", _self_digest(value, "selection_sha256"))


def _runtime_args(tmp_path, monkeypatch):
    archive = tmp_path / "specialist.zip"
    archive.write_bytes(b"specialist checkpoint fixture")
    original_digest, original_stat = admission.file_digest, Path.stat

    def digest(path):
        if Path(path) == archive:
            return admission._LEGACY_POLICY["artifacts"]["checkpoint"]["sha256"]
        return original_digest(path)

    def stat(path, *args, **kwargs):
        if path == archive:
            return SimpleNamespace(st_size=5_350_459_172)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(admission, "file_digest", digest)
    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(rlc_policy, "_verify_checkout", lambda *_: None)
    monkeypatch.setattr(rlc_policy, "_task_checkpoint", lambda *_: (1, "checkpoint_2"))
    return SimpleNamespace(
        policy_kind="rlc-specialist",
        policy_execution_variant="native",
        policy_archive=archive,
        policy_root=tmp_path / "rlc",
        upstream_root=tmp_path / "upstream",
        host="127.0.0.1",
        port=8000,
    )


def _historical_runtime(args) -> dict:
    from npa.workflows.behavior_challenge import rlc_specialist

    revision = UPSTREAM_COMMITS["3.9.2"]
    payload = {
        "schema": "npa.behavior.rlc-specialist-runtime.v2",
        "kind": "rlc-specialist",
        "task": {"name": "picking_up_trash", "id": 1},
        "runtime_files": admission._LEGACY_RUNTIME_FILES,
        "checkpoint": admission._LEGACY_POLICY["artifacts"]["checkpoint"],
        "source_commits": admission._source_commits(rlc_policy, rlc_specialist),
        "contracts": admission._runtime_contracts(rlc_specialist, revision),
    }
    return admission._identity(payload)


def _baseline_receipts(tmp_path, storage, upstream_commit=None):
    seals, evidence, panels = {}, {}, {}
    for arm, score in (
        ("native", 0.6),
        ("comet12", 0.4),
        ("comet50", 0.2),
    ):
        panels[arm] = _panel(
            admission._BASELINE_POLICIES[arm], "report", upstream_commit
        )
        seals[arm] = _seal(tmp_path, arm, panels[arm])
        prefix = f"s3://test/{arm}"
        _populate(storage, panels[arm], prefix, tmp_path / arm, score)
        evidence[arm] = _evidence(
            tmp_path, storage, arm, panels[arm], prefix, seals[arm]["sha256"]
        )
    return seals, evidence, panels


def _selection_receipts(tmp_path, candidate, seals, evidence, panels):
    unseal_value = {
        "schema": "npa.behavior.baseline-results-unseal.v1",
        "status": "candidate_frozen_then_three_baselines_verified_then_unsealed",
        "candidate_selection_sha256": candidate["sha256"],
        "baseline_seal_sha256": {arm: ref["sha256"] for arm, ref in seals.items()},
        "baseline_evidence_sha256": {
            arm: ref["sha256"] for arm, ref in evidence.items()
        },
    }
    unseal = _write(
        tmp_path / "unseal.json", _self_digest(unseal_value, "unseal_sha256")
    )
    metrics = {
        arm: admission._baseline_metrics(json.loads(Path(ref["path"]).read_text()))
        for arm, ref in evidence.items()
    }
    bstar_value = {
        "schema": "npa.behavior.reporting-bstar-selection.v1",
        "status": "recomputed_unique_released_baseline_selected",
        "unseal_sha256": unseal["sha256"],
        "criteria": [
            "maximum_mean_q",
            "maximum_full_successes",
            "minimum_protocol_failures",
        ],
        "metrics": metrics,
        "selected_arm": "native",
        "selected_policy_identity_sha256": panels["native"]["policy"][
            "identity_sha256"
        ],
    }
    bstar = _write(tmp_path / "bstar.json", _self_digest(bstar_value, "bstar_sha256"))
    return unseal, bstar


def _report_admission(equivalence, dev, candidate, baselines, report):
    seals, evidence, unseal, bstar = baselines
    value = {
        "schema": "npa.behavior.rlc-specialist-report-admission.v1",
        "status": "complete_verified_local_report_authorized",
        "legacy_policy": admission._LEGACY_POLICY,
        "equivalence_receipt_sha256": equivalence["sha256"],
        "development_evidence": dev,
        "candidate_selection": candidate,
        "baseline_seals": seals,
        "baseline_evidence": evidence,
        "unseal": unseal,
        "bstar": bstar,
        "report_panel": report,
        "scope": "local_task1_report_first_rollout_only",
        "official_24gb_qualified": False,
    }
    return _self_digest(value, "admission_sha256")


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    args = _runtime_args(tmp_path, monkeypatch)
    legacy = UPSTREAM_COMMITS["3.9.2"]
    development = _panel(admission._LEGACY_POLICY, "development", legacy)
    equivalence = _write(
        tmp_path / "equivalence.json",
        _equivalence(
            _historical_runtime(args),
            development["upstream_commit"],
        ),
    )
    args.policy_specialist_equivalence_receipt = Path(equivalence["path"])
    args.policy_specialist_equivalence_sha256 = equivalence["sha256"]
    storage = MemoryStorage()
    report = _panel(admission._LEGACY_POLICY, "report", legacy)
    _populate(storage, development, "s3://test/dev", tmp_path / "dev", 0.4)
    dev = _evidence(
        tmp_path, storage, "development", development, "s3://test/dev", None
    )
    candidate = _selection(tmp_path, development, dev["sha256"])
    seals, evidence, panels = _baseline_receipts(tmp_path, storage, legacy)
    unseal, bstar = _selection_receipts(tmp_path, candidate, seals, evidence, panels)
    receipt = _report_admission(
        equivalence, dev, candidate, (seals, evidence, unseal, bstar), report
    )
    admission_ref = _write(tmp_path / "admission.json", receipt)
    args.policy_specialist_report_admission = Path(admission_ref["path"])
    args.policy_specialist_report_admission_sha256 = admission_ref["sha256"]
    return args, report, receipt, storage, tmp_path / "admission-workspace"


def _rewrite_admission(args, receipt):
    value = deepcopy(receipt)
    value["admission_sha256"] = canonical_digest(
        {key: item for key, item in value.items() if key != "admission_sha256"}
    )
    args.policy_specialist_report_admission_sha256 = _write(
        args.policy_specialist_report_admission, value
    )["sha256"]


def _verify_legacy_receipts(args, panel, storage, workspace):
    equivalence, equivalence_sha, report_admission, admission_sha = (
        admission._load_report_receipts(args)
    )
    admission._preflight_report_receipts(equivalence, equivalence_sha, report_admission)
    runtime = _historical_runtime(args)
    admission._verify_equivalence(equivalence, runtime, panel["upstream_commit"])
    admission._verify_admission(
        storage, workspace, report_admission, panel, equivalence_sha
    )
    admission._record_verified_report_scope(
        args, runtime, panel["panel_id"], equivalence_sha, admission_sha
    )
    return runtime


def test_legacy_receipt_validator_reconstructs_complete_original_evidence(admitted):
    args, panel, _, storage, workspace = admitted
    runtime = _verify_legacy_receipts(args, panel, storage, workspace)
    assert runtime["identity_sha256"]
    assert args._specialist_report_receipts["admission"]


def test_current_equivalence_is_not_invented():
    with pytest.raises(ValueError, match="no qualification receipt"):
        admission._equivalence_contract(UPSTREAM_COMMITS["3.9.3"])


def test_legacy_equivalence_contract_remains_exactly_versioned():
    schema, status, changes, claims = admission._equivalence_contract(
        UPSTREAM_COMMITS["3.9.2"]
    )

    assert schema == "npa.behavior.rlc-specialist-equivalence.v1"
    assert status == "reviewed_legacy_v1_to_runtime_v2_equivalent"
    assert changes == admission._LEGACY_CONTROL_PLANE_CHANGES
    assert claims["runtime_files_byte_identical"] is True
    assert "singleton_wire_semantics_qualified" not in claims


def test_legacy_qualification_cannot_authorize_the_current_runtime():
    with pytest.raises(ValueError, match="report is held"):
        admission._hold_specialist_report(UPSTREAM_COMMITS["3.9.2"])


def test_legacy_runtime_identity_rejects_current_local_files(tmp_path, monkeypatch):
    args = _runtime_args(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="runtime bytes differ"):
        admission.specialist_runtime_identity(args, UPSTREAM_COMMITS["3.9.2"])


@pytest.mark.parametrize("version", ["3.9.2", "3.9.3"])
def test_report_holds_before_receipts_or_storage(tmp_path, version):
    args = SimpleNamespace(policy_kind="rlc-specialist")
    panel = _panel(
        admission._LEGACY_POLICY,
        "report",
        UPSTREAM_COMMITS[version],
    )

    with pytest.raises(ValueError, match="report is held"):
        admission.verify_specialist_report_admission(
            args, panel, None, tmp_path / "must-not-exist"
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_specialist_policy_scope_rejects_a_different_evaluator_revision(admitted):
    args, panel, _, storage, workspace = admitted
    _verify_legacy_receipts(args, panel, storage, workspace)
    plan = campaign_runner._managed_plan(panel, panel["cases"][0])
    changed = UPSTREAM_COMMITS["3.9.2"]
    plan["upstream_commit"] = changed
    plan["recipe"]["upstream_commit"] = changed

    with pytest.raises(ValueError, match="not admitted"):
        admission.verify_specialist_policy_scope(args, plan)


def test_development_needs_no_report_receipts(tmp_path):
    args = SimpleNamespace(policy_kind="rlc-specialist")
    panel = _panel(admission._LEGACY_POLICY, "development")
    assert (
        admission.verify_specialist_report_admission(
            args, panel, MemoryStorage(), tmp_path
        )
        is None
    )


def test_downloaded_video_tamper_rejected(admitted):
    args, panel, receipt, storage, workspace = admitted
    evidence = json.loads(Path(receipt["development_evidence"]["path"]).read_text())
    video = next(
        row["uri"] for row in evidence["inventory"] if row["uri"].endswith(".mp4")
    )
    storage.objects[video] = (b"corrupt video", "changed")
    with pytest.raises(ValueError, match="SHA-256 verification"):
        _verify_legacy_receipts(args, panel, storage, workspace)


def test_bstar_is_recomputed_from_original_aggregates(admitted):
    args, panel, receipt, storage, workspace = admitted
    bstar_path = Path(receipt["bstar"]["path"])
    bstar = json.loads(bstar_path.read_text())
    bstar["selected_arm"] = "comet12"
    bstar["bstar_sha256"] = canonical_digest(
        {key: value for key, value in bstar.items() if key != "bstar_sha256"}
    )
    receipt["bstar"] = _write(bstar_path, bstar)
    _rewrite_admission(args, receipt)
    with pytest.raises(ValueError, match="recomputed selection"):
        _verify_legacy_receipts(args, panel, storage, workspace)


def test_candidate_selection_must_bind_exact_development_evidence(admitted):
    args, panel, receipt, storage, workspace = admitted
    candidate_path = Path(receipt["candidate_selection"]["path"])
    candidate = json.loads(candidate_path.read_text())
    candidate["development_evidence_sha256"] = "0" * 64
    candidate["selection_sha256"] = canonical_digest(
        {key: value for key, value in candidate.items() if key != "selection_sha256"}
    )
    receipt["candidate_selection"] = _write(candidate_path, candidate)
    _rewrite_admission(args, receipt)
    with pytest.raises(ValueError, match="development-only evidence"):
        _verify_legacy_receipts(args, panel, storage, workspace)


def test_unseal_must_bind_all_frozen_evidence_files(admitted):
    args, panel, receipt, storage, workspace = admitted
    unseal_path = Path(receipt["unseal"]["path"])
    unseal = json.loads(unseal_path.read_text())
    unseal["baseline_evidence_sha256"]["native"] = "0" * 64
    unseal["unseal_sha256"] = canonical_digest(
        {key: value for key, value in unseal.items() if key != "unseal_sha256"}
    )
    receipt["unseal"] = _write(unseal_path, unseal)
    _rewrite_admission(args, receipt)
    with pytest.raises(ValueError, match="unseal ordering differs"):
        _verify_legacy_receipts(args, panel, storage, workspace)


def test_receipt_reference_rejects_symlinks(admitted, tmp_path):
    args, panel, receipt, storage, workspace = admitted
    target = Path(receipt["candidate_selection"]["path"])
    link = tmp_path / "candidate-link.json"
    link.symlink_to(target)
    receipt["candidate_selection"] = {
        "path": str(link),
        "sha256": admission.file_digest(target),
    }
    _rewrite_admission(args, receipt)
    with pytest.raises(ValueError, match="path or SHA-256 is invalid"):
        _verify_legacy_receipts(args, panel, storage, workspace)


def test_report_panel_requires_exact_specialist_identity(admitted):
    args, _, receipt, storage, workspace = admitted
    changed = _panel(
        _policy("other-specialist", "e"),
        "report",
        UPSTREAM_COMMITS["3.9.2"],
    )
    receipt["report_panel"] = changed
    _rewrite_admission(args, receipt)
    with pytest.raises(ValueError, match="frozen candidate"):
        _verify_legacy_receipts(args, changed, storage, workspace)


def test_swapped_baseline_arms_reject_policy_identity(admitted):
    args, panel, receipt, storage, workspace = admitted
    receipt["baseline_evidence"]["comet12"], receipt["baseline_evidence"]["comet50"] = (
        receipt["baseline_evidence"]["comet50"],
        receipt["baseline_evidence"]["comet12"],
    )
    _rewrite_admission(args, receipt)
    with pytest.raises(ValueError, match="comet12 baseline policy identity differs"):
        _verify_legacy_receipts(args, panel, storage, workspace)


def test_started_panel_is_rejected_before_original_recovery(tmp_path):
    storage = MemoryStorage()
    panel = _panel(admission._LEGACY_POLICY, "development")
    store = CaseStore(storage, "s3://test/started", panel["panel_id"])
    version = store.start(store.claim(panel["cases"][0], "worker-0"))
    prefix = store.artifact_prefix(version)
    storage.objects[f"{prefix}/validation.json"] = (b"{}", "original")
    storage.objects[f"{prefix}/provenance.json"] = (b"{}", "original")
    declared = {
        "panel": panel,
        "state_prefix": "s3://test/started",
        "seal_sha256": None,
    }
    with pytest.raises(ValueError, match="Complete panel required"):
        admission._actual_panel_evidence(storage, declared, tmp_path / "verify")
    assert store.read(panel["cases"][0]).record["state"] == "started"
    assert not (tmp_path / "verify").exists()


def test_report_validation_precedes_case_store_creation(monkeypatch, tmp_path):
    panel = _panel(admission._LEGACY_POLICY, "report")
    events = []
    args = SimpleNamespace(
        policy_kind="rlc-specialist",
        upstream_root=tmp_path,
        workspace=tmp_path / "workspace",
        output_path="s3://example/state",
        host="127.0.0.1",
        port=8000,
    )
    monkeypatch.setattr(campaign_runner, "verify_upstream", lambda *_: None)
    monkeypatch.setattr(
        campaign_runner.StorageClient, "from_environment", MemoryStorage
    )
    monkeypatch.setattr(campaign_runner, "_worker_declarations", lambda *_: (panel, {}))
    monkeypatch.setattr(
        admission,
        "verify_specialist_report_admission",
        lambda *_: (_ for _ in ()).throw(ValueError("not admitted")),
    )
    monkeypatch.setattr(campaign_runner, "CaseStore", lambda *_: events.append("store"))
    with pytest.raises(ValueError, match="not admitted"):
        campaign_runner.evaluate_partition(args)
    assert events == []


def test_occupied_endpoint_rejected_before_case_store(monkeypatch, tmp_path):
    panel = _panel(admission._LEGACY_POLICY, "report")
    events = []
    args = SimpleNamespace(
        policy_kind="rlc-specialist",
        upstream_root=tmp_path,
        workspace=tmp_path / "workspace",
        output_path="s3://example/state",
        host="localhost",
        port=8000,
    )
    monkeypatch.setattr(campaign_runner, "verify_upstream", lambda *_: None)
    monkeypatch.setattr(
        campaign_runner.StorageClient, "from_environment", MemoryStorage
    )
    monkeypatch.setattr(campaign_runner, "_worker_declarations", lambda *_: (panel, {}))
    monkeypatch.setattr(admission, "verify_specialist_report_admission", lambda *_: {})
    monkeypatch.setattr(
        "npa.workflows.behavior_challenge.policy._healthy", lambda *_: True
    )
    monkeypatch.setattr(campaign_runner, "CaseStore", lambda *_: events.append("store"))
    with pytest.raises(ValueError, match="already occupied"):
        campaign_runner.evaluate_partition(args)
    assert events == []


def test_non_policy_listener_is_rejected_before_case_store(monkeypatch, tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        monkeypatch.setattr(
            "npa.workflows.behavior_challenge.policy._healthy", lambda *_: False
        )
        args = SimpleNamespace(host="127.0.0.1", port=port)
        with pytest.raises(ValueError, match="another service"):
            admission.verify_specialist_preclaim_endpoint(
                args, _panel(admission._LEGACY_POLICY, "report")
            )


def test_equivalence_rejects_runtime_identity_change(admitted):
    args, panel, _, storage, workspace = admitted
    receipt = json.loads(args.policy_specialist_equivalence_receipt.read_text())
    receipt["runtime_identity"]["runtime_files"]["rlc_server.py"]["sha256"] = "0" * 64
    receipt["receipt_sha256"] = canonical_digest(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    args.policy_specialist_equivalence_sha256 = _write(
        args.policy_specialist_equivalence_receipt, receipt
    )["sha256"]
    report_admission = json.loads(args.policy_specialist_report_admission.read_text())
    report_admission["equivalence_receipt_sha256"] = (
        args.policy_specialist_equivalence_sha256
    )
    _rewrite_admission(args, report_admission)
    with pytest.raises(ValueError, match="equivalence differs"):
        _verify_legacy_receipts(args, panel, storage, workspace)


def test_bad_receipt_schema_rejects_before_runtime_identity(admitted, monkeypatch):
    args, panel, _, storage, workspace = admitted
    args.policy_specialist_equivalence_sha256 = _write(
        args.policy_specialist_equivalence_receipt,
        {"schema": "invalid-negative-receipt"},
    )["sha256"]
    monkeypatch.setattr(
        admission,
        "specialist_runtime_identity",
        lambda *_: (_ for _ in ()).throw(AssertionError("runtime identity called")),
    )
    with pytest.raises(ValueError, match="equivalence receipt fields differ"):
        _verify_legacy_receipts(args, panel, storage, workspace)


def test_bad_receipt_bytes_reject_before_runtime_identity(admitted, monkeypatch):
    args, panel, _, storage, workspace = admitted
    args.policy_specialist_equivalence_receipt.write_text("{}\n")
    monkeypatch.setattr(
        admission,
        "specialist_runtime_identity",
        lambda *_: (_ for _ in ()).throw(AssertionError("runtime identity called")),
    )
    with pytest.raises(ValueError, match="bytes differ from the frozen receipt"):
        _verify_legacy_receipts(args, panel, storage, workspace)


def test_report_receipts_never_enter_native_policy_command(admitted, tmp_path):
    args, _, _, _, _ = admitted
    args.policy_python = Path("/runtime/python")
    args.policy_root = Path("/source")
    args.policy_checkpoint = Path("/checkpoint")
    command = rlc_policy._command(
        args, 1, tmp_path, upstream_commit=UPSTREAM_COMMITS["3.9.3"]
    )
    assert command[-2:] == ["--execution-variant", "native"]
    assert not any("admission" in item or "equivalence" in item for item in command)
    adapters = rlc_policy._adapter_files(tmp_path, selected=False, specialist=True)
    admission.verify_staged_specialist_runtime(
        args, tmp_path, adapters, command, UPSTREAM_COMMITS["3.9.3"]
    )
    with pytest.raises(ValueError, match="Staged specialist runtime"):
        admission.verify_staged_specialist_runtime(
            args, tmp_path, adapters, command, UPSTREAM_COMMITS["3.9.2"]
        )


def test_report_policy_scope_requires_prior_full_panel_admission(monkeypatch):
    monkeypatch.setattr(rlc_policy, "_verify_checkout", lambda *_: None)
    monkeypatch.setattr(rlc_policy, "_task_checkpoint", lambda *_: (1, "checkpoint_2"))
    args = SimpleNamespace(
        policy_kind="rlc-specialist",
        policy_root=Path("/source"),
        upstream_root=Path("/upstream"),
    )
    plan = {
        "recipe": {"split": "report", "tasks": ["picking_up_trash"]},
        "cases": [{"task": "picking_up_trash", "instance_id": 301, "rollout_id": 0}],
    }
    with pytest.raises(ValueError, match="not admitted before case claim"):
        rlc_policy._verify_task(args, plan)
