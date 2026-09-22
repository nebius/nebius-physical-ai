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


def _panel(policy: dict, split: str) -> dict:
    tasks = [f"task-{index}" for index in range(100)]
    tasks[1] = "picking_up_trash"
    return declare_panel(policy, tasks, ["picking_up_trash"], split)


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


def _equivalence(runtime: dict) -> dict:
    value = {
        "schema": "npa.behavior.rlc-specialist-equivalence.v1",
        "status": "reviewed_legacy_v1_to_runtime_v2_equivalent",
        "legacy_policy": admission._LEGACY_POLICY,
        "legacy_authorization": admission._LEGACY_AUTHORIZATION,
        "runtime_identity": runtime,
        "control_plane_changes": list(admission._CONTROL_PLANE_CHANGES),
        "claims": {
            "runtime_files_byte_identical": True,
            "checkpoint_bytes_identical": True,
            "native_command_identical": True,
            "observation_and_action_contract_identical": True,
            "fresh_process_lifecycle_identical": True,
            "official_24gb_qualified": False,
        },
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


def _baseline_receipts(tmp_path, storage):
    seals, evidence, panels = {}, {}, {}
    for arm, marker, score in (
        ("native", "b", 0.6),
        ("comet12", "c", 0.4),
        ("comet50", "d", 0.2),
    ):
        panels[arm] = _panel(_policy(arm, marker), "report")
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
    equivalence = _write(
        tmp_path / "equivalence.json",
        _equivalence(admission.specialist_runtime_identity(args)),
    )
    args.policy_specialist_equivalence_receipt = Path(equivalence["path"])
    args.policy_specialist_equivalence_sha256 = equivalence["sha256"]
    storage = MemoryStorage()
    development = _panel(admission._LEGACY_POLICY, "development")
    report = _panel(admission._LEGACY_POLICY, "report")
    _populate(storage, development, "s3://test/dev", tmp_path / "dev", 0.4)
    dev = _evidence(
        tmp_path, storage, "development", development, "s3://test/dev", None
    )
    candidate = _selection(tmp_path, development, dev["sha256"])
    seals, evidence, panels = _baseline_receipts(tmp_path, storage)
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


def test_complete_original_evidence_admits_exact_report_panel(admitted):
    args, panel, _, storage, workspace = admitted
    runtime = admission.verify_specialist_report_admission(
        args, panel, storage, workspace
    )
    assert runtime["identity_sha256"]
    assert args._specialist_report_receipts["admission"]


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
        admission.verify_specialist_report_admission(args, panel, storage, workspace)


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
        admission.verify_specialist_report_admission(args, panel, storage, workspace)


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
    with pytest.raises(ValueError, match="equivalence differs"):
        admission.verify_specialist_report_admission(args, panel, storage, workspace)


def test_report_receipts_never_enter_native_policy_command(admitted, tmp_path):
    args, _, _, _, _ = admitted
    args.policy_python = Path("/runtime/python")
    args.policy_root = Path("/source")
    args.policy_checkpoint = Path("/checkpoint")
    command = rlc_policy._command(args, 1, tmp_path)
    assert command[-2:] == ["--execution-variant", "native"]
    assert not any("admission" in item or "equivalence" in item for item in command)
    adapters = rlc_policy._adapter_files(tmp_path, selected=False, specialist=True)
    admission.verify_staged_specialist_runtime(args, tmp_path, adapters, command)


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
