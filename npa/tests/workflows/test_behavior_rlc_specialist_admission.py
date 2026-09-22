"""Verify specialist reporting requires immutable evidence before case claims."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workflows.behavior_challenge import campaign_runner
from npa.workflows.behavior_challenge.campaign import (
    aggregate_panel,
    bind_inspected_rollout,
    canonical_digest,
    declare_panel,
    freeze_policy_identity,
)
from npa.workflows.behavior_challenge import rlc_policy
from npa.workflows.behavior_challenge import rlc_specialist_admission as admission


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


def _verified(panel: dict, marker: str) -> dict:
    receipts = []
    for index, case in enumerate(panel["cases"]):
        stem = f"{case['task']}_{case['instance_id']}_0"
        receipts.append(
            bind_inspected_rollout(
                panel,
                {
                    **case,
                    "q_score": index / 10,
                    "video_frames": index + 1,
                    "success": False,
                    "steps": index + 1,
                    "files": {
                        f"json/{stem}.json": marker * 64,
                        f"videos/{stem}.mp4": marker * 64,
                    },
                },
            )
        )
    return {
        "schema": "npa.behavior.verified-panel.v1",
        "aggregate": aggregate_panel(panel, receipts),
        "verification": {
            "all_original_bytes_downloaded_and_hashed": True,
            "all_original_videos_fully_decoded": True,
            "case_count": 10,
        },
    }


def _candidate(panel: dict, verified: dict) -> dict:
    value = {
        "schema": "npa.behavior.specialist-candidate-selection.v1",
        "status": "development_only_candidate_selected",
        "policy_identity_sha256": admission._LEGACY_POLICY["identity_sha256"],
        "development_panel_id": panel["panel_id"],
        "development_aggregate_sha256": verified["aggregate"]["aggregate_sha256"],
        "selection_inputs": ["development"],
        "report_evidence_used": False,
        "baseline_evidence_used": False,
        "frozen_before_baseline_unseal": True,
        "immutable": True,
    }
    return {**value, "selection_sha256": canonical_digest(value)}


def _baseline(candidate_sha256: str) -> dict:
    arms = {}
    for name, marker in (("native", "b"), ("comet12", "c"), ("comet50", "d")):
        panel = _panel(_policy(name, marker), "report")
        arms[name] = {
            "policy_kind": name,
            "panel": panel,
            "verified_panel": _verified(panel, marker),
        }
    value = {
        "schema": "npa.behavior.reporting-baseline-selection.v1",
        "status": "complete_frozen_reporting_baseline_selected",
        "candidate_selection_sha256": candidate_sha256,
        "arms": arms,
        "selected_arm": "native",
        "selected_policy_identity_sha256": arms["native"]["panel"]["policy"][
            "identity_sha256"
        ],
    }
    return {**value, "selection_sha256": canonical_digest(value)}


def _write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return admission.file_digest(path)


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    archive = tmp_path / "specialist.zip"
    archive.write_bytes(b"specialist checkpoint fixture")
    original_digest = admission.file_digest

    def digest(path):
        if Path(path) == archive:
            return admission._LEGACY_POLICY["artifacts"]["checkpoint"]["sha256"]
        return original_digest(path)

    monkeypatch.setattr(admission, "file_digest", digest)
    monkeypatch.setattr(rlc_policy, "_verify_checkout", lambda *_: None)
    monkeypatch.setattr(rlc_policy, "_task_checkpoint", lambda *_: (1, "checkpoint_2"))
    monkeypatch.setattr(
        Path,
        "stat",
        lambda self, *args, **kwargs: (
            SimpleNamespace(st_size=5_350_459_172)
            if self == archive
            else _ORIGINAL_STAT(self, *args, **kwargs)
        ),
    )
    args = SimpleNamespace(
        policy_kind="rlc-specialist",
        policy_execution_variant="native",
        policy_archive=archive,
        policy_root=tmp_path / "rlc",
        upstream_root=tmp_path / "upstream",
    )
    runtime = admission.specialist_runtime_identity(args)
    equivalence = _equivalence(runtime)
    equivalence_path = tmp_path / "equivalence.json"
    args.policy_specialist_equivalence_receipt = equivalence_path
    args.policy_specialist_equivalence_sha256 = _write(equivalence_path, equivalence)
    development = _panel(admission._LEGACY_POLICY, "development")
    verified = _verified(development, "a")
    candidate = _candidate(development, verified)
    report = _panel(admission._LEGACY_POLICY, "report")
    value = {
        "schema": "npa.behavior.rlc-specialist-report-admission.v1",
        "status": "complete_verified_local_report_authorized",
        "legacy_policy": admission._LEGACY_POLICY,
        "equivalence_receipt_sha256": args.policy_specialist_equivalence_sha256,
        "development_panel": development,
        "development_verified_panel": verified,
        "candidate_selection": candidate,
        "baseline_selection": _baseline(candidate["selection_sha256"]),
        "report_panel": report,
        "scope": "local_task1_report_first_rollout_only",
        "official_24gb_qualified": False,
    }
    receipt = {**value, "admission_sha256": canonical_digest(value)}
    path = tmp_path / "admission.json"
    args.policy_specialist_report_admission = path
    args.policy_specialist_report_admission_sha256 = _write(path, receipt)
    return args, report, receipt


_ORIGINAL_STAT = Path.stat


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
    return {**value, "receipt_sha256": canonical_digest(value)}


def _rewrite_admission(args, receipt):
    value = deepcopy(receipt)
    value["admission_sha256"] = canonical_digest(
        {key: item for key, item in value.items() if key != "admission_sha256"}
    )
    args.policy_specialist_report_admission_sha256 = _write(
        args.policy_specialist_report_admission, value
    )


def test_complete_frozen_evidence_admits_exact_report_panel(admitted):
    args, panel, _ = admitted

    runtime = admission.verify_specialist_report_admission(args, panel)

    assert runtime["identity_sha256"]
    assert (
        runtime["contracts"]["episode_lifecycle"] == "fresh-managed-process-per-case-v1"
    )
    assert (
        args._specialist_report_receipts["admission"]
        == args.policy_specialist_report_admission_sha256
    )


def test_development_needs_no_report_receipts():
    args = SimpleNamespace(policy_kind="rlc-specialist")
    panel = _panel(admission._LEGACY_POLICY, "development")

    assert admission.verify_specialist_report_admission(args, panel) is None


def test_report_rejects_missing_receipt_paths():
    args = SimpleNamespace(
        policy_kind="rlc-specialist", policy_execution_variant="native"
    )
    panel = _panel(admission._LEGACY_POLICY, "report")

    with pytest.raises(ValueError, match="requires frozen equivalence and admission"):
        admission.verify_specialist_report_admission(args, panel)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda value: value["development_verified_panel"]["aggregate"][
                "case_receipts"
            ].pop(),
            "every prescribed case",
        ),
        (
            lambda value: value["candidate_selection"].update(
                report_evidence_used=True
            ),
            "development-only",
        ),
        (
            lambda value: value["baseline_selection"]["arms"].pop("comet50"),
            "fields differ",
        ),
        (lambda value: value.update(official_24gb_qualified=True), "scope differs"),
    ],
)
def test_report_rejects_incomplete_or_scope_changed_evidence(admitted, mutate, message):
    args, panel, receipt = admitted
    mutate(receipt)
    _rewrite_admission(args, receipt)

    with pytest.raises(ValueError, match=message):
        admission.verify_specialist_report_admission(args, panel)


def test_equivalence_rejects_any_runtime_identity_change(admitted):
    args, panel, _ = admitted
    receipt = json.loads(args.policy_specialist_equivalence_receipt.read_text())
    receipt["runtime_identity"]["runtime_files"]["rlc_server.py"]["sha256"] = "0" * 64
    receipt["receipt_sha256"] = canonical_digest(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    args.policy_specialist_equivalence_sha256 = _write(
        args.policy_specialist_equivalence_receipt, receipt
    )

    with pytest.raises(ValueError, match="equivalence differs"):
        admission.verify_specialist_report_admission(args, panel)


def test_runtime_identity_rejects_changed_staged_development_byte(
    admitted, monkeypatch
):
    args, _, _ = admitted
    expected = deepcopy(admission._LEGACY_RUNTIME_FILES)
    expected["rlc_server.py"]["sha256"] = "0" * 64
    monkeypatch.setattr(admission, "_LEGACY_RUNTIME_FILES", expected)

    with pytest.raises(ValueError, match="staged runtime bytes differ"):
        admission.specialist_runtime_identity(args)


def test_runtime_identity_rejects_changed_upstream_task_mapping(admitted, monkeypatch):
    args, _, _ = admitted
    monkeypatch.setattr(rlc_policy, "_task_checkpoint", lambda *_: (0, "checkpoint_2"))

    with pytest.raises(ValueError, match="task mapping differs"):
        admission.specialist_runtime_identity(args)


def test_report_receipts_never_enter_native_policy_command(admitted, tmp_path):
    args, _, _ = admitted
    args.policy_python = Path("/runtime/python")
    args.policy_root = Path("/source")
    args.policy_checkpoint = Path("/checkpoint")
    args.port = 8000

    command = rlc_policy._command(args, 1, tmp_path)

    assert command[-2:] == ["--execution-variant", "native"]
    assert "--specialist-state-contract" in command
    assert not any("admission" in item or "equivalence" in item for item in command)

    adapters = rlc_policy._adapter_files(tmp_path, selected=False, specialist=True)
    admission.verify_staged_specialist_runtime(args, tmp_path, adapters, command)
    changed = [*command[:-1], "adaptive-short-chunk"]
    with pytest.raises(ValueError, match="native command differs"):
        admission.verify_staged_specialist_runtime(args, tmp_path, adapters, changed)


def test_report_provenance_keeps_runtime_and_authorization_separate(admitted, tmp_path):
    args, panel, _ = admitted
    admission.verify_specialist_report_admission(args, panel)
    args.policy_python = Path("/runtime/python")
    args.policy_root = Path("/source")
    args.policy_checkpoint = Path("/checkpoint")
    args.port = 8000
    command = rlc_policy._command(args, 1, tmp_path)
    plan = {
        "recipe": {
            "split": "report",
            "tasks": ["picking_up_trash"],
            "policy_checkpoint_sha256": admission._LEGACY_POLICY["artifacts"][
                "checkpoint"
            ]["sha256"],
        },
        "cases": [panel["cases"][0]],
    }

    rlc_policy._record_specialist(tmp_path, command, {}, plan, args=args)

    record = json.loads((tmp_path / "policy-provenance.json").read_text())
    authorization = record["report_authorization"]
    assert record["status"] == "local_report_admitted_memory_unverified"
    assert authorization["legacy_policy"] == admission._LEGACY_POLICY
    assert authorization["runtime_identity"]["schema"].endswith("runtime.v2")
    assert authorization["official_24gb_qualified"] is False


def test_report_validation_precedes_case_store_creation(monkeypatch, tmp_path):
    panel = _panel(admission._LEGACY_POLICY, "report")
    partition = {"unused": True}
    events = []
    args = SimpleNamespace(
        policy_kind="rlc-specialist",
        upstream_root=tmp_path,
        workspace=tmp_path / "workspace",
        output_path="s3://example/state",
    )
    monkeypatch.setattr(campaign_runner, "verify_upstream", lambda *_: None)
    monkeypatch.setattr(
        campaign_runner.StorageClient, "from_environment", lambda: object()
    )
    monkeypatch.setattr(
        campaign_runner, "_worker_declarations", lambda *_: (panel, partition)
    )
    monkeypatch.setattr(
        admission,
        "verify_specialist_report_admission",
        lambda *_: (_ for _ in ()).throw(ValueError("not admitted")),
    )
    monkeypatch.setattr(campaign_runner, "CaseStore", lambda *_: events.append("store"))

    with pytest.raises(ValueError, match="not admitted"):
        campaign_runner.evaluate_partition(args)
    assert events == []


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
