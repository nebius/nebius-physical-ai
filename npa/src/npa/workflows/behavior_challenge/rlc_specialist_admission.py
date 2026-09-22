"""Admit the frozen released RLC specialist to one local reporting panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import socket

from .campaign import (
    canonical_digest,
    validate_panel,
)
from .protocol import UPSTREAM_COMMIT, WRAPPER, file_digest

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TASK = "picking_up_trash"
_LEGACY_POLICY = {
    "schema": "npa.behavior.policy-identity.v1",
    "policy_id": "rlc-specialist-task1-release5f035",
    "artifacts": {
        "checkpoint": {
            "sha256": "a64eba16c1609c3db4b47bc1b09b9e64e3edcf34dc4fa0bdc539043622942005",
            "bytes": 5_350_459_172,
        },
        "serving": {
            "sha256": "8bc05146ae245691f678a40961bc00269296c626774a89b065bd092fd0229a97",
            "bytes": 1_805,
        },
    },
    "identity_sha256": "43c0510ea49c34b90b3969abcca36aa20e4fb85a324e8696e69f70871c71dff6",
}
_LEGACY_AUTHORIZATION = {
    "source_commit": "2f45ffda78fbaf3ed899dd88440d72d018f36f5f",
    "rlc_policy_sha256": "c5734b8024ddfdd50b81fd7f2f941bad979838a417e6ac19bb3455ec2896dd21",
    "reporting_authorized": False,
}
_RUNTIME_FILES = (
    "rlc_server.py",
    "rlc_observations.py",
    "rlc_execution.py",
    "rlc_correlation.py",
    "rlc_specialist.py",
    "rlc-specialist-checkpoint.json",
)
_LEGACY_RUNTIME_FILES = {
    "rlc_server.py": {
        "sha256": "78caf7eadccbf3803832f58a7451d644d5d938fc2efce59e89c453af2521c2ee",
        "bytes": 7_790,
    },
    "rlc_observations.py": {
        "sha256": "770c1e3ef814b5baa587b4f685112f34230f3277a4679386897f8adcce76437b",
        "bytes": 1_768,
    },
    "rlc_execution.py": {
        "sha256": "6b56676a73a68b5c4d4d2b31170531b9021756df4acec98fb2b269438eecc3fb",
        "bytes": 21_421,
    },
    "rlc_correlation.py": {
        "sha256": "c8e2b9367b29e504be9813b8282365b05c0ffb9cd0c79f13e92fce42eacbcf9c",
        "bytes": 8_938,
    },
    "rlc_specialist.py": {
        "sha256": "9b741fe93dd2a4d8acdbbb37d41cb36698615f02fa29ce92f66bc4c98ab74345",
        "bytes": 6_068,
    },
    "rlc-specialist-checkpoint.json": {
        "sha256": "f561adc5dafb69446863920831ed430b64495892b4f572d03d8c8832b37ef75a",
        "bytes": 3_520,
    },
}
_CONTROL_PLANE_CHANGES = (
    "__main__.py",
    "campaign_runner.py",
    "campaign_workflow.py",
    "policy.py",
    "rlc_policy.py",
    "rlc_specialist_admission.py",
)


def _json_identity(path: Path, expected: str, label: str) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file() or _SHA256.fullmatch(expected) is None:
        raise ValueError(f"{label} path or SHA-256 is invalid")
    digest = file_digest(path)
    if digest != expected:
        raise ValueError(f"{label} bytes differ from the frozen receipt")
    try:
        value = json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, digest


def _identity(payload: dict) -> dict:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "identity_sha256": hashlib.sha256(encoded).hexdigest()}


def specialist_runtime_identity(args) -> dict:
    """Fingerprint only inputs that can change specialist policy behavior.

    Args:
        args: Specialist paths and native execution settings.
    Returns:
        Canonical runtime identity independent of campaign authorization code.
    Raises:
        ValueError: The specialist task or execution setting is unsupported.
        OSError: A runtime or checkpoint input cannot be read.
    """
    from . import rlc_policy, rlc_specialist

    if (
        args.policy_kind != "rlc-specialist"
        or args.policy_execution_variant != "native"
    ):
        raise ValueError("Specialist runtime identity requires native execution")
    _verify_source_checkouts(args, rlc_policy)
    archive = Path(args.policy_archive)
    checkpoint = {"sha256": file_digest(archive), "bytes": archive.stat().st_size}
    if checkpoint != _LEGACY_POLICY["artifacts"]["checkpoint"]:
        raise ValueError(
            "Specialist checkpoint differs from the frozen development policy"
        )
    payload = {
        "schema": "npa.behavior.rlc-specialist-runtime.v2",
        "kind": "rlc-specialist",
        "task": {"name": _TASK, "id": 1},
        "runtime_files": _verified_runtime_files(),
        "checkpoint": checkpoint,
        "source_commits": _source_commits(rlc_policy, rlc_specialist),
        "contracts": _runtime_contracts(rlc_specialist),
    }
    return _identity(payload)


def _verify_source_checkouts(args, rlc_policy) -> None:
    for relative, revision in (
        (".", rlc_policy.SOURCE_COMMIT),
        ("openpi", rlc_policy.OPENPI_COMMIT),
        ("BEHAVIOR-1K", rlc_policy.BEHAVIOR_COMMIT),
    ):
        rlc_policy._verify_checkout(args.policy_root / relative, revision)
    task_id, _ = rlc_policy._task_checkpoint(
        args.policy_root, args.upstream_root, _TASK
    )
    if task_id != 1:
        raise ValueError("Specialist task mapping differs from frozen task1")


def _verified_runtime_files() -> dict[str, dict]:
    root = Path(__file__).parent
    actual = {
        name: {
            "sha256": file_digest(root / name),
            "bytes": (root / name).stat().st_size,
        }
        for name in _RUNTIME_FILES
    }
    if actual != _LEGACY_RUNTIME_FILES:
        raise ValueError("Specialist staged runtime bytes differ from development")
    return actual


def verify_staged_specialist_runtime(
    args, output: Path, adapters: dict[str, str], command: list[str]
) -> None:
    """Verify the files and argv that the specialist process will consume.

    Args:
        args: Verified managed-policy paths and loopback port.
        output: Per-case policy directory containing staged source files.
        adapters: SHA-256 map returned by the staging helper.
        command: Native specialist server command.
    Returns:
        None.
    Raises:
        ValueError: Staged bytes, file set, or command semantics differ.
    """
    expected_hashes = {
        name: identity["sha256"] for name, identity in _LEGACY_RUNTIME_FILES.items()
    }
    actual = {
        name: {
            "sha256": file_digest(output / name),
            "bytes": (output / name).stat().st_size,
        }
        for name in adapters
    }
    if adapters != expected_hashes or actual != _LEGACY_RUNTIME_FILES:
        raise ValueError("Staged specialist runtime differs from development")
    _verify_native_command(args, output, command)


def _verify_native_command(args, output: Path, command: list[str]) -> None:
    fixed = {
        0: str(args.policy_python),
        1: str(output / "rlc_server.py"),
        2: "--source-root",
        3: str(args.policy_root),
        4: "--checkpoint",
        5: str(args.policy_checkpoint),
        6: "--task-id",
        7: "1",
        8: "--port",
        9: str(args.port),
        10: "--specialist-state-contract",
        11: "--execution-variant",
        12: "native",
    }
    if len(command) != 13 or any(
        command[index] != value for index, value in fixed.items()
    ):
        raise ValueError("Specialist native command differs from development")


def _source_commits(rlc_policy, specialist) -> dict[str, str]:
    return {
        "rlc": rlc_policy.SOURCE_COMMIT,
        "openpi": rlc_policy.OPENPI_COMMIT,
        "behavior": rlc_policy.BEHAVIOR_COMMIT,
        "specialist": specialist.SOURCE_COMMIT,
        "model_revision": specialist.MODEL_REVISION,
    }


def _runtime_contracts(specialist) -> dict:
    return {
        "entrypoint": "rlc_server.py",
        "command_argv": [
            "{policy_python}",
            "{output}/rlc_server.py",
            "--source-root",
            "{policy_root}",
            "--checkpoint",
            "{policy_checkpoint}",
            "--task-id",
            "1",
            "--port",
            "{port}",
            "--specialist-state-contract",
            "--execution-variant",
            "native",
        ],
        "specialist_state_contract": True,
        "execution_variant": "native",
        "episode_lifecycle": "fresh-managed-process-per-case-v1",
        "normalization": specialist.NORMALIZATION,
        "correlation_sha256": specialist.CORRELATION_SHA256,
        "topology_sha256": specialist.TOPOLOGY_SHA256,
        "allowed_observations": "three_rgb_plus_61_proprio",
        "action_shape": 23,
        "upstream_commit": UPSTREAM_COMMIT,
        "wrapper": WRAPPER,
    }


def _receipt_digest(receipt: dict, field: str, label: str) -> None:
    digest = receipt.get(field)
    payload = {key: value for key, value in receipt.items() if key != field}
    if _SHA256.fullmatch(digest or "") is None or digest != canonical_digest(payload):
        raise ValueError(f"{label} digest differs")


def _verify_equivalence(receipt: dict, runtime: dict) -> None:
    expected_keys = {
        "schema",
        "status",
        "legacy_policy",
        "legacy_authorization",
        "runtime_identity",
        "control_plane_changes",
        "claims",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError("Specialist equivalence receipt fields differ")
    claims = {
        "runtime_files_byte_identical": True,
        "checkpoint_bytes_identical": True,
        "native_command_identical": True,
        "observation_and_action_contract_identical": True,
        "fresh_process_lifecycle_identical": True,
        "official_24gb_qualified": False,
    }
    if (
        receipt["schema"] != "npa.behavior.rlc-specialist-equivalence.v1"
        or receipt["status"] != "reviewed_legacy_v1_to_runtime_v2_equivalent"
        or receipt["legacy_policy"] != _LEGACY_POLICY
        or receipt["legacy_authorization"] != _LEGACY_AUTHORIZATION
        or receipt["runtime_identity"] != runtime
        or receipt["control_plane_changes"] != list(_CONTROL_PLANE_CHANGES)
        or receipt["claims"] != claims
    ):
        raise ValueError("Specialist legacy-to-runtime equivalence differs")
    _receipt_digest(receipt, "receipt_sha256", "Specialist equivalence receipt")


def _task_panel(value: object, split: str, label: str) -> dict:
    panel = validate_panel(value)
    expected_instances = range(311, 321) if split == "development" else range(301, 311)
    instances = [case["instance_id"] for case in panel["cases"]]
    if (
        panel["split"] != split
        or panel["selected_tasks"] != [_TASK]
        or panel["case_count"] != 10
        or instances != list(expected_instances)
        or any(case["rollout_id"] != 0 for case in panel["cases"])
    ):
        raise ValueError(f"{label} is not the exact task1 first-rollout panel")
    return panel


def _reference(value: object, label: str) -> tuple[dict, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference fields differ")
    path = Path(value["path"])
    if not path.is_absolute():
        raise ValueError(f"{label} reference path must be absolute")
    return _json_identity(path, value["sha256"], label)


def _bytes_identity(payload: bytes, uri: str) -> dict:
    return {
        "uri": uri,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _remote_identity(storage, uri: str) -> tuple[dict, bytes]:
    saved = storage.read_bytes_with_etag(uri)
    if saved is None:
        raise ValueError(f"Original evidence is missing at {uri}")
    return _bytes_identity(saved[0], uri), saved[0]


def _local_identity(path: Path, uri: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Verified original artifact must be a regular file")
    return {"uri": uri, "sha256": file_digest(path), "bytes": path.stat().st_size}


def _case_inventory(store, version, output: Path) -> list[dict]:
    prefix = store.artifact_prefix(version)
    validation, validation_bytes = _remote_identity(
        store.storage, f"{prefix}/validation.json"
    )
    provenance, provenance_bytes = _remote_identity(
        store.storage, f"{prefix}/provenance.json"
    )
    validation_record = json.loads(validation_bytes)
    provenance_record = json.loads(provenance_bytes)
    if not isinstance(validation_record.get("files"), dict):
        raise ValueError("Validation original file map is invalid")
    if not isinstance(provenance_record, dict):
        raise ValueError("Provenance original file map is invalid")
    rows = [validation, provenance]
    rows.extend(
        _local_identity(output / relative, f"{prefix}/{relative}")
        for relative in sorted(validation_record["files"])
    )
    rows.extend(
        _local_identity(output / name, f"{prefix}/provenance/{name}")
        for name in sorted(provenance_record)
    )
    return rows


def _actual_panel_evidence(storage, declared: dict, workspace: Path) -> dict:
    from .campaign_runner import aggregate_stored_panel, _case_directory
    from .case_store import CaseStore

    panel = validate_panel(declared["panel"])
    store = CaseStore(storage, declared["state_prefix"], panel["panel_id"])
    verified = aggregate_stored_panel(panel, store, workspace)
    inventory = []
    for case in panel["cases"]:
        version = store.read(case)
        if version is None:
            raise ValueError("Complete panel required before evidence verification")
        inventory.extend(
            _case_inventory(store, version, _case_directory(workspace, version))
        )
    uris = [row["uri"] for row in inventory]
    if len(uris) != len(set(uris)):
        raise ValueError("Verified panel inventory contains duplicate artifact URIs")
    payload = {
        "schema": "npa.behavior.verified-panel-evidence.v1",
        "panel": panel,
        "state_prefix": declared["state_prefix"],
        "seal_sha256": declared["seal_sha256"],
        "aggregate": verified["aggregate"],
        "inventory": sorted(inventory, key=lambda row: row["uri"]),
    }
    return {**payload, "receipt_sha256": canonical_digest(payload)}


def _verify_panel_evidence(
    storage, reference: object, workspace: Path, label: str
) -> tuple[dict, str]:
    declared, digest = _reference(reference, label)
    keys = {
        "schema",
        "panel",
        "state_prefix",
        "seal_sha256",
        "aggregate",
        "inventory",
        "receipt_sha256",
    }
    if not isinstance(declared, dict) or set(declared) != keys:
        raise ValueError(f"{label} fields differ")
    actual = _actual_panel_evidence(storage, declared, workspace)
    if declared != actual:
        raise ValueError(f"{label} differs from downloaded original evidence")
    return actual, digest


def _verify_candidate_selection(value: object, panel: dict, evidence_sha: str) -> str:
    keys = {
        "schema",
        "status",
        "policy_identity_sha256",
        "development_panel_id",
        "development_evidence_sha256",
        "selection_inputs",
        "selection_sha256",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Specialist candidate selection fields differ")
    if (
        value["schema"] != "npa.behavior.specialist-candidate-selection.v1"
        or value["status"] != "development_only_candidate_selected"
        or value["policy_identity_sha256"] != _LEGACY_POLICY["identity_sha256"]
        or value["development_panel_id"] != panel["panel_id"]
        or value["development_evidence_sha256"] != evidence_sha
        or value["selection_inputs"] != ["development"]
    ):
        raise ValueError(
            "Specialist candidate selection is not immutable development-only evidence"
        )
    _receipt_digest(value, "selection_sha256", "Specialist candidate selection")
    return value["selection_sha256"]


def _verify_seal(reference: object, arm: str, panel: dict) -> str:
    value, digest = _reference(reference, f"{arm} baseline seal")
    expected = {
        "schema",
        "status",
        "arm",
        "panel_id",
        "policy_identity_sha256",
        "seal_sha256",
    }
    if (
        set(value) != expected
        or value["schema"] != "npa.behavior.baseline-results-seal.v1"
        or value["status"] != "frozen_before_results_unseal"
        or value["arm"] != arm
        or value["panel_id"] != panel["panel_id"]
        or value["policy_identity_sha256"] != panel["policy"]["identity_sha256"]
    ):
        raise ValueError(f"{arm} baseline seal differs")
    _receipt_digest(value, "seal_sha256", f"{arm} baseline seal")
    return digest


def _same_cohort(panels: dict[str, dict], report: dict) -> None:
    fields = (
        "upstream_commit",
        "wrapper",
        "registry_sha256",
        "split",
        "selected_tasks",
        "cases",
    )
    expected = {field: report[field] for field in fields}
    if any(
        {field: panel[field] for field in fields} != expected
        for panel in panels.values()
    ):
        raise ValueError(
            "Reporting baselines and specialist must use one frozen cohort"
        )


def _verify_unseal(
    value: object, candidate_sha: str, seals: dict, evidence: dict
) -> None:
    keys = {
        "schema",
        "status",
        "candidate_selection_sha256",
        "baseline_seal_sha256",
        "baseline_evidence_sha256",
        "unseal_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value["schema"] != "npa.behavior.baseline-results-unseal.v1"
        or value["status"]
        != "candidate_frozen_then_three_baselines_verified_then_unsealed"
        or value["candidate_selection_sha256"] != candidate_sha
        or value["baseline_seal_sha256"] != seals
        or value["baseline_evidence_sha256"] != evidence
    ):
        raise ValueError("Baseline unseal ordering differs")
    _receipt_digest(value, "unseal_sha256", "Baseline unseal")


def _baseline_metrics(evidence: dict) -> dict:
    aggregate = evidence["aggregate"]
    if type(aggregate["success_count"]) is not int:
        raise ValueError("B* requires full-success counts for every baseline")
    return {
        "mean_q": aggregate["mean_q"],
        "full_successes": aggregate["success_count"],
        "protocol_failures": 0,
    }


def _select_bstar(metrics: dict[str, dict]) -> str:
    ranks = {
        arm: (row["mean_q"], row["full_successes"], -row["protocol_failures"])
        for arm, row in metrics.items()
    }
    best = max(ranks.values())
    winners = [arm for arm, rank in ranks.items() if rank == best]
    if len(winners) != 1:
        raise ValueError("Frozen B* criteria do not identify a unique baseline")
    return winners[0]


def _verify_bstar(value: object, unseal_file_sha: str, evidence: dict) -> None:
    metrics = {arm: _baseline_metrics(receipt) for arm, receipt in evidence.items()}
    selected = _select_bstar(metrics)
    keys = {
        "schema",
        "status",
        "unseal_sha256",
        "criteria",
        "metrics",
        "selected_arm",
        "selected_policy_identity_sha256",
        "bstar_sha256",
    }
    criteria = ["maximum_mean_q", "maximum_full_successes", "minimum_protocol_failures"]
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value["schema"] != "npa.behavior.reporting-bstar-selection.v1"
        or value["status"] != "recomputed_unique_released_baseline_selected"
        or value["unseal_sha256"] != unseal_file_sha
        or value["criteria"] != criteria
        or value["metrics"] != metrics
        or value["selected_arm"] != selected
        or value["selected_policy_identity_sha256"]
        != evidence[selected]["panel"]["policy"]["identity_sha256"]
    ):
        raise ValueError("Frozen B* receipt differs from recomputed selection")
    _receipt_digest(value, "bstar_sha256", "Frozen B* selection")


def _baseline_inputs(
    storage, receipt: dict, workspace: Path
) -> tuple[dict, dict, dict, dict]:
    arms = {"native", "comet12", "comet50"}
    if set(receipt["baseline_seals"]) != arms:
        raise ValueError("Admission requires all three baseline seals")
    if set(receipt["baseline_evidence"]) != arms:
        raise ValueError("Admission requires all three baseline evidence receipts")
    evidence, evidence_shas, seal_shas, panels = {}, {}, {}, {}
    for arm in sorted(arms):
        evidence[arm], evidence_shas[arm] = _verify_panel_evidence(
            storage,
            receipt["baseline_evidence"][arm],
            workspace / arm,
            f"{arm} evidence",
        )
        panels[arm] = _task_panel(evidence[arm]["panel"], "report", f"{arm} panel")
        seal_shas[arm] = _verify_seal(receipt["baseline_seals"][arm], arm, panels[arm])
        if evidence[arm]["seal_sha256"] != seal_shas[arm]:
            raise ValueError(f"{arm} evidence is not bound to its frozen seal")
    return evidence, evidence_shas, seal_shas, panels


def _verify_admission_chain(
    storage, workspace: Path, receipt: dict, report: dict
) -> None:
    development, development_sha = _verify_panel_evidence(
        storage,
        receipt["development_evidence"],
        workspace / "development",
        "Development evidence",
    )
    development_panel = _task_panel(
        development["panel"], "development", "Development panel"
    )
    if development["seal_sha256"] is not None:
        raise ValueError("Development evidence must precede baseline sealing")
    selection, selection_sha = _reference(
        receipt["candidate_selection"], "Candidate selection"
    )
    _verify_candidate_selection(selection, development_panel, development_sha)
    evidence, evidence_shas, seal_shas, panels = _baseline_inputs(
        storage, receipt, workspace
    )
    _same_cohort(panels, report)
    identities = {value["policy"]["identity_sha256"] for value in panels.values()}
    if len(identities) != 3 or _LEGACY_POLICY["identity_sha256"] in identities:
        raise ValueError("Reporting baseline policy identities are not distinct")
    if development_panel["registry_sha256"] != report["registry_sha256"]:
        raise ValueError("Development and reporting registries differ")
    development_ids = {case["case_id"] for case in development_panel["cases"]}
    if development_ids & {case["case_id"] for case in report["cases"]}:
        raise ValueError("Development and reporting cases overlap")
    unseal, unseal_file_sha = _reference(receipt["unseal"], "Baseline unseal")
    _verify_unseal(unseal, selection_sha, seal_shas, evidence_shas)
    bstar, _ = _reference(receipt["bstar"], "Frozen B* selection")
    _verify_bstar(bstar, unseal_file_sha, evidence)
    if development_panel["policy"] != _LEGACY_POLICY:
        raise ValueError("Development panel uses another specialist policy")


def _verify_admission(
    storage,
    workspace: Path,
    receipt: dict,
    panel: dict,
    equivalence_sha256: str,
) -> None:
    keys = {
        "schema",
        "status",
        "legacy_policy",
        "equivalence_receipt_sha256",
        "development_evidence",
        "candidate_selection",
        "baseline_seals",
        "baseline_evidence",
        "unseal",
        "bstar",
        "report_panel",
        "scope",
        "official_24gb_qualified",
        "admission_sha256",
    }
    if set(receipt) != keys:
        raise ValueError("Specialist report admission fields differ")
    report = _task_panel(receipt["report_panel"], "report", "Specialist report panel")
    if report != panel:
        raise ValueError("Specialist report panel differs from frozen candidate")
    _verify_admission_chain(storage, workspace, receipt, report)
    if not _admission_header_matches(receipt, equivalence_sha256):
        raise ValueError("Specialist report authorization scope differs")
    _receipt_digest(receipt, "admission_sha256", "Specialist report admission")


def _admission_header_matches(receipt: dict, equivalence_sha256: str) -> bool:
    return (
        receipt["schema"] == "npa.behavior.rlc-specialist-report-admission.v1"
        and receipt["status"] == "complete_verified_local_report_authorized"
        and receipt["legacy_policy"] == _LEGACY_POLICY
        and receipt["equivalence_receipt_sha256"] == equivalence_sha256
        and receipt["scope"] == "local_task1_report_first_rollout_only"
        and receipt["official_24gb_qualified"] is False
    )


def _report_paths(args) -> tuple[Path, str, Path, str]:
    fields = (
        "policy_specialist_equivalence_receipt",
        "policy_specialist_equivalence_sha256",
        "policy_specialist_report_admission",
        "policy_specialist_report_admission_sha256",
    )
    values = tuple(getattr(args, field, None) for field in fields)
    if not all(values):
        raise ValueError(
            "Specialist report requires frozen equivalence and admission receipts"
        )
    return values


def _load_report_receipts(args, runtime: dict) -> tuple[dict, str, str]:
    equivalent_path, equivalent_sha, admission_path, admission_sha = _report_paths(args)
    equivalence, equivalence_file_sha = _json_identity(
        equivalent_path, equivalent_sha, "Specialist equivalence receipt"
    )
    _verify_equivalence(equivalence, runtime)
    report_admission, _ = _json_identity(
        admission_path, admission_sha, "Specialist report admission"
    )
    return report_admission, equivalence_file_sha, admission_sha


def verify_specialist_report_admission(
    args, panel: dict, storage, workspace: Path
) -> dict | None:
    """Validate report authorization before policy startup or any case claim.

    Args:
        args: Campaign worker arguments containing pinned receipt paths and hashes.
        panel: Actual immutable panel loaded by the worker.
        storage: Storage client used to retrieve original campaign evidence.
        workspace: Empty local directory for evidence reconstruction.
    Returns:
        Validated runtime identity for report, or ``None`` for development.
    Raises:
        ValueError: Report evidence, identity, scope, or receipt bytes differ.
        OSError: Runtime inputs cannot be read.
    """
    valid_panel = validate_panel(panel)
    if getattr(args, "policy_kind", None) != "rlc-specialist":
        return None
    if valid_panel["split"] == "development":
        if any(_report_paths_present(args)):
            raise ValueError("Specialist development does not accept report receipts")
        return None
    runtime = specialist_runtime_identity(args)
    report_admission, equivalence_sha, admission_sha = _load_report_receipts(
        args, runtime
    )
    _verify_admission(
        storage, workspace, report_admission, valid_panel, equivalence_sha
    )
    args._specialist_report_runtime_identity = runtime
    args._specialist_report_panel_id = valid_panel["panel_id"]
    args._specialist_report_receipts = {
        "equivalence": equivalence_sha,
        "admission": admission_sha,
    }
    return runtime


def verify_specialist_preclaim_endpoint(args, panel: dict) -> None:
    """Reject external or occupied specialist endpoints before case-store creation.

    Args:
        args: Campaign worker arguments containing the loopback host and port.
        panel: Actual immutable panel loaded by the worker.
    Returns:
        None.
    Raises:
        ValueError: Host, case ports, or existing endpoint state is unsafe.
    """
    from .policy import _healthy

    valid = validate_panel(panel)
    ports = {case.get("policy_port") for case in valid["cases"]}
    if getattr(args, "host", None) not in {"127.0.0.1", "localhost"}:
        raise ValueError("Specialist report policy host must be loopback")
    if ports - {None, args.port}:
        raise ValueError("Specialist report panel contains another policy port")
    if _healthy(args.port):
        raise ValueError("Specialist report policy endpoint is already occupied")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", args.port))
    except OSError as exc:
        raise ValueError(
            "Specialist report policy port is occupied by another service"
        ) from exc


def verify_specialist_report_token(args, panel: dict) -> None:
    """Recheck the in-process token created by full pre-claim admission.

    Args:
        args: Campaign worker arguments admitted earlier in this process.
        panel: Actual panel about to enter managed policy execution.
    Returns:
        None.
    Raises:
        ValueError: Full admission did not occur for this exact panel.
    """
    valid = validate_panel(panel)
    if (
        getattr(args, "_specialist_report_panel_id", None) != valid["panel_id"]
        or not isinstance(
            getattr(args, "_specialist_report_runtime_identity", None), dict
        )
        or not isinstance(getattr(args, "_specialist_report_receipts", None), dict)
    ):
        raise ValueError("Specialist report lacks its pre-claim admission token")


def _report_paths_present(args) -> tuple[object, ...]:
    return tuple(
        getattr(args, field, None)
        for field in (
            "policy_specialist_equivalence_receipt",
            "policy_specialist_equivalence_sha256",
            "policy_specialist_report_admission",
            "policy_specialist_report_admission_sha256",
        )
    )


def verify_specialist_policy_scope(args, plan: dict) -> dict:
    """Require a prior full campaign admission for one specialist report case.

    Args:
        args: Worker arguments previously admitted against its full panel.
        plan: Single-case managed-policy plan.
    Returns:
        Validated runtime identity retained from full-panel admission.
    Raises:
        ValueError: The plan is not the admitted task1 reporting scope.
    """
    runtime = getattr(args, "_specialist_report_runtime_identity", None)
    receipts = getattr(args, "_specialist_report_receipts", None)
    recipe = plan.get("recipe", {})
    cases = plan.get("cases", [])
    if (
        not isinstance(runtime, dict)
        or not isinstance(receipts, dict)
        or recipe.get("split") != "report"
        or recipe.get("tasks") != [_TASK]
        or len(cases) != 1
        or cases[0].get("task") != _TASK
        or cases[0].get("rollout_id") != 0
        or cases[0].get("instance_id") not in range(301, 311)
    ):
        raise ValueError("Specialist report policy was not admitted before case claim")
    return runtime


def specialist_report_provenance(args, plan: dict) -> dict:
    """Return separate runtime-equivalence and report-authorization provenance.

    Args:
        args: Admitted specialist worker arguments.
        plan: Current managed-policy plan.
    Returns:
        Immutable legacy, runtime, and receipt identities for report evidence.
    Raises:
        ValueError: The current report plan was not admitted.
    """
    runtime = verify_specialist_policy_scope(args, plan)
    return {
        "legacy_policy": _LEGACY_POLICY,
        "runtime_identity": runtime,
        "equivalence_receipt_sha256": args._specialist_report_receipts["equivalence"],
        "report_admission_sha256": args._specialist_report_receipts["admission"],
        "official_24gb_qualified": False,
    }
