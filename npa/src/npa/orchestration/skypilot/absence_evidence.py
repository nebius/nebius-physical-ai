"""Validate original Sky submit evidence independently of the MK8s recovery contract."""

from __future__ import annotations

import json

from npa.cluster.absent_evidence import digest, pinned_bytes, require
from npa.orchestration.skypilot.absence_native import native_scope
from npa.orchestration.skypilot.absence_target import validate_target
from npa.orchestration.skypilot.absence_trace import validate_trace


def _json(manifest: dict, key: str) -> dict:
    value = json.loads(pinned_bytes(manifest[key]))
    require(isinstance(value, dict), "Original evidence must be an object")
    return value


def _ledger(manifest: dict, journal: dict, response: dict, trace: dict) -> dict:
    require(
        manifest["submission_ledger"]["path"] == trace["ledger_path"],
        "Wrong original ledger path",
    )
    require(
        manifest["original_kubeconfig"]["path"] == trace["kubeconfig_path"],
        "Wrong original target path",
    )
    ledger = _json(manifest, "submission_ledger")
    require(
        ledger.get("schema_version") == "npa.workflow.submission.v1"
        and ledger.get("project") == journal["project_alias"]
        and ledger.get("run_id") == journal["requested_name"],
        "Original submission ledger differs from operation",
    )
    # The rendered response and persisted ledger use different remedy prose.
    # Every structured identity/state field must still match exactly.
    retained = {
        key: value
        for key, value in ledger["launch"].items()
        if key != "operator_remedy"
    }
    returned = {
        key: value
        for key, value in response["launch_transaction"].items()
        if key != "operator_remedy"
    }
    require(
        retained == returned,
        "Original launch receipts disagree",
    )
    return ledger


def _naming_source(manifest: dict) -> None:
    from npa.orchestration.skypilot.absence_naming_contract import NAMING_SOURCE_SHA256

    require(manifest.get("sky_version") == "0.12.2", "Uncovered native Sky version")
    files = manifest["naming_source"]
    require(
        set(files) == set(NAMING_SOURCE_SHA256), "Incomplete naming source contract"
    )
    for name, expected in NAMING_SOURCE_SHA256.items():
        require(
            digest(pinned_bytes(files[name])) == expected,
            "Native naming source differs",
        )


def load_evidence(manifest: dict) -> dict:
    """Require original producing records and complete native naming coverage.

    Args:
        manifest: Private pinned original files and existing reader authority.
    Returns:
        Bound journal, original native scope, and registered target.
    Raises:
        ValueError: Evidence is missing, ambiguous, changed, or unsupported.
    """
    require(
        manifest.get("schema") in {"npa.sky.absence.v1", "npa.sky.zero-id-absence.v1"},
        "Unsupported Sky absence schema",
    )
    journal = _json(manifest, "original_journal")
    require(
        journal["operation_id"] == manifest["operation_id"]
        and journal["command"] == "npa workbench workflow submit"
        and journal["resource_type"] == "workflow-submit",
        "Wrong original producing operation",
    )
    response = _json(manifest, "producer_response")
    _naming_source(manifest)
    bound = _original_scope(manifest, journal, response)
    registration = validate_target(manifest, bound["scope"])
    require(
        registration["project_id"] == journal["project_id"],
        "Foreign registered project",
    )
    return {
        "journal": journal,
        **bound,
        "registration": registration,
        "historical_workload_outcome": "unknown",
    }


def _original_scope(manifest, journal, response) -> dict:
    if manifest["schema"] == "npa.sky.zero-id-absence.v1":
        from npa.orchestration.skypilot.absence_zero_evidence import zero_evidence

        return zero_evidence(manifest, journal, response, _ledger)
    trace = validate_trace(manifest, journal, response)
    ledger = _ledger(manifest, journal, response, trace)
    require(
        ledger["launch"]["state"] == "indeterminate",
        "Not an indeterminate legacy launch",
    )
    return {"scope": native_scope(manifest, journal, ledger, trace), "trace": trace}
