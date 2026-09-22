"""Require the original failed, issued, never-accepted Sky launch evidence."""

from npa.cluster.absent_evidence import require
from npa.orchestration.skypilot.absence_zero_native import zero_native_scope
from npa.orchestration.skypilot.absence_zero_trace import validate_zero_trace


def validate_zero_launch(journal: dict, response: dict, ledger: dict) -> None:
    """Reject accepted or uncertain launches before any fresh reads or mutation.

    Args:
        journal: Original failed operation generation.
        response: Original terminal CLI response.
        ledger: Persisted original launch with matching structured fields.
    Returns:
        None.
    Raises:
        ValueError: An identity, success, ambiguity, or contradictory receipt exists.
    """
    launch = ledger["launch"]
    require(
        response.get("status") == "failed" and journal["result"] == "failed",
        "Original failed operation required",
    )
    require(
        ledger.get("launch_state") == "planned"
        and journal.get("phase") == "recovery-required"
        and journal.get("resources") == [],
        "Original operation has an accepted or uncovered resource",
    )
    require(
        launch.get("schema_version") == "npa.skypilot.launch-transaction.v1"
        and launch.get("state") == "terminal_failure"
        and type(launch.get("launch_sequence")) is int
        and launch["launch_sequence"] == 1
        and launch.get("job_id") == ""
        and launch.get("existence") == "absent"
        and launch.get("recovery_decision") == "verified_absent_no_retry"
        and launch.get("reconciliation_error") == "",
        "Not a single issued, authoritatively absent launch",
    )
    _absence_decisions(launch)
    require(
        bool(response.get("error")) and response["error"] == journal["last_error"],
        "Original failure differs from journal",
    )


def _absence_decisions(launch: dict) -> None:
    absent = {
        "error": "",
        "job_id": "",
        "state": "absent",
        "status": "",
        "workload_evidence": "",
        "workload_observable": False,
    }
    require(
        launch.get("reconciliations") == [absent, absent],
        "Original absence decisions are incomplete or contradictory",
    )
    controller = launch["controller"]
    require(
        controller.get("state") == "absent"
        and controller.get("execution_probe")
        == {
            "error": "",
            "healthy": True,
            "outcome": "controller_absent",
            "pod_count": 0,
        },
        "Original controller absence was not independently probed",
    )


def zero_evidence(manifest: dict, journal: dict, response: dict, ledger_loader) -> dict:
    """Bind original failure while explicitly preserving historical read limits.

    Args:
        manifest: Pinned private original files.
        journal: Original failed operation generation.
        response: Original failed response.
        ledger_loader: Common original ledger consistency validator.
    Returns:
        Validated scope and trace, without an execution or success claim.
    Raises:
        ValueError: Any original source, identity or absence predicate fails.
    """
    trace = validate_zero_trace(manifest, journal, response)
    ledger = ledger_loader(manifest, journal, response, trace)
    validate_zero_launch(journal, response, ledger)
    return {
        "trace": trace,
        "scope": zero_native_scope(manifest, journal, ledger, trace),
        "historical_read_limits": {
            "initial_decision": "controller-derived absence, not a queue call",
            "second_decision": "original source-bound queue reconciliation",
            "historical_queue_stdout_retained": False,
            "authority": "Fresh provider and complete metadata reads prove current absence only",
        },
    }
