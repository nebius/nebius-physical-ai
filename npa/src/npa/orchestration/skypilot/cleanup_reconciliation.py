"""Bind read-only task metadata observations without authorizing cleanup or API stop."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from npa.literal_values import require_boolean, require_integer

_FORMAT = "npa.skypilot.task-local-reconciliation.v1"
_CONTRACT = "npa.skypilot.task-local-reconciliation-contract.v1"
_OBSERVATION = "npa.skypilot.task-local-observation.v1"
_QUIESCENCE = "npa.skypilot.same-run-client-quiescence.v1"
_ROLES = {"cancel", "degraded", "remote_absence", "api_identity", "controller_identity"}
_TARGET_FIELDS = {
    "run_id",
    "project_alias",
    "project_id",
    "cluster_id",
    "context",
    "controller_name",
    "controller_uid",
}
_API_FIELDS = {"root", "marker", "interpreter", "uid", "pid", "start_ticks"}
_CLIENT_FIELDS = _API_FIELDS | {"run_id"}
_OWNER_FIELDS = {
    "schema_version",
    "project_alias",
    "project_id",
    "cluster_id",
    "cluster_name",
    "context",
    "context_fingerprint",
    "mode",
    "namespace",
    "name",
    "operation_id",
}
_TERMINAL = {
    "SUCCEEDED",
    "CANCELLED",
    "FAILED",
    "FAILED_SETUP",
    "FAILED_PRECHECKS",
    "FAILED_NO_RESOURCE",
    "FAILED_CONTROLLER",
}
_FOREIGN_ERROR = (
    "controller cleanup succeeded but the exact local ownership record could not be "
    "cleared: Refusing to clear controller ownership for a different cluster id."
)
_CONTRACT_FIELDS = {
    "schema",
    "target",
    "api",
    "task_root",
    "state_db",
    "jobs",
    "clients",
    "original_sha256",
    "observation_sha256",
    "quiescence_sha256",
    "foreign_owner_sha256",
    "operation",
}
_OBSERVATION_FIELDS = {
    "schema",
    "started_at",
    "finished_at",
    "target",
    "operation",
    "task_db",
    "owner_before",
    "owner_after",
    "api_before",
    "api_after",
}
_DATABASE_FIELDS = {
    "observed_at",
    "path",
    "root",
    "uid",
    "access",
    "readable",
    "complete",
    "target_names",
    "rows",
    "errors",
    "path_identity_verified",
    "symlink_components",
}
_QUIESCENCE_FIELDS = {
    "schema",
    "observed_at",
    "target",
    "api",
    "scope",
    "complete",
    "unreadable_pids",
    "unknown_clients",
    "active_clients",
    "clients",
    "jobs",
    "errors",
}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _object(value: Any, field: str, keys: set[str] | None = None) -> dict:
    _require(type(value) is dict, f"{field}: object required")
    if keys is not None:
        _require(set(value) == keys, f"{field}: exact fields required")
    return value


def _text(value: Any, field: str) -> str:
    _require(
        type(value) is str and bool(value.strip()), f"{field}: nonempty text required"
    )
    _require(
        value == value.strip() and not any(ord(c) < 32 for c in value),
        f"{field}: canonical text required",
    )
    return value


def _flag(value: Any, expected: bool, field: str) -> None:
    require_boolean(value, field=field)
    _require(value is expected, f"{field}: must be {expected}")


def _hash(value: Any, field: str) -> str:
    _require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{field}: lowercase SHA256 required",
    )
    return value


def _member(value: Any, values: set[str], field: str) -> None:
    _require(type(value) is str and value in values, f"{field}: not terminal")


def _timestamp(value: Any, field: str) -> datetime:
    _text(value, field)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field}: ISO8601 timestamp required") from error
    _require(parsed.tzinfo is not None, f"{field}: timezone required")
    return parsed


def _path(value: Any, field: str) -> PurePosixPath:
    _text(value, field)
    path = PurePosixPath(value)
    _require(
        path.is_absolute()
        and not value.startswith("//")
        and str(path) == value
        and ".." not in path.parts,
        f"{field}: canonical absolute path required",
    )
    _require(len(path.parts) > 2, f"{field}: broad root forbidden")
    return path


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        _require(key not in result, f"JSON: duplicate key {key}")
        result[key] = value
    return result


def _decode(raw: bytes, field: str) -> dict:
    _require(type(raw) is bytes and bool(raw), f"{field}: original bytes required")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda token: _require(False, f"JSON: {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field}: unreadable JSON") from error
    return _object(value, field)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _api(value: Any, field: str) -> dict:
    record = _object(value, field, _API_FIELDS)
    for key in ("root", "interpreter"):
        _path(record[key], f"{field}.{key}")
    _text(record["marker"], f"{field}.marker")
    for key in ("uid", "pid", "start_ticks"):
        require_integer(
            record[key], field=f"{field}.{key}", minimum=1 if key != "uid" else 0
        )
    return record


def _client(value: Any, field: str) -> dict:
    record = _object(value, field, _CLIENT_FIELDS)
    _api({key: record[key] for key in _API_FIELDS}, field)
    _text(record["run_id"], f"{field}.run_id")
    return record


def _jobs(value: Any, field: str) -> list[dict]:
    _require(type(value) is list and bool(value), f"{field}: nonempty list required")
    for job in value:
        _object(job, field, {"job_id", "job_name"})
        for key in job:
            _text(job[key], f"{field}.{key}")
    for key in ("job_id", "job_name"):
        _require(
            len({job[key] for job in value}) == len(value), f"{field}: duplicate {key}"
        )
    return value


def _contract(raw: bytes) -> dict:
    contract = _decode(raw, "contract")
    _object(contract, "contract", _CONTRACT_FIELDS)
    _require(contract["schema"] == _CONTRACT, "contract.schema: unsupported")
    _require(
        contract["operation"] == "observe_only",
        "contract.operation: mutation forbidden",
    )
    target = _object(contract["target"], "target", _TARGET_FIELDS)
    for key, value in target.items():
        _text(value, f"target.{key}")
    api = _api(contract["api"], "api")
    root = _path(contract["task_root"], "task_root")
    for key, value in (("api.root", api["root"]), ("state_db", contract["state_db"])):
        path = _path(value, key)
        _require(root in path.parents, f"{key}: outside task_root")
    _jobs(contract["jobs"], "jobs")
    _contract_clients(contract)
    hashes = _object(contract["original_sha256"], "original_sha256", _ROLES)
    for key, value in hashes.items():
        _hash(value, f"original_sha256.{key}")
    for key in ("observation_sha256", "quiescence_sha256", "foreign_owner_sha256"):
        _hash(contract[key], key)
    return contract


def _contract_clients(contract: dict) -> None:
    clients = contract["clients"]
    _require(
        type(clients) is list and bool(clients), "clients: known submit client required"
    )
    for client in clients:
        _client(client, "clients")
        for key in ("root", "marker", "uid"):
            _require(
                client[key] == contract["api"][key], f"clients.{key}: API mismatch"
            )
        _require(
            client["run_id"] == contract["target"]["run_id"], "clients.run_id: mismatch"
        )
        _require(
            client["pid"] != contract["api"]["pid"], "clients.pid: API is not a client"
        )
    _require(
        len({client["pid"] for client in clients}) == len(clients),
        "clients: duplicate PID",
    )


def _originals(contract: dict, originals: dict[str, bytes]) -> tuple[dict, datetime]:
    _object(originals, "originals", _ROLES)
    decoded = {}
    for role, raw in originals.items():
        decoded[role] = _decode(raw, f"originals.{role}")
        _require(
            _digest(raw) == contract["original_sha256"][role],
            f"originals.{role}: hash mismatch",
        )
    _original_api(contract, decoded["api_identity"])
    _cancel(contract, decoded["cancel"])
    _degraded(contract, decoded["degraded"])
    controller = decoded["controller_identity"]
    for field, target_field in (
        ("cloud_name", "controller_name"),
        ("controller_uid", "controller_uid"),
    ):
        _require(
            controller.get(field) == contract["target"][target_field],
            f"controller_identity.{field}: mismatch",
        )
    created = _timestamp(controller.get("at"), "controller_identity.at")
    observed = _remote_absence(contract, decoded["remote_absence"])
    _other_controllers(controller, decoded["remote_absence"], contract)
    _require(created <= observed, "remote_absence.at: before controller identity")
    return decoded, observed


def _other_controllers(controller: dict, remote: dict, contract: dict) -> None:
    _flag(
        controller.get("controller_absent_before_task"),
        True,
        "controller_identity.controller_absent_before_task",
    )
    others = controller.get("other_controller_uids")
    _require(
        type(others) is list, "controller_identity.other_controller_uids: list required"
    )
    for uid in others:
        _text(uid, "controller_identity.other_controller_uids")
    _require(
        len(set(others)) == len(others),
        "controller_identity.other_controller_uids: duplicates",
    )
    _require(
        contract["target"]["controller_uid"] not in others,
        "controller_identity.other_controller_uids: includes target",
    )
    _require(
        remote["other_controllers"] == len(others),
        "remote_absence.other_controllers: mismatch",
    )


def _original_api(contract: dict, record: dict) -> None:
    _object(record, "api_identity", _API_FIELDS - {"uid"})
    for key, value in record.items():
        _require(
            type(value) is type(contract["api"][key]) and value == contract["api"][key],
            f"api_identity.{key}: mismatch",
        )


def _cancel(contract: dict, record: dict) -> None:
    _require(
        record.get("run_id") == contract["target"]["run_id"], "cancel.run_id: mismatch"
    )
    _require(record.get("outcome") == "terminal", "cancel.outcome: not terminal")
    _member(record.get("status"), _TERMINAL, "cancel.status")
    _require(record.get("errors", []) == [], "cancel.errors: unresolved")
    for key in ("durable_absence_conflict_errors", "durable_absence_conflict_job_ids"):
        _require(record.get(key) == [], f"cancel.{key}: unresolved")
    expected = contract["jobs"]
    _require(
        record.get("sky_job_ids") == [job["job_id"] for job in expected],
        "cancel.sky_job_ids: mismatch",
    )
    jobs = record.get("jobs")
    _require(
        type(jobs) is list and len(jobs) == len(expected), "cancel.jobs: incomplete"
    )
    for actual, identity in zip(jobs, expected):
        _object(actual, "cancel.jobs")
        for key, value in identity.items():
            _require(actual.get(key) == value, f"cancel.jobs.{key}: mismatch")
        _member(
            actual.get("live_outcome"),
            {"durable_terminal", "cancelled", "terminal"},
            "cancel.jobs.live_outcome",
        )
        _member(actual.get("live_status"), _TERMINAL, "cancel.jobs.live_status")
        _require(actual.get("live_error") == "", "cancel.jobs.live_error: unresolved")
        states = actual.get("persisted_states")
        _require(
            type(states) is list
            and bool(states)
            and all(type(s) is str and s in _TERMINAL for s in states),
            "cancel.jobs.persisted_states: not terminal",
        )


def _degraded(contract: dict, record: dict) -> None:
    for key in ("project_alias", "project_id", "cluster_id", "context"):
        _require(
            record.get(key) == contract["target"][key], f"degraded.{key}: mismatch"
        )
    _require(
        record.get("outcome") == "degraded_local_metadata", "degraded.outcome: mismatch"
    )
    for key, expected in (
        ("remote_absence_verified", True),
        ("local_metadata_cleared", False),
        ("overall_verified", False),
        ("verified", False),
    ):
        _flag(record.get(key), expected, f"degraded.{key}")
    _require(
        record.get("errors") == [_FOREIGN_ERROR],
        "degraded.errors: not exact foreign-owner refusal",
    )


def _remote_absence(contract: dict, record: dict) -> datetime:
    for key, expected in (
        ("remote_absence_verified", True),
        ("own_uid_absent", True),
        ("other_controller_uids_preserved", True),
        ("local_metadata_cleared", False),
        ("overall_verified", False),
    ):
        _flag(record.get(key), expected, f"remote_absence.{key}")
    require_integer(
        record.get("other_controllers"),
        field="remote_absence.other_controllers",
        minimum=0,
    )
    _require(
        record.get("controller_outcome") == "degraded_local_metadata",
        "remote_absence.controller_outcome: mismatch",
    )
    _remote_commands(contract, record)
    return _timestamp(record.get("at"), "remote_absence.at")


def _remote_commands(contract: dict, record: dict) -> None:
    commands = _object(
        record.get("commands"), "remote_absence.commands", {"cancel", "controller"}
    )
    for name, role, outcome in (
        ("cancel", "cancel", "terminal"),
        ("controller", "degraded", "degraded_local_metadata"),
    ):
        command = _object(
            commands[name],
            f"remote_absence.{name}",
            {"returncode", "outcome", "sha256"},
        )
        require_integer(
            command["returncode"],
            field=f"remote_absence.{name}.returncode",
            minimum=0,
            maximum=0,
        )
        _require(
            command["outcome"] == outcome, f"remote_absence.{name}.outcome: mismatch"
        )
        _require(
            command["sha256"] == contract["original_sha256"][role],
            f"remote_absence.{name}.sha256: mismatch",
        )


def _window(record: dict, lower: datetime, upper: datetime, field: str) -> datetime:
    observed = _timestamp(record.get("observed_at"), f"{field}.observed_at")
    _require(
        lower <= observed <= upper, f"{field}.observed_at: outside observation window"
    )
    return observed


def _observation(
    contract: dict, raw: bytes, original_time: datetime, assembled: datetime
) -> dict:
    record = _decode(raw, "observation")
    _require(
        _digest(raw) == contract["observation_sha256"], "observation: hash mismatch"
    )
    _object(record, "observation", _OBSERVATION_FIELDS)
    _require(record["schema"] == _OBSERVATION, "observation.schema: unsupported")
    _require(
        record["operation"] == "observe_only",
        "observation.operation: mutation forbidden",
    )
    _require(record["target"] == contract["target"], "observation.target: mismatch")
    start = _timestamp(record["started_at"], "observation.started_at")
    end = _timestamp(record["finished_at"], "observation.finished_at")
    _require(
        original_time <= start <= end <= assembled, "observation: invalid chronology"
    )
    _database(contract, record["task_db"], start, end)
    _owners(contract, record, start, end)
    _api_observations(contract, record, start, end)
    return record


def _api_observations(
    contract: dict, record: dict, start: datetime, end: datetime
) -> None:
    for field in ("api_before", "api_after"):
        observation = _object(
            record[field],
            field,
            {"observed_at", "identity", "state", "identity_verified"},
        )
        at = _window(observation, start, end, field)
        expected_time = start if field == "api_before" else end
        _require(
            at == expected_time, f"{field}.observed_at: must bracket all observations"
        )
        _api(observation["identity"], field)
        _require(
            observation["identity"] == contract["api"],
            f"{field}.identity: lifetime mismatch",
        )
        _require(
            observation["state"] == "ready", f"{field}.state: not original live API"
        )
        _flag(observation["identity_verified"], True, f"{field}.identity_verified")


def _database(contract: dict, value: Any, start: datetime, end: datetime) -> None:
    record = _object(value, "task_db", _DATABASE_FIELDS)
    _window(record, start, end, "task_db")
    _require(record["path"] == contract["state_db"], "task_db.path: mismatch")
    _require(record["root"] == contract["task_root"], "task_db.root: mismatch")
    uid = require_integer(record["uid"], field="task_db.uid", minimum=0)
    _require(uid == contract["api"]["uid"], "task_db.uid: mismatch")
    _require(record["access"] == "read_only", "task_db.access: mutation forbidden")
    for key in ("readable", "complete", "path_identity_verified"):
        _flag(record[key], True, f"task_db.{key}")
    _require(
        record["symlink_components"] == [], "task_db.symlink_components: ambiguous path"
    )
    _require(
        record["target_names"] == [contract["target"]["controller_name"]],
        "task_db.target_names: mismatch",
    )
    _require(record["errors"] == [], "task_db.errors: unresolved")
    rows = record["rows"]
    _require(type(rows) is list, "task_db.rows: list required")
    for row in rows:
        _object(row, "task_db.rows", {"name"})
        _text(row["name"], "task_db.rows.name")
    names = [row["name"] for row in rows]
    _require(len(names) == len(set(names)), "task_db.rows: ambiguous duplicates")
    _require(
        contract["target"]["controller_name"] not in names,
        "task_db.rows: target still present",
    )


def _owners(contract: dict, observation: dict, start: datetime, end: datetime) -> None:
    times = []
    for field in ("owner_before", "owner_after"):
        record = _object(
            observation[field],
            field,
            {"observed_at", "identity", "readable", "complete", "writes"},
        )
        times.append(_window(record, start, end, field))
        for key in ("readable", "complete"):
            _flag(record[key], True, f"{field}.{key}")
        _require(record["writes"] == [], f"{field}.writes: mutation forbidden")
        _owner_identity(contract, record["identity"], field)
    _require(times[0] <= times[1], "owner: reversed observations")
    database_time = _timestamp(
        observation["task_db"]["observed_at"], "task_db.observed_at"
    )
    _require(
        times[0] <= database_time <= times[1],
        "owner: must bracket task database observation",
    )
    _require(
        observation["owner_before"]["identity"]
        == observation["owner_after"]["identity"],
        "owner: identity changed",
    )
    _require(
        _digest(_canonical(observation["owner_before"]["identity"]))
        == contract["foreign_owner_sha256"],
        "owner: original foreign identity mismatch",
    )


def _owner_identity(contract: dict, value: Any, field: str) -> None:
    owner = _object(value, f"{field}.identity", _OWNER_FIELDS)
    for key, item in owner.items():
        if key == "operation_id" and item == "":
            continue
        _text(item, f"{field}.identity.{key}")
    _require(
        owner["schema_version"] == "npa.controller-owner.v1",
        f"{field}.schema_version: unsupported",
    )
    _require(
        owner["project_id"] == contract["target"]["project_id"],
        f"{field}.project_id: mismatch",
    )
    _require(
        owner["cluster_id"] != contract["target"]["cluster_id"],
        f"{field}.cluster_id: not foreign",
    )


def _quiescence(contract: dict, raw: bytes, observation: dict) -> dict:
    record = _decode(raw, "quiescence")
    _require(_digest(raw) == contract["quiescence_sha256"], "quiescence: hash mismatch")
    _object(record, "quiescence", _QUIESCENCE_FIELDS)
    _require(record["schema"] == _QUIESCENCE, "quiescence.schema: unsupported")
    _require(record["target"] == contract["target"], "quiescence.target: mismatch")
    _api(record["api"], "quiescence.api")
    _require(record["api"] == contract["api"], "quiescence.api: lifetime mismatch")
    start = _timestamp(
        observation["api_before"]["observed_at"], "api_before.observed_at"
    )
    end = _timestamp(observation["api_after"]["observed_at"], "api_after.observed_at")
    _window(record, start, end, "quiescence")
    _require(
        record["scope"] == "all_same_run_api_clients_and_jobs",
        "quiescence.scope: incomplete",
    )
    _flag(record["complete"], True, "quiescence.complete")
    for field in ("unreadable_pids", "unknown_clients", "active_clients", "errors"):
        _require(record[field] == [], f"quiescence.{field}: unresolved")
    _quiescent_clients(contract, record["clients"])
    _quiescent_jobs(contract, record["jobs"])
    return record


def _quiescent_clients(contract: dict, rows: Any) -> None:
    _require(
        type(rows) is list and len(rows) == len(contract["clients"]),
        "quiescence.clients: incomplete",
    )
    for row, expected in zip(rows, contract["clients"]):
        _object(
            row, "quiescence.clients", {"identity", "state", "exact_lifetime_absent"}
        )
        _client(row["identity"], "quiescence.clients.identity")
        _require(row["identity"] == expected, "quiescence.clients.identity: mismatch")
        _require(row["state"] == "exited", "quiescence.clients.state: not exited")
        _flag(
            row["exact_lifetime_absent"],
            True,
            "quiescence.clients.exact_lifetime_absent",
        )


def _quiescent_jobs(contract: dict, rows: Any) -> None:
    _require(
        type(rows) is list and len(rows) == len(contract["jobs"]),
        "quiescence.jobs: incomplete",
    )
    for row, expected in zip(rows, contract["jobs"]):
        _object(
            row,
            "quiescence.jobs",
            {"job_id", "job_name", "status", "exact_identity_verified"},
        )
        for key, value in expected.items():
            _require(row[key] == value, f"quiescence.jobs.{key}: mismatch")
        _require(
            type(row["status"]) is str and row["status"] in _TERMINAL,
            "quiescence.jobs.status: not terminal",
        )
        _flag(
            row["exact_identity_verified"],
            True,
            "quiescence.jobs.exact_identity_verified",
        )


def reconcile_task_metadata(
    contract: bytes,
    originals: dict[str, bytes],
    observation: bytes,
    quiescence: bytes,
    *,
    assembled_at: str,
) -> dict:
    """Join authenticated caller-supplied observations; perform no I/O or mutation.

    Args:
        contract: Trusted, externally authenticated v1 scope and exact input hashes.
        originals: Unmodified cancel, degraded, remote absence and identity bytes.
        observation: Read-only v1 task database, foreign owner and API observations.
        quiescence: Separate closed-world same-run client/job observation bytes.
        assembled_at: Actual timezone-qualified assembly timestamp, not observation time.
    Returns:
        A new observation-scoped receipt, never stop or qualification authority.
    Raises:
        ValueError: Missing, ambiguous, changed, unreadable or nonliteral evidence.
    """
    scope = _contract(contract)
    _, original_time = _originals(scope, originals)
    assembled = _timestamp(assembled_at, "assembled_at")
    observed = _observation(scope, observation, original_time, assembled)
    clients = _quiescence(scope, quiescence, observed)
    return _receipt(scope, contract, observed, clients, assembled_at)


def _receipt(
    scope: dict, contract: bytes, observed: dict, clients: dict, assembled_at: str
) -> dict:
    return {
        "format": _FORMAT,
        "assembled_at": assembled_at,
        "contract_sha256": _digest(contract),
        "target": scope["target"],
        "api": scope["api"],
        "original_sha256": scope["original_sha256"],
        "observation_sha256": scope["observation_sha256"],
        "quiescence_sha256": scope["quiescence_sha256"],
        "task_local_metadata_absent": True,
        "foreign_owner_preserved": True,
        "foreign_owner_sha256": _digest(
            _canonical(observed["owner_after"]["identity"])
        ),
        "observed_at": {
            "task_db": observed["task_db"]["observed_at"],
            "foreign_owner_before": observed["owner_before"]["observed_at"],
            "foreign_owner_after": observed["owner_after"]["observed_at"],
            "api_before": observed["api_before"]["observed_at"],
            "api_after": observed["api_after"]["observed_at"],
            "client_quiescence": clients["observed_at"],
        },
        "caller_quiescence_verified": True,
        "original_local_metadata_cleared": False,
        "original_overall_verified": False,
        "ownership_stores_written": False,
        "stop_authorized": False,
        "cleanup_v2_pass": False,
        "scope": "retained_observations_only_not_current_authority",
    }


def validate_reconciliation(
    receipt: bytes,
    contract: bytes,
    originals: dict[str, bytes],
    observation: bytes,
    quiescence: bytes,
) -> dict:
    """Recompute a receipt from its exact original and observation bytes.

    Args:
        receipt: Versioned reconciliation JSON bytes to verify.
        contract: Externally trusted scope and exact hashes, not receipt-selected policy.
        originals: Unmodified original receipts under their required roles.
        observation: Exact read-only observation bytes.
        quiescence: Exact separate caller quiescence bytes.
    Returns:
        The recomputed receipt, with no authority beyond its recorded observations.
    Raises:
        ValueError: Any evidence mismatch, extra authority field or invalid input.
    """
    supplied = _decode(receipt, "receipt")
    expected = reconcile_task_metadata(
        contract,
        originals,
        observation,
        quiescence,
        assembled_at=supplied.get("assembled_at"),
    )
    _require(
        _canonical(supplied) == _canonical(expected),
        "receipt: recomputed content mismatch",
    )
    return expected
