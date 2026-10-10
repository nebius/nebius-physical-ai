"""Exercise task-local reconciliation using synthetic, offline ownership evidence."""

from copy import deepcopy
import hashlib
import json
import re

import pytest

from npa.orchestration.skypilot.cleanup_reconciliation import (
    reconcile_task_metadata,
    validate_reconciliation,
)

_START = "2026-01-01T10:00:00+00:00"
_MIDDLE = "2026-01-01T10:00:01+00:00"
_END = "2026-01-01T10:00:02+00:00"
_ASSEMBLED = "2026-01-01T10:00:03+00:00"
_ORIGINAL = "2026-01-01T09:00:00+00:00"
_FOREIGN_ERROR = (
    "controller cleanup succeeded but the exact local ownership record could not be "
    "cleared: Refusing to clear controller ownership for a different cluster id."
)


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value):
    return hashlib.sha256(_raw(value)).hexdigest()


def _identity():
    return {
        "run_id": "synthetic-run",
        "project_alias": "synthetic-project",
        "project_id": "synthetic-project-id",
        "cluster_id": "synthetic-owned-cluster",
        "context": "synthetic-owned-context",
        "controller_name": "synthetic-owned-controller",
        "controller_uid": "synthetic-controller-uid",
    }


def _api():
    return {
        "root": "/synthetic/task/local-api",
        "marker": "synthetic-marker",
        "interpreter": "/synthetic/venv/bin/python",
        "uid": 1000,
        "pid": 700,
        "start_ticks": 123456,
    }


def _owner():
    return {
        "schema_version": "npa.controller-owner.v1",
        "project_alias": "synthetic-project",
        "project_id": "synthetic-project-id",
        "cluster_id": "synthetic-foreign-cluster",
        "cluster_name": "synthetic-foreign",
        "context": "synthetic-foreign-context",
        "context_fingerprint": "synthetic-fingerprint",
        "mode": "kubernetes",
        "namespace": "synthetic-namespace",
        "name": "synthetic-foreign-controller",
        "operation_id": "",
    }


def _degraded(target):
    degraded = {
        key: target[key]
        for key in ("project_alias", "project_id", "cluster_id", "context")
    }
    degraded.update(
        outcome="degraded_local_metadata",
        remote_absence_verified=True,
        local_metadata_cleared=False,
        overall_verified=False,
        verified=False,
        errors=[_FOREIGN_ERROR],
    )
    return degraded


def _cancel(target, jobs):
    return {
        "run_id": target["run_id"],
        "outcome": "terminal",
        "status": "SUCCEEDED",
        "durable_absence_conflict_errors": [],
        "durable_absence_conflict_job_ids": [],
        "sky_job_ids": [job["job_id"] for job in jobs],
        "jobs": [
            dict(
                job,
                live_outcome="durable_terminal",
                live_status="SUCCEEDED",
                live_error="",
                persisted_states=["SUCCEEDED"],
            )
            for job in jobs
        ],
    }


def _remote(cancel, degraded):
    return {
        "at": _ORIGINAL,
        "remote_absence_verified": True,
        "own_uid_absent": True,
        "other_controller_uids_preserved": True,
        "other_controllers": 4,
        "local_metadata_cleared": False,
        "overall_verified": False,
        "controller_outcome": "degraded_local_metadata",
        "commands": {
            "cancel": {"returncode": 0, "outcome": "terminal", "sha256": _sha(cancel)},
            "controller": {
                "returncode": 0,
                "outcome": "degraded_local_metadata",
                "sha256": _sha(degraded),
            },
        },
    }


def _originals(target, api, jobs):
    cancel, degraded = _cancel(target, jobs), _degraded(target)
    return {
        "cancel": cancel,
        "degraded": degraded,
        "remote_absence": _remote(cancel, degraded),
        "api_identity": {key: value for key, value in api.items() if key != "uid"},
        "controller_identity": {
            "at": _ORIGINAL,
            "cloud_name": target["controller_name"],
            "controller_uid": target["controller_uid"],
            "controller_absent_before_task": True,
            "other_controller_uids": [
                "synthetic-foreign-a",
                "synthetic-foreign-b",
                "synthetic-foreign-c",
                "synthetic-foreign-d",
            ],
        },
    }


def _owner_observation():
    return {
        "observed_at": _MIDDLE,
        "identity": _owner(),
        "readable": True,
        "complete": True,
        "writes": [],
    }


def _observation(target, api):
    return {
        "schema": "npa.skypilot.task-local-observation.v1",
        "started_at": _START,
        "finished_at": _END,
        "target": deepcopy(target),
        "operation": "observe_only",
        "task_db": {
            "observed_at": _MIDDLE,
            "path": "/synthetic/task/sky-runtime/.sky/state.db",
            "root": "/synthetic/task",
            "uid": 1000,
            "access": "read_only",
            "readable": True,
            "complete": True,
            "target_names": [target["controller_name"]],
            "rows": [],
            "errors": [],
            "path_identity_verified": True,
            "symlink_components": [],
        },
        "owner_before": _owner_observation(),
        "owner_after": _owner_observation(),
        "api_before": {
            "observed_at": _START,
            "identity": deepcopy(api),
            "state": "ready",
            "identity_verified": True,
        },
        "api_after": {
            "observed_at": _END,
            "identity": deepcopy(api),
            "state": "ready",
            "identity_verified": True,
        },
    }


def _quiescence(target, api, clients, jobs):
    return {
        "schema": "npa.skypilot.same-run-client-quiescence.v1",
        "observed_at": _MIDDLE,
        "target": deepcopy(target),
        "api": deepcopy(api),
        "scope": "all_same_run_api_clients_and_jobs",
        "complete": True,
        "unreadable_pids": [],
        "unknown_clients": [],
        "active_clients": [],
        "errors": [],
        "clients": [
            {
                "identity": deepcopy(client),
                "state": "exited",
                "exact_lifetime_absent": True,
            }
            for client in clients
        ],
        "jobs": [
            dict(job, status="SUCCEEDED", exact_identity_verified=True) for job in jobs
        ],
    }


def _population():
    target, api = _identity(), _api()
    jobs = [
        {"job_id": "1", "job_name": "synthetic-reconstruct"},
        {"job_id": "2", "job_name": "synthetic-render"},
    ]
    clients = [
        dict(
            api,
            interpreter="/synthetic/client/bin/python",
            pid=701,
            start_ticks=123457,
            run_id=target["run_id"],
        )
    ]
    contract = {
        "schema": "npa.skypilot.task-local-reconciliation-contract.v1",
        "target": target,
        "api": api,
        "task_root": "/synthetic/task",
        "state_db": "/synthetic/task/sky-runtime/.sky/state.db",
        "jobs": jobs,
        "clients": clients,
        "original_sha256": {},
        "observation_sha256": "",
        "quiescence_sha256": "",
        "foreign_owner_sha256": _sha(_owner()),
        "operation": "observe_only",
    }
    return {
        "contract": contract,
        "originals": _originals(target, api, jobs),
        "observation": _observation(target, api),
        "quiescence": _quiescence(target, api, clients, jobs),
    }


def _inputs(population):
    contract = deepcopy(population["contract"])
    contract["original_sha256"] = {
        key: _sha(value) for key, value in population["originals"].items()
    }
    contract["observation_sha256"] = _sha(population["observation"])
    contract["quiescence_sha256"] = _sha(population["quiescence"])
    return {
        "contract": _raw(contract),
        "originals": {
            key: _raw(value) for key, value in population["originals"].items()
        },
        "observation": _raw(population["observation"]),
        "quiescence": _raw(population["quiescence"]),
    }


def _produce(population):
    return reconcile_task_metadata(**_inputs(population), assembled_at=_ASSEMBLED)


def _set(population, path, value):
    target = population
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _refusal(population, reason):
    with pytest.raises(ValueError) as error:
        _produce(population)
    assert str(error.value) == reason


def test_genuine_shape_join_preserves_failed_flags_and_confers_no_authority(
    monkeypatch,
):
    population = _population()
    original = deepcopy(population)
    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: pytest.fail("unexpected filesystem operation"),
    )
    monkeypatch.setattr(
        "os.kill", lambda *args, **kwargs: pytest.fail("unexpected signal")
    )
    monkeypatch.setattr(
        "subprocess.Popen", lambda *args, **kwargs: pytest.fail("unexpected process")
    )
    receipt = _produce(population)
    assert receipt["task_local_metadata_absent"] is True
    assert receipt["foreign_owner_preserved"] is True
    assert receipt["original_local_metadata_cleared"] is False
    assert receipt["original_overall_verified"] is False
    assert receipt["stop_authorized"] is False
    assert receipt["cleanup_v2_pass"] is False
    assert receipt["ownership_stores_written"] is False
    assert "orphan_count" not in receipt
    assert "local_runtime_disposition" not in receipt
    assert receipt["observed_at"]["task_db"] == _MIDDLE
    assert validate_reconciliation(_raw(receipt), **_inputs(population)) == receipt
    assert population == original


def test_foreign_database_rows_are_preserved_not_deleted():
    population = _population()
    population["observation"]["task_db"]["rows"] = [
        {"name": "synthetic-foreign-controller"}
    ]
    assert _produce(population)["task_local_metadata_absent"] is True
    assert population["observation"]["task_db"]["rows"] == [
        {"name": "synthetic-foreign-controller"}
    ]


_MUTATIONS = [
    (("contract", "operation"), "clear", "contract.operation: mutation forbidden"),
    (("contract", "schema"), "v2", "contract.schema: unsupported"),
    (("contract", "task_root"), "/", "task_root: broad root forbidden"),
    (("contract", "task_root"), "/synthetic", "task_root: broad root forbidden"),
    (
        ("contract", "task_root"),
        "/synthetic/../task",
        "task_root: canonical absolute path required",
    ),
    (
        ("contract", "task_root"),
        "//synthetic/task",
        "task_root: canonical absolute path required",
    ),
    (("contract", "state_db"), "/another/task/state.db", "state_db: outside task_root"),
    (("contract", "api", "root"), "/another/task/api", "api.root: outside task_root"),
    (("contract", "clients"), [], "clients: known submit client required"),
    (("contract", "jobs"), [], "jobs: nonempty list required"),
    (("contract", "jobs", 1, "job_id"), "1", "jobs: duplicate job_id"),
    (
        ("contract", "jobs", 1, "job_name"),
        "synthetic-reconstruct",
        "jobs: duplicate job_name",
    ),
    (("contract", "clients", 0, "pid"), 700, "clients.pid: API is not a client"),
    (("observation", "schema"), "v2", "observation.schema: unsupported"),
    (
        ("observation", "operation"),
        "rebind",
        "observation.operation: mutation forbidden",
    ),
    (
        ("observation", "task_db", "path"),
        "/another/task/state.db",
        "task_db.path: mismatch",
    ),
    (("observation", "task_db", "root"), "/another/task", "task_db.root: mismatch"),
    (("observation", "task_db", "uid"), 1001, "task_db.uid: mismatch"),
    (
        ("observation", "task_db", "access"),
        "read_write",
        "task_db.access: mutation forbidden",
    ),
    (("observation", "task_db", "target_names"), [], "task_db.target_names: mismatch"),
    (
        ("observation", "task_db", "rows"),
        [{"name": "synthetic-owned-controller"}],
        "task_db.rows: target still present",
    ),
    (
        ("observation", "task_db", "rows"),
        [{"name": "foreign"}, {"name": "foreign"}],
        "task_db.rows: ambiguous duplicates",
    ),
    (("observation", "task_db", "rows"), None, "task_db.rows: list required"),
    (
        ("observation", "task_db", "errors"),
        ["unreadable"],
        "task_db.errors: unresolved",
    ),
    (
        ("observation", "task_db", "symlink_components"),
        ["parent"],
        "task_db.symlink_components: ambiguous path",
    ),
    (
        ("observation", "owner_after", "identity", "cluster_id"),
        "another-foreign",
        "owner: identity changed",
    ),
    (
        ("observation", "owner_before", "identity", "cluster_id"),
        "synthetic-owned-cluster",
        "owner_before.cluster_id: not foreign",
    ),
    (
        ("observation", "owner_before", "identity", "project_id"),
        "other-project",
        "owner_before.project_id: mismatch",
    ),
    (
        ("observation", "owner_before", "identity", "schema_version"),
        "v2",
        "owner_before.schema_version: unsupported",
    ),
    (
        ("observation", "owner_after", "writes"),
        ["clear"],
        "owner_after.writes: mutation forbidden",
    ),
    (
        ("observation", "api_after", "state"),
        "stopped",
        "api_after.state: not original live API",
    ),
    (("quiescence", "schema"), "v2", "quiescence.schema: unsupported"),
    (
        ("quiescence", "scope"),
        "known_command_patterns_only",
        "quiescence.scope: incomplete",
    ),
    (
        ("quiescence", "unreadable_pids"),
        [100],
        "quiescence.unreadable_pids: unresolved",
    ),
    (
        ("quiescence", "unknown_clients"),
        [100],
        "quiescence.unknown_clients: unresolved",
    ),
    (("quiescence", "active_clients"), [701], "quiescence.active_clients: unresolved"),
    (("quiescence", "errors"), ["ambiguous"], "quiescence.errors: unresolved"),
    (("quiescence", "clients"), [], "quiescence.clients: incomplete"),
    (
        ("quiescence", "clients", 0, "state"),
        "unknown",
        "quiescence.clients.state: not exited",
    ),
    (("quiescence", "jobs"), [], "quiescence.jobs: incomplete"),
    (
        ("quiescence", "jobs", 0, "job_id"),
        "unknown",
        "quiescence.jobs.job_id: mismatch",
    ),
    (
        ("quiescence", "jobs", 0, "job_name"),
        "unknown",
        "quiescence.jobs.job_name: mismatch",
    ),
    (
        ("quiescence", "jobs", 0, "status"),
        "WINDING_DOWN",
        "quiescence.jobs.status: not terminal",
    ),
    (
        ("quiescence", "jobs", 0, "status"),
        "FUTURE_STATUS",
        "quiescence.jobs.status: not terminal",
    ),
    (("quiescence", "jobs", 0, "status"), [], "quiescence.jobs.status: not terminal"),
    (("originals", "cancel", "run_id"), "other-run", "cancel.run_id: mismatch"),
    (("originals", "cancel", "outcome"), "partial", "cancel.outcome: not terminal"),
    (("originals", "cancel", "status"), "SUBMITTED", "cancel.status: not terminal"),
    (("originals", "cancel", "status"), [], "cancel.status: not terminal"),
    (("originals", "cancel", "jobs"), [], "cancel.jobs: incomplete"),
    (
        ("originals", "cancel", "jobs", 0, "live_outcome"),
        "unknown",
        "cancel.jobs.live_outcome: not terminal",
    ),
    (
        ("originals", "cancel", "jobs", 0, "live_status"),
        "unknown",
        "cancel.jobs.live_status: not terminal",
    ),
    (
        ("originals", "cancel", "jobs", 0, "live_error"),
        "failed",
        "cancel.jobs.live_error: unresolved",
    ),
    (
        ("originals", "cancel", "jobs", 0, "persisted_states"),
        ["RUNNING"],
        "cancel.jobs.persisted_states: not terminal",
    ),
    (("originals", "cancel", "sky_job_ids"), ["1"], "cancel.sky_job_ids: mismatch"),
    (
        ("originals", "cancel", "durable_absence_conflict_errors"),
        ["conflict"],
        "cancel.durable_absence_conflict_errors: unresolved",
    ),
    (("originals", "degraded", "outcome"), "cleaned", "degraded.outcome: mismatch"),
    (
        ("originals", "degraded", "errors"),
        [],
        "degraded.errors: not exact foreign-owner refusal",
    ),
    (
        ("originals", "remote_absence", "controller_outcome"),
        "cleaned",
        "remote_absence.controller_outcome: mismatch",
    ),
    (
        ("originals", "remote_absence", "commands", "cancel", "sha256"),
        "0" * 64,
        "remote_absence.cancel.sha256: mismatch",
    ),
    (
        ("originals", "remote_absence", "commands", "controller", "outcome"),
        "cleaned",
        "remote_absence.controller.outcome: mismatch",
    ),
]


@pytest.mark.parametrize(
    "path,value,reason",
    _MUTATIONS,
    ids=["/".join(map(str, row[0])) for row in _MUTATIONS],
)
def test_exact_reason_hostile_controls(path, value, reason):
    population = _population()
    _set(population, path, value)
    _refusal(population, reason)


@pytest.mark.parametrize("field", list(_identity()))
@pytest.mark.parametrize("section", ["observation", "quiescence"])
def test_all_target_identities_are_exact(field, section):
    population = _population()
    population[section]["target"][field] = "different"
    _refusal(population, f"{section}.target: mismatch")


@pytest.mark.parametrize(
    "field", ["root", "marker", "interpreter", "uid", "pid", "start_ticks"]
)
@pytest.mark.parametrize("section", ["api_before", "api_after", "quiescence", "client"])
def test_all_process_lifetime_fields_are_exact(field, section):
    population = _population()
    replacement = {
        "root": "/different/api",
        "marker": "different",
        "interpreter": "/different/python",
        "uid": 1001,
        "pid": 999,
        "start_ticks": 999,
    }
    if section == "client":
        population["quiescence"]["clients"][0]["identity"][field] = replacement[field]
        reason = "quiescence.clients.identity: mismatch"
    elif section == "quiescence":
        population["quiescence"]["api"][field] = replacement[field]
        reason = "quiescence.api: lifetime mismatch"
    else:
        population["observation"][section]["identity"][field] = replacement[field]
        reason = f"{section}.identity: lifetime mismatch"
    _refusal(population, reason)


_FLAGS = (
    [
        (("observation", "task_db", key), True, f"task_db.{key}")
        for key in ("readable", "complete", "path_identity_verified")
    ]
    + [
        (("observation", owner, key), True, f"{owner}.{key}")
        for owner in ("owner_before", "owner_after")
        for key in ("readable", "complete")
    ]
    + [
        (("observation", api, "identity_verified"), True, f"{api}.identity_verified")
        for api in ("api_before", "api_after")
    ]
    + [
        (("quiescence", "complete"), True, "quiescence.complete"),
        (
            ("quiescence", "clients", 0, "exact_lifetime_absent"),
            True,
            "quiescence.clients.exact_lifetime_absent",
        ),
        (
            ("quiescence", "jobs", 0, "exact_identity_verified"),
            True,
            "quiescence.jobs.exact_identity_verified",
        ),
    ]
    + [
        (("originals", "degraded", key), value, f"degraded.{key}")
        for key, value in (
            ("remote_absence_verified", True),
            ("local_metadata_cleared", False),
            ("overall_verified", False),
            ("verified", False),
        )
    ]
    + [
        (("originals", "remote_absence", key), value, f"remote_absence.{key}")
        for key, value in (
            ("own_uid_absent", True),
            ("remote_absence_verified", True),
            ("other_controller_uids_preserved", True),
            ("local_metadata_cleared", False),
            ("overall_verified", False),
        )
    ]
)


@pytest.mark.parametrize("path,expected,field", _FLAGS)
@pytest.mark.parametrize("wrong", [None, 0, 1, "true", "false", [], {}])
def test_boolean_evidence_never_coerces(path, expected, field, wrong):
    population = _population()
    _set(population, path, wrong)
    _refusal(population, f"{field} must be a literal boolean")


@pytest.mark.parametrize("path,expected,field", _FLAGS)
def test_opposite_boolean_refuses(path, expected, field):
    population = _population()
    _set(population, path, not expected)
    _refusal(population, f"{field}: must be {expected}")


@pytest.mark.parametrize(
    "role",
    [
        "cancel",
        "degraded",
        "remote_absence",
        "api_identity",
        "controller_identity",
        "observation",
        "quiescence",
    ],
)
def test_raw_bytes_bound_not_only_parsed_population(role):
    inputs = _inputs(_population())
    target = inputs if role in ("observation", "quiescence") else inputs["originals"]
    target[role] += b" "
    prefix = role if target is inputs else f"originals.{role}"
    with pytest.raises(ValueError, match=f"^{re.escape(prefix)}: hash mismatch$"):
        reconcile_task_metadata(**inputs, assembled_at=_ASSEMBLED)


def _required_containers():
    population = _population()
    inputs = _inputs(population)
    population["contract"] = json.loads(inputs["contract"])
    paths = [
        ("contract",),
        ("contract", "target"),
        ("contract", "api"),
        ("contract", "clients", 0),
        ("contract", "jobs", 0),
        ("contract", "original_sha256"),
        ("observation",),
        ("observation", "task_db"),
        ("quiescence",),
        ("quiescence", "api"),
        ("quiescence", "clients", 0),
        ("quiescence", "clients", 0, "identity"),
        ("quiescence", "jobs", 0),
    ]
    paths += [
        ("observation", name)
        for name in ("owner_before", "owner_after", "api_before", "api_after")
    ]
    paths += [
        ("observation", name, "identity")
        for name in ("owner_before", "owner_after", "api_before", "api_after")
    ]
    cases = []
    for path in paths:
        value = population
        for key in path:
            value = value[key]
        for key in value:
            cases.append((path, key))
    return cases


@pytest.mark.parametrize("path,key", _required_containers())
def test_every_required_versioned_field_rejects_omission(path, key):
    population = _population()
    inputs = _inputs(population)
    role = path[0]
    decoded = json.loads(inputs[role])
    container = decoded
    for part in path[1:]:
        container = container[part]
    del container[key]
    inputs[role] = _raw(decoded)
    if role != "contract":
        contract = json.loads(inputs["contract"])
        contract[f"{role}_sha256"] = hashlib.sha256(inputs[role]).hexdigest()
        inputs["contract"] = _raw(contract)
    with pytest.raises(ValueError) as error:
        reconcile_task_metadata(**inputs, assembled_at=_ASSEMBLED)
    assert (
        str(error.value) == f"{_missing_field_container(path)}: exact fields required"
    )


def _missing_field_container(path):
    if path[0] == "contract":
        return path[1] if len(path) > 1 else "contract"
    parts = [str(part) for part in path if not isinstance(part, int)]
    if parts[0] == "observation" and len(parts) > 1:
        parts = parts[1:]
        if parts[0] in ("api_before", "api_after"):
            parts = parts[:1]
    return ".".join(parts)


@pytest.mark.parametrize(
    "field", ["root", "marker", "interpreter", "pid", "start_ticks"]
)
def test_original_api_identity_cannot_be_substituted(field):
    population = _population()
    population["originals"]["api_identity"][field] = "substituted"
    _refusal(population, f"api_identity.{field}: mismatch")


@pytest.mark.parametrize(
    "field", ["project_alias", "project_id", "cluster_id", "context"]
)
def test_original_degraded_identity_cannot_be_substituted(field):
    population = _population()
    population["originals"]["degraded"][field] = "substituted"
    _refusal(population, f"degraded.{field}: mismatch")


@pytest.mark.parametrize("field", ["cloud_name", "controller_uid"])
def test_original_controller_identity_cannot_be_substituted(field):
    population = _population()
    population["originals"]["controller_identity"][field] = "substituted"
    _refusal(population, f"controller_identity.{field}: mismatch")


@pytest.mark.parametrize("field", ["uid", "pid", "start_ticks"])
@pytest.mark.parametrize("wrong", [True, False, "1000", 1000.0, None])
def test_process_identifiers_are_literal_integers(field, wrong):
    population = _population()
    population["contract"]["api"][field] = wrong
    _refusal(population, f"api.{field} must be a literal integer")


@pytest.mark.parametrize(
    "field",
    [
        "stop_authorized",
        "cleanup_v2_pass",
        "original_local_metadata_cleared",
        "original_overall_verified",
    ],
)
def test_consumer_rejects_promoted_authority_and_original_flags(field):
    population = _population()
    receipt = _produce(population)
    receipt[field] = True
    with pytest.raises(ValueError, match="^receipt: recomputed content mismatch$"):
        validate_reconciliation(_raw(receipt), **_inputs(population))


@pytest.mark.parametrize(
    "field,value",
    [
        ("orphan_count", 0),
        ("local_runtime_disposition", "removed"),
        ("status", "pass"),
        ("stop_authorized", 0),
    ],
)
def test_missing_docker_builder_evidence_never_produces_qualification(field, value):
    population = _population()
    receipt = _produce(population)
    receipt[field] = value
    with pytest.raises(ValueError, match="^receipt: recomputed content mismatch$"):
        validate_reconciliation(_raw(receipt), **_inputs(population))


def test_stable_but_different_foreign_owner_is_not_original_identity():
    population = _population()
    for key in ("owner_before", "owner_after"):
        population["observation"][key]["identity"]["context"] = (
            "different-stable-context"
        )
    _refusal(population, "owner: original foreign identity mismatch")


@pytest.mark.parametrize(
    "raw,reason",
    [
        (b'{"x":1,"x":2}', "JSON: duplicate key x"),
        (b'{"x":NaN}', "JSON: NaN"),
        (b"{", "contract: unreadable JSON"),
        (b"\xff", "contract: unreadable JSON"),
        (b"[]", "contract: object required"),
        (None, "contract: original bytes required"),
    ],
)
def test_unreadable_or_ambiguous_contract_refuses(raw, reason):
    inputs = _inputs(_population())
    inputs["contract"] = raw
    with pytest.raises(ValueError) as error:
        reconcile_task_metadata(**inputs, assembled_at=_ASSEMBLED)
    assert str(error.value) == reason


@pytest.mark.parametrize(
    "path,value,reason",
    [
        (
            ("observation", "started_at"),
            "2025-01-01T00:00:00+00:00",
            "observation: invalid chronology",
        ),
        (
            ("observation", "finished_at"),
            "2027-01-01T00:00:00+00:00",
            "observation: invalid chronology",
        ),
        (
            ("observation", "task_db", "observed_at"),
            _ORIGINAL,
            "task_db.observed_at: outside observation window",
        ),
        (
            ("quiescence", "observed_at"),
            _ASSEMBLED,
            "quiescence.observed_at: outside observation window",
        ),
        (
            ("observation", "owner_before", "observed_at"),
            "2026-01-01T10:00:01",
            "owner_before.observed_at: timezone required",
        ),
        (
            ("observation", "owner_before", "observed_at"),
            "not-time",
            "owner_before.observed_at: ISO8601 timestamp required",
        ),
    ],
)
def test_observation_times_are_preserved_not_fabricated(path, value, reason):
    population = _population()
    _set(population, path, value)
    _refusal(population, reason)


def test_input_mutation_request_never_passes_unknown_field_boundary():
    population = _population()
    population["observation"]["clear_foreign_owner"] = True
    _refusal(population, "observation: exact fields required")


@pytest.mark.parametrize("field", ["uid", "pid", "start_ticks"])
def test_process_identity_bounds(field):
    population = _population()
    population["contract"]["api"][field] = -1 if field == "uid" else 0
    _refusal(population, f"api.{field} must be at least {0 if field == 'uid' else 1}")


@pytest.mark.parametrize("key", ["root", "marker", "uid", "run_id"])
def test_known_client_must_belong_to_same_api_and_run(key):
    population = _population()
    replacement = {
        "root": "/another/task/api",
        "marker": "other-marker",
        "interpreter": "/another/python",
        "uid": 1001,
        "run_id": "other-run",
    }
    population["contract"]["clients"][0][key] = replacement[key]
    _refusal(
        population,
        f"clients.{key}: {'mismatch' if key == 'run_id' else 'API mismatch'}",
    )


def test_duplicate_client_population_refuses():
    population = _population()
    population["contract"]["clients"] *= 2
    _refusal(population, "clients: duplicate PID")


@pytest.mark.parametrize("key", list(_owner()))
def test_every_foreign_owner_field_is_stable(key):
    population = _population()
    if key == "schema_version":
        reason = "owner_after.schema_version: unsupported"
    elif key == "project_id":
        reason = "owner_after.project_id: mismatch"
    else:
        reason = "owner: identity changed"
    population["observation"]["owner_after"]["identity"][key] = "different"
    _refusal(population, reason)


@pytest.mark.parametrize(
    "field", ["observation_sha256", "quiescence_sha256", "foreign_owner_sha256"]
)
@pytest.mark.parametrize("wrong", [None, "", "A" * 64, "0" * 63, 123])
def test_contract_hash_shape_is_strict(field, wrong):
    inputs = _inputs(_population())
    contract = json.loads(inputs["contract"])
    contract[field] = wrong
    inputs["contract"] = _raw(contract)
    with pytest.raises(ValueError, match=f"^{field}: lowercase SHA256 required$"):
        reconcile_task_metadata(**inputs, assembled_at=_ASSEMBLED)


@pytest.mark.parametrize(
    "role",
    ["cancel", "degraded", "remote_absence", "api_identity", "controller_identity"],
)
@pytest.mark.parametrize("wrong", [None, "", "0" * 63, "A" * 64])
def test_every_original_hash_shape_is_strict(role, wrong):
    inputs = _inputs(_population())
    contract = json.loads(inputs["contract"])
    contract["original_sha256"][role] = wrong
    inputs["contract"] = _raw(contract)
    with pytest.raises(
        ValueError, match=f"^original_sha256.{role}: lowercase SHA256 required$"
    ):
        reconcile_task_metadata(**inputs, assembled_at=_ASSEMBLED)


@pytest.mark.parametrize(
    "role",
    ["cancel", "degraded", "remote_absence", "api_identity", "controller_identity"],
)
def test_original_receipt_omission_refuses(role):
    inputs = _inputs(_population())
    del inputs["originals"][role]
    with pytest.raises(ValueError, match="^originals: exact fields required$"):
        reconcile_task_metadata(**inputs, assembled_at=_ASSEMBLED)


@pytest.mark.parametrize(
    "role",
    ["cancel", "degraded", "remote_absence", "api_identity", "controller_identity"],
)
def test_unreadable_original_refuses_before_hash_join(role):
    inputs = _inputs(_population())
    inputs["originals"][role] = b"{"
    with pytest.raises(ValueError, match=f"^originals.{role}: unreadable JSON$"):
        reconcile_task_metadata(**inputs, assembled_at=_ASSEMBLED)


@pytest.mark.parametrize(
    "path,value,reason",
    [
        (
            ("originals", "remote_absence", "commands", "controller", "returncode"),
            1,
            "remote_absence.controller.returncode must be at most 0",
        ),
        (
            ("originals", "remote_absence", "commands", "cancel", "returncode"),
            False,
            "remote_absence.cancel.returncode must be a literal integer",
        ),
        (
            ("originals", "controller_identity", "controller_absent_before_task"),
            False,
            "controller_identity.controller_absent_before_task: must be True",
        ),
        (
            ("originals", "controller_identity", "other_controller_uids"),
            None,
            "controller_identity.other_controller_uids: list required",
        ),
        (
            ("originals", "controller_identity", "other_controller_uids"),
            ["duplicate", "duplicate"],
            "controller_identity.other_controller_uids: duplicates",
        ),
        (
            ("originals", "controller_identity", "other_controller_uids"),
            ["synthetic-controller-uid"],
            "controller_identity.other_controller_uids: includes target",
        ),
        (
            ("originals", "remote_absence", "other_controllers"),
            3,
            "remote_absence.other_controllers: mismatch",
        ),
        (
            ("originals", "remote_absence", "other_controllers"),
            True,
            "remote_absence.other_controllers must be a literal integer",
        ),
        (
            ("observation", "api_before", "observed_at"),
            _MIDDLE,
            "api_before.observed_at: must bracket all observations",
        ),
        (
            ("observation", "api_after", "observed_at"),
            _MIDDLE,
            "api_after.observed_at: must bracket all observations",
        ),
        (
            ("observation", "owner_before", "observed_at"),
            _END,
            "owner: reversed observations",
        ),
        (
            ("observation", "owner_after", "observed_at"),
            _START,
            "owner: reversed observations",
        ),
        (
            ("observation", "task_db", "observed_at"),
            _START,
            "owner: must bracket task database observation",
        ),
    ],
)
def test_additional_causal_identity_and_chronology_controls(path, value, reason):
    population = _population()
    _set(population, path, value)
    _refusal(population, reason)


def test_timezone_equivalent_observations_compare_instants_not_lexical_text():
    population = _population()
    population["observation"]["api_before"]["observed_at"] = "2026-01-01T11:00:00+01:00"
    receipt = _produce(population)
    assert receipt["observed_at"]["api_before"] == "2026-01-01T11:00:00+01:00"


@pytest.mark.parametrize(
    "status",
    [
        "SUCCEEDED",
        "CANCELLED",
        "FAILED",
        "FAILED_SETUP",
        "FAILED_PRECHECKS",
        "FAILED_NO_RESOURCE",
        "FAILED_CONTROLLER",
    ],
)
def test_terminal_jobs_are_closed_world_not_success_only(status):
    population = _population()
    population["quiescence"]["jobs"][0]["status"] = status
    assert _produce(population)["caller_quiescence_verified"] is True


def test_new_receipt_does_not_satisfy_existing_r7_three_boolean_predicate():
    population = _population()
    receipt = _produce(population)
    assert not all(
        receipt.get(key) is True
        for key in (
            "overall_verified",
            "remote_absence_verified",
            "local_metadata_cleared",
        )
    )
    assert population["originals"]["degraded"]["overall_verified"] is False


@pytest.mark.parametrize("field", list(_identity()))
@pytest.mark.parametrize("value", [None, "", True, []])
def test_original_scope_requires_each_literal_identity(field, value):
    population = _population()
    population["contract"]["target"][field] = value
    _refusal(population, f"target.{field}: nonempty text required")


@pytest.mark.parametrize("field", list(_identity()))
@pytest.mark.parametrize("section", ["observation", "quiescence"])
def test_current_scope_requires_every_identity_field(field, section):
    population = _population()
    del population[section]["target"][field]
    _refusal(population, f"{section}.target: mismatch")


_ORIGINAL_REQUIRED = (
    [
        ("cancel", "run_id", "cancel.run_id: mismatch"),
        ("cancel", "outcome", "cancel.outcome: not terminal"),
        ("cancel", "status", "cancel.status: not terminal"),
        (
            "cancel",
            "durable_absence_conflict_errors",
            "cancel.durable_absence_conflict_errors: unresolved",
        ),
        (
            "cancel",
            "durable_absence_conflict_job_ids",
            "cancel.durable_absence_conflict_job_ids: unresolved",
        ),
        ("cancel", "sky_job_ids", "cancel.sky_job_ids: mismatch"),
        ("cancel", "jobs", "cancel.jobs: incomplete"),
        ("degraded", "outcome", "degraded.outcome: mismatch"),
        ("degraded", "errors", "degraded.errors: not exact foreign-owner refusal"),
        ("remote_absence", "at", "remote_absence.at: nonempty text required"),
        (
            "remote_absence",
            "other_controllers",
            "remote_absence.other_controllers must be a literal integer",
        ),
        (
            "remote_absence",
            "controller_outcome",
            "remote_absence.controller_outcome: mismatch",
        ),
        ("remote_absence", "commands", "remote_absence.commands: object required"),
        ("controller_identity", "at", "controller_identity.at: nonempty text required"),
        (
            "controller_identity",
            "controller_absent_before_task",
            "controller_identity.controller_absent_before_task must be a literal boolean",
        ),
        (
            "controller_identity",
            "other_controller_uids",
            "controller_identity.other_controller_uids: list required",
        ),
    ]
    + [
        ("degraded", field, f"degraded.{field}: mismatch")
        for field in ("project_alias", "project_id", "cluster_id", "context")
    ]
    + [
        ("api_identity", field, "api_identity: exact fields required")
        for field in ("root", "marker", "interpreter", "pid", "start_ticks")
    ]
    + [
        ("controller_identity", field, f"controller_identity.{field}: mismatch")
        for field in ("cloud_name", "controller_uid")
    ]
    + [
        (path[1], path[2], f"{field} must be a literal boolean")
        for path, _, field in _FLAGS
        if path[0] == "originals"
    ]
)


@pytest.mark.parametrize("role,field,reason", _ORIGINAL_REQUIRED)
def test_original_required_field_omission_is_explicit(role, field, reason):
    population = _population()
    del population["originals"][role][field]
    _refusal(population, reason)


@pytest.mark.parametrize(
    "field,reason",
    [
        ("job_id", "cancel.jobs.job_id: mismatch"),
        ("job_name", "cancel.jobs.job_name: mismatch"),
        ("live_outcome", "cancel.jobs.live_outcome: not terminal"),
        ("live_status", "cancel.jobs.live_status: not terminal"),
        ("live_error", "cancel.jobs.live_error: unresolved"),
        ("persisted_states", "cancel.jobs.persisted_states: not terminal"),
    ],
)
def test_original_job_field_omission_is_explicit(field, reason):
    population = _population()
    del population["originals"]["cancel"]["jobs"][0][field]
    _refusal(population, reason)


@pytest.mark.parametrize("command", ["cancel", "controller"])
@pytest.mark.parametrize("field", ["returncode", "outcome", "sha256"])
def test_original_command_field_omission_is_explicit(command, field):
    population = _population()
    del population["originals"]["remote_absence"]["commands"][command][field]
    _refusal(population, f"remote_absence.{command}: exact fields required")
