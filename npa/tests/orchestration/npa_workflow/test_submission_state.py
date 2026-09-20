from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.submission_state import (
    audit_project_submissions,
    inspect_submission_state,
    load_submission_state,
    record_submission_plan,
    submission_lock,
    submission_proves_never_launched,
    submission_state_path,
    update_submission_state,
)


def _receipt_bytes(*, omit: tuple[str, ...] = (), **overrides: object) -> bytes:
    payload: dict[str, object] = {
        "schema_version": "npa.workflow.submission.v1",
        "project": "demo",
        "run_id": "run-1",
        "workflow": {"name": "sim2real"},
        "launch": {"status": "launching", "kind": "runtime"},
    }
    payload.update(overrides)
    for key in omit:
        payload.pop(key, None)
    return json.dumps(payload, sort_keys=True).encode()


UNVERIFIABLE_RECEIPTS = (
    pytest.param("truncated_json", b'{"schema_version":', id="truncated-json"),
    pytest.param("invalid_utf8", b'\xff\xfe{"schema_version":', id="invalid-utf8"),
    pytest.param("zero_bytes", b"", id="zero-bytes"),
    pytest.param("non_object", b"[]", id="non-object"),
    pytest.param("empty_object", b"{}", id="empty-object"),
    pytest.param(
        "wrong_schema",
        _receipt_bytes(schema_version="npa.workflow.submission.v0"),
        id="wrong-schema",
    ),
    pytest.param(
        "missing_schema",
        _receipt_bytes(omit=("schema_version",)),
        id="missing-schema",
    ),
    pytest.param(
        "wrong_project",
        _receipt_bytes(project="other"),
        id="wrong-project",
    ),
    pytest.param("wrong_run", _receipt_bytes(run_id="run-2"), id="wrong-run"),
)


@pytest.mark.parametrize("operation", ["update", "update_locked", "plan"])
@pytest.mark.parametrize(("case", "body"), UNVERIFIABLE_RECEIPTS)
def test_mutation_rejects_unverifiable_existing_receipt_without_replacing_bytes(
    operation: str,
    case: str,
    body: bytes,
) -> None:
    path = submission_state_path("demo", "run-1")
    path.parent.mkdir(parents=True)
    path.write_bytes(body)

    with pytest.raises(
        ValueError, match="existing workflow submission receipt is unavailable"
    ):
        if operation == "update":
            update_submission_state(
                "demo", "run-1", {"artifact_load": {"status": "ok"}}
            )
        elif operation == "update_locked":
            with submission_lock("demo", "run-1"):
                update_submission_state(
                    "demo",
                    "run-1",
                    {"artifact_load": {"status": "ok"}},
                    locked=True,
                )
        else:
            record_submission_plan(
                "demo",
                "run-1",
                workflow={"name": "sim2real"},
                planning={"state": "durable"},
            )

    assert path.read_bytes() == body, case


@pytest.mark.parametrize("operation", ["update", "update_locked", "plan"])
def test_mutation_rejects_symlink_receipt_without_replacing_link(
    tmp_path: Path, operation: str
) -> None:
    path = submission_state_path("demo", "run-1")
    path.parent.mkdir(parents=True)
    target = tmp_path / "retained-receipt.json"
    body = _receipt_bytes()
    target.write_bytes(body)
    path.symlink_to(target)

    with pytest.raises(
        ValueError, match="existing workflow submission receipt is unavailable"
    ):
        if operation == "update":
            update_submission_state(
                "demo", "run-1", {"artifact_load": {"status": "ok"}}
            )
        elif operation == "update_locked":
            with submission_lock("demo", "run-1"):
                update_submission_state(
                    "demo",
                    "run-1",
                    {"artifact_load": {"status": "ok"}},
                    locked=True,
                )
        else:
            record_submission_plan(
                "demo",
                "run-1",
                workflow={"name": "sim2real"},
                planning={"state": "durable"},
            )

    assert path.is_symlink()
    assert target.read_bytes() == body


def test_unverifiable_receipt_error_never_includes_receipt_contents() -> None:
    secret = "synthetic-secret-that-must-not-appear"
    path = submission_state_path("demo", "run-1")
    path.parent.mkdir(parents=True)
    body = f'{{"aws_secret_access_key":"{secret}",'.encode()
    path.write_bytes(body)

    with pytest.raises(ValueError) as exc_info:
        update_submission_state("demo", "run-1", {"launch_state": "submitted"})

    assert secret not in str(exc_info.value)
    assert path.read_bytes() == body


def test_resume_planning_preserves_run_location_and_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    workflow = {"name": "sim2real", "run_prefix_uri": "s3://bucket/custom/run-1"}
    launch = {"status": "launching", "kind": "runtime"}
    update_submission_state(
        "demo",
        "run-1",
        {
            "workflow": workflow,
            "launch": launch,
            "launch_state": "submitted",
        },
    )

    receipt = record_submission_plan(
        "demo",
        "run-1",
        workflow={"name": "sim2real"},
        planning={"state": "durable"},
        launch_state="reserved",
    )

    assert receipt["workflow"] == workflow
    assert receipt["launch"] == launch
    assert receipt["launch_state"] == "submitted"
    assert not submission_proves_never_launched(receipt, project="demo", run_id="run-1")
    assert audit_project_submissions("demo").outcome == "launch_evidence"


def test_new_plan_proves_no_launch_and_rejects_identity_change(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    receipt = record_submission_plan(
        "demo",
        "run-1",
        workflow={"name": "sim2real"},
        planning={"state": "durable"},
    )
    assert submission_proves_never_launched(receipt, project="demo", run_id="run-1")
    with pytest.raises(ValueError, match="workflow identity"):
        record_submission_plan(
            "demo",
            "run-1",
            workflow={"name": "other"},
            planning={"state": "durable"},
        )
    assert load_submission_state("demo", "run-1") == receipt


def test_submission_state_is_owner_only_and_restart_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    update_submission_state("demo", "run-1", {"source_uri": "s3://bucket/src/"})
    state = load_submission_state("demo", "run-1")

    assert state["source_uri"] == "s3://bucket/src/"
    assert state["schema_version"] == "npa.workflow.submission.v1"
    assert submission_state_path("demo", "run-1").stat().st_mode & 0o777 == 0o600


def test_submission_state_rejects_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(ValueError, match="must not contain"):
        update_submission_state("demo", "run-1", {"aws_secret_access_key": "nope"})


def test_submission_state_allows_only_names_under_image_pull_secret_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    payload = update_submission_state(
        "demo",
        "run-image-pull-secret",
        {
            "workflow": {
                "resources_profile": {
                    "kubernetes": {
                        "pod_config": {
                            "spec": {"imagePullSecrets": [{"name": "registry-pull"}]}
                        }
                    }
                }
            }
        },
    )

    assert payload["workflow"]["resources_profile"]["kubernetes"]["pod_config"]["spec"][
        "imagePullSecrets"
    ] == [{"name": "registry-pull"}]
    with pytest.raises(ValueError, match="must not contain credentials"):
        update_submission_state(
            "demo",
            "run-malformed-image-pull-secret",
            {"imagePullSecrets": [{"password": "forbidden-inline-value"}]},
        )


def test_concurrent_updates_do_not_corrupt_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def write(index: int) -> None:
        update_submission_state("demo", "run-1", {f"field_{index}": index})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(20)))

    state = load_submission_state("demo", "run-1")
    assert all(state[f"field_{index}"] == index for index in range(20))


def test_locked_update_supports_a_multi_step_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    with submission_lock("demo", "run-1"):
        update_submission_state(
            "demo", "run-1", {"launch_state": "reserved"}, locked=True
        )

    assert load_submission_state("demo", "run-1")["launch_state"] == "reserved"


def test_inspection_distinguishes_absent_and_corrupt_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert inspect_submission_state("demo", "run-1").outcome == "absent"

    update_submission_state("demo", "run-1", {"launch": {"status": "launching"}})
    assert inspect_submission_state("demo", "run-1").outcome == "found"

    submission_state_path("demo", "run-1").write_text("not-json", encoding="utf-8")
    inspected = inspect_submission_state("demo", "run-1")
    assert inspected.outcome == "unavailable"
    assert "invalid receipt JSON" in inspected.error

    submission_state_path("demo", "run-1").write_bytes(b"\xff\xfe")
    inspected = inspect_submission_state("demo", "run-1")
    assert inspected.outcome == "unavailable"
    assert load_submission_state("demo", "run-1") == {}


def test_project_audit_requires_every_exact_ledger_to_prove_no_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert audit_project_submissions("demo").outcome == "absent"

    update_submission_state("demo", "reserved", {"launch_state": "reserved"})
    audit = audit_project_submissions("demo")
    assert audit.outcome == "not_submitted"
    assert audit.ledger_count == 1

    update_submission_state("demo", "launched", {"launch": {"status": "launching"}})
    assert audit_project_submissions("demo").outcome == "launch_evidence"


def test_project_audit_rejects_symlinks_and_unavailable_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    update_submission_state("demo", "reserved", {"launch_state": "reserved"})
    foreign = submission_state_path("other", "foreign")
    foreign.parent.mkdir(parents=True)
    foreign.write_text("{}", encoding="utf-8")
    link = submission_state_path("demo", "linked")
    link.symlink_to(foreign)
    assert audit_project_submissions("demo").outcome == "unavailable"

    link.unlink()
    submission_state_path("demo", "corrupt").write_text("not-json", encoding="utf-8")
    assert audit_project_submissions("demo").outcome == "unavailable"

    submission_state_path("demo", "corrupt").write_bytes(b"\xff\xfe")
    assert audit_project_submissions("demo").outcome == "unavailable"
