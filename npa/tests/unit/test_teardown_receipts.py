from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from npa import teardown_receipts as receipts


def _root(monkeypatch, tmp_path: Path) -> Path:  # noqa: ANN001
    root = tmp_path / "audit"
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(root))
    return root


def test_receipt_is_atomic_sanitized_and_survives_project_removal(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    root = _root(monkeypatch, tmp_path)
    project_config = tmp_path / "config.yaml"
    project_config.write_text("project: demo\n", encoding="utf-8")

    path = receipts.record_teardown_event(
        phase="cluster",
        resource="cluster-demo",
        terminal_state="verified_deleted",
        project_alias="demo",
        project_id="project-demo",
        context="ctx-demo",
        precheck={"access_token": "ghp_abcdefghijklmnopqrstuvwxyz123456"},
        action={"environment": {"AWS_SECRET_ACCESS_KEY": "secret"}},
        verification={"endpoint": "https://private.example.invalid"},
    )
    project_config.unlink()

    payload = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert payload["schema_version"] == receipts.SCHEMA_VERSION
    assert payload["events"][-1]["terminal_state"] == "verified_deleted"
    assert "ghp_" not in serialized
    assert "secret" not in serialized
    assert "private.example" not in serialized
    assert (
        receipts.latest_phase_states(project_alias="demo")["cluster"]["terminal_state"]
        == "verified_deleted"
    )
    assert list(root.glob(".*.tmp")) == []


def test_concurrent_events_are_not_lost(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    _root(monkeypatch, tmp_path)

    def write(index: int) -> None:
        receipts.record_teardown_event(
            phase=f"phase-{index}",
            resource=f"resource-{index}",
            terminal_state="completed",
            project_alias="demo",
            project_id="project-demo",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(24)))

    [payload] = receipts.list_teardown_receipts()
    assert len(payload["events"]) == 24


def test_secret_detection_preserves_container_types_and_rejects_secret_keys() -> None:
    assert not receipts.receipt_contains_secret(("safe", {"status": "completed"}))
    assert receipts.receipt_contains_secret({"access_token": "value"})
    assert receipts.receipt_contains_secret(
        "command failed: ghp_abcdefghijklmnopqrstuvwxyz"
    )
    assert receipts.receipt_contains_secret(
        "provider error at https://private.example.invalid: token=unsafe"
    )


def test_receipt_directory_symlink_is_refused(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    real = tmp_path / "real"
    real.mkdir()
    root = tmp_path / "linked"
    root.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(root))

    with pytest.raises(receipts.TeardownReceiptError, match="symlink"):
        receipts.record_teardown_event(
            phase="cluster",
            resource="demo",
            terminal_state="completed",
        )


def test_retry_does_not_regress_terminal_phase_to_unknown(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    _root(monkeypatch, tmp_path)
    receipts.record_teardown_event(
        phase="bucket",
        resource="bucket-demo",
        terminal_state="verified_deleted",
        project_alias="demo",
    )

    assert (
        receipts.latest_phase_states(project_alias="demo")["bucket"]["terminal_state"]
        == "verified_deleted"
    )


def test_global_receipt_never_proves_completion_for_an_explicit_project(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    _root(monkeypatch, tmp_path)
    receipts.record_teardown_event(
        phase="cluster",
        resource="unknown-cluster",
        terminal_state="verified_deleted",
    )

    assert receipts.latest_phase_states(project_alias="demo") == {}


def test_prune_is_explicit_age_gated_and_preserves_uncertainty(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    _root(monkeypatch, tmp_path)
    terminal = receipts.record_teardown_event(
        phase="cluster",
        resource="cluster-demo",
        terminal_state="verified_deleted",
        project_alias="terminal",
    )
    unresolved = receipts.record_teardown_event(
        phase="controller",
        resource="controller-demo",
        terminal_state="verification_failed",
        project_alias="unresolved",
    )
    payload = json.loads(terminal.read_text(encoding="utf-8"))
    payload["updated_at"] = (
        datetime.now(timezone.utc) - timedelta(days=100)
    ).isoformat()
    receipts._write_atomic(terminal, payload)

    removed, retained = receipts.prune_teardown_receipts(older_than_days=90)

    assert removed == [terminal]
    assert not terminal.exists()
    assert unresolved.exists()
    assert any("unresolved/uncertain" in item for item in retained)


def test_v2_identity_is_private_immutable_and_path_safe(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    root = _root(monkeypatch, tmp_path)
    path = receipts.record_teardown_event(
        phase="agent",
        resource="agent",
        terminal_state="in_progress",
        project_id="project-1",
        identity={
            "project_id": "project-1",
            "instance_id": "instance-1",
            "terraform_backends": [
                {"bucket": "state-bucket", "endpoint": "https://storage.example"}
            ],
        },
    )

    payload = receipts.load_teardown_receipt(path.stem)
    assert payload["identity"]["agents"][0]["instance_id"] == "instance-1"
    assert payload["identity"]["terraform_backends"][0]["endpoint"].startswith(
        "https://"
    )
    assert path.stat().st_mode & 0o777 == 0o600
    assert root.stat().st_mode & 0o777 == 0o700
    with pytest.raises(receipts.TeardownReceiptError, match="opaque ID"):
        receipts.load_teardown_receipt("../credentials")
    with pytest.raises(receipts.TeardownReceiptError, match="secret-shaped"):
        receipts.record_teardown_event(
            phase="agent",
            resource="agent",
            terminal_state="failed",
            project_id="project-1",
            identity={"access_key_secret": "must-not-persist"},
        )


def test_v1_receipt_loads_and_migrates_additively(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    root = _root(monkeypatch, tmp_path)
    root.mkdir(mode=0o700)
    path = receipts.receipt_path(project_alias="demo", project_id="project-1")
    path.write_text(
        json.dumps(
            {
                "schema_version": "npa.teardown.receipt.v1",
                "receipt_id": path.stem,
                "project_alias": "demo",
                "project_id": "project-1",
                "created_at": "2026-08-01T00:00:00Z",
                "updated_at": "2026-08-01T00:00:00Z",
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    loaded = receipts.load_teardown_receipt(path.stem)
    assert loaded["schema_version"] == receipts.SCHEMA_VERSION
    assert loaded["identity"] == {}

    receipts.record_teardown_event(
        phase="project_config",
        resource="demo",
        terminal_state="completed",
        project_alias="demo",
        project_id="project-1",
        identity={"project_id": "project-1"},
    )
    assert (
        json.loads(path.read_text(encoding="utf-8"))["schema_version"]
        == receipts.SCHEMA_VERSION
    )


def test_one_project_receipt_keeps_multiple_resource_identities(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    _root(monkeypatch, tmp_path)
    for name, instance_id in (("agent-a", "instance-a"), ("agent-b", "instance-b")):
        path = receipts.record_teardown_event(
            phase="agent",
            resource=name,
            terminal_state="verified_absent",
            project_id="project-1",
            identity={
                "project_id": "project-1",
                "agent_name": name,
                "instance_id": instance_id,
            },
        )
    payload = receipts.load_teardown_receipt(path.stem)
    assert {
        (item["agent_name"], item["instance_id"])
        for item in payload["identity"]["agents"]
    } == {("agent-a", "instance-a"), ("agent-b", "instance-b")}


def test_retry_cluster_ids_coexist_for_one_context_and_resolve_exactly(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    from npa.cleanup_identity import resolve_cleanup_identity

    _root(monkeypatch, tmp_path)
    path = None
    for cluster_id in ("cluster-first", "cluster-retry"):
        path = receipts.record_teardown_event(
            phase="cluster",
            resource="gpu-context",
            terminal_state="verified_absent",
            project_id="project-1",
            context="gpu-context",
            identity={
                "project_id": "project-1",
                "context": "gpu-context",
                "cluster_id": cluster_id,
            },
        )
    assert path is not None
    payload = receipts.load_teardown_receipt(path.stem)
    assert {item["cluster_id"] for item in payload["identity"]["clusters"]} == {
        "cluster-first",
        "cluster-retry",
    }

    selected = resolve_cleanup_identity(
        explicit={"project_id": "project-1", "cluster_id": "cluster-retry"},
        receipt_id=path.stem,
        phase="cluster",
        resource="gpu-context",
    )
    assert selected.get("cluster_id") == "cluster-retry"


def test_controller_observation_does_not_poison_recreated_cluster_identity(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    _root(monkeypatch, tmp_path)
    first = receipts.record_teardown_event(
        phase="controller",
        resource="gpu-context",
        terminal_state="verified_absent",
        project_id="project-1",
        identity={
            "project_id": "project-1",
            "context": "gpu-context",
            "cluster_id": "cluster-first",
            "operation_id": "cluster-operation-first",
            "cluster_absent": True,
        },
    )
    legacy = json.loads(first.read_text(encoding="utf-8"))
    legacy["identity"]["cluster_absent"] = True
    legacy["identity"]["operation_id"] = legacy["identity"]["clusters"][0].pop(
        "operation_id"
    )
    receipts._write_atomic(first, legacy)
    second = receipts.record_teardown_event(
        phase="controller",
        resource="gpu-context",
        terminal_state="in_progress",
        project_id="project-1",
        identity={
            "project_id": "project-1",
            "context": "gpu-context",
            "cluster_id": "cluster-retry",
            "operation_id": "cluster-operation-retry",
            "cluster_absent": False,
        },
    )

    assert first == second
    payload = receipts.load_teardown_receipt(second.stem)
    assert "cluster_absent" not in payload["identity"]
    assert {
        (item["cluster_id"], item["operation_id"])
        for item in payload["identity"]["clusters"]
    } == {
        ("cluster-first", "cluster-operation-first"),
        ("cluster-retry", "cluster-operation-retry"),
    }
