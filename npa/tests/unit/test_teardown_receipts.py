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
