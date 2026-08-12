from __future__ import annotations

import json
from pathlib import Path
import shutil

from typer.testing import CliRunner

from npa.cli import cleanup as cleanup_cli
from npa.cli.main import app
from npa import teardown_receipts


runner = CliRunner()


def _home(monkeypatch, tmp_path: Path) -> Path:  # noqa: ANN001
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv(
        "NPA_TEARDOWN_RECEIPT_DIR", str(home / ".npa" / "teardown-receipts")
    )
    return home


def _sky_state(home: Path) -> tuple[Path, Path]:
    sky = home / ".sky"
    sky.mkdir(parents=True)
    venv = home / ".npa" / "skypilot-venv"
    venv.mkdir(parents=True)
    return sky, venv


def test_active_jobs_preserve_both_skypilot_state_stores(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    home = _home(monkeypatch, tmp_path)
    sky, venv = _sky_state(home)
    monkeypatch.setattr(cleanup_cli, "_nonterminal_jobs", lambda sky_bin: (["8"], ""))

    result = runner.invoke(app, ["cleanup", "--yes", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["cleanup_failed"] is True
    assert sky.exists() and venv.exists()
    event = teardown_receipts.latest_phase_states()["workflow_audit"]
    assert event["terminal_state"] == "active"
    assert event["verification"]["nonterminal_job_ids"] == ["8"]


def test_verified_job_audit_is_durable_before_isolated_runtime_removal(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    home = _home(monkeypatch, tmp_path)
    sky, venv = _sky_state(home)
    monkeypatch.setattr(cleanup_cli, "_nonterminal_jobs", lambda sky_bin: ([], ""))
    real_rmtree = shutil.rmtree
    observed: list[str] = []
    tracked_identities = {
        (sky.stat().st_dev, sky.stat().st_ino),
        (venv.stat().st_dev, venv.stat().st_ino),
    }

    def guarded_rmtree(path: Path) -> None:
        identity = Path(path).stat(follow_symlinks=False)
        if (identity.st_dev, identity.st_ino) in tracked_identities:
            observed.append(
                teardown_receipts.latest_phase_states()["workflow_audit"][
                    "terminal_state"
                ]
            )
        real_rmtree(path)

    monkeypatch.setattr(shutil, "rmtree", guarded_rmtree)

    result = runner.invoke(app, ["cleanup", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    # The isolated NPA SkyPilot virtualenv may be removed only after the audit;
    # machine-shared ~/.sky is never project cleanup residue.
    assert observed == ["verified_absent"]
    assert sky.exists() and not venv.exists()
    payload = json.loads(result.output)
    assert payload["audit_receipts_are_operational_residue"] is False
    assert payload["operational_residue_present"] is False
    phases = teardown_receipts.latest_phase_states()
    assert phases["workflow_audit"]["terminal_state"] == "verified_absent"
    assert phases["local_cleanup"]["terminal_state"] == "completed"


def test_receipt_failure_preserves_the_only_job_audit_state(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    home = _home(monkeypatch, tmp_path)
    sky, venv = _sky_state(home)
    monkeypatch.setattr(cleanup_cli, "_nonterminal_jobs", lambda sky_bin: ([], ""))
    monkeypatch.setattr(
        teardown_receipts,
        "record_teardown_event",
        lambda **kwargs: (_ for _ in ()).throw(OSError("receipt disk unavailable")),
    )

    result = runner.invoke(app, ["cleanup", "--yes"])

    assert result.exit_code == 1
    assert sky.exists() and venv.exists()
    assert "durable managed-job audit receipt failed" in result.output


def test_local_transaction_receipt_failure_preserves_non_sky_cache(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    home = _home(monkeypatch, tmp_path)
    cache = home / ".npa" / "terraform-plugin-cache"
    cache.mkdir(parents=True)
    (cache / "artifact").write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(cleanup_cli, "_nonterminal_jobs", lambda sky_bin: ([], ""))
    real_record = teardown_receipts.record_teardown_event

    def fail_local_start(**kwargs):  # noqa: ANN001
        if kwargs.get("phase") == "local_cleanup":
            raise OSError("receipt disk unavailable")
        return real_record(**kwargs)

    monkeypatch.setattr(teardown_receipts, "record_teardown_event", fail_local_start)

    result = runner.invoke(app, ["cleanup", "--yes"])

    assert result.exit_code == 1
    assert cache.exists()
    assert "Preserved local state" in result.output


def test_final_audit_uses_receipts_after_config_and_resources_are_removed(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    _home(monkeypatch, tmp_path)
    for phase in (
        "workflow_audit",
        "agent",
        "cluster",
        "bucket",
        "project_config",
        "local_cleanup",
    ):
        teardown_receipts.record_teardown_event(
            phase=phase,
            resource=f"{phase}-fixture",
            terminal_state="verified_deleted"
            if phase != "local_cleanup"
            else "completed",
        )
    monkeypatch.setattr(
        cleanup_cli,
        "_nonterminal_jobs",
        lambda sky_bin: ([], "SkyPilot is not installed, so jobs could not be checked"),
    )

    result = runner.invoke(app, ["cleanup", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    phases = {item["resource"]: item for item in payload["phases"]}
    for resource in (
        "workflow runs",
        "agent VM",
        "SkyPilot controller",
        "Kubernetes cluster",
        "object-storage bucket",
        "project configuration",
        "local caches and known credentials",
    ):
        assert phases[resource]["operator_action_required"] is False
        assert phases[resource]["operator_action_remains"] is False
    assert payload["retained_audit_receipts"] == 1
    assert payload["audit_receipts_retained"] is True
    assert payload["audit_receipts_are_operational_residue"] is False
    assert payload["operational_residue_present"] is False
    assert payload["residue_present"] is False
    assert payload["local_state"] == "fully_cleaned"
    assert payload["verification_unresolved"] is False


def test_uncertain_receipt_remains_visible_and_actionable(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    _home(monkeypatch, tmp_path)
    teardown_receipts.record_teardown_event(
        phase="controller",
        resource="controller-fixture",
        terminal_state="verification_failed",
        errors=["RBAC denied"],
    )
    monkeypatch.setattr(cleanup_cli, "_nonterminal_jobs", lambda sky_bin: ([], ""))

    result = runner.invoke(app, ["cleanup", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["operational_residue_present"] is False
    assert payload["audit_receipts_retained"] is True
    assert payload["verification_unresolved"] is True
    controller_phase = next(
        item for item in payload["phases"] if item["resource"] == "SkyPilot controller"
    )
    assert controller_phase["operator_action_required"] is True
    assert "verification_failed" in controller_phase["observed_state"]


def test_receipt_listing_human_and_json_and_pruning_confirmation(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    _home(monkeypatch, tmp_path)
    teardown_receipts.record_teardown_event(
        phase="cluster", resource="cluster-fixture", terminal_state="verified_deleted"
    )

    human = runner.invoke(app, ["cleanup", "--list-receipts"])
    machine = runner.invoke(app, ["cleanup", "--list-receipts", "--json"])
    refused = runner.invoke(app, ["cleanup", "--prune-receipts"])

    assert human.exit_code == 0
    assert "retained indefinitely" in human.output
    assert json.loads(machine.output)["result"] == "receipts_listed"
    assert refused.exit_code == 2
    assert "requires --yes" in refused.output
