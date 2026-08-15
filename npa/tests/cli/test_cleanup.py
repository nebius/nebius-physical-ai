"""`npa cleanup` — local residue report + wipe after teardown."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from npa.cli import cleanup as cleanup_cli
from npa.cli.main import app

runner = CliRunner()


def _seed_residue() -> tuple[Path, Path, Path, Path]:
    """Create the caches a teardown leaves behind under the isolated HOME."""
    home = Path.home()
    npa = home / ".npa"
    sky_venv = npa / "skypilot-venv" / "bin"
    sky_venv.mkdir(parents=True)
    (sky_venv / "sky").write_text("#!/bin/sh\n")
    tf_cache = npa / "terraform-plugin-cache" / "registry.terraform.io"
    tf_cache.mkdir(parents=True)
    (tf_cache / "provider").write_text("x" * 1024)
    sky_home = home / ".sky" / "state"
    sky_home.mkdir(parents=True)
    (sky_home / "db").write_text("y" * 2048)
    empty_alias = npa / "agents" / "test-rtx"
    empty_alias.mkdir(parents=True)
    return (
        npa / "skypilot-venv",
        npa / "terraform-plugin-cache",
        home / ".sky",
        empty_alias,
    )


def _seed_project_config() -> None:
    from npa.clients import config as config_module

    config_module.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config_module.CONFIG_PATH.write_text(
        yaml.safe_dump(
            {
                "default_project": "prod",
                "projects": {
                    "prod": {"project_id": "project-a", "tenant_id": "tenant-a"}
                },
            }
        )
    )


def test_cleanup_reports_residue_without_removing(monkeypatch) -> None:
    from npa.clients import config as config_module

    sky_venv, tf_cache, sky_home, empty_alias = _seed_residue()
    # A persisted sky_bin should be reported but not touched without --yes.
    config_module.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config_module.CONFIG_PATH.write_text(
        yaml.safe_dump({"skypilot": {"sky_bin": "/x/sky"}})
    )

    result = runner.invoke(app, ["cleanup"])

    assert result.exit_code == 0, result.output
    assert "SkyPilot venv" in result.output
    assert "Terraform provider cache" in result.output
    assert "~/.sky" in result.output
    assert "Re-run with --yes" in result.output
    # Nothing removed in report mode.
    assert (
        sky_venv.exists()
        and tf_cache.exists()
        and sky_home.exists()
        and empty_alias.exists()
    )


def test_cleanup_yes_removes_local_caches_but_keeps_tokens(monkeypatch) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    sky_venv, tf_cache, sky_home, empty_alias = _seed_residue()
    config_module.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config_module.CONFIG_PATH.write_text(
        yaml.safe_dump({"skypilot": {"sky_bin": "/x/sky"}})
    )
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump({"tokens": {"HF_TOKEN": "hf_keep"}})
    )
    monkeypatch.setattr(cleanup_cli, "_nonterminal_jobs", lambda sky_bin: ([], ""))

    result = runner.invoke(app, ["cleanup", "--yes"])

    assert result.exit_code == 0, result.output
    assert not sky_venv.exists()
    assert not tf_cache.exists()
    assert sky_home.exists()
    assert "Preserved shared SkyPilot state" in result.output
    assert not empty_alias.exists()
    # sky_bin cleared from config; tokens untouched.
    saved_config = yaml.safe_load(config_module.CONFIG_PATH.read_text()) or {}
    assert "sky_bin" not in saved_config.get("skypilot", {})
    assert (
        yaml.safe_load(credentials_module.CREDENTIALS_PATH.read_text())["tokens"][
            "HF_TOKEN"
        ]
        == "hf_keep"
    )


def test_cleanup_keep_sky_leaves_dot_sky(monkeypatch) -> None:
    sky_venv, _tf, sky_home, _empty = _seed_residue()
    monkeypatch.setattr(cleanup_cli, "_nonterminal_jobs", lambda sky_bin: ([], ""))

    result = runner.invoke(app, ["cleanup", "--yes", "--keep-sky"])

    assert result.exit_code == 0, result.output
    assert not sky_venv.exists()
    assert sky_home.exists()  # ~/.sky preserved


def test_cleanup_unreadable_queue_preserves_sky_state_but_continues_unrelated_cleanup(
    monkeypatch,
) -> None:
    sky_venv, tf_cache, sky_home, _empty = _seed_residue()
    monkeypatch.setattr(
        cleanup_cli,
        "_nonterminal_jobs",
        lambda sky_bin: ([], "could not read the managed-job queue: RBAC denied"),
    )

    result = runner.invoke(app, ["cleanup", "--yes", "--json"])

    assert result.exit_code == 1
    payload = __import__("json").loads(result.output)
    assert payload["managed_job_queue_state"] == "unreadable_or_unverified"
    assert payload["verification_unresolved"] is True
    assert sky_venv.exists() and sky_home.exists()
    assert not tf_cache.exists()


def test_cleanup_explicit_skip_is_not_failure_and_preserves_unattested_sky() -> None:
    sky_venv, tf_cache, sky_home, _empty = _seed_residue()

    result = runner.invoke(app, ["cleanup", "--yes", "--skip-jobs", "--json"])

    assert result.exit_code == 0, result.output
    payload = __import__("json").loads(result.output)
    assert payload["managed_job_queue_state"] == "SKIPPED_BY_OPERATOR"
    assert payload["result"] == "completed_with_preserved_sky"
    assert payload["verification_unresolved"] is True
    assert sky_venv.exists() and sky_home.exists()
    assert not tf_cache.exists()


def test_full_cleanup_accepts_exact_receipt_scope_after_alias_removal(
    monkeypatch,
) -> None:
    from npa import teardown_receipts

    monkeypatch.setattr("npa.clients.config.resolve_environment", lambda _project: None)
    monkeypatch.setattr(
        cleanup_cli,
        "_storage_iam_full_check",
        lambda *_a, **_k: ("verified absent", False, "verified_absent", "owned"),
    )
    teardown_receipts.record_teardown_event(
        phase="project_destroy_workflows",
        resource="all",
        terminal_state="completed",
        project_id="project-a",
    )

    result = runner.invoke(
        app,
        [
            "cleanup",
            "--project",
            "project-a",
            "--full",
            "--yes",
            "--include-sky",
            "--skip-jobs",
            "--attest-no-active-jobs",
            "--json",
        ],
    )

    assert result.exit_code != 2, result.output
    assert "durably receipted project" not in result.output
    assert __import__("json").loads(result.output)["managed_job_queue_state"] == (
        "SKIPPED_BY_OPERATOR"
    )
    assert __import__("json").loads(result.output)["verification_unresolved"] is False


def test_project_cleanup_preserves_global_runtime_and_provider_cache(
    monkeypatch,
) -> None:
    sky_venv, tf_cache, _sky_home, _empty = _seed_residue()
    _seed_project_config()
    monkeypatch.setattr(cleanup_cli, "_nonterminal_jobs", lambda _sky_bin: ([], ""))

    result = runner.invoke(
        app, ["cleanup", "--project", "prod", "--keep-sky", "--yes", "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = __import__("json").loads(result.output)
    assert payload["preserved_shared_runtime"] is True
    assert sky_venv.is_dir()
    assert tf_cache.is_dir()
    assert payload["removed_bytes"] == 0


def test_cleanup_json_distinguishes_terminal_only_queue_from_verified_empty(
    monkeypatch,
) -> None:
    _seed_residue()
    monkeypatch.setattr(
        cleanup_cli,
        "_nonterminal_jobs",
        lambda sky_bin: ([], "", "verified_terminal_only"),
    )

    result = runner.invoke(app, ["cleanup", "--json"])

    assert result.exit_code == 0, result.output
    payload = __import__("json").loads(result.output)
    assert payload["managed_job_queue_state"] == "verified_terminal_only"
    assert payload["nonterminal_job_ids"] == []


def test_cleanup_iam_note_names_the_storage_service_account(monkeypatch) -> None:
    from npa.clients import credentials as credentials_module

    credentials_module.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump(
            {"nebius": {"service_account_id": "serviceaccount-lerobot-training"}}
        )
    )

    result = runner.invoke(app, ["cleanup"])

    assert result.exit_code == 0, result.output
    assert "serviceaccount-lerobot-training" in result.output
    assert "npa storage service-account reconcile" in result.output
    assert "nebius iam service-account delete" not in result.output


def test_cleanup_iam_note_prefers_owned_storage_lifecycle_record(monkeypatch) -> None:
    from npa.clients import credentials as credentials_module

    credentials_module.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump(
            {
                "nebius": {"service_account_id": "serviceaccount-agent"},
                "storage_iam": {
                    "service_account_id": "serviceaccount-storage",
                    "service_account_name": "lerobot-training",
                    "service_account_project_id": "project-a",
                    "service_account_managed_by": "npa",
                },
            }
        )
    )

    result = runner.invoke(app, ["cleanup"])

    assert result.exit_code == 0, result.output
    assert "serviceaccount-storage" in result.output
    assert "npa storage service-account delete" in result.output
    assert "serviceaccount-agent" not in result.output


def test_cleanup_is_quiet_when_nothing_is_left() -> None:
    result = runner.invoke(app, ["cleanup"])

    assert result.exit_code == 0, result.output
    assert "No local NPA/SkyPilot residue" in result.output


def test_cleanup_full_reports_credentials_without_removing_them() -> None:
    from npa.clients import credentials as credentials_module

    credentials_module.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump(
            {
                "tokens": {
                    "HF_TOKEN": "hf_remove",
                    "NEBIUS_TOKEN_FACTORY_KEY": "tf_remove",
                },
                "ngc": {"api_key": "nvapi_remove"},
            }
        )
    )

    result = runner.invoke(app, ["cleanup", "--full", "--skip-jobs"])

    assert result.exit_code == 0, result.output
    assert "Hugging Face token" in result.output
    assert "Token Factory key" in result.output
    assert "NGC credentials" in result.output
    assert "--full --yes" in result.output
    saved = yaml.safe_load(credentials_module.CREDENTIALS_PATH.read_text())
    assert saved["tokens"]["HF_TOKEN"] == "hf_remove"


def test_cleanup_full_yes_removes_known_tokens_but_preserves_unrelated_credentials() -> (
    None
):
    from npa.clients import credentials as credentials_module

    credentials_module.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump(
            {
                "tokens": {
                    "HF_TOKEN": "hf_remove",
                    "NEBIUS_TOKEN_FACTORY_KEY": "tf_remove",
                    "CUSTOM_VENDOR_TOKEN": "keep",
                },
                "ngc": {"api_key": "nvapi_remove", "org": "remove", "extra": "keep"},
                "huggingface": {"token": "hf_alias_remove", "cache": "keep"},
                "token_factory": {"api_key": "tf_alias_remove", "endpoint": "keep"},
                "ssh": {"host": "example.invalid", "key_path": "/keys/id"},
            }
        )
    )

    result = runner.invoke(app, ["cleanup", "--full", "--yes", "--skip-jobs"])

    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(credentials_module.CREDENTIALS_PATH.read_text())
    assert saved["tokens"] == {"CUSTOM_VENDOR_TOKEN": "keep"}
    assert saved["ngc"] == {"extra": "keep"}
    assert saved["huggingface"] == {"cache": "keep"}
    assert saved["token_factory"] == {"endpoint": "keep"}
    assert saved["ssh"]["host"] == "example.invalid"
    assert "Removed locally stored Hugging Face token" in result.output


def test_cleanup_full_yes_removes_only_empty_npa_owned_tree() -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    npa_dir = Path.home() / ".npa"
    (npa_dir / "clusters").mkdir(parents=True)
    config_module.CONFIG_PATH.write_text("projects: {}\n")
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump({"tokens": {"HF_TOKEN": "hf_remove"}})
    )

    result = runner.invoke(app, ["cleanup", "--full", "--yes", "--skip-jobs"])

    assert result.exit_code == 0, result.output
    assert not config_module.CONFIG_PATH.exists()
    assert not credentials_module.CREDENTIALS_PATH.exists()
    assert npa_dir.is_dir()
    # The hardened private-YAML writer intentionally retains its owner-only
    # serialization lock even after the empty credential document is pruned.
    assert {path.name for path in npa_dir.iterdir()} == {
        "credentials.yaml.lock",
        "teardown-receipts",
    }
    assert (npa_dir / "credentials.yaml.lock").stat().st_mode & 0o777 == 0o600
    assert "Retained audit receipts" in result.output


def test_cleanup_full_preserves_nonempty_config_and_cluster_data() -> None:
    from npa.clients import config as config_module

    npa_dir = Path.home() / ".npa"
    cluster_file = npa_dir / "clusters" / "keep" / "kubeconfig"
    cluster_file.parent.mkdir(parents=True)
    cluster_file.write_text("apiVersion: v1\n")
    config_module.CONFIG_PATH.write_text(yaml.safe_dump({"custom": {"keep": True}}))

    result = runner.invoke(app, ["cleanup", "--full", "--yes", "--skip-jobs"])

    assert result.exit_code == 0, result.output
    assert cluster_file.exists()
    assert yaml.safe_load(config_module.CONFIG_PATH.read_text()) == {
        "custom": {"keep": True}
    }
    assert npa_dir.exists()


def test_cleanup_full_is_idempotent() -> None:
    first = runner.invoke(app, ["cleanup", "--full", "--yes", "--skip-jobs"])
    second = runner.invoke(app, ["cleanup", "--full", "--yes", "--skip-jobs"])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "No local NPA/SkyPilot residue" in second.output


def test_cleanup_runbook_is_complete_without_dead_skypilot_uninstall() -> None:
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert result.exit_code == 0, result.output
    assert "npa storage service-account delete --project <alias> --yes" in result.output
    assert "npa cleanup --full --yes" in result.output
    assert "npa skypilot uninstall" not in result.output


def test_cleanup_never_suggests_invalid_nebius_yes_flag() -> None:
    result = runner.invoke(app, ["cleanup", "--skip-jobs"])

    assert result.exit_code == 0, result.output
    assert "nebius iam service-account delete --id <id> --yes" not in result.output
    assert "npa storage service-account reconcile" in result.output


def test_cleanup_full_reports_a_credential_removal_failure(monkeypatch) -> None:
    from npa.cli import cleanup as cleanup_cli
    from npa.clients import credentials as credentials_module

    credentials_module.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump({"tokens": {"HF_TOKEN": "hf_keep_on_failure"}})
    )
    monkeypatch.setattr(cleanup_cli, "_clear_full_credentials", lambda: [])

    result = runner.invoke(app, ["cleanup", "--full", "--yes", "--skip-jobs"])

    assert result.exit_code == 1
    assert "could not be removed" in result.output
    assert "full local cleanup was incomplete" in result.output
    assert credentials_module.CREDENTIALS_PATH.exists()


def _seed_legacy_source_terraform_cache(monkeypatch, tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    terraform_dir = repo / "deploy" / "cluster"
    cache = terraform_dir / ".terraform" / "providers"
    cache.mkdir(parents=True)
    (cache / "provider").write_text("provider-bytes")
    for name in ("main.tf", "versions.tf", ".terraform.lock.hcl"):
        (terraform_dir / name).write_text("# test\n")
    monkeypatch.chdir(repo)
    return terraform_dir / ".terraform"


def test_cleanup_full_removes_only_validated_source_terraform_cache(
    monkeypatch, tmp_path: Path
) -> None:
    cache = _seed_legacy_source_terraform_cache(monkeypatch, tmp_path)

    report = runner.invoke(app, ["cleanup", "--full", "--skip-jobs"])
    assert report.exit_code == 0, report.output
    assert "Legacy source-checkout Terraform cache" in report.output
    assert cache.exists()

    cleanup = runner.invoke(app, ["cleanup", "--full", "--yes", "--skip-jobs"])
    assert cleanup.exit_code == 0, cleanup.output
    assert not cache.exists()

    subsequent = runner.invoke(app, ["cleanup", "--full", "--yes", "--skip-jobs"])
    assert subsequent.exit_code == 0, subsequent.output
    assert "No local NPA/SkyPilot residue" in subsequent.output


def test_cleanup_failure_keeps_terraform_residue_visible_on_the_next_run(
    monkeypatch, tmp_path: Path
) -> None:
    cache = _seed_legacy_source_terraform_cache(monkeypatch, tmp_path)
    from npa.cli.cluster import terraform_runtime

    monkeypatch.setattr(
        terraform_runtime,
        "remove_terraform_residue",
        lambda _item: "filesystem busy",
    )

    failed = runner.invoke(app, ["cleanup", "--full", "--yes", "--skip-jobs"])
    assert failed.exit_code == 1, failed.output
    assert "filesystem busy" in failed.output
    assert cache.exists()

    subsequent = runner.invoke(app, ["cleanup", "--full", "--skip-jobs"])
    assert subsequent.exit_code == 0, subsequent.output
    assert "Legacy source-checkout Terraform cache" in subsequent.output
    assert "No local NPA/SkyPilot residue" not in subsequent.output


def test_cleanup_full_reports_owned_storage_iam_as_partial(
    monkeypatch,
) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    _seed_project_config()

    credentials_module.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump(
            {
                "storage_iam": {
                    "service_account_id": "serviceaccount-storage",
                    "service_account_name": "lerobot-training",
                    "service_account_project_id": "project-a",
                    "service_account_managed_by": "npa",
                }
            }
        )
    )
    monkeypatch.setattr(
        nebius_module,
        "get_service_account_identity",
        lambda account_id, **_kwargs: nebius_module.ServiceAccountIdentity(
            account_id, "lerobot-training", "project-a", "tenant-a", ""
        ),
    )

    result = runner.invoke(app, ["cleanup", "--full", "--yes", "--skip-jobs"])

    assert result.exit_code == 2, result.output
    assert "verified present" in result.output
    assert "Full cleanup is partial" in result.output
    assert credentials_module.CREDENTIALS_PATH.exists()


def test_cleanup_full_distinguishes_provider_verification_failure(
    monkeypatch,
) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    _seed_project_config()

    credentials_module.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump(
            {
                "storage_iam": {
                    "service_account_id": "serviceaccount-storage",
                    "service_account_name": "lerobot-training",
                    "service_account_project_id": "project-a",
                    "service_account_managed_by": "npa",
                }
            }
        )
    )
    monkeypatch.setattr(
        nebius_module,
        "get_service_account_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            nebius_module.NebiusError("Unauthenticated")
        ),
    )

    result = runner.invoke(app, ["cleanup", "--full", "--yes", "--skip-jobs"])

    assert result.exit_code == 2, result.output
    assert "provider/auth verification failure" in result.output
    assert credentials_module.CREDENTIALS_PATH.exists()


def test_cleanup_full_prunes_provenance_after_verified_absence(
    monkeypatch,
) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    _seed_project_config()

    credentials_module.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump(
            {
                "storage_iam": {
                    "service_account_id": "serviceaccount-storage",
                    "service_account_name": "lerobot-training",
                    "service_account_project_id": "project-a",
                    "service_account_managed_by": "npa",
                }
            }
        )
    )
    monkeypatch.setattr(
        nebius_module, "get_service_account_identity", lambda *_args, **_kwargs: None
    )

    result = runner.invoke(app, ["cleanup", "--full", "--yes", "--skip-jobs"])

    assert result.exit_code == 0, result.output
    assert "verified absence" in result.output
    assert not credentials_module.CREDENTIALS_PATH.exists()


def test_cleanup_full_json_repeats_monotonically_while_iam_is_unresolved(
    monkeypatch,
) -> None:
    import json

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    _seed_project_config()
    credentials_module.CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    credentials_module.CREDENTIALS_PATH.write_text(
        yaml.safe_dump(
            {
                "nebius": {"service_account_id": "serviceaccount-storage"},
                "storage_iam": {
                    "service_account_id": "serviceaccount-storage",
                    "service_account_name": "lerobot-training",
                    "service_account_project_id": "project-a",
                    "service_account_managed_by": "",
                },
            }
        )
    )
    monkeypatch.setattr(
        nebius_module,
        "get_service_account_identity",
        lambda account_id, **_kwargs: nebius_module.ServiceAccountIdentity(
            account_id, "lerobot-training", "project-a", "tenant-a", ""
        ),
    )

    command = ["cleanup", "--full", "--yes", "--skip-jobs", "--json"]
    first = runner.invoke(app, command)
    second = runner.invoke(app, command)

    assert first.exit_code == second.exit_code == 2
    first_payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    assert (
        first_payload["result"]
        == second_payload["result"]
        == ("locally_clean_cloud_iam_unresolved")
    )
    assert first_payload["iam_verification_required"] is True
    assert second_payload["project_retained"] is True
    assert [phase["phase"] for phase in first_payload["phases"]] == list(range(1, 10))
    assert all(
        {
            "phase",
            "resource",
            "observed_state",
            "recommended_npa_command",
            "safety_status",
            "ownership_status",
            "operator_action_required",
        }
        <= phase.keys()
        for phase in first_payload["phases"]
    )
    commands = "\n".join(
        phase["recommended_npa_command"] for phase in first_payload["phases"]
    )
    assert "npa workflow cancel" in commands
    assert "npa agent destroy" in commands
    assert "npa cluster down" in commands
    assert "npa storage" in commands
    assert all(
        forbidden not in commands
        for forbidden in ("kubectl ", "terraform ", "sky jobs", "nebius ")
    )
    config = yaml.safe_load(config_module.CONFIG_PATH.read_text())
    assert "prod" in config["projects"]
    assert (
        config["projects"]["prod"]["storage_iam_verification_required"][
            "service_account_id"
        ]
        == "serviceaccount-storage"
    )

    monkeypatch.setattr(
        nebius_module, "get_service_account_identity", lambda *_args, **_kwargs: None
    )
    resolved = runner.invoke(app, command)
    assert resolved.exit_code == 0, resolved.output
    resolved_payload = json.loads(resolved.output)
    assert resolved_payload["result"] == "fully_cleaned"
    iam_phase = next(
        phase
        for phase in resolved_payload["phases"]
        if phase["resource"] == "storage IAM"
    )
    assert iam_phase["observed_state"] == "verified_deleted_or_absent"
    assert iam_phase["operator_action_required"] is False
    local_phase = next(
        phase
        for phase in resolved_payload["phases"]
        if phase["resource"] == "local caches and known credentials"
    )
    assert local_phase["operator_action_required"] is False
    config = yaml.safe_load(config_module.CONFIG_PATH.read_text())
    assert "storage_iam_verification_required" not in config["projects"]["prod"]
