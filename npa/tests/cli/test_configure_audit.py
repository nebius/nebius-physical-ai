from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import typer
from typer.testing import CliRunner
import yaml

from npa.cli import main as cli_main
from npa.cli.main import app
from npa.clients import config as config_module
from npa.clients import credentials as credentials_module
from npa.clients.huggingface import HFAccessResult
from npa.clients.project_credential_store import SCHEMA_VERSION


runner = CliRunner()


def test_init_and_configure_expose_the_same_option_contract() -> None:
    configure_help = runner.invoke(app, ["configure", "--help"])
    init_help = runner.invoke(app, ["init", "--help"])

    assert configure_help.exit_code == 0
    assert init_help.exit_code == 0
    root = typer.main.get_command(app)
    configure_options = {
        option
        for parameter in root.commands["configure"].params
        for option in parameter.opts + parameter.secondary_opts
    }
    init_options = {
        option
        for parameter in root.commands["init"].params
        for option in parameter.opts + parameter.secondary_opts
    }
    assert configure_options == init_options
    assert "Deprecated alias" in init_help.output
    assert "--allow-visible-secret-input" in configure_options


def _snapshot(root: Path) -> dict[str, tuple[int, bytes]]:
    return {
        str(path.relative_to(root)): (path.stat().st_mode & 0o777, path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _subprocess_environment(home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "PATH": os.environ["PATH"],
        "NPA_CONFIG_DIR": str(home / ".npa"),
        "HF_TOKEN": "hf_synthetic_process_secret",
        "NGC_API_KEY": "nvapi-synthetic-process-secret",
        "NPA_OPERATION_JOURNAL_DIR": str(home / "operation-state"),
        "NPA_TEARDOWN_RECEIPT_DIR": str(home / "teardown-receipts"),
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["configure", "--show", "--src-s3-uri", "s3://synthetic/source"],
        ["configure", "--show", "--forget-project", "saved"],
        ["configure", "--show", "--save-env-credentials"],
        [
            "configure",
            "--show",
            "--tenant-id",
            "tenant-b",
            "--project-id",
            "project-b",
            "--region",
            "region-b",
            "--project-alias",
            "project-b",
        ],
        ["configure", "--show", "--interactive"],
        ["configure", "--show", "--provision"],
        ["configure", "--env", "--no-provision"],
        [
            "configure",
            "--src-s3-uri",
            "s3://synthetic/source",
            "--forget-project",
            "saved",
        ],
        ["init", "--show", "--save-env-credentials"],
    ],
)
def test_rejected_mode_combinations_are_read_only_in_subprocess(
    tmp_path: Path, arguments: list[str]
) -> None:
    home = tmp_path / "home"
    dot_npa = home / ".npa"
    dot_npa.mkdir(parents=True)
    (dot_npa / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "default_project": "saved",
                "projects": {
                    "saved": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "region-a",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (dot_npa / "credentials.yaml").write_text(
        yaml.safe_dump({"tokens": {"HF_TOKEN": "hf_existing_synthetic"}}),
        encoding="utf-8",
    )
    (dot_npa / "credentials.yaml").chmod(0o600)
    for directory, marker in (
        (home / "operation-state", "operation-marker"),
        (home / "teardown-receipts", "receipt-marker"),
    ):
        directory.mkdir()
        (directory / marker).write_text("unchanged\n", encoding="utf-8")
    before = _snapshot(home)

    result = subprocess.run(
        [sys.executable, "-m", "npa.cli.main", *arguments],
        text=True,
        capture_output=True,
        env=_subprocess_environment(home),
        check=False,
    )

    assert result.returncode == 2, (result.stdout, result.stderr)
    assert _snapshot(home) == before
    assert "hf_synthetic_process_secret" not in result.stdout + result.stderr
    assert "nvapi-synthetic-process-secret" not in result.stdout + result.stderr


def test_forced_interactive_non_tty_refuses_visible_secret_input_before_mutation(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    (home / ".npa").mkdir(parents=True)
    (home / ".npa" / "config.yaml").write_text("projects: {}\n", encoding="utf-8")
    before = _snapshot(home)

    result = subprocess.run(
        [sys.executable, "-m", "npa.cli.main", "configure", "--interactive"],
        input="synthetic-visible-secret\n",
        text=True,
        capture_output=True,
        env=_subprocess_environment(home),
        check=False,
    )

    assert result.returncode == 2
    assert "Refusing forced interactive setup on non-TTY" in result.stderr
    assert "synthetic-visible-secret" not in result.stdout + result.stderr
    assert _snapshot(home) == before


def _point_configure_at_tmp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path]:
    config_path = tmp_path / "config.yaml"
    credentials_path = tmp_path / "credentials.yaml"
    operations = tmp_path / "operations"
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(operations))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    for name in credentials_module.SUPPORTED_ENV_CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("NPA_ALLOW_VISIBLE_SECRET_INPUT", raising=False)
    return config_path, credentials_path, operations


def _config_document(project_id: str, *, alias: str = "selected") -> dict:
    return {
        "default_project": alias,
        "projects": {
            alias: {
                "project_id": project_id,
                "tenant_id": f"tenant-{project_id}",
                "region": "synthetic-region1",
            }
        },
    }


def _storage(bucket: str) -> dict[str, str]:
    return {
        "bucket": f"s3://{bucket}/",
        "endpoint_url": "https://storage.synthetic.invalid",
        "aws_access_key_id": f"access-{bucket}",
        "aws_secret_access_key": f"secret-{bucket}",
    }


@pytest.mark.parametrize("empty_record", [True, False])
def test_show_env_never_uses_another_projects_compatibility_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, empty_record: bool
) -> None:
    config_path, credentials_path, _operations = _point_configure_at_tmp(
        monkeypatch, tmp_path
    )
    config_path.write_text(
        yaml.safe_dump(_config_document("project-b")), encoding="utf-8"
    )
    projects = {
        "project-a": {"project_id": "project-a", "storage": _storage("bucket-a")}
    }
    if empty_record:
        projects["project-b"] = {"project_id": "project-b"}
    credentials_path.write_text(
        yaml.safe_dump(
            {
                # Deliberately stale compatibility view for project A.
                "storage": _storage("bucket-a"),
                "project_credentials": {
                    "schema_version": SCHEMA_VERSION,
                    "current_project_id": "project-a",
                    "projects": projects,
                },
            }
        ),
        encoding="utf-8",
    )
    credentials_path.chmod(0o600)
    monkeypatch.setattr(
        "npa.cluster.state.list_local_clusters",
        lambda: [type("State", (), {"name": "cluster-a", "project_id": "project-a"})()],
    )

    result = runner.invoke(app, ["configure", "--show", "--env"])

    assert result.exit_code == 0, result.output
    assert "NPA_PROJECT_ID=project-b" in result.output
    assert "NPA_BUCKET" not in result.output
    assert "NPA_S3_ENDPOINT" not in result.output
    assert "NPA_KUBE_CONTEXT" not in result.output
    assert "bucket-a" not in result.output


@pytest.mark.parametrize("contents", ["projects: {}\n", "projects: [broken\n"])
def test_show_env_fails_closed_without_readable_exact_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contents: str
) -> None:
    config_path, _credentials_path, _operations = _point_configure_at_tmp(
        monkeypatch, tmp_path
    )
    config_path.write_text(contents, encoding="utf-8")

    result = runner.invoke(app, ["configure", "--env"])

    assert result.exit_code == 2
    assert "refusing" in result.output.lower()


def test_show_hides_storage_when_multiple_projects_have_no_exact_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, credentials_path, _operations = _point_configure_at_tmp(
        monkeypatch, tmp_path
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "missing",
                "projects": {
                    "a": {"project_id": "project-a"},
                    "b": {"project_id": "project-b"},
                },
            }
        ),
        encoding="utf-8",
    )
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {"HF_TOKEN": "hf_synthetic_present"},
                "storage": _storage("wrong-project-bucket"),
            }
        ),
        encoding="utf-8",
    )
    credentials_path.chmod(0o600)

    result = runner.invoke(app, ["configure", "--show"])

    assert result.exit_code == 0, result.output
    assert "no exact default selected; storage hidden" in result.output
    assert "wrong-project-bucket" not in result.output
    assert "HF token:" in result.output


def test_show_reports_credential_presence_without_project_stanzas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, credentials_path, _operations = _point_configure_at_tmp(
        monkeypatch, tmp_path
    )
    config_path.write_text("projects: {}\n", encoding="utf-8")
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {
                    "HF_TOKEN": "hf_synthetic_present",
                    "NEBIUS_TOKEN_FACTORY_KEY": "v1.synthetic-present",
                },
                "ngc": {"api_key": "nvapi-synthetic-present"},
            }
        ),
        encoding="utf-8",
    )
    credentials_path.chmod(0o600)

    result = runner.invoke(app, ["configure", "--show"])

    assert result.exit_code == 0, result.output
    assert "(no projects" in result.output
    assert "HF token:" in result.output and "set" in result.output
    assert "Token Factory key:" in result.output
    assert "NGC API key:" in result.output
    assert "hf_synthetic_present" not in result.output


def test_known_project_no_provision_ignores_and_deselects_stale_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, credentials_path, _operations = _point_configure_at_tmp(
        monkeypatch, tmp_path
    )
    config_path.write_text(
        yaml.safe_dump(_config_document("project-a", alias="old")), encoding="utf-8"
    )
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "storage": _storage("bucket-a"),
                "storage_iam": {
                    "service_account_id": "service-account-a",
                    "service_account_project_id": "project-a",
                    "service_account_managed_by": "npa",
                },
            }
        ),
        encoding="utf-8",
    )

    def forbidden(*args, **kwargs):
        pytest.fail("--no-provision must not contact providers or probe storage")

    monkeypatch.setattr("npa.clients.nebius.get_iam_token", forbidden)
    monkeypatch.setattr("npa.clients.nebius.set_profile_project", forbidden)
    monkeypatch.setattr("npa.clients.storage_validation.probe_storage_write", forbidden)
    monkeypatch.setattr(cli_main, "_provision_object_storage", forbidden)

    result = runner.invoke(
        app,
        [
            "configure",
            "--no-interactive",
            "--no-provision",
            "--tenant-id",
            "tenant-b",
            "--project-id",
            "project-b",
            "--region",
            "synthetic-region1",
            "--project-alias",
            "new",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        len([line for line in result.output.splitlines() if line.startswith("[NOTE]")])
        == 1
    )
    saved = yaml.safe_load(credentials_path.read_text())
    assert "storage" not in saved
    root = saved["project_credentials"]
    assert root["current_project_id"] == "project-b"
    assert root["projects"]["project-a"]["storage"]["bucket"] == "s3://bucket-a/"
    assert "storage" not in root["projects"]["project-b"]
    configured = yaml.safe_load(config_path.read_text())
    assert configured["default_project"] == "new"
    assert configured["projects"]["new"]["project_id"] == "project-b"


def test_known_project_no_provision_preserves_but_deactivates_exact_stale_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.clients.project_credential_store import write_project_credentials

    config_path, credentials_path, _operations = _point_configure_at_tmp(
        monkeypatch, tmp_path
    )
    config_path.write_text(
        yaml.safe_dump(_config_document("project-a")), encoding="utf-8"
    )
    write_project_credentials(
        "project-a",
        {"storage": _storage("stale-bucket-a")},
        alias="selected",
        path=credentials_path,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("project-only setup must not authenticate or probe stale storage")

    monkeypatch.setattr("npa.clients.nebius.get_iam_token", forbidden)
    monkeypatch.setattr("npa.clients.nebius.set_profile_project", forbidden)
    monkeypatch.setattr("npa.clients.storage_validation.probe_storage_write", forbidden)

    result = runner.invoke(
        app,
        [
            "configure",
            "--no-provision",
            "--tenant-id",
            "tenant-project-a",
            "--project-id",
            "project-a",
            "--region",
            "synthetic-region1",
            "--project-alias",
            "selected",
        ],
    )

    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(credentials_path.read_text())
    exact = saved["project_credentials"]["projects"]["project-a"]
    assert exact["storage"]["bucket"] == "s3://stale-bucket-a/"
    assert exact["storage_selected"] is False
    assert "storage" not in saved
    assert config_module.resolve_project_storage("selected").checkpoint_bucket == ""

    display = runner.invoke(app, ["configure", "--show", "--env"])
    assert display.exit_code == 0, display.output
    assert "NPA_BUCKET" not in display.output
    assert "stale-bucket-a" not in display.output


def test_known_project_alias_repoint_is_rejected_before_journal_or_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, credentials_path, operations = _point_configure_at_tmp(
        monkeypatch, tmp_path
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "shared",
                "projects": {
                    "shared": {
                        "project_id": "project-a",
                        "terraform_state": {"bucket": "state-a"},
                        "workbenches": {"agent": {"id": "vm-a"}},
                        "container_registry": "registry-a.synthetic.invalid/repo",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    credentials_path.write_text("tokens: {}\n", encoding="utf-8")
    before = (config_path.read_bytes(), credentials_path.read_bytes())

    result = runner.invoke(
        app,
        [
            "configure",
            "--no-provision",
            "--tenant-id",
            "tenant-b",
            "--project-id",
            "project-b",
            "--region",
            "synthetic-region1",
            "--project-alias",
            "shared",
        ],
    )

    assert result.exit_code == 2
    assert "Choose a new --project-alias." in " ".join(result.output.split())
    assert (config_path.read_bytes(), credentials_path.read_bytes()) == before
    assert not operations.exists()


@pytest.mark.parametrize(
    ("error_kind", "status_code", "expected"),
    [
        ("authentication", 401, "HF has no access"),
        ("entitlement", 403, "HF has no access"),
        ("catalog_drift", 404, "HF has no access"),
        ("transient", None, "unverified"),
    ],
)
def test_credential_import_access_outcomes_are_one_advisory_and_never_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_kind: str,
    status_code: int | None,
    expected: str,
) -> None:
    _config_path, credentials_path, _operations = _point_configure_at_tmp(
        monkeypatch, tmp_path
    )
    monkeypatch.setenv("HF_TOKEN", "hf_synthetic_import")

    def validate(token, repo, repo_type="model", **kwargs):
        return HFAccessResult(
            repo=repo,
            ok=False,
            status_code=status_code,
            error="synthetic result",
            error_kind=error_kind,
        )

    monkeypatch.setattr("npa.clients.huggingface.validate_hf_access", validate)
    monkeypatch.setattr(
        "npa.workbench.nurec.nurec.check_ngc_image_access",
        lambda key, **kwargs: "reachable",
    )
    monkeypatch.setattr(
        cli_main,
        "_run_interactive_configure",
        lambda **kwargs: pytest.fail("credential import must not prompt"),
    )

    result = runner.invoke(app, ["configure", "--save-env-credentials"])

    assert result.exit_code == 0, result.output
    notes = [line for line in result.output.splitlines() if line.startswith("[NOTE]")]
    assert len(notes) == 1
    assert expected in notes[0]
    assert "hf_synthetic_import" not in result.output
    assert yaml.safe_load(credentials_path.read_text())["tokens"]["HF_TOKEN"] == (
        "hf_synthetic_import"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["configure", "--src-s3-uri", "s3://synthetic/source"],
        [
            "configure",
            "--no-provision",
            "--tenant-id",
            "tenant-a",
            "--project-id",
            "project-a",
            "--region",
            "synthetic-region1",
            "--project-alias",
            "project-a",
        ],
    ],
)
def test_successful_prompt_free_configuration_paths_emit_one_advisory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arguments: list[str]
) -> None:
    _point_configure_at_tmp(monkeypatch, tmp_path)

    result = runner.invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert (
        len([line for line in result.output.splitlines() if line.startswith("[NOTE]")])
        == 1
    )


def test_successful_exit_after_config_write_commits_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, _credentials_path, operations = _point_configure_at_tmp(
        monkeypatch, tmp_path
    )

    def completed_write(uri: str) -> None:
        config_module.write_config({"src_s3_uri": uri})
        raise cli_main.typer.Exit(code=0)

    monkeypatch.setattr(cli_main, "_store_src_s3_uri", completed_write)

    result = runner.invoke(app, ["configure", "--src-s3-uri", "s3://synthetic/source"])

    assert result.exit_code == 0, result.output
    assert config_path.exists()
    journals = list(operations.glob("*/journal.json"))
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text())
    assert journal["phase"] == "committed"
    assert journal["lifecycle"] == "succeeded"
    assert "recovery-required" not in journals[0].read_text()


def test_failed_exit_still_marks_configure_journal_recoverable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _config_path, _credentials_path, operations = _point_configure_at_tmp(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(
        cli_main,
        "_store_src_s3_uri",
        lambda uri: (_ for _ in ()).throw(cli_main.typer.Exit(code=1)),
    )

    result = runner.invoke(app, ["configure", "--src-s3-uri", "s3://synthetic/source"])

    assert result.exit_code == 1
    journals = list(operations.glob("*/journal.json"))
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text())
    assert journal["phase"] == "recovery-required"
    assert journal["last_error"] == "Exit"
