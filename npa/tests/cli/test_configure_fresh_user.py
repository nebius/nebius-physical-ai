from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli import main as cli_main
from npa.cli.main import app
from npa.clients import config as config_module
from npa.clients import credentials as credentials_module
from npa.clients import nebius
from npa.clients.huggingface import HFAccessResult


runner = CliRunner()


def _point_configure_at_tmp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    config_path = tmp_path / "config.yaml"
    credentials_path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(tmp_path / "operations"))
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    for name in credentials_module.SUPPORTED_ENV_CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    return config_path, credentials_path


def _known_project_args(*, provision: bool) -> list[str]:
    return [
        "configure",
        "--no-interactive",
        "--provision" if provision else "--no-provision",
        "--tenant-id",
        "tenant-synthetic",
        "--project-id",
        "project-synthetic",
        "--region",
        "eu-north1",
        "--project-alias",
        "fresh",
    ]


def test_display_rejects_provisioning_before_any_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, credentials_path = _point_configure_at_tmp(monkeypatch, tmp_path)
    config_path.write_text("projects: {}\n", encoding="utf-8")
    credentials_path.write_text("tokens: {}\n", encoding="utf-8")
    credentials_path.chmod(0o600)
    before = (config_path.read_bytes(), credentials_path.read_bytes())

    result = runner.invoke(app, ["configure", "--show", "--provision"])

    assert result.exit_code == 2
    assert "display mode is read-only or self-contained" in " ".join(
        result.output.split()
    )
    assert (config_path.read_bytes(), credentials_path.read_bytes()) == before


def test_known_project_no_provision_is_provider_free_and_deselects_old_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, credentials_path = _point_configure_at_tmp(monkeypatch, tmp_path)
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "storage": {
                    "bucket": "s3://old-synthetic-bucket/",
                    "endpoint_url": "https://storage.example.invalid",
                    "aws_access_key_id": "old-access",
                    "aws_secret_access_key": "old-secret",
                }
            }
        ),
        encoding="utf-8",
    )
    credentials_path.chmod(0o600)

    def forbidden(*_args, **_kwargs):
        pytest.fail("project-only configure must not contact a provider")

    monkeypatch.setattr(nebius, "get_iam_token", forbidden)
    monkeypatch.setattr(nebius, "set_profile_project", forbidden)
    monkeypatch.setattr(nebius, "get_project_name", forbidden)
    monkeypatch.setattr(cli_main, "_provision_object_storage", forbidden)
    monkeypatch.setattr(
        "npa.clients.storage_validation.probe_storage_write", forbidden
    )
    monkeypatch.setattr(
        cli_main,
        "_saved_model_access_note",
        lambda: "[NOTE] Credential access is informational.",
        raising=False,
    )

    result = runner.invoke(app, _known_project_args(provision=False))

    assert result.exit_code == 0, result.output
    assert "Intent: save project configuration only" in result.output
    assert "Summary:" in result.output
    assert "storage=not selected" in result.output
    saved_config = yaml.safe_load(config_path.read_text())
    assert saved_config["default_project"] == "fresh"
    assert saved_config["projects"]["fresh"]["project_id"] == "project-synthetic"
    saved_credentials = yaml.safe_load(credentials_path.read_text())
    assert "storage" not in saved_credentials
    root = saved_credentials["project_credentials"]
    assert root["current_project_id"] == "project-synthetic"
    assert root["projects"]["project-synthetic"]["storage_selected"] is False
    assert "old-synthetic-bucket" not in result.output


def test_known_project_no_provision_disables_retained_exact_project_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _config_path, credentials_path = _point_configure_at_tmp(monkeypatch, tmp_path)
    retained_storage = {
        "bucket": "s3://retained-synthetic-bucket/",
        "endpoint_url": "https://storage.example.invalid",
        "aws_access_key_id": "retained-access",
        "aws_secret_access_key": "retained-secret",
    }
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "project_credentials": {
                    "schema_version": "npa.project-credentials.v2",
                    "current_project_id": "project-synthetic",
                    "projects": {
                        "project-synthetic": {
                            "project_id": "project-synthetic",
                            "aliases": ["fresh"],
                            "storage_selected": True,
                            "storage": retained_storage,
                        }
                    },
                },
                "storage": retained_storage,
            }
        ),
        encoding="utf-8",
    )
    credentials_path.chmod(0o600)

    def forbidden(*_args, **_kwargs):
        pytest.fail("project-only configure must not contact a provider")

    monkeypatch.setattr(nebius, "get_iam_token", forbidden)
    monkeypatch.setattr(nebius, "set_profile_project", forbidden)
    monkeypatch.setattr(nebius, "get_project_name", forbidden)
    monkeypatch.setattr(cli_main, "_provision_object_storage", forbidden)
    monkeypatch.setattr(
        "npa.clients.storage_validation.probe_storage_write", forbidden
    )

    result = runner.invoke(app, _known_project_args(provision=False))

    assert result.exit_code == 0, result.output
    saved_credentials = yaml.safe_load(credentials_path.read_text())
    retained_record = saved_credentials["project_credentials"]["projects"][
        "project-synthetic"
    ]
    assert retained_record["storage_selected"] is False
    assert retained_record["storage"] == retained_storage
    resolved = config_module.resolve_project_storage("fresh")
    assert resolved.checkpoint_bucket == ""
    assert resolved.endpoint_url == ""
    assert resolved.aws_access_key_id == ""
    assert resolved.aws_secret_access_key == ""


def test_known_project_provision_is_explicit_and_summarized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_configure_at_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr(nebius, "get_iam_token", lambda: "synthetic-iam")
    monkeypatch.setattr(nebius, "set_profile_project", lambda *_args: True)
    provision_calls: list[str] = []

    def provision(*_args, **kwargs):
        provision_calls.append(kwargs["project_id"])
        return {
            "aws_access_key_id": "synthetic-access",
            "aws_secret_access_key": "synthetic-secret",
            "endpoint_url": "https://storage.example.invalid",
            "bucket": "s3://new-synthetic-bucket/",
            "_validated": "true",
            "_disposition": "created",
        }

    monkeypatch.setattr(cli_main, "_provision_object_storage", provision)
    monkeypatch.setattr(
        cli_main,
        "_saved_model_access_note",
        lambda: "[NOTE] Credential access is informational.",
        raising=False,
    )

    result = runner.invoke(app, _known_project_args(provision=True))

    assert result.exit_code == 0, result.output
    assert provision_calls == ["project-synthetic"]
    assert "Intent: provision object storage explicitly" in result.output
    assert "Summary:" in result.output
    assert "storage=created" in result.output


@pytest.mark.parametrize("failure_phase", ["authentication", "storage-probe"])
def test_failed_known_project_provision_preserves_prior_credential_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_phase: str,
) -> None:
    _config_path, credentials_path = _point_configure_at_tmp(monkeypatch, tmp_path)
    prior_storage = {
        "aws_access_key_id": "prior-access",
        "aws_secret_access_key": "prior-secret",
        "endpoint_url": "https://storage.example.invalid",
        "bucket": "s3://prior-bucket/",
    }
    target_storage = {
        "aws_access_key_id": "target-access",
        "aws_secret_access_key": "target-secret",
        "endpoint_url": "https://storage.example.invalid",
        "bucket": "s3://target-bucket/",
    }
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "storage": prior_storage,
                "storage_iam": {
                    "service_account_id": "prior-account",
                    "service_account_name": "lerobot-training",
                    "service_account_project_id": "prior-project",
                    "service_account_managed_by": "npa",
                },
                "project_credentials": {
                    "schema_version": "npa.project-credentials.v2",
                    "current_project_id": "prior-project",
                    "projects": {
                        "prior-project": {
                            "project_id": "prior-project",
                            "storage_selected": True,
                            "storage": prior_storage,
                            "storage_iam": {
                                "service_account_id": "prior-account",
                                "service_account_name": "lerobot-training",
                                "service_account_project_id": "prior-project",
                                "service_account_managed_by": "npa",
                            },
                        },
                        "project-synthetic": {
                            "project_id": "project-synthetic",
                            "storage_selected": False,
                            "storage": target_storage,
                            "storage_iam": {
                                "service_account_id": "target-account",
                                "service_account_name": "lerobot-training",
                                "service_account_project_id": "project-synthetic",
                                "service_account_managed_by": "npa",
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    credentials_path.chmod(0o600)

    if failure_phase == "authentication":
        monkeypatch.setattr(
            nebius,
            "get_iam_token",
            lambda: (_ for _ in ()).throw(nebius.NebiusError("authentication failed")),
        )
    else:
        monkeypatch.setattr(nebius, "get_iam_token", lambda: "synthetic-iam")
        monkeypatch.setattr(
            "npa.clients.storage_validation.probe_storage_write",
            lambda **_kwargs: type("Probe", (), {"ok": False})(),
        )
        monkeypatch.setattr(cli_main, "_provision_object_storage", lambda *_a, **_k: None)

    result = runner.invoke(app, _known_project_args(provision=True))

    assert result.exit_code == 2, result.output
    saved = yaml.safe_load(credentials_path.read_text())
    root = saved["project_credentials"]
    assert root["current_project_id"] == "prior-project"
    assert root["projects"]["project-synthetic"]["storage_selected"] is False
    assert saved["storage"]["bucket"] == "s3://prior-bucket/"


def test_interactive_no_provision_has_clear_provider_free_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _point_configure_at_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: False)

    def forbidden(*_args, **_kwargs):
        pytest.fail("interactive --no-provision must not contact a provider")

    monkeypatch.setattr(nebius, "get_project_name", forbidden)
    monkeypatch.setattr(nebius, "get_iam_token", forbidden)
    monkeypatch.setattr(nebius, "set_profile_project", forbidden)
    monkeypatch.setattr(cli_main, "_provision_object_storage", forbidden)
    monkeypatch.setattr(
        "npa.clients.storage_validation.probe_storage_write", forbidden
    )
    access_probe_calls: list[tuple[str, str]] = []

    def access_note(hf_token: str, ngc_key: str) -> str:
        access_probe_calls.append((hf_token, ngc_key))
        return "[NOTE] HF token missing; NGC key missing (informational)."

    monkeypatch.setattr(cli_main, "_model_access_note", access_note)
    answers = "\n".join(
        ["tenant-synthetic", "project-synthetic", "", "", "", ""]
    ) + "\n"

    result = runner.invoke(
        app,
        [
            "configure",
            "--interactive",
            "--no-provision",
        ],
        input=answers,
    )

    assert result.exit_code == 0, result.output
    assert "Intent: interactive project configuration only" in result.output
    assert "Summary:" in result.output
    assert "storage=not selected" in result.output
    assert "access probes skipped by --no-provision" in result.output
    assert access_probe_calls == []


@pytest.mark.parametrize(
    ("hf_identity", "hf_asset_status", "ngc_outcome", "expected"),
    [
        (None, None, None, ("HF token missing", "NGC key missing")),
        (
            HFAccessResult(repo="whoami-v2", ok=True, status_code=200),
            200,
            "reachable",
            ("HF token valid", "gated access confirmed", "NGC key valid"),
        ),
        (
            HFAccessResult(repo="whoami-v2", ok=False, status_code=401),
            None,
            "auth-401",
            ("HF token rejected", "NGC key rejected"),
        ),
        (
            HFAccessResult(repo="whoami-v2", ok=True, status_code=200),
            403,
            "entitlement-required",
            ("HF token valid", "gated access missing", "NGC entitlement denied"),
        ),
        (
            HFAccessResult(repo="whoami-v2", ok=False, error="network unavailable"),
            None,
            "unreachable",
            ("HF provider/network unavailable", "NGC provider/network unavailable"),
        ),
    ],
)
def test_access_advisory_reports_each_outcome_without_gating_or_secret_leak(
    monkeypatch: pytest.MonkeyPatch,
    hf_identity: HFAccessResult | None,
    hf_asset_status: int | None,
    ngc_outcome: str | None,
    expected: tuple[str, ...],
) -> None:
    hf_token = "hf_synthetic_secret" if hf_identity is not None else ""
    ngc_key = "nvapi-synthetic-secret" if ngc_outcome is not None else ""
    monkeypatch.setattr(
        "npa.clients.huggingface.validate_hf_identity",
        lambda *_args, **_kwargs: hf_identity,
    )

    def probe(_validator, _token, assets, **_kwargs):
        return {
            (asset.repo, asset.repo_type, asset.revision, asset.probe_path): HFAccessResult(
                repo=asset.repo,
                ok=hf_asset_status == 200,
                status_code=hf_asset_status,
                error=f"provider diagnostic {hf_token}",
            )
            for asset in assets
        }

    monkeypatch.setattr(cli_main, "_probe_hf_assets_parallel", probe)
    monkeypatch.setattr(
        "npa.workbench.nurec.nurec.check_ngc_image_access",
        lambda *_args, **_kwargs: ngc_outcome,
    )

    note = cli_main._model_access_note(hf_token, ngc_key)

    assert note.startswith("[NOTE] Access checks are informational")
    assert all(fragment in note for fragment in expected)
    if hf_token:
        assert hf_token not in note
    if ngc_key:
        assert ngc_key not in note


def test_access_probe_exceptions_are_informative_and_never_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("provider failed with hf_synthetic_secret")

    monkeypatch.setattr(
        "npa.clients.huggingface.validate_hf_identity", unavailable
    )
    monkeypatch.setattr(
        "npa.workbench.nurec.nurec.check_ngc_image_access", unavailable
    )

    note = cli_main._model_access_note(
        "hf_synthetic_secret", "nvapi-synthetic-secret"
    )

    assert "HF provider/network unavailable" in note
    assert "NGC provider/network unavailable" in note
    assert "hf_synthetic_secret" not in note
    assert "nvapi-synthetic-secret" not in note


def test_interactive_configure_survives_access_advisory_construction_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, _credentials_path = _point_configure_at_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    monkeypatch.setattr(nebius, "current_project_id", lambda: "project-synthetic")
    monkeypatch.setattr(nebius, "current_tenant_id", lambda: "tenant-synthetic")
    monkeypatch.setattr(nebius, "list_projects_in_tenant", lambda *_args: [])
    monkeypatch.setattr(
        cli_main,
        "_provision_object_storage",
        lambda *_args, **_kwargs: {
            "aws_access_key_id": "synthetic-access",
            "aws_secret_access_key": "synthetic-storage-secret",
            "endpoint_url": "https://storage.example.invalid",
            "bucket": "s3://new-synthetic-bucket/",
            "_validated": "true",
            "_disposition": "created",
        },
    )
    monkeypatch.setattr(
        cli_main,
        "_prompt_setup_tokens",
        lambda *_args, **_kwargs: (
            "hf_synthetic_secret",
            "",
            "nvapi-synthetic-secret",
        ),
    )

    def asset_discovery_failed() -> list[object]:
        raise RuntimeError("asset discovery echoed hf_synthetic_secret")

    monkeypatch.setattr(
        "npa.workbench.model_access.gated_hf_assets", asset_discovery_failed
    )

    result = runner.invoke(
        app,
        ["configure", "--interactive", "--provision"],
        input="\n\n\n",
    )

    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(config_path.read_text())
    selected = saved["projects"][saved["default_project"]]
    assert selected["project_id"] == "project-synthetic"
    assert "configuration=saved" in result.output
    assert "Access checks are informational" in result.output
    assert "advisory unavailable" in result.output
    assert "hf_synthetic_secret" not in result.output
    assert "nvapi-synthetic-secret" not in result.output


def test_environment_credential_import_saves_then_reports_advisory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _config_path, credentials_path = _point_configure_at_tmp(monkeypatch, tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_synthetic_import_secret")
    monkeypatch.setenv("NGC_API_KEY", "nvapi-synthetic-import-secret")
    monkeypatch.setattr(
        cli_main,
        "_saved_model_access_note",
        lambda: "[NOTE] Access checks are informational; HF token rejected; NGC key rejected.",
        raising=False,
    )

    result = runner.invoke(app, ["configure", "--save-env-credentials"])

    assert result.exit_code == 0, result.output
    assert "values redacted" in result.output
    assert "Access checks are informational" in result.output
    assert "hf_synthetic_import_secret" not in result.output
    assert "nvapi-synthetic-import-secret" not in result.output
    saved = yaml.safe_load(credentials_path.read_text())
    assert saved["tokens"]["HF_TOKEN"] == "hf_synthetic_import_secret"
    assert saved["ngc"]["api_key"] == "nvapi-synthetic-import-secret"


def test_prompt_free_project_and_environment_import_are_one_provider_free_intent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, credentials_path = _point_configure_at_tmp(monkeypatch, tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_synthetic_combined_secret")

    def forbidden(*_args, **_kwargs):
        pytest.fail("prompt-free project-only configure must not contact a provider")

    monkeypatch.setattr(nebius, "get_iam_token", forbidden)
    monkeypatch.setattr(nebius, "set_profile_project", forbidden)
    monkeypatch.setattr(cli_main, "_provision_object_storage", forbidden)
    access_probe_calls: list[str] = []

    def access_note() -> str:
        access_probe_calls.append("called")
        return "[NOTE] Access checks are informational; HF token valid."

    monkeypatch.setattr(cli_main, "_saved_model_access_note", access_note, raising=False)

    result = runner.invoke(
        app, [*_known_project_args(provision=False), "--save-env-credentials"]
    )

    assert result.exit_code == 0, result.output
    assert "values redacted" in result.output
    assert "Intent: save project configuration only" in result.output
    assert "mode=project-only" in result.output
    assert "hf_synthetic_combined_secret" not in result.output
    assert yaml.safe_load(config_path.read_text())["default_project"] == "fresh"
    assert (
        yaml.safe_load(credentials_path.read_text())["tokens"]["HF_TOKEN"]
        == "hf_synthetic_combined_secret"
    )
    assert "access probes skipped by --no-provision" in result.output
    assert access_probe_calls == []


def test_show_env_can_import_credentials_without_breaking_machine_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, credentials_path = _point_configure_at_tmp(monkeypatch, tmp_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "fresh",
                "projects": {
                    "fresh": {
                        "tenant_id": "tenant-synthetic",
                        "project_id": "project-synthetic",
                        "region": "eu-north1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", "hf_synthetic_machine_secret")
    monkeypatch.setattr(
        cli_main,
        "_saved_model_access_note",
        lambda: "[NOTE] Access checks are informational; HF token valid.",
    )

    result = runner.invoke(
        app,
        ["configure", "--show", "--env", "--save-env-credentials"],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "NPA_PROJECT_ALIAS=fresh",
        "NPA_PROJECT_ID=project-synthetic",
        "NPA_TENANT_ID=tenant-synthetic",
        "NPA_REGION=eu-north1",
    ]
    assert "Credential fields persisted" in result.stderr
    assert "hf_synthetic_machine_secret" not in result.output
    assert (
        yaml.safe_load(credentials_path.read_text())["tokens"]["HF_TOKEN"]
        == "hf_synthetic_machine_secret"
    )


@pytest.mark.parametrize("command", ["configure", "init"])
def test_env_can_import_credentials_without_requiring_show(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str
) -> None:
    config_path, credentials_path = _point_configure_at_tmp(monkeypatch, tmp_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "fresh",
                "projects": {
                    "fresh": {
                        "tenant_id": "tenant-synthetic",
                        "project_id": "project-synthetic",
                        "region": "eu-north1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", "hf_synthetic_legacy_machine_secret")
    monkeypatch.setattr(
        cli_main,
        "_saved_model_access_note",
        lambda: "[NOTE] Access checks are informational; HF token valid.",
    )

    result = runner.invoke(
        app,
        [command, "--env", "--save-env-credentials"],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "NPA_PROJECT_ALIAS=fresh",
        "NPA_PROJECT_ID=project-synthetic",
        "NPA_TENANT_ID=tenant-synthetic",
        "NPA_REGION=eu-north1",
    ]
    assert "Credential fields persisted" in result.stderr
    assert "hf_synthetic_legacy_machine_secret" not in result.output
    assert (
        yaml.safe_load(credentials_path.read_text())["tokens"]["HF_TOKEN"]
        == "hf_synthetic_legacy_machine_secret"
    )


def test_show_accepts_harmless_no_interactive_flag() -> None:
    result = runner.invoke(app, ["configure", "--show", "--no-interactive"])

    assert result.exit_code == 0, result.output
    assert "Current configuration" in result.output


def test_configure_help_describes_provider_free_and_explicit_provision_intent() -> None:
    result = runner.invoke(app, ["configure", "--help"], terminal_width=200)

    assert result.exit_code == 0, result.output
    compact = " ".join(result.output.replace("│", " ").split())
    assert "--provision is passed" in compact
    assert "--no-provision performs no provider calls or storage adoption" in compact


def test_generated_configure_bucket_names_are_utc_and_collision_safe() -> None:
    names = {
        cli_main._generated_configure_bucket_name("tenant", "project")
        for _ in range(3)
    }

    assert len(names) == 3
    assert all(
        re.fullmatch(r"npa-bucket-\d{8}t\d{6}z-[0-9a-f]{6}-[0-9a-f]{8}", name)
        for name in names
    )


def test_generated_exact_bucket_collision_is_reported_and_never_adopted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lookups: list[str] = []
    provisioned: list[str] = []
    monkeypatch.setattr(
        cli_main,
        "_generated_configure_bucket_name",
        lambda *_args: "fresh-collision",
    )

    def exists(_project_id: str, bucket_name: str) -> bool:
        lookups.append(bucket_name)
        return True

    monkeypatch.setattr(nebius, "bucket_exists", exists)

    def provision(**kwargs):
        provisioned.append(kwargs["bucket_name"])
        return (
            {
                "nebius_api_key": "synthetic-access",
                "nebius_secret_key": "synthetic-secret",
                "s3_bucket": kwargs["bucket_name"],
                "s3_endpoint": "https://storage.example.invalid",
            },
            type("Probe", (), {"summary": "synthetic probe ok"})(),
        )

    monkeypatch.setattr("npa.clients.storage_setup.provision_storage", provision)

    result = cli_main._provision_object_storage(
        nebius,
        lambda _label, *, default="", **_kwargs: default,
        project_id="project-synthetic",
        tenant_id="tenant-synthetic",
        region="eu-north1",
        interactive=False,
    )

    output = capsys.readouterr().out
    assert lookups == ["fresh-collision"]
    assert provisioned == []
    assert result is None
    assert "Generated name collision" in output
    assert "will not be adopted" in output


def test_exact_bucket_lookup_does_not_enumerate_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_json = Mock(
        return_value={
            "metadata": {"name": "exact-synthetic", "parent_id": "project-synthetic"}
        }
    )
    monkeypatch.setattr(nebius, "_run_json", run_json)

    item = nebius.get_bucket_by_name("project-synthetic", "exact-synthetic")

    assert item is not None
    args = run_json.call_args.args[0]
    assert args[:3] == ["storage", "bucket", "get-by-name"]
    assert args[args.index("--name") + 1] == "exact-synthetic"
    assert "--all" not in args


@pytest.mark.parametrize(
    "message",
    [
        "profile 'missing-profile' not found",
        "rpc error: code = NotFound desc = parent project not found",
    ],
)
def test_exact_bucket_lookup_does_not_treat_unrelated_not_found_as_absence(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    monkeypatch.setattr(
        nebius,
        "_run_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(nebius.NebiusError(message)),
    )

    with pytest.raises(nebius.NebiusError, match="not found"):
        nebius.get_bucket_by_name("project-synthetic", "exact-synthetic")


def test_exact_bucket_lookup_accepts_multiline_no_such_bucket_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = "\n".join(
        [
            "nebius storage bucket get-by-name failed (exit 13):",
            "rpc error: code = NotFound desc = NoSuchBucket: Bucket doesn't exist",
            "request = synthetic-request-id",
            "caused by service error:",
            "  Service storage error NoSuchBucket",
            "This issue may be linked to API request calls.",
            "  Trace ID: synthetic-trace-id",
        ]
    )
    monkeypatch.setattr(
        nebius,
        "_run_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(nebius.NebiusError(message)),
    )

    assert nebius.get_bucket_by_name("project-synthetic", "exact-synthetic") is None


def test_exact_existing_bucket_reuse_is_explicit_in_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(nebius, "bucket_exists", lambda *_args: True)
    monkeypatch.setattr(
        "npa.clients.storage_setup.provision_storage",
        lambda **kwargs: (
            {
                "nebius_api_key": "synthetic-access",
                "nebius_secret_key": "synthetic-secret",
                "s3_bucket": kwargs["bucket_name"],
                "s3_endpoint": "https://storage.example.invalid",
            },
            type("Probe", (), {"summary": "synthetic probe ok"})(),
        ),
    )

    result = cli_main._provision_object_storage(
        nebius,
        lambda _label, *, default="", **_kwargs: default,
        project_id="project-synthetic",
        tenant_id="tenant-synthetic",
        region="eu-north1",
        existing_bucket="exact-synthetic",
        interactive=False,
    )

    output = capsys.readouterr().out
    assert result is not None
    assert result["_disposition"] == "reused"
    assert "Explicitly reusing exact existing bucket 'exact-synthetic'" in output
    assert "Created object-storage bucket" not in output
