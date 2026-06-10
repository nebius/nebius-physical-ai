from __future__ import annotations

import json
import re
from importlib.metadata import version

import pytest
from typer.testing import CliRunner

from npa.cli import main as cli_main
from npa.cli.main import app
from npa.clients.serverless import NotEnoughResourcesError


runner = CliRunner()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--help"], "Nebius Physical AI workbench CLI"),
        (["workbench", "--help"], "Physical AI workbench tools"),
        (["workbench", "lerobot", "--help"], "LeRobot policy training"),
        (["workbench", "genesis", "--help"], "Genesis simulation"),
        (["adapter", "--help"], "Convert simulation data"),
        (["convert", "--help"], "standalone formats"),
        (["demo", "--help"], "Demo artifact bootstrap"),
        (["network", "--help"], "Network operations"),
        (["rerun", "--help"], "Host and share Rerun"),
        (["viz", "--help"], "visualization"),
        (["workflow", "--help"], "Multi-stage training workflow"),
        (["configure", "--help"], "credential and config setup guidance"),
        (["init", "--help"], "credential and config setup guidance"),
    ],
)
def test_help_smoke(args: list[str], expected: str) -> None:
    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert expected in result.output


def test_no_args_shows_top_level_help() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "Nebius Physical AI workbench CLI" in result.output
    assert "workbench" in result.output
    assert "adapter" in result.output
    assert "convert" in result.output
    assert "demo" in result.output
    assert "network" in result.output
    assert "rerun" in result.output
    assert "viz" in result.output
    assert "workflow" in result.output
    assert "configure" in result.output
    assert "init" in result.output


def test_version_flag_reports_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert re.match(r"^npa \d+\.\d+(\.\d+)?", result.stdout)
    assert result.stdout.strip() == f"npa {version('npa')}"


@pytest.mark.parametrize("command", ["configure", "init"])
def test_setup_guidance_commands_show_credentials_path(command: str) -> None:
    result = runner.invoke(app, [command])

    assert result.exit_code == 0
    assert "~/.npa/credentials.yaml" in result.output
    assert "HF_TOKEN" in result.output
    assert "ngc:" in result.output
    assert "api_key" in result.output
    assert "chmod 600" in result.output


def test_configure_show_includes_storage_and_registry() -> None:
    result = runner.invoke(app, ["configure", "--show"])

    assert result.exit_code == 0
    assert "storage:" in result.output
    assert "aws_access_key_id" in result.output
    assert "container registry" in result.output.lower()
    assert "~/.npa/config.yaml" in result.output


def _stub_nebius_defaults(monkeypatch, *, project="", tenant="", registry="") -> None:
    """Stop configure from touching real Nebius infra for profile-derived defaults."""
    import npa.clients.nebius as nebius_module

    monkeypatch.setattr(nebius_module, "current_project_id", lambda: project)
    monkeypatch.setattr(nebius_module, "current_tenant_id", lambda: tenant)
    monkeypatch.setattr(
        nebius_module, "discover_container_registry", lambda project_id: registry
    )


def test_configure_interactive_provisions_storage(monkeypatch, tmp_path) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: None)
    _stub_nebius_defaults(
        monkeypatch, project="project-12345", tenant="tenant-abcde"
    )

    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: False)

    bootstrap_calls: list[dict] = []

    def fake_bootstrap(
        project_id,
        tenant_id,
        region,
        *,
        bucket_name=None,
        bucket_max_size_bytes=0,
        on_status=None,
    ):
        bootstrap_calls.append(
            {
                "project_id": project_id,
                "tenant_id": tenant_id,
                "region": region,
                "bucket_name": bucket_name,
                "bucket_max_size_bytes": bucket_max_size_bytes,
            }
        )
        if on_status:
            on_status("Setting up S3 bucket...")
        return {
            "nebius_api_key": "AKIAPROVISIONED",
            "nebius_secret_key": "provisioned-secret",
            "s3_bucket": bucket_name,
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        }

    monkeypatch.setattr(nebius_module, "bootstrap_environment", fake_bootstrap)

    # Accept derived project/tenant + default region/registry; pick a custom
    # bucket name and a custom size; then HF + NGC.
    answers = "\n".join(
        [
            "",                  # project id (accept derived)
            "",                  # tenant id (accept derived)
            "",                  # region (default eu-north1)
            "",                  # registry (default)
            "my-bucket",         # bucket name (customer choice)
            "y",                 # set a size limit?
            "100",               # size in GB
            "hf_secret_token",   # HF token
            "nvapi_secret",      # NGC API key
        ]
    ) + "\n"

    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert len(bootstrap_calls) == 1
    call = bootstrap_calls[0]
    assert (call["project_id"], call["tenant_id"], call["region"]) == (
        "project-12345",
        "tenant-abcde",
        "eu-north1",
    )
    assert call["bucket_name"] == "my-bucket"
    assert call["bucket_max_size_bytes"] == 100 * 1024**3

    creds = yaml.safe_load(creds_path.read_text())
    assert creds["tokens"]["HF_TOKEN"] == "hf_secret_token"
    assert creds["ngc"]["api_key"] == "nvapi_secret"
    assert creds["storage"]["aws_access_key_id"] == "AKIAPROVISIONED"
    assert creds["storage"]["aws_secret_access_key"] == "provisioned-secret"
    assert creds["storage"]["endpoint_url"] == "https://storage.eu-north1.nebius.cloud"
    assert creds["storage"]["bucket"] == "s3://my-bucket/"

    cfg = yaml.safe_load(config_path.read_text())
    project = cfg["projects"]["eu-north1"]
    assert project["project_id"] == "project-12345"
    assert project["tenant_id"] == "tenant-abcde"
    assert project["container_registry"].startswith("cr.eu-north1.nebius.cloud/")
    assert oct(creds_path.stat().st_mode)[-3:] == "600"


def test_configure_provision_reuses_existing_bucket_without_size_prompt(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: None)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: True)

    sizes: list[int] = []

    def fake_bootstrap(
        project_id,
        tenant_id,
        region,
        *,
        bucket_name=None,
        bucket_max_size_bytes=0,
        on_status=None,
    ):
        sizes.append(bucket_max_size_bytes)
        return {
            "nebius_api_key": "AKIA",
            "nebius_secret_key": "secret",
            "s3_bucket": bucket_name,
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        }

    monkeypatch.setattr(nebius_module, "bootstrap_environment", fake_bootstrap)

    # No size prompt is shown when the bucket already exists.
    # proj, tenant, region, registry, bucket name (default), hf, ngc
    answers = "\n".join(["", "", "", "", "", "", ""]) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert "Reusing existing object-storage bucket" in result.output
    assert sizes == [0]
    creds = yaml.safe_load(creds_path.read_text())
    assert creds["storage"]["bucket"].startswith("s3://lerobot-")


def test_configure_provision_falls_back_to_manual_on_error(monkeypatch, tmp_path) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: None)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: False)

    def boom(*_args, **_kwargs):
        raise nebius_module.NebiusError("PermissionDenied")

    monkeypatch.setattr(nebius_module, "bootstrap_environment", boom)

    answers = "\n".join(
        [
            "",                  # project id (accept derived)
            "",                  # tenant id (accept derived)
            "",                  # region (default)
            "",                  # registry (default)
            "",                  # bucket name (accept suggested)
            "y",                 # set a size limit?
            "",                  # size GB (default 50)
            "AKIAMANUAL",        # S3 access key (fallback)
            "manual-secret",     # S3 secret (fallback)
            "",                  # S3 endpoint (default-by-region)
            "s3://manual-bucket/",  # S3 bucket (fallback)
            "hf_tok",            # HF token
            "",                  # NGC API key (skip)
        ]
    ) + "\n"

    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert "Could not auto-provision" in result.output
    creds = yaml.safe_load(creds_path.read_text())
    assert creds["storage"]["aws_access_key_id"] == "AKIAMANUAL"
    assert creds["storage"]["aws_secret_access_key"] == "manual-secret"
    assert creds["storage"]["endpoint_url"] == "https://storage.eu-north1.nebius.cloud"
    assert creds["storage"]["bucket"] == "s3://manual-bucket/"
    assert "api_key" not in creds.get("ngc", {})


def test_configure_no_provision_uses_manual_entry(monkeypatch, tmp_path) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: None)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")

    def must_not_call(*_args, **_kwargs):
        raise AssertionError("--no-provision must not call bootstrap_environment")

    monkeypatch.setattr(nebius_module, "bootstrap_environment", must_not_call)

    answers = "\n".join(
        [
            "",                  # project id
            "",                  # tenant id
            "me-central1",       # region
            "",                  # registry (default)
            "AKIAMANUAL",        # S3 access key
            "manual-secret",     # S3 secret
            "",                  # S3 endpoint (default-by-region)
            "s3://b/",           # S3 bucket
            "",                  # HF token
            "",                  # NGC API key
        ]
    ) + "\n"

    result = runner.invoke(
        app, ["configure", "--interactive", "--no-provision"], input=answers
    )

    assert result.exit_code == 0, result.output
    creds = yaml.safe_load(creds_path.read_text())
    assert creds["storage"]["aws_access_key_id"] == "AKIAMANUAL"
    # Endpoint default tracks the entered region.
    assert creds["storage"]["endpoint_url"] == "https://storage.me-central1.nebius.cloud"


def test_configure_interactive_skips_config_without_project(monkeypatch, tmp_path) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: None)
    _stub_nebius_defaults(monkeypatch)

    # Skip every field. With no project/tenant, provisioning is skipped and the
    # manual object-storage prompts run; only the defaulted endpoint remains.
    answers = "\n".join([""] * 10) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert creds_path.exists()
    assert not config_path.exists()
    creds = yaml.safe_load(creds_path.read_text())
    # Empty values are pruned; the defaulted endpoint is still written.
    assert "HF_TOKEN" not in creds.get("tokens", {})
    assert creds["storage"]["endpoint_url"] == "https://storage.eu-north1.nebius.cloud"


def test_configure_non_tty_prints_guidance() -> None:
    # CliRunner stdin is not a TTY, so configure must fall back to guidance.
    result = runner.invoke(app, ["configure"])

    assert result.exit_code == 0
    assert "~/.npa/credentials.yaml" in result.output


def test_configure_creates_nebius_profile_when_missing(monkeypatch, tmp_path) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml")
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    # A nebius binary exists but no profile is ready until we "create" one.
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/nebius")
    readiness = iter([False, True])
    monkeypatch.setattr(cli_main, "_nebius_profile_ready", lambda **_: next(readiness))
    created: list[bool] = []

    def fake_create(**_):
        created.append(True)
        return True

    monkeypatch.setattr(cli_main, "_create_nebius_profile", fake_create)
    _stub_nebius_defaults(monkeypatch)

    # confirm profile, then skip all 10 interactive fields
    answers = "y\n" + "\n".join([""] * 10) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert created == [True]
    assert "Nebius CLI profile is ready." in result.output


def test_configure_detects_existing_nebius_profile(monkeypatch, tmp_path) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml")
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main, "_nebius_profile_ready", lambda **_: True)

    def fail_create(**_):
        raise AssertionError("must not create a profile when one already works")

    monkeypatch.setattr(cli_main, "_create_nebius_profile", fail_create)
    _stub_nebius_defaults(monkeypatch)

    result = runner.invoke(
        app, ["configure", "--interactive"], input="\n".join([""] * 10) + "\n"
    )

    assert result.exit_code == 0, result.output
    assert "Nebius CLI profile detected" in result.output


def test_nebius_profile_ready_uses_get_access_token(monkeypatch) -> None:
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/nebius")
    calls: list[list[str]] = []

    class _Result:
        returncode = 0

    def fake_runner(cmd, **kwargs):
        calls.append(cmd)
        return _Result()

    assert cli_main._nebius_profile_ready(runner=fake_runner) is True
    assert calls == [["nebius", "iam", "get-access-token"]]


def test_nebius_profile_not_ready_without_binary(monkeypatch) -> None:
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: None)
    assert cli_main._nebius_profile_ready() is False


def test_app_entry_typed_error_exits_one_without_traceback(monkeypatch, capsys) -> None:
    def fail() -> None:
        raise NotEnoughResourcesError(
            "capacity blocked",
            project_id="project-1",
            platform="gpu-h200-sxm",
            suggested_alternatives=["Retry in a few minutes"],
        )

    monkeypatch.setattr(cli_main, "app", fail)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.app_entry()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Not enough resources" in err
    assert "Retry in a few minutes" in err
    assert "Traceback" not in err


def test_app_entry_typed_error_json_mode(monkeypatch, capsys) -> None:
    def fail() -> None:
        raise NotEnoughResourcesError("capacity blocked", project_id="project-1")

    monkeypatch.setenv("NPA_ERROR_FORMAT", "json")
    monkeypatch.setattr(cli_main, "app", fail)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.app_entry()

    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"] == "NotEnoughResources"
    assert payload["project_id"] == "project-1"


def test_app_entry_unexpected_error_no_stacktrace_by_default(monkeypatch, capsys) -> None:
    def fail() -> None:
        raise RuntimeError("boom")

    monkeypatch.delenv("NPA_DEBUG", raising=False)
    monkeypatch.setattr(cli_main, "app", fail)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.app_entry()

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "Unexpected error: boom" in err
    assert "NPA_DEBUG=1" in err
    assert "Traceback" not in err


def test_app_entry_unexpected_error_stacktrace_with_debug(monkeypatch, capsys) -> None:
    def fail() -> None:
        raise RuntimeError("boom")

    monkeypatch.setenv("NPA_DEBUG", "1")
    monkeypatch.setattr(cli_main, "app", fail)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.app_entry()

    assert exc_info.value.code == 2
    assert "Traceback" in capsys.readouterr().err
