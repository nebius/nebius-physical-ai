from __future__ import annotations

import json
import re
from importlib.metadata import version
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli import main as cli_main
from npa.cli.main import app
from npa.clients.serverless import NotEnoughResourcesError


runner = CliRunner()


@pytest.fixture(autouse=True)
def _stub_model_access(monkeypatch):
    """Keep `npa configure`'s model-access NOTE off real HF and NGC APIs.

    `configure` runs live access checks after collecting tokens. Default every
    probe to "accessible" so unrelated tests stay hermetic; individual tests
    override these fakes to exercise the NOTE.
    """

    from npa.clients import huggingface
    from npa.clients.huggingface import HFAccessResult

    def _ok(
        token,
        repo,
        repo_type="model",
        revision="",
        probe_path="",
        *,
        timeout=10.0,
    ):
        return HFAccessResult(repo=repo, ok=True, status_code=200)

    monkeypatch.setattr(
        huggingface,
        "validate_hf_identity",
        lambda token, *, timeout=10.0: HFAccessResult(
            repo="whoami-v2", ok=True, status_code=200
        ),
    )
    monkeypatch.setattr(huggingface, "validate_hf_access", _ok)
    monkeypatch.setattr(
        "npa.workbench.nurec.nurec.check_ngc_image_access",
        lambda key, *, timeout=30.0: "reachable",
    )

    from npa.clients import storage_setup, storage_validation
    from npa.clients.storage_validation import StorageProbeResult

    probe = StorageProbeResult(
        True,
        "ok",
        "Writable S3 verified with a cleaned write/delete probe.",
        cleanup_attempted=True,
        cleanup_succeeded=True,
    )
    monkeypatch.setattr(storage_setup, "probe_storage_write", lambda **_kwargs: probe)
    monkeypatch.setattr(
        storage_validation, "probe_storage_write", lambda **_kwargs: probe
    )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--help"], "Nebius Physical AI workbench CLI"),
        (["workbench", "--help"], "Physical AI workbench tools"),
        (["workbench", "lerobot", "--help"], "LeRobot policy training"),
        (["workbench", "genesis", "--help"], "Genesis simulation"),
        (["adapter", "--help"], "Convert simulation data"),
        (["burst", "--help"], "multi-node SkyPilot GPU jobs"),
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


def test_configure_help_exposes_no_secret_value_flags() -> None:
    result = runner.invoke(app, ["configure", "--help"])

    assert result.exit_code == 0
    assert "--save-env-credentials" in result.output
    assert "--container-registry" not in result.output
    for forbidden in (
        "--hf-token",
        "--ngc-api-key",
        "--token-factory-key",
        "--aws-access-key-id",
        "--aws-secret-access-key",
        "--secret",
    ):
        assert forbidden not in result.output


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


def test_configure_show_includes_storage_and_public_image_guidance() -> None:
    result = runner.invoke(app, ["configure", "--show"])

    assert result.exit_code == 0
    assert "storage:" in result.output
    assert "aws_access_key_id" in result.output
    assert "container registry" not in result.output.lower()
    assert "GHCR" in result.output
    assert "~/.npa/config.yaml" in result.output


def _stub_nebius_defaults(
    monkeypatch, *, project="", tenant="", project_name=""
) -> list[tuple[str, str]]:
    """Stop configure from touching real Nebius infra for profile-derived defaults.

    Returns the list that records ``set_profile_project`` calls, so tests can
    assert whether configure re-pointed the operator's Nebius CLI profile.
    """
    import npa.clients.nebius as nebius_module

    monkeypatch.setattr(nebius_module, "current_project_id", lambda: project)
    monkeypatch.setattr(nebius_module, "current_tenant_id", lambda: tenant)
    monkeypatch.setattr(
        nebius_module, "get_project_tenant_id", lambda project_id: tenant
    )
    monkeypatch.setattr(
        nebius_module, "get_project_name", lambda project_id: project_name
    )
    monkeypatch.setattr(nebius_module, "list_tenants", lambda: [])
    bound: list[tuple[str, str]] = []

    def _set_profile_project(project_id, tenant_id=""):
        bound.append((project_id, tenant_id))
        return True

    monkeypatch.setattr(nebius_module, "set_profile_project", _set_profile_project)
    return bound


def test_configure_discovers_and_writes_multiple_projects(
    monkeypatch, tmp_path
) -> None:
    """With discoverable projects, configure picks from a list (no id typing)."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-prod", tenant="tenant-a")
    monkeypatch.setattr(
        nebius_module,
        "list_projects_in_tenant",
        lambda tenant_id: [
            {
                "id": "project-prod",
                "name": "prod",
                "tenant_id": "tenant-a",
                "region": "eu-north1",
            },
            {
                "id": "project-dev",
                "name": "dev",
                "tenant_id": "tenant-a",
                "region": "us-central1",
            },
        ],
    )

    def _must_not_provision(*_a, **_k):
        raise AssertionError("storage must not provision when the user opts out")

    monkeypatch.setattr(nebius_module, "bootstrap_environment", _must_not_provision)

    # select both, default = prod, decline storage, then HF/TF/NGC (all skipped).
    answers = "\n".join(["1,2", "prod", "N", "", "", ""]) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["default_project"] == "prod"
    assert set(cfg["projects"]) == {"prod", "dev"}
    assert cfg["projects"]["prod"]["project_id"] == "project-prod"
    assert cfg["projects"]["prod"]["tenant_id"] == "tenant-a"
    assert cfg["projects"]["prod"]["region"] == "eu-north1"
    assert cfg["projects"]["dev"]["project_id"] == "project-dev"
    assert cfg["projects"]["dev"]["region"] == "us-central1"
    # No storage stanza was written (opted out).
    creds = yaml.safe_load(creds_path.read_text())
    assert not creds.get("storage")
    # Opting out warns that the agent / Physical AI Data Factory need storage.
    assert "Physical AI Data Factory" in result.output


def test_configure_discovery_does_not_discover_or_write_registry(
    monkeypatch, tmp_path
) -> None:
    """Project discovery must not require a private registry for public images."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")

    def _must_not_discover_registry(_project_id):
        raise AssertionError("configure must not discover a container registry")

    monkeypatch.setattr(
        nebius_module, "discover_container_registry", _must_not_discover_registry
    )
    monkeypatch.setattr(
        nebius_module,
        "list_projects_in_tenant",
        lambda tenant_id: [
            {
                "id": "project-1",
                "name": "solo",
                "tenant_id": "tenant-1",
                "region": "us-central1",
            }
        ],
    )

    def _must_not_provision(*_a, **_k):
        raise AssertionError("storage must not provision when the user opts out")

    monkeypatch.setattr(nebius_module, "bootstrap_environment", _must_not_provision)

    # select the single project, decline storage, then HF/TF/NGC (all skipped).
    answers = "\n".join(["1", "N", "", "", ""]) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load(config_path.read_text())
    stanza = cfg["projects"]["solo"]
    assert stanza["region"] == "us-central1"
    assert "container_registry" not in stanza
    assert config_module.resolve_container_registry("solo") == (
        "ghcr.io/nebius/nebius-physical-ai"
    )


def test_configure_rerun_updates_the_existing_alias_for_a_project(
    monkeypatch, tmp_path
) -> None:
    """A re-run must update the stanza this project already has, not add a new one.

    Regression: the discovery path always derived the alias from the Nebius
    project name and repointed `default_project` at it, so a config whose alias
    was `prod` gained a second `tle-workbench` stanza — one without the
    `workbenches` endpoints or `terraform_state` pointer the deploy paths read.
    """
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "prod",
                "projects": {
                    "prod": {
                        "project_id": "project-1",
                        "tenant_id": "tenant-1",
                        "region": "eu-north1",
                        "terraform_state": {"bucket": "tfstate-prod"},
                        "workbenches": {"b200": {"endpoint": "http://vm:8080"}},
                    }
                },
            }
        )
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")
    monkeypatch.setattr(
        nebius_module,
        "list_projects_in_tenant",
        lambda tenant_id: [
            {
                "id": "project-1",
                "name": "tle-workbench",
                "tenant_id": "tenant-1",
                "region": "eu-north1",
            }
        ],
    )
    monkeypatch.setattr(
        nebius_module,
        "bootstrap_environment",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("opted out of storage")),
    )

    answers = "\n".join(["1", "N", "", "", ""]) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["default_project"] == "prod"
    assert set(cfg["projects"]) == {"prod"}
    assert cfg["projects"]["prod"]["terraform_state"] == {"bucket": "tfstate-prod"}
    assert (
        cfg["projects"]["prod"]["workbenches"]["b200"]["endpoint"] == "http://vm:8080"
    )


def test_configure_never_writes_a_project_into_another_projects_alias(
    monkeypatch, tmp_path
) -> None:
    """A derived alias that collides with a different project must not merge into it."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "prod",
                "projects": {
                    "prod": {
                        "project_id": "project-old",
                        "tenant_id": "tenant-1",
                        "terraform_state": {"bucket": "tfstate-old"},
                    }
                },
            }
        )
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-new", tenant="tenant-1")
    monkeypatch.setattr(
        nebius_module,
        "list_projects_in_tenant",
        lambda tenant_id: [
            {
                "id": "project-new",
                "name": "prod",  # same slug as the existing alias, different project
                "tenant_id": "tenant-1",
                "region": "eu-north1",
            }
        ],
    )
    monkeypatch.setattr(
        nebius_module,
        "bootstrap_environment",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("opted out of storage")),
    )

    answers = "\n".join(["1", "N", "", "", ""]) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load(config_path.read_text())
    # The old stanza keeps its own project and Terraform state.
    assert cfg["projects"]["prod"]["project_id"] == "project-old"
    assert cfg["projects"]["prod"]["terraform_state"] == {"bucket": "tfstate-old"}
    new_alias = cfg["default_project"]
    assert new_alias != "prod"
    assert cfg["projects"][new_alias]["project_id"] == "project-new"
    assert "terraform_state" not in cfg["projects"][new_alias]


def test_configure_show_prints_the_saved_configuration(monkeypatch, tmp_path) -> None:
    """--show printed only the empty template, not what is actually configured."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "tle-workbench",
                "projects": {
                    "tle-workbench": {
                        "project_id": "project-1",
                        "tenant_id": "tenant-1",
                        "region": "us-central1",
                        "container_registry": "registry.example/customer",
                    },
                    "other": {"project_id": "project-2"},
                },
            }
        )
    )
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {"HF_TOKEN": "hf_secret"},
                "storage": {
                    "bucket": "s3://npa-bucket-test/",
                    "endpoint_url": "https://storage.eu-north1.nebius.cloud",
                    "access_key_id": "AKTEST",
                    "secret_access_key": "SKTEST",
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)

    result = runner.invoke(app, ["configure", "--show"])

    assert result.exit_code == 0, result.output
    # The values the PAIDF quickstart placeholders need.
    assert "tle-workbench" in result.output
    assert "project-1" in result.output
    assert "tenant-1" in result.output
    assert "us-central1" in result.output
    assert "registry.example/customer" not in result.output
    assert "s3://npa-bucket-test/" in result.output
    assert "other" in result.output  # the non-default alias is listed too
    # Secrets are reported as present, never echoed.
    assert "hf_secret" not in result.output
    assert "AKTEST" not in result.output
    assert "HF token:" in result.output
    assert "NGC API key:" in result.output


def test_configure_show_env_emits_shell_assignments(monkeypatch, tmp_path) -> None:
    """The runbook eval's this instead of asking for three hand-substitutions."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    from npa.cluster import state as state_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "tle-workbench",
                "projects": {
                    "tle-workbench": {
                        "project_id": "project-1",
                        "tenant_id": "tenant-1",
                        "region": "us-central1",
                        "container_registry": "registry.example/customer",
                    }
                },
            }
        )
    )
    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {"HF_TOKEN": "hf_secret"},
                "storage": {
                    "bucket": "s3://npa-bucket-test/checkpoints/",
                    "endpoint_url": "https://storage.eu-north1.nebius.cloud",
                    "access_key_id": "AKTEST",
                    "secret_access_key": "SKTEST",
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)
    monkeypatch.setattr(
        state_module,
        "list_local_clusters",
        lambda: [SimpleNamespace(name="npa-cluster")],
    )

    result = runner.invoke(app, ["configure", "--show", "--env"])

    assert result.exit_code == 0, result.output
    values = dict(
        line.split("=", 1) for line in result.output.strip().splitlines() if "=" in line
    )
    assert "NPA_REGISTRY" not in values
    assert values["NPA_PROJECT_ALIAS"] == "tle-workbench"
    assert values["NPA_PROJECT_ID"] == "project-1"
    assert values["NPA_REGION"] == "us-central1"
    # The bare bucket name is what `--var bucket=` wants, not the s3:// URI.
    assert values["NPA_BUCKET"] == "npa-bucket-test"
    assert values["NPA_BUCKET_URI"] == "s3://npa-bucket-test/checkpoints/"
    assert values["NPA_KUBE_CONTEXT"] == "npa-cluster"
    # No secrets, and no prose that would break `eval`.
    assert "hf_secret" not in result.output
    assert "AKTEST" not in result.output
    assert "Credential setup" not in result.output


def test_configure_show_env_is_eval_safe_in_clean_subprocess(tmp_path) -> None:
    """Machine stdout stays shell-only even when credential diagnostics exist."""

    import os
    import subprocess
    import sys

    import yaml

    home = tmp_path / "home with spaces"
    npa_dir = home / ".npa"
    npa_dir.mkdir(parents=True)
    (npa_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "default_project": "project alias",
                "projects": {
                    "project alias": {
                        "project_id": "project-1",
                        "tenant_id": "tenant-1",
                        "region": "eu-north1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (npa_dir / "credentials.yaml").write_text(
        yaml.safe_dump(
            {
                "storage": {
                    "bucket": "s3://bucket/with-prefix/",
                    "endpoint_url": "https://storage.eu-north1.nebius.cloud",
                }
            }
        ),
        encoding="utf-8",
    )
    (npa_dir / "credentials.yaml").chmod(0o600)
    stderr_path = tmp_path / "stderr.txt"
    script = (
        'set -eu; output="$("$NPA_TEST_PYTHON" -m npa.cli.main configure --show --env '
        '2>"$NPA_TEST_ERR")"; '
        'eval "$output"; test "$NPA_PROJECT_ALIAS" = "project alias"; '
        'test "$NPA_BUCKET" = bucket; printf "%s" "$output"'
    )
    env = {
        "HOME": str(home),
        "PATH": os.environ["PATH"],
        "NPA_TEST_PYTHON": sys.executable,
        "NPA_TEST_ERR": str(stderr_path),
        "HF_TOKEN": "hf_never-print-this",
    }
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Credential environment sources detected" not in completed.stdout
    assert "hf_never-print-this" not in completed.stdout
    assert "Credential environment sources detected" in stderr_path.read_text()


def test_configure_show_env_scopes_kube_context_to_configured_project(
    monkeypatch, tmp_path
) -> None:
    """--show --env must not emit an unrelated project's cluster context.

    Regression: `_saved_kube_context` returned the most recent local cluster of
    any project, so a runbook could `eval` a context pointing at the wrong infra.
    """
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    from npa.cluster import state as state_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "mine",
                "projects": {
                    "mine": {"project_id": "project-1", "region": "us-central1"}
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(
        state_module,
        "list_local_clusters",
        # The unrelated cluster is listed last (most recent) on purpose.
        lambda: [
            SimpleNamespace(name="my-cluster", project_id="project-1"),
            SimpleNamespace(name="someone-elses", project_id="project-OTHER"),
        ],
    )

    result = runner.invoke(app, ["configure", "--show", "--env"])

    assert result.exit_code == 0, result.output
    values = dict(
        line.split("=", 1) for line in result.output.strip().splitlines() if "=" in line
    )
    assert values["NPA_KUBE_CONTEXT"] == "my-cluster"
    assert "someone-elses" not in result.output


def test_configure_stores_hf_and_ngc_tokens_without_prompting(
    monkeypatch, tmp_path
) -> None:
    """Scripted setup persists environment secrets without putting them in argv."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    creds_path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("NGC_API_KEY", "nvapi-test")
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "v1.test")

    result = runner.invoke(
        app,
        [
            "configure",
            "--show",
            "--save-env-credentials",
        ],
    )

    assert result.exit_code == 0, result.output
    assert all(
        secret not in result.output for secret in ("hf_test", "nvapi-test", "v1.test")
    )
    saved = yaml.safe_load(creds_path.read_text())
    assert saved["tokens"]["HF_TOKEN"] == "hf_test"
    assert saved["tokens"]["NEBIUS_TOKEN_FACTORY_KEY"] == "v1.test"
    assert saved["ngc"]["api_key"] == "nvapi-test"


def test_configure_env_no_interactive_confirms_persistence(
    monkeypatch, tmp_path
) -> None:
    """Noninteractive persistence is explicit and does not expose a secret argv."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    creds_path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setenv("HF_TOKEN", "hf_flagvalue")

    result = runner.invoke(
        app,
        ["configure", "--save-env-credentials", "--no-interactive"],
    )

    assert result.exit_code == 0, result.output
    assert "saved to" in result.output
    # The whole "Credential setup" template no longer prints as if nothing happened.
    assert "Credential setup" not in result.output
    assert (
        yaml.safe_load(creds_path.read_text())["tokens"]["HF_TOKEN"] == "hf_flagvalue"
    )


def test_configure_no_tokens_no_interactive_still_shows_template(
    monkeypatch, tmp_path
) -> None:
    """With nothing stored, `--no-interactive` still prints the setup guidance."""
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")

    result = runner.invoke(app, ["configure", "--no-interactive"])

    assert result.exit_code == 0, result.output
    assert "Credential setup" in result.output


def test_configure_noninteractive_detected_env_is_truthful_when_not_persisted(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    credentials_path = tmp_path / "credentials.yaml"
    secret = "hf_process_only_secret"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setenv("HF_TOKEN", secret)

    result = runner.invoke(app, ["configure", "--no-interactive"])

    assert result.exit_code == 0, result.output
    assert "HF_TOKEN" in result.output
    assert "persistence: skipped" in result.output
    assert "--save-env-credentials" in result.output
    assert secret not in result.output
    assert not credentials_path.exists()


def test_prompt_setup_tokens_keeps_flagged_tokens_without_reprompting() -> None:
    """A token passed via flag is kept, not re-prompted (an empty Enter would wipe it) — bug 6."""
    from types import SimpleNamespace

    existing = SimpleNamespace(
        hf_token="hf_kept", token_factory_api_key="v1.kept", ngc_api_key="nvapi-kept"
    )
    asked: list[str] = []

    def ask(prompt, *, default="", secret=False):
        asked.append(prompt)
        return ""  # a bare Enter — would wipe the key if it reached the store

    hf, tf, ngc = cli_main._prompt_setup_tokens(
        ask, existing, skip={"HF_TOKEN", "NEBIUS_TOKEN_FACTORY_KEY", "NGC_API_KEY"}
    )

    assert (hf, tf, ngc) == ("hf_kept", "v1.kept", "nvapi-kept")
    assert asked == []  # none of the three were re-prompted


def test_prompt_setup_tokens_still_prompts_for_unflagged_tokens() -> None:
    from types import SimpleNamespace

    existing = SimpleNamespace(hf_token="", token_factory_api_key="", ngc_api_key="")
    asked: list[str] = []

    def ask(prompt, *, default="", secret=False):
        asked.append(prompt)
        return "hf_new" if "HF_TOKEN" in prompt else ""

    hf, _tf, _ngc = cli_main._prompt_setup_tokens(ask, existing, skip={"NGC_API_KEY"})

    assert hf == "hf_new"
    assert any("HF_TOKEN" in p for p in asked)
    assert not any("NGC_API_KEY" in p for p in asked)  # NGC was skipped


def test_configure_token_factory_key_without_a_nebius_profile(
    monkeypatch, tmp_path
) -> None:
    """Storing only the Token Factory key must not report that nothing was written."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    creds_path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    # No authenticated Nebius CLI profile: the laptop-only inference case.
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: False)
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "v1.test-key")

    result = runner.invoke(
        app,
        ["configure", "--show", "--save-env-credentials"],
    )

    assert result.exit_code == 0, result.output
    assert "Nothing was written under ~/.npa." not in result.output
    assert (
        yaml.safe_load(creds_path.read_text())["tokens"]["NEBIUS_TOKEN_FACTORY_KEY"]
        == "v1.test-key"
    )


def _fresh_configure_paths(monkeypatch, tmp_path):
    """Point configure at empty tmp dotfiles and a ready Nebius profile."""
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    return creds_path, config_path


def test_configure_discovery_recovers_tenant_from_profile_project(
    monkeypatch, tmp_path
) -> None:
    """A profile with parent-id but no tenant-id still discovers projects.

    Regression: `nebius config get tenant-id` is empty for federation profiles
    and single-project profiles, and discovery bailed out silently, dropping the
    operator into hand-typing tenant/project ids.
    """
    import yaml
    import npa.clients.nebius as nebius_module

    creds_path, config_path = _fresh_configure_paths(monkeypatch, tmp_path)
    _stub_nebius_defaults(monkeypatch, project="project-prod", tenant="")
    # Only the project -> parent-tenant lookup knows the tenant.
    monkeypatch.setattr(
        nebius_module, "get_project_tenant_id", lambda project_id: "tenant-a"
    )
    seen_tenants: list[str] = []

    def _list_projects(tenant_id):
        seen_tenants.append(tenant_id)
        return [
            {
                "id": "project-prod",
                "name": "prod",
                "tenant_id": "tenant-a",
                "region": "eu-north1",
            }
        ]

    monkeypatch.setattr(nebius_module, "list_projects_in_tenant", _list_projects)

    # select the single project; accept the profile bind; decline storage;
    # HF/TF/NGC empty.
    answers = "\n".join(["1", "", "N", "", "", ""]) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert seen_tenants == ["tenant-a"]
    assert "the parent tenant of project project-prod" in result.output
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["projects"]["prod"]["tenant_id"] == "tenant-a"
    assert creds_path.exists()


def test_configure_project_scoped_profile_uses_profile_defaults(
    monkeypatch, tmp_path
) -> None:
    """A tenant-list denial must not strand a project-scoped profile."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-scoped", tenant="tenant-scoped")
    # list_projects_in_tenant is best-effort and returns [] when the profile has
    # no tenant-wide list permission.
    monkeypatch.setattr(nebius_module, "list_projects_in_tenant", lambda _tenant: [])

    result = runner.invoke(
        app,
        ["configure", "--interactive", "--provision"],
        input="\n".join([""] * 12) + "\n",
    )

    assert result.exit_code == 0, result.output
    assert "expected for project-scoped IAM access" in result.output
    config = yaml.safe_load(config_path.read_text())
    stanza = next(iter(config["projects"].values()))
    assert stanza["tenant_id"] == "tenant-scoped"
    assert stanza["project_id"] == "project-scoped"


def test_configure_discovery_prompts_when_several_tenants(
    monkeypatch, tmp_path
) -> None:
    """No tenant-id and no parent-id: pick from the listable tenants."""
    import npa.clients.nebius as nebius_module

    _fresh_configure_paths(monkeypatch, tmp_path)
    _stub_nebius_defaults(monkeypatch, project="", tenant="")
    monkeypatch.setattr(
        nebius_module,
        "list_tenants",
        lambda: [
            {"id": "tenant-a", "name": "alpha", "region": "eu-north1"},
            {"id": "tenant-b", "name": "beta", "region": "us-central1"},
        ],
    )
    seen_tenants: list[str] = []

    def _list_projects(tenant_id):
        seen_tenants.append(tenant_id)
        return [
            {
                "id": "project-beta",
                "name": "beta-proj",
                "tenant_id": tenant_id,
                "region": "us-central1",
            }
        ]

    monkeypatch.setattr(nebius_module, "list_projects_in_tenant", _list_projects)

    # pick tenant 2, select the single project, accept the profile bind,
    # decline storage, no tokens.
    answers = "\n".join(["2", "1", "", "N", "", "", ""]) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert seen_tenants == ["tenant-b"]
    assert "tenant-a" in result.output and "tenant-b" in result.output


def test_configure_explains_why_discovery_was_skipped(monkeypatch, tmp_path) -> None:
    """With no tenant recoverable, say so instead of silently going manual."""
    import npa.clients.nebius as nebius_module

    _fresh_configure_paths(monkeypatch, tmp_path)
    _stub_nebius_defaults(monkeypatch, project="", tenant="")
    monkeypatch.setattr(nebius_module, "list_tenants", lambda: [])

    def _must_not_discover(_tenant_id):
        raise AssertionError("discovery must not run without a tenant")

    monkeypatch.setattr(nebius_module, "list_projects_in_tenant", _must_not_discover)
    monkeypatch.setattr(
        nebius_module,
        "bucket_exists",
        lambda _project_id, bucket_name: bucket_name == "b",
    )
    monkeypatch.setattr(
        nebius_module,
        "bootstrap_environment",
        lambda *_a, **_k: {
            "nebius_api_key": "AK",
            "nebius_secret_key": "SK",
            "s3_bucket": "b",
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        },
    )

    # manual: tenant, project, region, profile-bind(n), bucket, tokens
    answers = "\n".join(
        ["tenant-x", "project-x", "", "n", "b", "", "", ""]
    ) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert "Skipping project discovery" in result.output
    assert "nebius config set tenant-id" in result.output


def test_configure_secret_prompts_are_visible_on_non_tty(monkeypatch, tmp_path) -> None:
    """Piped stdin must not ask getpass to hide input (GetPassWarning)."""
    import npa.clients.nebius as nebius_module

    _fresh_configure_paths(monkeypatch, tmp_path)
    _stub_nebius_defaults(monkeypatch, project="", tenant="")
    monkeypatch.setattr(nebius_module, "list_tenants", lambda: [])
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: True)
    monkeypatch.setattr(
        nebius_module,
        "bootstrap_environment",
        lambda *_a, **_k: {
            "nebius_api_key": "AK",
            "nebius_secret_key": "SK",
            "s3_bucket": "b",
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        },
    )

    hide_flags: list[bool] = []
    real_prompt = cli_main.typer.prompt

    def _record_prompt(label, **kwargs):
        hide_flags.append(bool(kwargs.get("hide_input")))
        return real_prompt(label, **kwargs)

    monkeypatch.setattr(cli_main.typer, "prompt", _record_prompt)
    # CliRunner always pipes stdin, so isatty() is False here.
    answers = (
        "\n".join(["tenant-x", "project-x", "", "", "n", "", "hf_x", "", ""]) + "\n"
    )
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert hide_flags and not any(hide_flags), (
        "secret prompts must not hide input on a pipe"
    )
    assert result.output.count("stdin is not a terminal") == 1


def test_configure_binds_nebius_profile_to_selected_project(
    monkeypatch, tmp_path
) -> None:
    """Accepting the prompt re-points the Nebius CLI profile at the project."""
    import npa.clients.nebius as nebius_module

    _fresh_configure_paths(monkeypatch, tmp_path)
    bound = _stub_nebius_defaults(monkeypatch, project="project-other", tenant="")
    monkeypatch.setattr(nebius_module, "get_project_tenant_id", lambda _p: "")
    monkeypatch.setattr(nebius_module, "list_tenants", lambda: [])
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: True)
    monkeypatch.setattr(
        nebius_module,
        "bootstrap_environment",
        lambda *_a, **_k: {
            "nebius_api_key": "AK",
            "nebius_secret_key": "SK",
            "s3_bucket": "b",
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        },
    )

    answers = "\n".join(
        ["tenant-x", "project-x", "", "", "b", "", "", "", ""]
    ) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert bound == [("project-x", "tenant-x")]
    assert "now points at project-x" in result.output


def test_configure_declining_profile_binding_leaves_it_alone(
    monkeypatch, tmp_path
) -> None:
    import npa.clients.nebius as nebius_module

    _fresh_configure_paths(monkeypatch, tmp_path)
    bound = _stub_nebius_defaults(monkeypatch, project="project-other", tenant="")
    monkeypatch.setattr(nebius_module, "get_project_tenant_id", lambda _p: "")
    monkeypatch.setattr(nebius_module, "list_tenants", lambda: [])
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: True)
    monkeypatch.setattr(
        nebius_module,
        "bootstrap_environment",
        lambda *_a, **_k: {
            "nebius_api_key": "AK",
            "nebius_secret_key": "SK",
            "s3_bucket": "b",
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        },
    )

    answers = "\n".join(
        ["tenant-x", "project-x", "", "n", "b", "", "", ""]
    ) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert bound == []
    assert "nebius config set parent-id project-x" in result.output


def test_configure_manual_alias_uses_project_name_not_region(
    monkeypatch, tmp_path
) -> None:
    """A region-shaped alias (`us-central1`) is confusing next to `region:`."""
    import yaml
    import npa.clients.nebius as nebius_module

    _creds_path, config_path = _fresh_configure_paths(monkeypatch, tmp_path)
    _stub_nebius_defaults(
        monkeypatch, project="", tenant="", project_name="TLE Workbench"
    )
    monkeypatch.setattr(nebius_module, "list_tenants", lambda: [])
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: True)
    monkeypatch.setattr(
        nebius_module,
        "bootstrap_environment",
        lambda *_a, **_k: {
            "nebius_api_key": "AK",
            "nebius_secret_key": "SK",
            "s3_bucket": "b",
            "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        },
    )

    answers = (
        "\n".join(["tenant-x", "project-x", "us-central1", "", "n", "", "", "", ""])
        + "\n"
    )
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["default_project"] == "tle-workbench"
    assert cfg["projects"]["tle-workbench"]["region"] == "us-central1"


def test_configure_manual_alias_falls_back_to_region(monkeypatch, tmp_path) -> None:
    """With no readable project name the region alias is still used."""
    import yaml
    import npa.clients.nebius as nebius_module

    _creds_path, config_path = _fresh_configure_paths(monkeypatch, tmp_path)
    _stub_nebius_defaults(monkeypatch, project="", tenant="", project_name="")
    monkeypatch.setattr(nebius_module, "list_tenants", lambda: [])
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: True)
    monkeypatch.setattr(
        nebius_module,
        "bootstrap_environment",
        lambda *_a, **_k: {
            "nebius_api_key": "AK",
            "nebius_secret_key": "SK",
            "s3_bucket": "b",
            "s3_endpoint": "https://storage.us-central1.nebius.cloud",
        },
    )

    answers = (
        "\n".join(["tenant-x", "project-x", "us-central1", "", "n", "", "", "", ""])
        + "\n"
    )
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["default_project"] == "us-central1"


def test_configure_interactive_provisions_storage(monkeypatch, tmp_path) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-12345", tenant="tenant-abcde")

    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: False)

    bootstrap_calls: list[dict] = []

    def fake_bootstrap(
        project_id,
        tenant_id,
        region,
        *,
        bucket_name=None,
        bucket_max_size_bytes=0,
        bucket_storage_class="standard",
        on_status=None,
        on_resource_created=None,
    ):
        bootstrap_calls.append(
            {
                "project_id": project_id,
                "tenant_id": tenant_id,
                "region": region,
                "bucket_name": bucket_name,
                "bucket_max_size_bytes": bucket_max_size_bytes,
                "bucket_storage_class": bucket_storage_class,
            }
        )
        if on_status:
            on_status("Setting up S3 bucket...")
        if on_resource_created:
            on_resource_created(
                "service_account",
                {"id": "serviceaccount-storage", "name": "lerobot-training"},
            )
            on_resource_created("bucket", {"name": bucket_name})
            on_resource_created(
                "access_key",
                {"id": "accesskey-storage", "name": "lerobot-access-key"},
            )
        return {
            "nebius_api_key": "AKIAPROVISIONED",
            "nebius_secret_key": "provisioned-secret",
            "s3_bucket": bucket_name,
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
            "service_account_id": "serviceaccount-storage",
            "service_account_name": "lerobot-training",
            "service_account_project_id": project_id,
            "service_account_managed_by": "npa",
        }

    monkeypatch.setattr(nebius_module, "bootstrap_environment", fake_bootstrap)

    # Enter project/tenant + default region; pick a custom bucket name and size;
    # then HF + Token Factory + NGC.
    answers = (
        "\n".join(
            [
                "tenant-abcde",  # tenant id
                "project-12345",  # project id
                "",  # region (default eu-north1)
                "my-bucket",  # bucket name (customer choice)
                "",  # storage class (standard default)
                "100",  # size in GB
                "hf_secret_token",  # HF token
                "nebius_secret_key",  # Nebius Token Factory API key
                "nvapi_secret",  # NGC API key
            ]
        )
        + "\n"
    )

    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert "project alias: eu-north1" in result.output
    assert "-p eu-north1" in result.output
    assert len(bootstrap_calls) == 1
    call = bootstrap_calls[0]
    assert (call["project_id"], call["tenant_id"], call["region"]) == (
        "project-12345",
        "tenant-abcde",
        "eu-north1",
    )
    assert call["bucket_name"] == "my-bucket"
    assert call["bucket_max_size_bytes"] == 100 * 1024**3
    assert call["bucket_storage_class"] == "standard"

    creds = yaml.safe_load(creds_path.read_text())
    assert creds["tokens"]["HF_TOKEN"] == "hf_secret_token"
    assert "NEBIUS_AI_CLOUD_KEY" not in creds["tokens"]
    assert creds["tokens"]["NEBIUS_TOKEN_FACTORY_KEY"] == "nebius_secret_key"
    assert creds["ngc"]["api_key"] == "nvapi_secret"
    assert creds["storage"]["aws_access_key_id"] == "AKIAPROVISIONED"
    assert creds["storage"]["aws_secret_access_key"] == "provisioned-secret"
    assert creds["storage"]["endpoint_url"] == "https://storage.eu-north1.nebius.cloud"
    assert creds["storage"]["bucket"] == "s3://my-bucket/"
    assert creds["nebius"] == {"service_account_id": "serviceaccount-storage"}
    assert creds["storage_iam"] == {
        "service_account_id": "serviceaccount-storage",
        "service_account_name": "lerobot-training",
        "service_account_project_id": "project-12345",
        "service_account_managed_by": "npa",
    }

    cfg = yaml.safe_load(config_path.read_text())
    project = cfg["projects"]["eu-north1"]
    assert cfg["default_project"] == "eu-north1"
    assert project["project_id"] == "project-12345"
    assert project["tenant_id"] == "tenant-abcde"
    assert "container_registry" not in project
    assert oct(creds_path.stat().st_mode)[-3:] == "600"


def test_configure_provision_reuses_explicit_bucket_without_size_prompt(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
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
        bucket_storage_class="standard",
        on_status=None,
        on_resource_created=None,
    ):
        sizes.append(bucket_max_size_bytes)
        return {
            "nebius_api_key": "AKIA",
            "nebius_secret_key": "secret",
            "s3_bucket": bucket_name,
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        }

    monkeypatch.setattr(nebius_module, "bootstrap_environment", fake_bootstrap)

    # proj, tenant, region, exact existing bucket, HF, token factory, NGC
    answers = "\n".join(
        ["tenant-1", "project-1", "", "existing-bucket", "", "", ""]
    ) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert "Reusing existing object-storage bucket" in result.output
    assert sizes == [0]
    creds = yaml.safe_load(creds_path.read_text())
    assert creds["storage"]["bucket"] == "s3://existing-bucket/"


def _run_reuse_bucket_configure(monkeypatch, tmp_path, *, hf_token: str, ngc_key: str):
    """Drive a successful reuse-existing-bucket `npa configure` and return output.

    Uses the reuse path (bucket_exists=True) so there are no storage-class/size
    prompts; answers are: project, tenant, region, bucket-name(reuse), HF,
    Token Factory, and NGC.
    """

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: True)

    def fake_bootstrap(
        project_id,
        tenant_id,
        region,
        *,
        bucket_name=None,
        bucket_max_size_bytes=0,
        bucket_storage_class="standard",
        on_status=None,
        on_resource_created=None,
    ):
        return {
            "nebius_api_key": "AKIA",
            "nebius_secret_key": "secret",
            "s3_bucket": bucket_name,
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        }

    monkeypatch.setattr(nebius_module, "bootstrap_environment", fake_bootstrap)

    answers = (
        "\n".join(
            [
                "tenant-1",
                "project-1",
                "",
                "existing-bucket",
                hf_token,
                "",
                ngc_key,
            ]
        )
        + "\n"
    )
    return runner.invoke(app, ["configure", "--interactive"], input=answers)


def _note_line(output: str) -> str:
    lines = [line for line in output.splitlines() if line.startswith("[NOTE]")]
    assert lines, f"expected a [NOTE] line in output:\n{output}"
    assert len(lines) == 1, f"expected exactly one [NOTE] line, got: {lines}"
    return lines[0]


def test_configure_prints_model_access_note_all_ok(monkeypatch, tmp_path) -> None:
    # The autouse fixture makes both canonical live access probes succeed.
    result = _run_reuse_bucket_configure(
        monkeypatch, tmp_path, hf_token="hf_good", ngc_key="nvapi-good"
    )
    assert result.exit_code == 0, result.output
    note = _note_line(result.output)
    assert "Access checks are informational" in note
    assert "HF token valid; gated access confirmed" in note
    assert "NGC key valid; repository access confirmed" in note


def test_configure_ngc_audit_defers_registry_credential_validity_to_provider(
    monkeypatch, caplog
) -> None:
    secret = "registry-credential"
    observed: list[str] = []

    def validate(key: str, *, timeout: float = 30.0) -> str:
        del timeout
        observed.append(key)
        return "reachable"

    monkeypatch.setattr(
        "npa.workbench.nurec.nurec.check_ngc_image_access", validate
    )
    caplog.set_level("DEBUG", logger="npa.cli.main")
    note = cli_main._model_access_note("hf_good", secret)

    assert observed == [secret]
    assert "NGC key valid; repository access confirmed" in note
    assert secret not in note
    assert secret not in caplog.text


def test_configure_hf_probe_preserves_gated_dataset_type(monkeypatch, tmp_path) -> None:
    from npa.clients import huggingface
    from npa.clients.huggingface import HFAccessResult

    observed: dict[str, str] = {}

    def _record(
        token,
        repo,
        repo_type="model",
        revision="",
        probe_path="",
        *,
        timeout=10.0,
    ):
        observed[repo] = repo_type
        return HFAccessResult(repo=repo, ok=True, status_code=200)

    monkeypatch.setattr(huggingface, "validate_hf_access", _record)

    result = _run_reuse_bucket_configure(
        monkeypatch, tmp_path, hf_token="hf_synthetic", ngc_key="nvapi-synthetic"
    )

    assert result.exit_code == 0, result.output
    assert observed["nvidia/PhysicalAI-Autonomous-Vehicles"] == "dataset"


def test_configure_note_lists_inaccessible_hf_models(monkeypatch, tmp_path) -> None:
    from npa.clients import huggingface
    from npa.clients.huggingface import HFAccessResult

    denied = "nvidia/Cosmos-Reason2-2B"

    def _deny_one(
        token,
        repo,
        repo_type="model",
        revision="",
        probe_path="",
        *,
        timeout=10.0,
    ):
        if repo == denied:
            return HFAccessResult(
                repo=repo, ok=False, status_code=403, error="no access"
            )
        return HFAccessResult(repo=repo, ok=True, status_code=200)

    monkeypatch.setattr(huggingface, "validate_hf_access", _deny_one)

    result = _run_reuse_bucket_configure(
        monkeypatch, tmp_path, hf_token="hf_partial", ngc_key="nvapi-good"
    )
    assert result.exit_code == 0, result.output
    note = _note_line(result.output)
    assert "HF has no access to:" in note
    assert denied in note
    assert "huggingface.co" in note


def test_configure_note_reports_ngc_entitlement_rejection(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "npa.workbench.nurec.nurec.check_ngc_image_access",
        lambda key, *, timeout=30.0: "entitlement-required",
    )
    secret = "nvapi-synthetic-rejected"

    result = _run_reuse_bucket_configure(
        monkeypatch, tmp_path, hf_token="hf_synthetic", ngc_key=secret
    )

    assert result.exit_code == 0, result.output
    note = _note_line(result.output)
    assert "NGC repository entitlement denied for: nurec" in note
    assert "npa workbench health access" in note
    assert secret not in note


def test_configure_note_lists_ngc_blocked_when_key_missing(
    monkeypatch, tmp_path
) -> None:
    # HF probes all succeed (autouse); NGC key omitted -> NVIDIA pulls blocked.
    result = _run_reuse_bucket_configure(
        monkeypatch, tmp_path, hf_token="hf_good", ngc_key=""
    )
    assert result.exit_code == 0, result.output
    note = _note_line(result.output)
    assert "NGC not configured" in note
    assert "nurec" in note
    assert "groot" not in note and "cosmos" not in note


def test_configure_note_keeps_optional_hf_and_ngc_credentials_non_blocking(
    monkeypatch, tmp_path
) -> None:
    result = _run_reuse_bucket_configure(
        monkeypatch, tmp_path, hf_token="", ngc_key=""
    )

    assert result.exit_code == 0, result.output
    note = _note_line(result.output)
    assert "NGC not configured" in note
    assert "model(s) unverified" in note
    assert "npa workbench health access" in note


def test_configure_note_never_breaks_on_probe_error(
    monkeypatch, tmp_path, caplog
) -> None:
    from npa.clients import huggingface

    secret = "hf_synthetic_not_for_logs"

    def _boom(
        token,
        repo,
        repo_type="model",
        revision="",
        probe_path="",
        *,
        timeout=10.0,
    ):
        raise RuntimeError(f"network exploded for {token}")

    monkeypatch.setattr(huggingface, "validate_hf_access", _boom)
    caplog.set_level("DEBUG", logger="npa.cli.main")

    result = _run_reuse_bucket_configure(
        monkeypatch, tmp_path, hf_token=secret, ngc_key="nvapi-good"
    )
    # A probe blowing up must not break configure; each affected model is simply
    # reported as unverified on the single NOTE line.
    assert result.exit_code == 0, result.output
    note = _note_line(result.output)
    assert note.startswith("[NOTE]")
    assert "unverified" in note
    assert secret not in note
    assert secret not in caplog.text


def _prepopulate_config(monkeypatch, tmp_path):
    """Write an existing config + credentials and point configure at them."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "prod",
                "projects": {
                    "prod": {
                        "project_id": "project-existing",
                        "tenant_id": "tenant-existing",
                        "region": "eu-north1",
                        "container_registry": "registry.example/customer-existing",
                    }
                },
            }
        )
    )
    creds_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {
                    "HF_TOKEN": "hf_existing",
                    "NEBIUS_TOKEN_FACTORY_KEY": "tf_existing",
                },
                "ngc": {"api_key": "nvapi-existing"},
                "storage": {
                    "aws_access_key_id": "AK_existing",
                    "aws_secret_access_key": "SK_existing",
                    "endpoint_url": "https://storage.eu-north1.nebius.cloud",
                    "bucket": "s3://npa-bucket-existing/",
                },
                "storage_setup": {
                    "version": 1,
                    "projects": {
                        "project-existing": {
                            "status": "complete",
                            "bucket_name": "npa-bucket-existing",
                        }
                    },
                },
            }
        )
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(
        monkeypatch, project="project-existing", tenant="tenant-existing"
    )
    monkeypatch.setattr(
        nebius_module,
        "bucket_exists",
        lambda _project_id, bucket_name: bucket_name == "npa-bucket-existing",
    )
    return creds_path, config_path, nebius_module


def test_configure_rerun_all_defaults_is_idempotent(monkeypatch, tmp_path) -> None:
    import yaml

    creds_path, config_path, nebius_module = _prepopulate_config(monkeypatch, tmp_path)

    def _must_not_provision(*_a, **_k):
        raise AssertionError(
            "bootstrap_environment should not run on an all-defaults re-run"
        )

    monkeypatch.setattr(nebius_module, "bootstrap_environment", _must_not_provision)

    # project, tenant, region, keep-storage, HF, TF, NGC
    answers = "\n".join([""] * 7) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output

    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["default_project"] == "prod"
    prod = cfg["projects"]["prod"]
    assert prod["project_id"] == "project-existing"
    assert prod["tenant_id"] == "tenant-existing"
    assert prod["region"] == "eu-north1"
    assert prod["container_registry"] == "registry.example/customer-existing"

    creds = yaml.safe_load(creds_path.read_text())
    assert creds["tokens"]["HF_TOKEN"] == "hf_existing"
    assert "NEBIUS_AI_CLOUD_KEY" not in creds["tokens"]
    assert creds["tokens"]["NEBIUS_TOKEN_FACTORY_KEY"] == "tf_existing"
    assert creds["ngc"]["api_key"] == "nvapi-existing"
    # Storage preserved verbatim — no new access key minted.
    assert creds["storage"]["aws_access_key_id"] == "AK_existing"
    assert creds["storage"]["aws_secret_access_key"] == "SK_existing"
    assert creds["storage"]["bucket"] == "s3://npa-bucket-existing/"


def test_configure_rerun_updates_selected_values(monkeypatch, tmp_path) -> None:
    import yaml

    creds_path, config_path, nebius_module = _prepopulate_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        nebius_module,
        "bootstrap_environment",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should keep storage")),
    )

    # Update the HF token; keep project identity and verified storage.
    answers = (
        "\n".join(
            [
                "",  # tenant id (keep)
                "",  # project id (keep)
                "",  # region (keep)
                "",  # keep existing storage? -> Y
                "hf_new",  # HF token (update)
                "",  # token factory (keep)
                "",  # NGC (keep)
            ]
        )
        + "\n"
    )
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output

    cfg = yaml.safe_load(config_path.read_text())
    # Updated stanza lives under the preserved alias.
    prod = cfg["projects"]["prod"]
    assert cfg["default_project"] == "prod"
    assert prod["project_id"] == "project-existing"
    assert prod["tenant_id"] == "tenant-existing"

    creds = yaml.safe_load(creds_path.read_text())
    assert creds["tokens"]["HF_TOKEN"] == "hf_new"
    assert creds["storage"]["aws_access_key_id"] == "AK_existing"


def test_configure_rerun_can_reprovision_storage_when_declined(
    monkeypatch, tmp_path
) -> None:
    import yaml

    creds_path, config_path, nebius_module = _prepopulate_config(monkeypatch, tmp_path)

    calls: list[dict] = []

    def fake_bootstrap(
        project_id,
        tenant_id,
        region,
        *,
        bucket_name=None,
        bucket_max_size_bytes=0,
        bucket_storage_class="standard",
        on_status=None,
        on_resource_created=None,
        allow_existing_bucket=True,
    ):
        calls.append(
            {
                "bucket_name": bucket_name,
                "allow_existing_bucket": allow_existing_bucket,
            }
        )
        return {
            "nebius_api_key": "AK_new",
            "nebius_secret_key": "SK_new",
            "s3_bucket": bucket_name,
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        }

    monkeypatch.setattr(nebius_module, "bootstrap_environment", fake_bootstrap)

    # Decline keep-existing storage -> provision; the editable bucket prompt
    # defaults to a fresh project-scoped proposal, never the declined name.
    answers = (
        "\n".join(
            [
                "",  # tenant id (keep)
                "",  # project id (keep)
                "",  # region (keep)
                "n",  # keep existing storage? -> no
                "",  # bucket name (fresh project-scoped default)
                "",  # storage class (standard default)
                "",  # size cap (recommended default)
                "",  # HF (keep)
                "",  # token factory (keep)
                "",  # NGC (keep)
            ]
        )
        + "\n"
    )
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert re.fullmatch(
        r"npa-bucket-\d{8}t\d{6}z-[0-9a-f]{6}-[0-9a-f]{8}",
        calls[0]["bucket_name"],
    )
    assert calls[0]["bucket_name"] != "npa-bucket-existing"
    assert calls[0]["allow_existing_bucket"] is False
    creds = yaml.safe_load(creds_path.read_text())
    assert creds["storage"]["aws_access_key_id"] == "AK_new"


def test_configure_provision_falls_back_to_manual_on_error(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: False)

    def boom(*_args, **_kwargs):
        # Mirror the real Nebius object-storage authorization failure.
        raise nebius_module.NebiusError(
            "nebius storage bucket list failed (exit 15):\n"
            "Error: rpc error: code = PermissionDenied desc = AccessDenied: Access denied"
        )

    monkeypatch.setattr(nebius_module, "bootstrap_environment", boom)

    answers = (
        "\n".join(
            [
                "tenant-1",  # tenant id
                "project-1",  # project id
                "",  # region (default)
                "provision-bucket",  # bucket name
                "",  # storage class (standard default)
                "",  # size GB (default 50)
                "n",  # skip object storage? -> no, enter manually
                "AKIAMANUAL",  # S3 access key (fallback)
                "manual-secret",  # S3 secret (fallback)
                "",  # S3 endpoint (default-by-region)
                "s3://manual-bucket/",  # S3 bucket (fallback)
                "hf_tok",  # HF token
                "",  # Token Factory API key (skip)
                "",  # NGC API key (skip)
            ]
        )
        + "\n"
    )

    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert "Could not auto-provision" in result.output
    # Access-denied provisioning should surface actionable IAM guidance, not a
    # bare rpc dump, before falling back to manual entry.
    assert "access denied" in result.output.lower()
    assert "storage.object-editor" in result.output
    assert "NPA_ALLOW_EDITORS_STORAGE_FALLBACK=1" in result.output
    assert "re-run `npa configure`" in result.output
    creds = yaml.safe_load(creds_path.read_text())
    assert creds["storage"]["aws_access_key_id"] == "AKIAMANUAL"
    assert creds["storage"]["aws_secret_access_key"] == "manual-secret"
    assert creds["storage"]["endpoint_url"] == "https://storage.eu-north1.nebius.cloud"
    assert creds["storage"]["bucket"] == "s3://manual-bucket/"
    assert "api_key" not in creds.get("ngc", {})


def _bootstrap_capture(calls: list[dict]):
    def fake_bootstrap(
        project_id,
        tenant_id,
        region,
        *,
        bucket_name=None,
        bucket_max_size_bytes=0,
        bucket_storage_class="standard",
        on_status=None,
        on_resource_created=None,
    ):
        calls.append(
            {
                "bucket_name": bucket_name,
                "bucket_max_size_bytes": bucket_max_size_bytes,
                "bucket_storage_class": bucket_storage_class,
            }
        )
        return {
            "nebius_api_key": "AKIA",
            "nebius_secret_key": "secret",
            "s3_bucket": bucket_name,
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        }

    return fake_bootstrap


def test_configure_skips_storage_and_still_writes_tokens_on_provision_failure(
    monkeypatch, tmp_path
) -> None:
    """Storage AccessDenied must not dead-end: skip storage, still write tokens."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")
    # Existence search + provisioning both denied.
    monkeypatch.setattr(
        nebius_module,
        "bucket_exists",
        lambda *_a, **_k: (_ for _ in ()).throw(
            nebius_module.NebiusError("AccessDenied")
        ),
    )

    def boom(*_a, **_k):
        raise nebius_module.NebiusError("AccessDenied: Access denied")

    monkeypatch.setattr(nebius_module, "bootstrap_environment", boom)

    # proj, tenant, region, bucket, skip-storage (Enter=Y), HF, TF, NGC.
    answers = (
        "\n".join(
            [
                "tenant-1",
                "project-1",
                "",
                "npa-tle-727",
                "",
                "hf_tok",
                "",
                "",
            ]
        )
        + "\n"
    )
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert "Skip object storage for now and finish setup?" in result.output
    assert "Skipping object storage" in result.output
    assert (
        "Setup incomplete: writable object storage is not configured" in result.output
    )
    assert "provision-if-absent --project" in result.output
    assert "Setup complete" not in result.output
    creds = yaml.safe_load(creds_path.read_text())
    assert creds["tokens"]["HF_TOKEN"] == "hf_tok"
    # No storage stanza (or empty) was written.
    assert not creds.get("storage")


def test_configure_accepts_region_without_registry_prompt(monkeypatch, tmp_path) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: True)
    monkeypatch.setattr(nebius_module, "bootstrap_environment", _bootstrap_capture([]))

    answers = (
        "\n".join(
            [
                "tenant-1",
                "project-1",
                "us-central1",
                "my-bucket",
                "",  # hf
                "",  # tf
                "",  # ngc
            ]
        )
        + "\n"
    )
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert "Container registry" not in result.output
    assert config_module.resolve_environment().region == "us-central1"


def test_normalize_pasted_secret_strips_quotes_and_auth_prefixes() -> None:
    n = cli_main._normalize_pasted_secret
    assert n('  "hf_abc123"  ') == "hf_abc123"
    assert n("Bearer v1.xyz") == "v1.xyz"
    assert n("bearer v1.xyz") == "v1.xyz"
    assert n("Authorization: Bearer nvapi-xyz") == "nvapi-xyz"
    assert n("'v1.abc'") == "v1.abc"
    assert n('"Bearer v1.abc"') == "v1.abc"
    # A bare token is unchanged.
    assert n("hf_plain") == "hf_plain"
    assert n("") == ""
    # AWS keys may legitimately begin with words that look like auth schemes;
    # the S3 prompt opts into quote-only normalization.
    assert n('"Bearer valid-aws-secret"', strip_auth_wrapper=False) == (
        "Bearer valid-aws-secret"
    )
    assert n("Authorization:valid-access-key", strip_auth_wrapper=False) == (
        "Authorization:valid-access-key"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        ("-1", 0),
        ("1", 1024**3),
        ("0.5", 512 * 1024**2),
        ("0.000000001", 1),
    ],
)
def test_gb_to_bytes_has_explicit_gib_flooring_boundaries(
    value: str, expected: int
) -> None:
    assert cli_main._gb_to_bytes(value) == expected


def test_gb_to_bytes_invalid_or_nonfinite_uses_recommended_cap() -> None:
    expected = int(cli_main.RECOMMENDED_BUCKET_SIZE_GB) * 1024**3
    assert cli_main._gb_to_bytes("invalid") == expected
    assert cli_main._gb_to_bytes("nan") == expected
    assert cli_main._gb_to_bytes("inf") == expected


def test_configure_normalizes_tokens_and_warns_on_bad_token_factory_key(
    monkeypatch, tmp_path
) -> None:
    """Pasted Bearer/quoted tokens are stored bare; a non-v1. TF key warns."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: True)
    monkeypatch.setattr(nebius_module, "bootstrap_environment", _bootstrap_capture([]))

    # proj, tenant, region, bucket, hf (Bearer+quoted),
    # token factory (quoted, not v1.), ngc (Bearer), alias.
    answers = (
        "\n".join(
            [
                "tenant-1",
                "project-1",
                "",
                "my-bucket",
                'Bearer "hf_abc123"',
                '"nebius-iam-looking-token"',
                "Bearer nvapi-xyz",
                "",
            ]
        )
        + "\n"
    )
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert "does not look like a Token Factory key" in result.output
    creds = yaml.safe_load(creds_path.read_text())
    assert creds["tokens"]["HF_TOKEN"] == "hf_abc123"
    assert creds["tokens"]["NEBIUS_TOKEN_FACTORY_KEY"] == "nebius-iam-looking-token"
    assert creds["ngc"]["api_key"] == "nvapi-xyz"


def test_configure_typed_existing_bucket_is_reused_without_create_prompts(
    monkeypatch, tmp_path
) -> None:
    """A typed name that already exists is reused; no storage-class/size prompts."""
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")

    searched: list[tuple[str, str]] = []

    def fake_bucket_exists(project_id, bucket_name):
        searched.append((project_id, bucket_name))
        return True

    monkeypatch.setattr(nebius_module, "bucket_exists", fake_bucket_exists)
    calls: list[dict] = []
    monkeypatch.setattr(
        nebius_module, "bootstrap_environment", _bootstrap_capture(calls)
    )

    # proj, tenant, region, bucket name, HF, TF, NGC.
    # No storage-class/size answers: none should be prompted for a reused bucket.
    answers = (
        "\n".join(
            [
                "tenant-1",
                "project-1",
                "",
                "my-existing-bucket",
                "hf_tok",
                "",
                "",
            ]
        )
        + "\n"
    )
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert searched == [("project-1", "my-existing-bucket")]
    assert (
        "Reusing existing object-storage bucket 'my-existing-bucket'" in result.output
    )
    assert "New bucket storage class" not in result.output
    assert calls and calls[0]["bucket_name"] == "my-existing-bucket"
    assert calls[0]["bucket_max_size_bytes"] == 0


def test_configure_bucket_search_failure_fails_closed_before_create(
    monkeypatch, tmp_path
) -> None:
    """When existence can't be verified, npa skips create prompts and get-or-creates."""
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")

    def boom_exists(*_a, **_k):
        raise nebius_module.NebiusError("nebius storage bucket list failed (exit 15)")

    monkeypatch.setattr(nebius_module, "bucket_exists", boom_exists)
    calls: list[dict] = []
    monkeypatch.setattr(
        nebius_module, "bootstrap_environment", _bootstrap_capture(calls)
    )

    answers = (
        "\n".join(
            [
                "tenant-1",
                "project-1",
                "",
                "maybe-existing",
                "",  # skip object storage after the verification failure
                "hf_tok",
                "",
                "",
                "",
                "",
            ]
        )
        + "\n"
    )
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert "Could not verify whether 'maybe-existing' already exists" in result.output
    assert "will not create or adopt it" in result.output
    assert "New bucket storage class" not in result.output
    assert calls == []


def test_noninteractive_bucket_collision_uses_a_fresh_collision_safe_name(
    monkeypatch,
) -> None:
    import npa.clients.nebius as nebius_module

    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: False)
    requested: list[str] = []

    def bootstrap(*_args, bucket_name, **_kwargs):
        requested.append(bucket_name)
        if len(requested) == 1:
            raise nebius_module.NebiusError(
                f"Object-storage bucket name '{bucket_name}' is already taken"
            )
        return {
            "nebius_api_key": "access",
            "nebius_secret_key": "secret",
            "s3_bucket": bucket_name,
            "s3_endpoint": "https://storage.example",
        }

    monkeypatch.setattr(nebius_module, "bootstrap_environment", bootstrap)

    result = cli_main._provision_object_storage(
        nebius_module,
        lambda _label, *, default="", **_kwargs: default,
        project_id="project-a",
        tenant_id="tenant-a",
        region="eu-north1",
        existing_bucket="npa-bucket-collision",
        interactive=False,
    )

    assert result is not None
    assert requested[0] == "npa-bucket-collision"
    assert requested[1] != requested[0]
    assert re.fullmatch(
        r"npa-bucket-\d{8}t\d{6}z-[0-9a-f]{6}-[0-9a-f]{8}", requested[1]
    )


def test_configure_no_provision_skips_manual_storage_and_provider_calls(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")

    def must_not_call(*_args, **_kwargs):
        raise AssertionError("--no-provision must not call bootstrap_environment")

    monkeypatch.setattr(nebius_module, "bootstrap_environment", must_not_call)

    answers = (
        "\n".join(
            [
                "tenant-1",  # tenant id
                "project-1",  # project id
                "me-central1",  # region
                "AKIAMANUAL",  # S3 access key
                "manual-secret",  # S3 secret
                "",  # S3 endpoint (default-by-region)
                "s3://b/",  # S3 bucket
                "",  # HF token
                "",  # Token Factory API key
                "",  # NGC API key
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app, ["configure", "--interactive", "--no-provision"], input=answers
    )

    assert result.exit_code == 0, result.output
    # The alias is auto-derived from the region (no alias prompt).
    assert "project alias: me-central1" in result.output
    assert "-p me-central1" in result.output
    creds = yaml.safe_load(creds_path.read_text())
    assert "storage" not in creds
    assert "Object storage not selected" in result.output
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["default_project"] == "me-central1"
    assert cfg["projects"]["me-central1"]["region"] == "me-central1"


def test_configure_interactive_skips_config_without_project(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch)

    # Skip every field. With no project/tenant, provisioning is skipped and the
    # manual object-storage prompts run; only the defaulted endpoint remains.
    answers = "\n".join([""] * 12) + "\n"
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
    assert "npa configure --interactive" in result.output


def test_configure_token_factory_key_stores_under_tokens_nebius_token_factory_key(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.clients import credentials as credentials_module

    creds_path = tmp_path / "credentials.yaml"
    creds_path.write_text(yaml.safe_dump({"tokens": {"HF_TOKEN": "hf-existing"}}))
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "tf-cli-key")

    result = runner.invoke(
        app,
        ["configure", "--show", "--save-env-credentials"],
    )

    assert result.exit_code == 0, result.output
    assert "NEBIUS_TOKEN_FACTORY_KEY" in result.output
    stored = yaml.safe_load(creds_path.read_text())
    assert stored["tokens"]["NEBIUS_TOKEN_FACTORY_KEY"] == "tf-cli-key"
    assert stored["tokens"]["HF_TOKEN"] == "hf-existing"


def test_configure_interactive_does_not_migrate_legacy_token_factory_key(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    creds_path.write_text(
        yaml.safe_dump({"tokens": {"NEBIUS_TOKEN_FACTORY_API_KEY": "tf-legacy-key"}})
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch)

    answers = "\n".join([""] * 12) + "\n"
    result = runner.invoke(
        app,
        ["configure", "--interactive", "--no-provision"],
        input=answers,
    )

    assert result.exit_code == 0, result.output
    stored = yaml.safe_load(creds_path.read_text())
    assert stored["tokens"]["NEBIUS_TOKEN_FACTORY_API_KEY"] == "tf-legacy-key"
    assert "NEBIUS_TOKEN_FACTORY_KEY" not in stored["tokens"]


def test_configure_interactive_updates_selected_token_and_preserves_skipped_tokens(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    creds_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {
                    "HF_TOKEN": "hf-existing",
                    "NEBIUS_TOKEN_FACTORY_KEY": "tf-existing",
                    "NGC_API_KEY": "ngc-existing",
                },
                "storage": {
                    "aws_access_key_id": "AKIAEXISTING",
                    "aws_secret_access_key": "secret-existing",
                    "endpoint_url": "https://storage.eu-north1.nebius.cloud",
                    "bucket": "s3://existing-bucket/checkpoints/",
                },
            }
        )
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch)

    # Skip everything except HF token.
    answers = (
        "\n".join(
            [
                "",  # tenant id
                "",  # project id
                "",  # region
                "hf-updated",  # HF token
                "",  # Token Factory API key (unchanged)
                "",  # NGC API key (unchanged)
            ]
        )
        + "\n"
    )
    result = runner.invoke(
        app,
        ["configure", "--interactive", "--no-provision"],
        input=answers,
    )

    assert result.exit_code == 0, result.output
    stored = yaml.safe_load(creds_path.read_text())
    assert stored["tokens"]["HF_TOKEN"] == "hf-updated"
    assert "NEBIUS_AI_CLOUD_KEY" not in stored["tokens"]
    assert stored["tokens"]["NEBIUS_TOKEN_FACTORY_KEY"] == "tf-existing"
    assert stored["tokens"]["NGC_API_KEY"] == "ngc-existing"


def test_configure_creates_nebius_profile_when_missing(monkeypatch, tmp_path) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    # A nebius binary exists but no profile is ready until we "create" one.
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/nebius")
    readiness = iter([False, True])
    monkeypatch.setattr(cli_main, "_nebius_profile_ready", lambda **_: next(readiness))
    monkeypatch.setattr(cli_main, "_list_nebius_profiles", lambda **_: [])
    created: list[str] = []

    def fake_create(*, project_id="", **_):
        created.append(project_id)
        return True

    monkeypatch.setattr(cli_main, "_create_nebius_profile", fake_create)
    _stub_nebius_defaults(monkeypatch)

    # confirm profile, then skip all interactive fields (empty project => the
    # manual object-storage prompts run, so 12 fields follow the confirm).
    answers = "y\nproject-scoped\n" + "\n".join([""] * 12) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert created == ["project-scoped"]
    assert "Nebius CLI profile is ready." in result.output


@pytest.mark.parametrize(
    ("project_id", "expected"),
    [
        (
            "project-scoped",
            ["nebius", "profile", "create", "--parent-id", "project-scoped"],
        ),
        ("  ", ["nebius", "profile", "create"]),
    ],
)
def test_create_nebius_profile_scopes_known_project(project_id, expected) -> None:
    calls: list[list[str]] = []

    class _Result:
        returncode = 0

    def fake_runner(command, **kwargs):
        calls.append(command)
        assert kwargs == {"check": False}
        return _Result()

    assert cli_main._create_nebius_profile(project_id=project_id, runner=fake_runner)
    assert calls == [expected]


def test_configure_detects_existing_nebius_profile(monkeypatch, tmp_path) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main, "_nebius_profile_ready", lambda **_: True)

    def fail_create(**_):
        raise AssertionError("must not create a profile when one already works")

    monkeypatch.setattr(cli_main, "_create_nebius_profile", fail_create)
    _stub_nebius_defaults(monkeypatch)

    # Empty project => provisioning is skipped and the manual object-storage
    # prompts run, so 12 fields are prompted for.
    result = runner.invoke(
        app, ["configure", "--interactive"], input="\n".join([""] * 12) + "\n"
    )

    assert result.exit_code == 0, result.output
    assert "Nebius CLI profile detected" in result.output


def test_configure_existing_profile_writes_config_with_explicit_ids(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_nebius_profile_ready", lambda **_: True)
    monkeypatch.setattr(cli_main, "_create_nebius_profile", lambda **_: False)
    _stub_nebius_defaults(
        monkeypatch,
        project="project-from-profile",
        tenant="tenant-from-profile",
    )
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: True)
    monkeypatch.setattr(
        nebius_module,
        "bootstrap_environment",
        lambda *_a, **_k: {
            "nebius_api_key": "AKIAEXISTING",
            "nebius_secret_key": "existing-secret",
            "s3_bucket": "existing-bucket",
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        },
    )

    answers = (
        "\n".join(
            [
                "tenant-from-profile",  # tenant id (entered explicitly)
                "project-from-profile",  # project id (entered explicitly)
                "",  # region (accept eu-north1 default)
                "existing-bucket",  # exact existing bucket name
                "hf_from_profile",  # HF token
                "",  # Token Factory API key (skip)
                "",  # NGC API key (skip)
            ]
        )
        + "\n"
    )
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert "Nebius CLI profile detected" in result.output
    config = yaml.safe_load(config_path.read_text())
    assert config["projects"]["eu-north1"]["project_id"] == "project-from-profile"
    assert config["projects"]["eu-north1"]["tenant_id"] == "tenant-from-profile"
    assert config["projects"]["eu-north1"]["region"] == "eu-north1"
    assert "container_registry" not in config["projects"]["eu-north1"]


def test_configure_uses_default_region_without_registry_discovery(
    monkeypatch, tmp_path
) -> None:
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch, project="project-1", tenant="tenant-1")

    def _must_not_discover_registry(_project_id):
        raise AssertionError("configure must not discover a container registry")

    monkeypatch.setattr(
        nebius_module, "discover_container_registry", _must_not_discover_registry
    )
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: True)
    monkeypatch.setattr(nebius_module, "bootstrap_environment", _bootstrap_capture([]))

    answers = "\n".join(
        ["tenant-1", "project-1", "", "existing-bucket", "", "", ""]
    ) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    config = yaml.safe_load(config_path.read_text())
    stanza = next(iter(config["projects"].values()))
    assert "container_registry" not in stanza
    assert stanza["region"] == "eu-north1"


def test_configure_stale_profile_shows_activate_guidance(monkeypatch, tmp_path) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/nebius")
    monkeypatch.setattr(cli_main, "_nebius_profile_ready", lambda **_: False)
    monkeypatch.setattr(cli_main, "_list_nebius_profiles", lambda **_: ["agent-sa"])
    monkeypatch.setattr(cli_main, "_create_nebius_profile", lambda **_: False)
    _stub_nebius_defaults(monkeypatch)

    answers = "n\n" + "\n".join([""] * 11) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    # Without an authenticated profile, default (provisioning) configure aborts
    # up front instead of falling through unanswerable prompts.
    assert result.exit_code == 1, result.output
    assert "profiles exist but" in result.output
    assert "nebius profile activate" in result.output
    assert "Skipped Nebius profile creation" in result.output
    assert (
        "auto-provisioning needs an authenticated Nebius CLI profile" in result.output
    )
    assert "--no-provision" in result.output


def test_list_nebius_profiles_parses_profile_names(monkeypatch) -> None:
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/nebius")

    class _Result:
        returncode = 0
        stdout = "agent-sa [default]\nagent-service\n"

    def fake_runner(cmd, **kwargs):
        assert cmd == ["nebius", "profile", "list"]
        return _Result()

    assert cli_main._list_nebius_profiles(runner=fake_runner) == [
        "agent-sa",
        "agent-service",
    ]


def test_configure_user_declines_profile_creation(monkeypatch, tmp_path) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/nebius")
    monkeypatch.setattr(cli_main, "_nebius_profile_ready", lambda **_: False)
    monkeypatch.setattr(cli_main, "_list_nebius_profiles", lambda **_: [])

    def fail_create(**_):
        raise AssertionError("must not create a profile when the user declines")

    monkeypatch.setattr(cli_main, "_create_nebius_profile", fail_create)
    _stub_nebius_defaults(monkeypatch)

    answers = "n\n" + "\n".join([""] * 11) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 1, result.output
    assert "Skipped Nebius profile creation" in result.output
    assert "Re-run `npa configure`" in result.output
    assert (
        "auto-provisioning needs an authenticated Nebius CLI profile" in result.output
    )


def test_configure_profile_creation_fails_verification(monkeypatch, tmp_path) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/nebius")
    readiness = iter([False, False])
    monkeypatch.setattr(cli_main, "_nebius_profile_ready", lambda **_: next(readiness))
    monkeypatch.setattr(cli_main, "_list_nebius_profiles", lambda **_: [])
    monkeypatch.setattr(cli_main, "_create_nebius_profile", lambda **_: True)
    _stub_nebius_defaults(monkeypatch)

    answers = "y\n" + "\n".join([""] * 11) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 1, result.output
    assert "Could not verify a Nebius profile" in result.output
    assert "Re-run `npa configure`" in result.output
    assert (
        "auto-provisioning needs an authenticated Nebius CLI profile" in result.output
    )


def test_configure_profile_create_subprocess_fails(monkeypatch, tmp_path) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/nebius")
    monkeypatch.setattr(cli_main, "_nebius_profile_ready", lambda **_: False)
    monkeypatch.setattr(cli_main, "_list_nebius_profiles", lambda **_: [])
    monkeypatch.setattr(cli_main, "_create_nebius_profile", lambda **_: False)
    _stub_nebius_defaults(monkeypatch)

    answers = "y\n" + "\n".join([""] * 11) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 1, result.output
    assert "Could not verify a Nebius profile" in result.output
    assert (
        "auto-provisioning needs an authenticated Nebius CLI profile" in result.output
    )


def test_configure_full_interactive_bootstraps_profile_and_provisions(
    monkeypatch, tmp_path
) -> None:
    """Interactive configure without stubbing _ensure_nebius_profile."""
    import yaml

    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    import npa.clients.nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/nebius")
    readiness = iter([False, True])
    monkeypatch.setattr(cli_main, "_nebius_profile_ready", lambda **_: next(readiness))
    monkeypatch.setattr(cli_main, "_list_nebius_profiles", lambda **_: [])
    created: list[str] = []

    def fake_create(*, project_id="", **_):
        created.append(project_id)
        return True

    monkeypatch.setattr(cli_main, "_create_nebius_profile", fake_create)
    _stub_nebius_defaults(monkeypatch, project="project-12345", tenant="tenant-abcde")
    monkeypatch.setattr(nebius_module, "bucket_exists", lambda *_a, **_k: False)
    monkeypatch.setattr(
        nebius_module,
        "bootstrap_environment",
        lambda project_id, tenant_id, region, **kwargs: {
            "nebius_api_key": "AKIAPROVISIONED",
            "nebius_secret_key": "provisioned-secret",
            "s3_bucket": kwargs.get("bucket_name"),
            "s3_endpoint": "https://storage.eu-north1.nebius.cloud",
        },
    )

    answers = (
        "\n".join(
            [
                "y",  # create Nebius profile
                "project-12345",  # bind profile before browser auth/discovery
                "tenant-abcde",  # tenant id
                "project-12345",  # project id
                "",  # region (default eu-north1)
                "",  # bucket name (Enter = default)
                "",  # storage class (standard default)
                "",  # size GB (default 50)
                "hf_secret_token",  # HF token
                "",  # Token Factory API key (skip)
                "",  # NGC API key (skip)
            ]
        )
        + "\n"
    )
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    assert result.exit_code == 0, result.output
    assert created == ["project-12345"]
    assert "Nebius CLI profile is ready." in result.output
    assert "No bucket name provided" in result.output
    assert "project alias: eu-north1" in result.output
    creds = yaml.safe_load(creds_path.read_text())
    assert creds["storage"]["aws_access_key_id"] == "AKIAPROVISIONED"
    assert creds["tokens"]["HF_TOKEN"] == "hf_secret_token"


def test_configure_missing_nebius_cli_shows_install_guidance(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli_main, "_nebius_profile_ready", lambda **_: False)
    _stub_nebius_defaults(monkeypatch)

    answers = "\n".join([""] * 11) + "\n"
    result = runner.invoke(app, ["configure", "--interactive"], input=answers)

    # No Nebius CLI => provisioning cannot proceed => abort up front (non-zero)
    # with install guidance and the --no-provision escape hatch, rather than
    # dropping into prompts a new user cannot answer and exiting 0 empty-handed.
    assert result.exit_code == 1, result.output
    assert "Nebius CLI not found" in result.output
    assert "re-run `npa configure`" in result.output.lower()
    assert "--no-provision" in result.output


def test_configure_interactive_abort_exits_nonzero_and_writes_nothing(
    monkeypatch, tmp_path
) -> None:
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module

    creds_path = tmp_path / "credentials.yaml"
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_main, "_ensure_nebius_profile", lambda: True)
    _stub_nebius_defaults(monkeypatch)

    # Only one answer then EOF: typer aborts on the next prompt. This used to
    # exit 0 having written nothing; it must now fail loudly.
    result = runner.invoke(
        app, ["configure", "--interactive", "--no-provision"], input="project-1\n"
    )

    assert result.exit_code == 1, result.output
    assert "cancelled before anything was written" in result.output
    assert not creds_path.exists()
    assert not config_path.exists()


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


def test_nebius_profile_ready_strips_ambient_token(monkeypatch) -> None:
    """The readiness probe must reflect the profile, not a stale inherited token.

    A stale NEBIUS_IAM_TOKEN / NEBIUS_IAM_TOKEN_FILE lets the CLI skip a real
    token exchange, so the probe must run with both stripped from the env.
    """
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/nebius")
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "stale-token")
    monkeypatch.setenv("NEBIUS_IAM_TOKEN_FILE", "/tmp/stale-token")
    monkeypatch.delenv("NPA_REUSE_IAM_TOKEN", raising=False)

    captured: dict = {}

    class _Result:
        returncode = 0

    def fake_runner(cmd, **kwargs):
        captured.update(kwargs)
        return _Result()

    assert cli_main._nebius_profile_ready(runner=fake_runner) is True
    passed_env = captured["env"]
    assert "NEBIUS_IAM_TOKEN" not in passed_env
    assert "NEBIUS_IAM_TOKEN_FILE" not in passed_env


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


def test_app_entry_unexpected_error_no_stacktrace_by_default(
    monkeypatch, capsys
) -> None:
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
