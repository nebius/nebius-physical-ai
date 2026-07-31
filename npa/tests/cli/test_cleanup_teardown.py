"""Teardown paths: bucket purge, agent IAM leftovers, cluster down without tfvars.

Regressions from a cleanup walkthrough where `npa` alone could not finish the job:
the `npa-agent` service account and its access key outlived `agent destroy`, the
bucket could not be deleted through npa at all (and a versioned bucket refuses an
immediate delete), and a bare `npa cluster down --force` failed with "No value for
required variable" because only `provision-if-absent` exported the TF_VARs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml
from typer.testing import CliRunner

from npa.cli.cluster import terraform_lifecycle as tf_mod
from npa.cli.main import app

runner = CliRunner()


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


# ── npa storage bucket delete ────────────────────────────────────────────────


def test_bucket_delete_schedules_a_purge_and_prunes_stale_credentials(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    creds_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {"HF_TOKEN": "hf_keep"},
                "storage": {
                    "bucket": "s3://npa-bucket-8a0bcf2c/",
                    "endpoint_url": "https://storage.eu-north1.nebius.cloud",
                    "access_key_id": "AKDEAD",
                    "secret_access_key": "SKDEAD",
                },
            }
        )
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(
        nebius_module,
        "get_bucket_by_name",
        lambda project_id, name: {"metadata": {"id": "bucket-abc", "name": name}},
    )
    deletes: list[tuple[str, str]] = []
    monkeypatch.setattr(
        nebius_module,
        "delete_bucket",
        lambda bucket_id, *, ttl="": deletes.append((bucket_id, ttl)),
    )

    result = runner.invoke(
        app,
        [
            "storage",
            "bucket",
            "delete",
            "--name",
            "npa-bucket-8a0bcf2c",
            "--project-id",
            "project-a",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    # A versioned bucket cannot be deleted immediately, so a purge is scheduled.
    assert deletes == [("bucket-abc", "1m")]
    assert "scheduled for purge" in result.output
    # The dead access key no longer sits in credentials.yaml waiting to be reused.
    saved = yaml.safe_load(creds_path.read_text())
    assert "access_key_id" not in saved["storage"]
    assert "bucket" not in saved["storage"]
    assert saved["storage"]["endpoint_url"].endswith("nebius.cloud")
    assert saved["tokens"]["HF_TOKEN"] == "hf_keep"
    assert creds_path.stat().st_mode & 0o077 == 0


def test_bucket_delete_keeps_credentials_for_another_bucket(monkeypatch, tmp_path: Path) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    creds_path.write_text(
        yaml.safe_dump({"storage": {"bucket": "s3://keep-me/", "access_key_id": "AKKEEP"}})
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(
        nebius_module,
        "get_bucket_by_name",
        lambda project_id, name: {"metadata": {"id": "bucket-other", "name": name}},
    )
    monkeypatch.setattr(nebius_module, "delete_bucket", lambda bucket_id, *, ttl="": None)

    result = runner.invoke(
        app,
        ["storage", "bucket", "delete", "--name", "other-bucket", "--project-id", "p", "--yes"],
    )

    assert result.exit_code == 0, result.output
    assert yaml.safe_load(creds_path.read_text())["storage"]["access_key_id"] == "AKKEEP"


def test_bucket_delete_reports_a_missing_bucket_without_failing(monkeypatch, tmp_path: Path) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml")
    monkeypatch.setattr(nebius_module, "get_bucket_by_name", lambda project_id, name: None)

    result = runner.invoke(
        app, ["storage", "bucket", "delete", "--name", "gone", "--project-id", "p", "--yes"]
    )

    assert result.exit_code == 0, result.output
    assert "does not exist" in result.output


def test_delete_bucket_client_passes_ttl() -> None:
    from npa.clients import nebius as nebius_module

    calls: list[list[str]] = []
    original = nebius_module._run
    try:
        nebius_module._run = lambda args, **kwargs: calls.append(args)  # type: ignore[assignment]
        nebius_module.delete_bucket("bucket-abc", ttl="1m")
        nebius_module.delete_bucket("bucket-def")
    finally:
        nebius_module._run = original  # type: ignore[assignment]

    assert calls[0] == ["storage", "bucket", "delete", "--id", "bucket-abc", "--ttl", "1m"]
    assert calls[1] == ["storage", "bucket", "delete", "--id", "bucket-def"]


# ── agent IAM leftovers ──────────────────────────────────────────────────────


def _iam_stubs(monkeypatch, *, sa_id: str = "serviceaccount-agent", keys=("accesskey-1",)):
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        nebius_module, "get_service_account_id_by_name", lambda project_id, name: sa_id or None
    )
    monkeypatch.setattr(
        nebius_module,
        "list_access_keys_for_service_account",
        lambda project_id, account: [
            {"id": key, "name": "npa-agent-access-key", "state": "ACTIVE"} for key in keys
        ],
    )
    deleted: list[str] = []
    monkeypatch.setattr(nebius_module, "delete_access_key", lambda key_id: deleted.append(key_id))
    monkeypatch.setattr(
        nebius_module, "delete_service_account", lambda account_id: deleted.append(account_id)
    )
    return deleted


def test_agent_iam_leftovers_are_reported_with_the_delete_commands(monkeypatch) -> None:
    from npa.cli.agent_iam import report_agent_iam

    _iam_stubs(monkeypatch)
    lines: list[str] = []

    deleted = report_agent_iam(
        project_id="project-a", remaining_agents=0, purge=False, on_status=lines.append
    )

    assert deleted == []
    joined = "\n".join(lines)
    assert "serviceaccount-agent" in joined
    assert "1 access key(s)" in joined
    assert "nebius iam v2 access-key delete --id accesskey-1" in joined
    assert "nebius iam service-account delete --id serviceaccount-agent" in joined


def test_agent_iam_purge_deletes_keys_then_the_account(monkeypatch) -> None:
    from npa.cli.agent_iam import report_agent_iam

    deleted = _iam_stubs(monkeypatch, keys=("accesskey-1", "accesskey-2"))
    lines: list[str] = []

    reported = report_agent_iam(
        project_id="project-a", remaining_agents=0, purge=True, on_status=lines.append
    )

    # Keys first: a service account cannot be removed while keys reference it.
    assert deleted == ["accesskey-1", "accesskey-2", "serviceaccount-agent"]
    assert len(reported) == 3


def test_agent_iam_purge_keeps_an_account_other_agents_use(monkeypatch) -> None:
    from npa.cli.agent_iam import report_agent_iam

    deleted = _iam_stubs(monkeypatch)
    lines: list[str] = []

    report_agent_iam(
        project_id="project-a", remaining_agents=2, purge=True, on_status=lines.append
    )

    assert deleted == []
    assert any("2 other agent(s)" in line for line in lines)


def test_agent_destroy_keep_iam_names_the_delete_commands(monkeypatch, tmp_path: Path) -> None:
    """With --keep-iam the account survives, so say exactly how to remove it.

    (Purging is the default since the cleanup report; see
    `test_agent_destroy_purges_iam_by_default` in test_teardown_inventory.py.)
    """
    from npa.cli import agent as agent_module
    from npa.clients import config as config_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "prod",
                "projects": {
                    "prod": {
                        "project_id": "project-a",
                        "tenant_id": "tenant-a",
                        "region": "eu-north1",
                        "agents": {"agent": {"public_ip": "203.0.113.50", "project_id": "project-a"}},
                    }
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(agent_module, "_destroy_agent_terraform", lambda *a, **k: None)
    monkeypatch.setattr(agent_module, "_cleanup_agent_local_files", lambda *a, **k: None)
    _iam_stubs(monkeypatch)

    result = runner.invoke(app, ["agent", "destroy", "--project", "prod", "--yes", "--keep-iam"])

    assert result.exit_code == 0, result.output
    assert "destroyed: prod/agent" in result.output
    # The leftovers are named, with the commands that remove them.
    assert "serviceaccount-agent" in result.output
    assert "nebius iam service-account delete" in result.output


def test_agent_destroy_purge_iam_removes_the_account(monkeypatch, tmp_path: Path) -> None:
    from npa.cli import agent as agent_module
    from npa.clients import config as config_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "prod",
                "projects": {
                    "prod": {
                        "project_id": "project-a",
                        "agents": {"agent": {"public_ip": "203.0.113.50", "project_id": "project-a"}},
                    }
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(agent_module, "_destroy_agent_terraform", lambda *a, **k: None)
    monkeypatch.setattr(agent_module, "_cleanup_agent_local_files", lambda *a, **k: None)
    deleted = _iam_stubs(monkeypatch)

    result = runner.invoke(
        app, ["agent", "destroy", "--project", "prod", "--yes", "--purge-iam"]
    )

    assert result.exit_code == 0, result.output
    assert deleted == ["accesskey-1", "serviceaccount-agent"]
    assert "Deleted access key accesskey-1" in result.output


def test_agent_iam_is_quiet_when_nothing_is_left(monkeypatch) -> None:
    from npa.cli.agent_iam import report_agent_iam

    _iam_stubs(monkeypatch, sa_id="")
    lines: list[str] = []

    assert report_agent_iam(
        project_id="project-a", remaining_agents=0, purge=True, on_status=lines.append
    ) == []
    assert lines == []


# ── npa cluster down without tfvars ──────────────────────────────────────────


def test_down_reads_project_settings_from_npa_config(monkeypatch, tmp_path: Path) -> None:
    """`provision-if-absent` exports the TF_VARs; a bare `down` had none."""
    from npa.clients import config as config_module

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
                    }
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    for var in ("TF_VAR_parent_id", "TF_VAR_tenant_id", "TF_VAR_region"):
        monkeypatch.delenv(var, raising=False)

    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)  # no terraform.tfvars at all
    envs: list[dict[str, str]] = []

    def fake_stream(args, **kwargs):
        envs.append(dict(kwargs.get("env") or {}))
        return _completed()

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", fake_stream)
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app, ["cluster", "down", "--terraform-dir", str(tf_dir), "--force"]
    )

    assert result.exit_code == 0, result.output
    assert "~/.npa/config.yaml" in result.output
    assert envs, "terraform was never invoked"
    # A machine with no SSH key can still tear down: the value cannot affect what
    # is destroyed, so validation gets a placeholder instead of a hard failure.
    destroy_env = envs[-1]
    assert destroy_env["TF_VAR_parent_id"] == "project-1"
    assert destroy_env["TF_VAR_tenant_id"] == "tenant-1"
    assert destroy_env["TF_VAR_region"] == "us-central1"


def test_down_uses_a_placeholder_key_when_the_machine_has_none(monkeypatch, tmp_path: Path) -> None:
    """Teardown must not be blocked by a missing SSH key; the value is irrelevant."""
    monkeypatch.delenv("NPA_SSH_PUBLIC_KEY", raising=False)
    args = tf_mod._ssh_public_key_var_args({}, {}, allow_placeholder=True)

    assert args[0] == "-var"
    assert "npa-teardown-placeholder" in args[1]

    # The provisioning path still refuses to guess a key.
    import pytest

    with pytest.raises(Exception, match="No SSH public key found"):
        tf_mod._ssh_public_key_var_args({}, {})


def test_cluster_destroy_points_at_the_terraform_teardown(monkeypatch, tmp_path: Path) -> None:
    """The API delete leaves the Terraform-managed network running."""
    from npa.cli.cluster import destroy as destroy_module

    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfstate").write_text("{}")
    monkeypatch.chdir(tmp_path)

    lines: list[str] = []
    monkeypatch.setattr(destroy_module.typer, "echo", lambda message="", **kwargs: lines.append(str(message)))

    destroy_module._warn_terraform_leftovers("npa-cluster")

    joined = "\n".join(lines)
    assert "npa-cluster-network" in joined
    assert "npa cluster down" in joined


def test_cluster_destroy_is_quiet_without_terraform_state(monkeypatch, tmp_path: Path) -> None:
    from npa.cli.cluster import destroy as destroy_module

    monkeypatch.chdir(tmp_path)
    lines: list[str] = []
    monkeypatch.setattr(destroy_module.typer, "echo", lambda message="", **kwargs: lines.append(str(message)))

    destroy_module._warn_terraform_leftovers("npa-cluster")

    assert lines == []


def test_down_does_not_override_explicit_tfvars(monkeypatch, tmp_path: Path) -> None:
    from npa.clients import config as config_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "p",
                "projects": {"p": {"project_id": "project-config", "tenant_id": "t", "region": "r"}},
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setenv("TF_VAR_region", "region-from-env")

    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text('parent_id = "project-tfvars"\n')
    envs: list[dict[str, str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(
        tf_mod, "_run_stream", lambda args, **kwargs: envs.append(dict(kwargs.get("env") or {})) or _completed()
    )
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(app, ["cluster", "down", "--terraform-dir", str(tf_dir), "--force"])

    assert result.exit_code == 0, result.output
    env = envs[-1]
    # tfvars owns parent_id, the environment owns region, config fills only tenant_id.
    assert "TF_VAR_parent_id" not in env
    assert env["TF_VAR_region"] == "region-from-env"
    assert env["TF_VAR_tenant_id"] == "t"
