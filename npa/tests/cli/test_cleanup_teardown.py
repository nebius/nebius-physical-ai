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

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.cluster import terraform_lifecycle as tf_mod
from npa.cli.main import app

runner = CliRunner()


def _completed(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


# ── npa storage bucket delete ────────────────────────────────────────────────


def test_bucket_delete_schedules_a_purge_and_prunes_stale_credentials(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    # The real schema `npa configure` writes: aws_* key names, an endpoint_url,
    # and the storage principal's service account id under `nebius:`.
    creds_path.write_text(
        yaml.safe_dump(
            {
                "tokens": {"HF_TOKEN": "hf_keep"},
                "storage": {
                    "bucket": "s3://npa-bucket-8a0bcf2c/",
                    "endpoint_url": "https://storage.eu-north1.nebius.cloud",
                    "aws_access_key_id": "AKDEAD",
                    "aws_secret_access_key": "SKDEAD",
                },
                "nebius": {"service_account_id": "serviceaccount-storage"},
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
    # Every stale secret for the deleted bucket is gone: the aws_* HMAC keys, the
    # endpoint, and the bucket. IAM evidence has its own lifecycle and remains
    # available for the ownership-gated follow-up command.
    saved = yaml.safe_load(creds_path.read_text())
    assert "storage" not in saved  # section emptied and dropped
    assert saved["nebius"]["service_account_id"] == "serviceaccount-storage"
    # Unrelated secrets are untouched.
    assert saved["tokens"]["HF_TOKEN"] == "hf_keep"
    assert creds_path.stat().st_mode & 0o077 == 0


def test_bucket_delete_wait_polls_until_the_bucket_is_gone(
    monkeypatch, tmp_path: Path
) -> None:
    """A scheduled purge is async; --wait blocks until Nebius has removed it."""
    import time as _time

    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(
        nebius_module, "delete_bucket", lambda bucket_id, *, ttl="": None
    )
    monkeypatch.setattr(_time, "sleep", lambda _s: None)

    calls = {"n": 0}

    def fake_get(project_id, name):
        calls["n"] += 1
        # Present for the first two polls, then purged.
        return {"metadata": {"id": "b", "name": name}} if calls["n"] < 3 else None

    monkeypatch.setattr(nebius_module, "get_bucket_by_name", fake_get)

    result = runner.invoke(
        app,
        [
            "storage",
            "bucket",
            "delete",
            "--name",
            "npa-bucket-x",
            "--project-id",
            "project-a",
            "--ttl",
            "1m",
            "--wait",
            "--keep-config",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "scheduled for purge" in result.output
    assert "Waiting up to" in result.output
    assert "is gone" in result.output
    assert calls["n"] >= 3


def test_bucket_delete_keeps_credentials_for_another_bucket(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    creds_path.write_text(
        yaml.safe_dump(
            {"storage": {"bucket": "s3://keep-me/", "access_key_id": "AKKEEP"}}
        )
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(
        nebius_module,
        "get_bucket_by_name",
        lambda project_id, name: {"metadata": {"id": "bucket-other", "name": name}},
    )
    monkeypatch.setattr(
        nebius_module, "delete_bucket", lambda bucket_id, *, ttl="": None
    )

    result = runner.invoke(
        app,
        [
            "storage",
            "bucket",
            "delete",
            "--name",
            "other-bucket",
            "--project-id",
            "p",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        yaml.safe_load(creds_path.read_text())["storage"]["access_key_id"] == "AKKEEP"
    )


def test_bucket_delete_requires_yes_without_a_tty_and_json_is_machine_readable(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(
        nebius_module,
        "get_bucket_by_name",
        lambda project_id, name: {"metadata": {"id": "bucket-a", "name": name}},
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        nebius_module,
        "delete_bucket",
        lambda bucket_id, *, ttl="": deleted.append(bucket_id),
    )

    result = runner.invoke(
        app,
        [
            "storage",
            "bucket",
            "delete",
            "--name",
            "private-bucket",
            "--project-id",
            "project-a",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.output)["result"] == "confirmation_required"
    assert deleted == []


def test_storage_credential_prune_replace_failure_preserves_complete_document(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_cli
    from npa.clients import credentials as credentials_module

    path = tmp_path / "credentials.yaml"
    original = yaml.safe_dump(
        {
            "tokens": {"HF_TOKEN": "hf_keep"},
            "storage": {
                "bucket": "s3://gone",
                "aws_access_key_id": "AK",
                "aws_secret_access_key": "SK",
            },
        }
    )
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", path)
    monkeypatch.setattr(
        credentials_module.os,
        "replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        storage_cli._prune_storage_credentials("gone")

    assert path.read_text(encoding="utf-8") == original


def test_storage_credential_prune_write_failure_preserves_complete_document(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_cli
    from npa.clients import credentials as credentials_module

    path = tmp_path / "credentials.yaml"
    original = "storage: {bucket: s3://gone, aws_access_key_id: AK}\n"
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", path)
    monkeypatch.setattr(
        credentials_module,
        "write_private_yaml",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )

    with pytest.raises(OSError, match="write failed"):
        storage_cli._prune_storage_credentials("gone")

    assert path.read_text(encoding="utf-8") == original


def test_storage_credential_prune_refuses_symlink_destination(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_cli
    from npa.clients import credentials as credentials_module

    target = tmp_path / "real.yaml"
    target.write_text("storage: {bucket: s3://gone, aws_access_key_id: AK}\n")
    link = tmp_path / "credentials.yaml"
    link.symlink_to(target)
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", link)

    with pytest.raises(OSError, match="symlink"):
        storage_cli._prune_storage_credentials("gone")

    assert "aws_access_key_id" in target.read_text()


def test_storage_credential_prune_refuses_nonowned_destination(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_cli
    from npa.clients import credentials as credentials_module

    path = tmp_path / "credentials.yaml"
    original = "storage: {bucket: s3://gone, aws_access_key_id: AK}\n"
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", path)
    real_euid = credentials_module.os.geteuid()
    monkeypatch.setattr(credentials_module.os, "geteuid", lambda: real_euid + 1)

    with pytest.raises(PermissionError, match="not owned"):
        storage_cli._prune_storage_credentials("gone")

    assert path.read_text(encoding="utf-8") == original


def test_bucket_delete_keeps_npa_ownership_proof_for_explicit_iam_teardown(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    creds_path.write_text(
        yaml.safe_dump(
            {
                "storage": {
                    "bucket": "s3://npa-bucket-owned/",
                    "aws_access_key_id": "AK",
                    "aws_secret_access_key": "SK",
                },
                "nebius": {
                    "service_account_id": "serviceaccount-storage",
                    "service_account_name": "lerobot-training",
                    "service_account_project_id": "project-a",
                    "service_account_managed_by": "npa",
                },
            }
        )
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)
    monkeypatch.setattr(
        nebius_module,
        "get_bucket_by_name",
        lambda project_id, name: {"metadata": {"id": "bucket-owned", "name": name}},
    )
    monkeypatch.setattr(
        nebius_module, "delete_bucket", lambda bucket_id, *, ttl="": None
    )

    result = runner.invoke(
        app,
        [
            "storage",
            "bucket",
            "delete",
            "--name",
            "npa-bucket-owned",
            "--project-id",
            "project-a",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(creds_path.read_text())
    assert "storage" not in saved
    assert saved["nebius"]["service_account_managed_by"] == "npa"
    assert saved["nebius"]["service_account_id"] == "serviceaccount-storage"


def test_bucket_then_iam_teardown_retires_exact_setup_journal(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_cli
    from npa.clients import credentials as credentials_module

    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "storage": {
                    "bucket": "s3://gone/",
                    "aws_access_key_id": "AK",
                    "aws_secret_access_key": "SK",
                },
                "storage_setup": {
                    "version": 1,
                    "projects": {
                        "project-a": {
                            "status": "complete",
                            "resources": {
                                "bucket": {
                                    "name": "gone",
                                    "project_id": "project-a",
                                    "created_by": "npa",
                                },
                                "service_account": {
                                    "id": "serviceaccount-storage",
                                    "name": "storage-account",
                                    "project_id": "project-a",
                                    "created_by": "npa",
                                },
                                "access_keys": {
                                    "accesskey-storage": {
                                        "id": "accesskey-storage",
                                        "created_by": "npa",
                                    }
                                },
                            },
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)

    storage_cli._prune_storage_credentials("gone")

    after_bucket = yaml.safe_load(credentials_path.read_text(encoding="utf-8"))
    resources = after_bucket["storage_setup"]["projects"]["project-a"]["resources"]
    assert "bucket" not in resources
    assert resources["service_account"]["id"] == "serviceaccount-storage"
    assert list(resources["access_keys"]) == ["accesskey-storage"]

    assert storage_cli._remove_storage_service_account_record("serviceaccount-storage")
    assert not credentials_path.exists()


def test_bucket_prune_retry_retires_matching_journal_without_live_credentials(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import storage as storage_cli
    from npa.clients import credentials as credentials_module

    credentials_path = tmp_path / "credentials.yaml"
    credentials_path.write_text(
        yaml.safe_dump(
            {
                "storage_setup": {
                    "version": 1,
                    "projects": {
                        "project-a": {
                            "resources": {
                                "bucket": {
                                    "name": "gone",
                                    "project_id": "project-a",
                                    "created_by": "npa",
                                }
                            }
                        },
                        "project-b": {
                            "resources": {
                                "bucket": {
                                    "name": "keep",
                                    "project_id": "project-b",
                                    "created_by": "npa",
                                }
                            }
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)

    storage_cli._prune_storage_credentials("gone")

    saved = yaml.safe_load(credentials_path.read_text(encoding="utf-8"))
    projects = saved["storage_setup"]["projects"]
    assert set(projects) == {"project-b"}
    assert projects["project-b"]["resources"]["bucket"]["name"] == "keep"


def test_bucket_delete_clears_terraform_state_for_that_bucket(
    monkeypatch, tmp_path: Path
) -> None:
    """The Terraform remote-state S3 keys for a deleted bucket are secrets too.

    They live in config.yaml (`projects.<alias>.terraform_state`), a separate
    file from the object-storage access key in credentials.yaml, and a bucket
    delete that cleaned only the latter left live-looking HMAC keys behind.
    """
    from npa.clients import config as config_module
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    creds_path = tmp_path / "credentials.yaml"
    creds_path.write_text(
        yaml.safe_dump(
            {"storage": {"bucket": "s3://npa-bucket-dead/", "aws_access_key_id": "AK"}}
        )
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "gone": {
                        "project_id": "project-a",
                        "terraform_state": {
                            "bucket": "npa-bucket-dead",
                            "endpoint": "https://storage.eu-north1.nebius.cloud",
                            "access_key": "TFAKDEAD",
                            "secret_key": "TFSKDEAD",
                        },
                    },
                    "other": {
                        "project_id": "project-b",
                        "terraform_state": {
                            "bucket": "npa-bucket-live",
                            "access_key": "KEEP",
                        },
                    },
                }
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        nebius_module,
        "get_bucket_by_name",
        lambda project_id, name: {"metadata": {"id": "bucket-dead", "name": name}},
    )
    monkeypatch.setattr(
        nebius_module, "delete_bucket", lambda bucket_id, *, ttl="": None
    )

    result = runner.invoke(
        app,
        [
            "storage",
            "bucket",
            "delete",
            "--name",
            "npa-bucket-dead",
            "--project-id",
            "project-a",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(config_path.read_text())
    # The deleted bucket's remote-state secrets are gone; the other project's stay.
    assert "terraform_state" not in saved["projects"]["gone"]
    assert saved["projects"]["other"]["terraform_state"]["access_key"] == "KEEP"
    assert "remote-state" in result.output


def test_configure_forget_project_removes_the_stanza(
    monkeypatch, tmp_path: Path
) -> None:
    """`npa configure --forget-project` is the inverse of writing a stanza."""
    from npa.clients import config as config_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "gone",
                "projects": {
                    "gone": {
                        "project_id": "project-a",
                        "terraform_state": {"access_key": "AK"},
                    },
                    "keep": {"project_id": "project-b"},
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)

    result = runner.invoke(app, ["configure", "--forget-project", "gone"])

    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(config_path.read_text())
    assert "gone" not in saved["projects"]
    assert "keep" in saved["projects"]
    # The default moved off the forgotten project.
    assert saved["default_project"] == "keep"


def test_configure_forget_project_is_quiet_when_absent(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import config as config_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"projects": {"keep": {"project_id": "p"}}}))
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)

    result = runner.invoke(app, ["configure", "--forget-project", "missing"])

    assert result.exit_code == 0, result.output
    assert "nothing to remove" in result.output
    assert (
        yaml.safe_load(config_path.read_text())["projects"]["keep"]["project_id"] == "p"
    )


def test_bucket_delete_reports_a_missing_bucket_without_failing(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    lookups: list[tuple[str, str]] = []
    monkeypatch.setattr(
        nebius_module,
        "get_bucket_by_name",
        lambda project_id, name: lookups.append((project_id, name)) or None,
    )

    result = runner.invoke(
        app,
        [
            "storage",
            "bucket",
            "delete",
            "--name",
            "s3://gone/prefix/",
            "--project-id",
            "p",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "does not exist" in result.output
    assert lookups == [("p", "gone")]


def test_bucket_delete_missing_bucket_still_requires_confirmation_before_local_prune(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    credentials_path = tmp_path / "credentials.yaml"
    original = "storage: {bucket: s3://gone, aws_access_key_id: AK}\n"
    credentials_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", credentials_path)
    monkeypatch.setattr(
        nebius_module, "get_bucket_by_name", lambda project_id, name: None
    )

    result = runner.invoke(
        app,
        [
            "storage",
            "bucket",
            "delete",
            "--name",
            "gone",
            "--project-id",
            "p",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.output)["result"] == "confirmation_required"
    assert credentials_path.read_text(encoding="utf-8") == original


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

    assert calls[0] == [
        "storage",
        "bucket",
        "delete",
        "--id",
        "bucket-abc",
        "--ttl",
        "1m",
    ]
    assert calls[1] == ["storage", "bucket", "delete", "--id", "bucket-def"]


def test_agent_destroy_removes_the_terraform_workdir() -> None:
    """destroy must also drop ~/.npa/workbenches/<alias>/<name>/ (the TF tree).

    It previously cleaned only ~/.npa/agents/<alias>/<name>/, leaving the whole
    Terraform workdir (provider cache + a local-backend tfstate) behind.
    """
    from npa.cli.agent import _cleanup_agent_local_files
    from npa.deploy import provisioner

    agent_dir = Path.home() / ".npa" / "agents" / "prod" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "auth.env").write_text("AGENT_PASSWORD=secret\n")

    tf_dir = provisioner.working_dir_path("prod", "agent")
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfstate").write_text("{}")
    (tf_dir / "backend.tf").write_text("")

    _cleanup_agent_local_files("prod", "agent")

    assert not agent_dir.exists()
    assert not tf_dir.exists()
    # The now-empty <alias> parents are pruned too (no empty tree left behind).
    assert not agent_dir.parent.exists()
    assert not tf_dir.parent.exists()


def test_agent_cleanup_keeps_alias_parent_with_a_sibling() -> None:
    """A second agent under the same alias must keep the shared <alias> parent."""
    from npa.cli.agent import _cleanup_agent_local_files
    from npa.deploy import provisioner

    for wb in ("agent", "other"):
        provisioner.working_dir_path("prod", wb).mkdir(parents=True)
        (Path.home() / ".npa" / "agents" / "prod" / wb).mkdir(parents=True)

    _cleanup_agent_local_files("prod", "agent")

    # The torn-down agent's trees are gone, the sibling's survive with the parent.
    assert not provisioner.working_dir_path("prod", "agent").exists()
    assert provisioner.working_dir_path("prod", "other").exists()
    assert provisioner.working_dir_path("prod", "other").parent.exists()


# ── agent IAM leftovers ──────────────────────────────────────────────────────


def _iam_stubs(
    monkeypatch, *, sa_id: str = "serviceaccount-agent", keys=("accesskey-1",)
):
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        nebius_module,
        "get_service_account_id_by_name",
        lambda project_id, name, **kwargs: sa_id or None,
    )
    monkeypatch.setattr(
        nebius_module,
        "list_access_keys_for_service_account",
        lambda project_id, account, **kwargs: [
            {"id": key, "name": "npa-agent-access-key", "state": "ACTIVE"}
            for key in keys
        ],
    )
    monkeypatch.setattr(
        nebius_module, "_run_json", lambda *args, **kwargs: {"items": []}
    )
    monkeypatch.setattr(
        nebius_module, "get_compute_instance_identity", lambda *args, **kwargs: None
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        nebius_module, "delete_access_key", lambda key_id: deleted.append(key_id)
    )
    monkeypatch.setattr(
        nebius_module,
        "delete_service_account",
        lambda account_id: deleted.append(account_id),
    )
    return deleted


def test_agent_iam_leftovers_without_provenance_are_reported_but_protected(
    monkeypatch,
) -> None:
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
    assert "no creation provenance" in joined
    assert "nebius iam v2 access-key delete" not in joined
    assert "nebius iam service-account delete" not in joined


def test_agent_iam_purge_deletes_keys_then_the_account(monkeypatch) -> None:
    from npa.cli.agent_iam import report_agent_iam

    deleted = _iam_stubs(monkeypatch, keys=("accesskey-1", "accesskey-2"))
    monkeypatch.setattr("npa.cli.agent_iam.agent_iam_owned", lambda *_args: True)
    monkeypatch.setattr("npa.cli.agent_iam.clear_agent_iam_record", lambda *_args: True)
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
    assert any("2 other local agent record(s)" in line for line in lines)


def test_agent_iam_purge_protects_same_project_peer_missing_from_local_config(
    monkeypatch,
) -> None:
    from npa.cli.agent_iam import report_agent_iam
    from npa.clients import nebius as nebius_module

    deleted = _iam_stubs(monkeypatch)
    monkeypatch.setattr("npa.cli.agent_iam.agent_iam_owned", lambda *_args: True)
    monkeypatch.setattr(
        nebius_module,
        "_run_json",
        lambda *args, **kwargs: {
            "items": [
                {
                    "metadata": {"id": "instance-peer", "name": "agent-peer"},
                    "spec": {
                        "account": {"service_account": {"id": "serviceaccount-agent"}}
                    },
                }
            ]
        },
    )
    lines: list[str] = []

    report_agent_iam(
        project_id="project-a", remaining_agents=0, purge=True, on_status=lines.append
    )

    assert deleted == []
    assert any("agent-peer (instance-peer)" in line for line in lines)


def test_agent_iam_purge_fails_closed_when_provider_inventory_is_forbidden(
    monkeypatch,
) -> None:
    from npa.cli.agent_iam import report_agent_iam
    from npa.clients import nebius as nebius_module

    deleted = _iam_stubs(monkeypatch)
    monkeypatch.setattr("npa.cli.agent_iam.agent_iam_owned", lambda *_args: True)
    monkeypatch.setattr(
        nebius_module,
        "_run_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            nebius_module.NebiusError("RBAC forbidden")
        ),
    )
    lines: list[str] = []

    report_agent_iam(
        project_id="project-a", remaining_agents=0, purge=True, on_status=lines.append
    )

    assert deleted == []
    assert any(
        "inventory is unresolved" in line and "RBAC forbidden" in line for line in lines
    )


def test_agent_iam_schema_invalid_inventory_uses_exact_terminal_graph_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli.agent_iam import report_agent_iam
    from npa.clients import nebius as nebius_module
    from npa.teardown_receipts import record_teardown_event

    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    deleted = _iam_stubs(monkeypatch)
    monkeypatch.setattr("npa.cli.agent_iam.agent_iam_owned", lambda *_args: True)
    monkeypatch.setattr("npa.cli.agent_iam.clear_agent_iam_record", lambda *_args: True)
    monkeypatch.setattr(nebius_module, "_run_json", lambda *_a, **_k: {"items": {}})
    record_teardown_event(
        phase="agent",
        resource="agent",
        terminal_state="verified_deleted",
        project_alias="prod",
        project_id="project-a",
        identity={
            "project_alias": "prod",
            "project_id": "project-a",
            "agent_name": "agent",
            "instance_id": "instance-agent",
            "service_account_id": "serviceaccount-agent",
        },
        verification={
            "terraform_destroy_completed": True,
            "terraform_dependency_graph": [
                "compute_instance",
                "boot_disk",
                "network",
                "subnet",
                "security_group",
                "public_ip",
            ],
            "exact_instance_absent": True,
        },
    )
    lines: list[str] = []

    reported = report_agent_iam(
        project_id="project-a",
        remaining_agents=0,
        purge=True,
        on_status=lines.append,
    )

    assert deleted == ["accesskey-1", "serviceaccount-agent"]
    assert len(reported) == 2


def test_agent_iam_vm_not_found_receipt_does_not_prove_dependency_graph(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli.agent_iam import report_agent_iam
    from npa.clients import nebius as nebius_module
    from npa.teardown_receipts import record_teardown_event

    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "receipts"))
    deleted = _iam_stubs(monkeypatch)
    monkeypatch.setattr("npa.cli.agent_iam.agent_iam_owned", lambda *_args: True)
    monkeypatch.setattr(nebius_module, "_run_json", lambda *_a, **_k: {"items": {}})
    record_teardown_event(
        phase="agent",
        resource="agent",
        terminal_state="verified_absent",
        project_alias="prod",
        project_id="project-a",
        identity={
            "project_alias": "prod",
            "project_id": "project-a",
            "agent_name": "agent",
            "instance_id": "instance-agent",
            "service_account_id": "serviceaccount-agent",
        },
        verification={"exact_instance_absent": True},
    )
    lines: list[str] = []

    report_agent_iam(
        project_id="project-a",
        remaining_agents=0,
        purge=True,
        on_status=lines.append,
    )

    assert deleted == []
    assert any("exact receipt fallback" in line for line in lines)


def test_agent_destroy_keep_iam_names_the_delete_commands(
    monkeypatch, tmp_path: Path
) -> None:
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
                        "agents": {
                            "agent": {
                                "public_ip": "203.0.113.50",
                                "project_id": "project-a",
                                "instance_id": "instance-agent",
                            }
                        },
                    }
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(agent_module, "_destroy_agent_terraform", lambda *a, **k: None)
    monkeypatch.setattr(
        agent_module, "_cleanup_agent_local_files", lambda *a, **k: None
    )
    _iam_stubs(monkeypatch)

    result = runner.invoke(
        app, ["agent", "destroy", "--project", "prod", "--yes", "--keep-iam"]
    )

    assert result.exit_code == 0, result.output
    assert "destroyed: prod/agent" in result.output
    # The leftovers are named, but an ownership-unproven familiar name is never
    # turned into deletion instructions.
    assert "serviceaccount-agent" in result.output
    assert "no creation provenance" in result.output
    assert "nebius iam service-account delete" not in result.output


def test_agent_destroy_purge_iam_removes_the_account(
    monkeypatch, tmp_path: Path
) -> None:
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
                        "agents": {
                            "agent": {
                                "public_ip": "203.0.113.50",
                                "project_id": "project-a",
                                "instance_id": "instance-agent",
                            }
                        },
                    }
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(agent_module, "_destroy_agent_terraform", lambda *a, **k: None)
    monkeypatch.setattr(
        agent_module, "_cleanup_agent_local_files", lambda *a, **k: None
    )
    deleted = _iam_stubs(monkeypatch)
    monkeypatch.setattr("npa.cli.agent_iam.agent_iam_owned", lambda *_args: True)
    monkeypatch.setattr("npa.cli.agent_iam.clear_agent_iam_record", lambda *_args: True)

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

    assert (
        report_agent_iam(
            project_id="project-a",
            remaining_agents=0,
            purge=True,
            on_status=lines.append,
        )
        == []
    )
    assert lines == []


# ── npa cluster down without tfvars ──────────────────────────────────────────


def test_down_reads_project_settings_from_npa_config(
    monkeypatch, tmp_path: Path
) -> None:
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
    (tf_dir / "terraform.tfstate").write_text("{}")
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
    monkeypatch.setattr(
        tf_mod, "_preflight_provider_lock", lambda *_args: "linux_amd64"
    )
    monkeypatch.setattr(
        "npa.terraform_lock.validate_provider_lock",
        lambda *_args, **_kwargs: "linux_amd64",
    )
    monkeypatch.setattr(
        "npa.terraform_lock.configure_plugin_cache",
        lambda *_args, **_kwargs: Path("/tmp/npa-test-terraform-cache"),
    )
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


def test_down_uses_a_placeholder_key_when_the_machine_has_none(
    monkeypatch, tmp_path: Path
) -> None:
    """Teardown must not be blocked by a missing SSH key; the value is irrelevant."""
    monkeypatch.delenv("NPA_SSH_PUBLIC_KEY", raising=False)
    args = tf_mod._ssh_public_key_var_args({}, {}, allow_placeholder=True)

    assert args[0] == "-var"
    assert "npa-teardown-placeholder" in args[1]

    # The provisioning path still refuses to guess a key.
    import pytest

    with pytest.raises(Exception, match="No SSH public key found"):
        tf_mod._ssh_public_key_var_args({}, {})


def test_cluster_destroy_points_at_the_terraform_teardown(
    monkeypatch, tmp_path: Path
) -> None:
    """The API delete leaves the Terraform-managed network running."""
    from npa.cli.cluster import destroy as destroy_module

    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfstate").write_text("{}")
    monkeypatch.chdir(tmp_path)

    lines: list[str] = []
    monkeypatch.setattr(
        destroy_module.typer,
        "echo",
        lambda message="", **kwargs: lines.append(str(message)),
    )

    destroy_module._warn_terraform_leftovers("npa-cluster")

    joined = "\n".join(lines)
    assert "npa-cluster-network" in joined
    assert "npa cluster down" in joined


def test_cluster_destroy_is_quiet_without_terraform_state(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli.cluster import destroy as destroy_module

    monkeypatch.chdir(tmp_path)
    lines: list[str] = []
    monkeypatch.setattr(
        destroy_module.typer,
        "echo",
        lambda message="", **kwargs: lines.append(str(message)),
    )

    destroy_module._warn_terraform_leftovers("npa-cluster")

    assert lines == []


def test_down_does_not_override_explicit_tfvars(monkeypatch, tmp_path: Path) -> None:
    from npa.clients import config as config_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "p",
                "projects": {
                    "p": {
                        "project_id": "project-config",
                        "tenant_id": "t",
                        "region": "r",
                    }
                },
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_path)
    monkeypatch.setenv("TF_VAR_region", "region-from-env")

    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text('parent_id = "project-tfvars"\n')
    (tf_dir / "terraform.tfstate").write_text("{}")
    envs: list[dict[str, str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(
        tf_mod, "_preflight_provider_lock", lambda *_args: "linux_amd64"
    )
    monkeypatch.setattr(
        "npa.terraform_lock.validate_provider_lock",
        lambda *_args, **_kwargs: "linux_amd64",
    )
    monkeypatch.setattr(
        "npa.terraform_lock.configure_plugin_cache",
        lambda *_args, **_kwargs: Path("/tmp/npa-test-terraform-cache"),
    )
    monkeypatch.setattr(
        tf_mod,
        "_run_stream",
        lambda args, **kwargs: (
            envs.append(dict(kwargs.get("env") or {})) or _completed()
        ),
    )
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app, ["cluster", "down", "--terraform-dir", str(tf_dir), "--force"]
    )

    assert result.exit_code == 0, result.output
    env = envs[-1]
    # tfvars owns parent_id, the environment owns region, config fills only tenant_id.
    assert "TF_VAR_parent_id" not in env
    assert env["TF_VAR_region"] == "region-from-env"
    assert env["TF_VAR_tenant_id"] == "t"
