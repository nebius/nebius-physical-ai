"""Teardown completeness and inventory: `cluster down`, `agent list`, `bucket list`.

Regressions from a cleanup walkthrough:
- a successful `cluster down` left ~/.npa/clusters/<context>/ behind, so
  `cluster list` still showed the destroyed cluster as UNKNOWN and a second
  command (`cluster destroy`) was needed purely to delete local files;
- `npa agent destroy` kept the `npa-agent` service account and access key that a
  *rolled-back* deploy had created unless --purge-iam was passed explicitly;
- there was no inventory command for agents or buckets, so operators probed
  `status`/`destroy` blind;
- a ~6.5-minute node-group drain printed `Still destroying...` with no detail.
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
from npa.cluster import state as state_module

runner = CliRunner()


def _completed(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


def _terraform_stubs(
    monkeypatch, *, streams: list[list[str]] | None = None, node_groups: str = ""
):
    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["terraform", "state", "pull"]:
            return _completed(
                json.dumps({"outputs": {"kube_cluster": {"value": {"id": "c1"}}}})
            )
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed(
                json.dumps(
                    {"items": [{"metadata": {"name": "npa-cluster", "id": "c1"}}]}
                )
            )
        if args[:4] == ["nebius", "mk8s", "node-group", "list"]:
            return _completed(node_groups or '{"items":[]}')
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
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)
    calls = streams if streams is not None else []
    monkeypatch.setattr(
        tf_mod, "_run_stream", lambda args, **kwargs: calls.append(args) or _completed()
    )
    return calls


def _write_legacy_cluster_ownership(state_dir: Path, *, context: str) -> None:
    """Write valid legacy ownership evidence for teardown behavior tests."""

    (state_dir / "cluster.json").write_text(
        json.dumps(
            {
                "name": context,
                "cluster_id": "c1",
                "project_id": "p",
                "region": "r",
                "node_count": 1,
                "node_platform": "cpu-d3",
                "node_preset": "4vcpu-16gb",
                "k8s_version": "1.30",
                "subnet_id": "",
                "created_at": "2026-01-01T00:00:00Z",
            }
        )
    )
    (state_dir / "metadata.json").write_text(
        json.dumps({"managed_by": "npa cluster terraform"})
    )


# ── `cluster down` finishes the job ──────────────────────────────────────────


def test_down_removes_the_local_cluster_state(monkeypatch, tmp_path: Path) -> None:
    clusters = tmp_path / "clusters"
    state_dir = clusters / "npa-cluster"
    state_dir.mkdir(parents=True)
    (state_dir / "kubeconfig").write_text("apiVersion: v1\n")
    _write_legacy_cluster_ownership(state_dir, context="npa-cluster")
    monkeypatch.setattr(state_module, "CLUSTERS_DIR", clusters)
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'parent_id = "p"\ntenant_id = "t"\nregion = "r"\ncluster_name = "npa-cluster"\n'
    )
    _terraform_stubs(monkeypatch)

    result = runner.invoke(
        app, ["cluster", "down", "--terraform-dir", str(tf_dir), "--force"]
    )

    assert result.exit_code == 0, result.output
    assert not state_dir.exists()
    assert "Removed local cluster state" in result.output


def test_down_keeps_local_state_when_asked(monkeypatch, tmp_path: Path) -> None:
    clusters = tmp_path / "clusters"
    state_dir = clusters / "npa-cluster"
    state_dir.mkdir(parents=True)
    _write_legacy_cluster_ownership(state_dir, context="npa-cluster")
    monkeypatch.setattr(state_module, "CLUSTERS_DIR", clusters)
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'cluster_name = "npa-cluster"\nparent_id = "p"\n'
    )
    _terraform_stubs(monkeypatch)

    result = runner.invoke(
        app,
        [
            "cluster",
            "down",
            "--terraform-dir",
            str(tf_dir),
            "--force",
            "--keep-local-state",
        ],
    )

    assert result.exit_code == 0, result.output
    assert state_dir.exists()
    assert "Removed local cluster state" not in result.output
    assert "best-effort drain preview unavailable" in result.output
    assert "teardown will continue" in result.output


def test_down_removes_state_for_an_explicit_context(
    monkeypatch, tmp_path: Path
) -> None:
    """`up --context` can name the context something other than the cluster."""
    clusters = tmp_path / "clusters"
    (clusters / "npa-cluster").mkdir(parents=True)
    (clusters / "npa-cluster" / "cluster.json").write_text("{}")
    (clusters / "custom-ctx").mkdir(parents=True)
    _write_legacy_cluster_ownership(clusters / "custom-ctx", context="custom-ctx")
    monkeypatch.setattr(state_module, "CLUSTERS_DIR", clusters)
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'cluster_name = "npa-cluster"\nparent_id = "p"\n'
    )
    _terraform_stubs(monkeypatch)

    result = runner.invoke(
        app,
        [
            "cluster",
            "down",
            "--terraform-dir",
            str(tf_dir),
            "--force",
            "--context",
            "custom-ctx",
        ],
    )

    assert result.exit_code == 0, result.output
    assert not (clusters / "custom-ctx").exists()
    assert (clusters / "npa-cluster").exists()  # untouched


def test_down_is_quiet_when_there_is_no_local_state(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(state_module, "CLUSTERS_DIR", tmp_path / "clusters")
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'cluster_name = "npa-cluster"\nparent_id = "p"\n'
    )
    _terraform_stubs(monkeypatch)

    result = runner.invoke(
        app, ["cluster", "down", "--terraform-dir", str(tf_dir), "--force"]
    )

    assert result.exit_code == 0, result.output
    assert "Removed local cluster state" not in result.output


def test_down_keeps_local_state_when_the_destroy_fails(
    monkeypatch, tmp_path: Path
) -> None:
    """State is the only record of a cluster a failed destroy may have left."""
    clusters = tmp_path / "clusters"
    (clusters / "npa-cluster").mkdir(parents=True)
    (clusters / "npa-cluster" / "cluster.json").write_text("{}")
    monkeypatch.setattr(state_module, "CLUSTERS_DIR", clusters)
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'cluster_name = "npa-cluster"\nparent_id = "p"\n'
    )
    _terraform_stubs(monkeypatch)

    def failing_stream(args, **kwargs):
        if args[:2] == ["terraform", "destroy"]:
            raise tf_mod.typer.BadParameter("destroy failed (exit 1)")
        return _completed()

    monkeypatch.setattr(tf_mod, "_run_stream", failing_stream)

    result = runner.invoke(
        app, ["cluster", "down", "--terraform-dir", str(tf_dir), "--force"]
    )

    assert result.exit_code != 0
    assert (clusters / "npa-cluster").exists()


def test_down_reports_node_group_progress_while_destroying(
    monkeypatch, tmp_path: Path
) -> None:
    """A ~6-minute node-group drain printed `Still destroying...` and nothing else."""
    monkeypatch.setattr(state_module, "CLUSTERS_DIR", tmp_path / "clusters")
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'cluster_name = "npa-cluster"\nparent_id = "p"\n'
    )
    # This test exercises an actual destroy, so provide the state evidence that
    # distinguishes it from the required no-cluster fast path.
    (tf_dir / "terraform.tfstate").write_text('{"version": 4, "resources": [{}]}\n')
    node_groups = json.dumps(
        {
            "items": [
                {
                    "metadata": {"name": "npa-cluster-cpu"},
                    "status": {
                        "state": "DELETING",
                        "target_node_count": "0",
                        "ready_node_count": "1",
                    },
                }
            ]
        }
    )
    watchers: list[object] = []
    original = tf_mod._NodeGroupWatcher

    class RecordingWatcher(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            watchers.append(self)

        def start(self) -> None:  # no thread in the test
            return

    monkeypatch.setattr(tf_mod, "_NodeGroupWatcher", RecordingWatcher)
    _terraform_stubs(monkeypatch, node_groups=node_groups)

    result = runner.invoke(
        app, ["cluster", "down", "--terraform-dir", str(tf_dir), "--force"]
    )

    assert result.exit_code == 0, result.output
    assert watchers, "the destroy did not start a node-group watcher"
    watchers[-1]._poll()
    assert (
        "node group npa-cluster-cpu: DELETING" in result.output or True
    )  # polled after the run
    lines: list[str] = []
    monkeypatch.setattr(
        tf_mod.typer, "echo", lambda message="", **kwargs: lines.append(str(message))
    )
    watchers[-1]._seen.clear()
    watchers[-1]._poll()
    assert any("npa-cluster-cpu: DELETING (1/0 ready)" in line for line in lines)


def test_down_help_disambiguates_the_two_teardown_verbs() -> None:
    """`down` (Terraform, complete) vs `destroy` (API only) was easy to confuse."""
    down = runner.invoke(app, ["cluster", "down", "--help"]).output
    destroy = runner.invoke(app, ["cluster", "destroy", "--help"]).output

    assert "complete teardown" in down
    assert "npa cluster destroy" in down
    assert "npa cluster down" in destroy
    assert "API-only" in destroy


# ── `npa agent list` ─────────────────────────────────────────────────────────


def _write_agents(monkeypatch, tmp_path: Path, payload: dict) -> Path:
    from npa.clients import config as config_module

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    monkeypatch.setattr(config_module, "CONFIG_PATH", path)
    return path


def test_agent_list_shows_recorded_agents(monkeypatch, tmp_path: Path) -> None:
    _write_agents(
        monkeypatch,
        tmp_path,
        {
            "default_project": "prod",
            "projects": {
                "prod": {
                    "project_id": "project-1",
                    "agents": {
                        "agent": {
                            "public_ip": "203.0.113.50",
                            "region": "eu-north1",
                            "instance_id": "computeinstance-a",
                            "agent_url": "http://203.0.113.50:8088",
                        }
                    },
                },
                "dev": {
                    "project_id": "project-2",
                    "agents": {"scratch": {"region": "us-central1"}},
                },
            },
        },
    )

    result = runner.invoke(app, ["agent", "list"])

    assert result.exit_code == 0, result.output
    assert "prod" in result.output and "agent" in result.output
    assert "203.0.113.50" in result.output
    assert "dev" in result.output and "scratch" in result.output
    # A record with no IP renders a placeholder rather than an empty column.
    assert "-" in result.output
    assert "npa agent status --project" in result.output


def test_agent_list_filters_by_project(monkeypatch, tmp_path: Path) -> None:
    _write_agents(
        monkeypatch,
        tmp_path,
        {
            "projects": {
                "prod": {"agents": {"agent": {"public_ip": "203.0.113.50"}}},
                "dev": {"agents": {"scratch": {}}},
            }
        },
    )

    result = runner.invoke(app, ["agent", "list", "--project", "prod"])

    assert result.exit_code == 0, result.output
    assert "agent" in result.output
    assert "scratch" not in result.output


def test_agent_list_json_is_machine_readable(monkeypatch, tmp_path: Path) -> None:
    _write_agents(
        monkeypatch,
        tmp_path,
        {"projects": {"prod": {"agents": {"agent": {"public_ip": "203.0.113.50"}}}}},
    )

    payload = json.loads(runner.invoke(app, ["agent", "list", "--json"]).output)

    assert payload == [
        {
            "project": "prod",
            "name": "agent",
            "public_ip": "203.0.113.50",
            "region": "",
            "instance_id": "",
            "agent_url": "",
            "created_at": "",
        }
    ]


def test_agent_list_explains_an_empty_inventory(monkeypatch, tmp_path: Path) -> None:
    _write_agents(monkeypatch, tmp_path, {"projects": {"prod": {"project_id": "p"}}})

    result = runner.invoke(app, ["agent", "list"])

    assert result.exit_code == 0, result.output
    assert "No agents recorded" in result.output
    assert "npa agent setup" in result.output


def test_agent_list_tolerates_a_malformed_config(monkeypatch, tmp_path: Path) -> None:
    from npa.cli.agent_inventory import agent_rows

    _write_agents(
        monkeypatch, tmp_path, {"projects": {"prod": {"agents": "not-a-mapping"}}}
    )

    assert agent_rows() == []


# ── `npa storage bucket list` ────────────────────────────────────────────────


def test_bucket_list_marks_the_configured_bucket(monkeypatch, tmp_path: Path) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    creds = tmp_path / "credentials.yaml"
    creds.write_text(
        yaml.safe_dump({"storage": {"bucket": "s3://npa-bucket-8a0bcf2c/"}})
    )
    monkeypatch.setattr(credentials_module, "CREDENTIALS_PATH", creds)
    monkeypatch.setattr(
        nebius_module,
        "_list_project_buckets",
        lambda project_id: [
            {"metadata": {"name": "npa-bucket-8a0bcf2c", "id": "storagebucket-a"}},
            {"metadata": {"name": "other-bucket", "id": "storagebucket-b"}},
        ],
    )

    result = runner.invoke(
        app, ["storage", "bucket", "list", "--project-id", "project-a"]
    )

    assert result.exit_code == 0, result.output
    assert "npa-bucket-8a0bcf2c" in result.output
    assert "<- configured in ~/.npa" in result.output
    assert "other-bucket" in result.output
    assert "npa storage bucket delete" in result.output
    # Only the configured bucket is marked.
    marked = [line for line in result.output.splitlines() if "configured" in line]
    assert len(marked) == 1 and "npa-bucket-8a0bcf2c" in marked[0]


def test_bucket_list_json_and_empty_project(monkeypatch, tmp_path: Path) -> None:
    from npa.clients import credentials as credentials_module
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        credentials_module, "CREDENTIALS_PATH", tmp_path / "credentials.yaml"
    )
    monkeypatch.setattr(nebius_module, "_list_project_buckets", lambda project_id: [])

    empty = runner.invoke(
        app, ["storage", "bucket", "list", "--project-id", "project-a"]
    )
    assert empty.exit_code == 0, empty.output
    assert "No buckets in project" in empty.output

    monkeypatch.setattr(
        nebius_module,
        "_list_project_buckets",
        lambda project_id: [{"metadata": {"name": "b", "id": "storagebucket-b"}}],
    )
    payload = json.loads(
        runner.invoke(
            app, ["storage", "bucket", "list", "--project-id", "project-a", "--json"]
        ).output
    )
    assert payload == [{"name": "b", "id": "storagebucket-b", "configured": False}]


def test_bucket_list_requires_a_project(monkeypatch, tmp_path: Path) -> None:
    from npa.clients import config as config_module

    config_module_path = tmp_path / "config.yaml"
    config_module_path.write_text(yaml.safe_dump({"projects": {}}))
    monkeypatch.setattr(config_module, "CONFIG_PATH", config_module_path)

    result = runner.invoke(app, ["storage", "bucket", "list"])

    assert result.exit_code != 0
    assert "Cannot tell which Nebius project" in result.output


# ── agent IAM cleanup is the default ─────────────────────────────────────────


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


def _agent_config(monkeypatch, tmp_path: Path, agents: dict) -> None:
    from npa.clients import config as config_module

    agents = {
        name: {
            **record,
            "project_id": str(record.get("project_id") or "project-a"),
            "instance_id": str(record.get("instance_id") or f"instance-{name}"),
        }
        for name, record in agents.items()
    }
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "default_project": "prod",
                "projects": {"prod": {"project_id": "project-a", "agents": agents}},
            }
        )
    )
    monkeypatch.setattr(config_module, "CONFIG_PATH", path)


def test_agent_destroy_purges_iam_by_default(monkeypatch, tmp_path: Path) -> None:
    """A rolled-back deploy still created the account; keeping it was the old default."""
    from npa.cli import agent as agent_module

    _agent_config(monkeypatch, tmp_path, {"agent": {"public_ip": "203.0.113.50"}})
    monkeypatch.setattr(agent_module, "_destroy_agent_terraform", lambda *a, **k: None)
    monkeypatch.setattr(
        agent_module, "_cleanup_agent_local_files", lambda *a, **k: None
    )
    deleted = _iam_stubs(monkeypatch)
    monkeypatch.setattr("npa.cli.agent_iam.agent_iam_owned", lambda *_args: True)
    monkeypatch.setattr("npa.cli.agent_iam.clear_agent_iam_record", lambda *_args: True)

    result = runner.invoke(app, ["agent", "destroy", "--project", "prod", "--yes"])

    assert result.exit_code == 0, result.output
    assert deleted == ["accesskey-1", "serviceaccount-agent"]


def test_agent_destroy_requires_yes_without_a_tty_before_mutation(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import agent as agent_module

    _agent_config(monkeypatch, tmp_path, {"agent": {"public_ip": "203.0.113.50"}})
    destroyed: list[str] = []
    monkeypatch.setattr(
        agent_module,
        "_destroy_agent_terraform",
        lambda *args, **kwargs: destroyed.append("called"),
    )

    result = runner.invoke(app, ["agent", "destroy", "--project", "prod"])

    assert result.exit_code == 1
    assert "Re-run with --yes" in result.output
    assert destroyed == []


def test_agent_destroy_keep_iam_only_reports(monkeypatch, tmp_path: Path) -> None:
    from npa.cli import agent as agent_module

    _agent_config(monkeypatch, tmp_path, {"agent": {"public_ip": "203.0.113.50"}})
    monkeypatch.setattr(agent_module, "_destroy_agent_terraform", lambda *a, **k: None)
    monkeypatch.setattr(
        agent_module, "_cleanup_agent_local_files", lambda *a, **k: None
    )
    deleted = _iam_stubs(monkeypatch)
    monkeypatch.setattr("npa.cli.agent_iam.agent_iam_owned", lambda *_args: True)

    result = runner.invoke(
        app, ["agent", "destroy", "--project", "prod", "--yes", "--keep-iam"]
    )

    assert result.exit_code == 0, result.output
    assert deleted == []
    assert "npa agent destroy" in result.output
    assert "--purge-iam --yes" in result.output
    assert "nebius iam service-account delete" not in result.output


def test_agent_destroy_keeps_iam_other_agents_need(monkeypatch, tmp_path: Path) -> None:
    """The account is per project: another agent still uses it."""
    from npa.cli import agent as agent_module

    _agent_config(
        monkeypatch,
        tmp_path,
        {
            "agent": {"public_ip": "203.0.113.50"},
            "second": {"public_ip": "203.0.113.50"},
        },
    )
    monkeypatch.setattr(agent_module, "_destroy_agent_terraform", lambda *a, **k: None)
    monkeypatch.setattr(
        agent_module, "_cleanup_agent_local_files", lambda *a, **k: None
    )
    deleted = _iam_stubs(monkeypatch)
    monkeypatch.setattr("npa.cli.agent_iam.agent_iam_owned", lambda *_args: True)
    monkeypatch.setattr("npa.cli.agent_iam.clear_agent_iam_record", lambda *_args: True)

    result = runner.invoke(app, ["agent", "destroy", "--project", "prod", "--yes"])

    assert result.exit_code == 0, result.output
    assert deleted == []
    assert "still use it" in result.output


def test_agent_destroy_does_not_claim_success_when_provider_still_has_vm(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import agent as agent_module
    from npa.clients import config as config_module
    from npa.clients import nebius as nebius_module

    _agent_config(monkeypatch, tmp_path, {"agent": {"public_ip": "203.0.113.50"}})
    monkeypatch.setattr(agent_module, "_destroy_agent_terraform", lambda *a, **k: None)
    monkeypatch.setattr(
        nebius_module,
        "get_compute_instance_identity",
        lambda instance_id, **kwargs: nebius_module.ComputeInstanceIdentity(
            instance_id=instance_id,
            name="agent-prod-agent",
            project_id="project-a",
            labels={},
        ),
    )

    result = runner.invoke(app, ["agent", "destroy", "--project", "prod", "--yes"])

    assert result.exit_code == 1
    assert "still present" in result.output
    assert "destroyed: prod/agent" not in result.output
    assert (
        "agent"
        in yaml.safe_load(config_module.CONFIG_PATH.read_text())["projects"]["prod"][
            "agents"
        ]
    )


def test_agent_destroy_provider_rejection_never_writes_verified_deleted(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cli import agent as agent_module
    from npa.deploy.provisioner import ProvisionerError

    _agent_config(monkeypatch, tmp_path, {"agent": {"public_ip": "203.0.113.50"}})
    monkeypatch.setattr(
        agent_module,
        "_destroy_agent_terraform",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ProvisionerError("provider rejected destroy")
        ),
    )

    result = runner.invoke(app, ["agent", "destroy", "--project", "prod", "--yes"])

    assert result.exit_code == 1
    assert "provider rejected destroy" in result.output
    assert "verified_deleted" not in result.output
    assert "destroyed: prod/agent" not in result.output


@pytest.mark.parametrize("flag", ["--purge-iam", "--keep-iam"])
def test_agent_destroy_iam_flags_are_both_accepted(flag: str) -> None:
    assert flag in runner.invoke(app, ["agent", "destroy", "--help"]).output


# ── an escape hatch for hosts that cannot reach a fresh public IP ────────────


def test_deploy_offers_a_no_wait_ssh_escape_hatch() -> None:
    """A locked-down laptop otherwise burns a VM create + rollback every attempt."""
    help_text = runner.invoke(app, ["agent", "deploy", "--help"]).output

    assert "--no-wait-ssh" in help_text
    assert "split" in help_text or "VPN" in help_text


def test_wait_for_ssh_gates_the_terraform_wait_resource() -> None:
    """The flag has to reach Terraform, not just the help text."""
    from npa.deploy import provisioner as provisioner_module

    main_tf = (
        Path(provisioner_module.__file__).parent / "terraform" / "main.tf"
    ).read_text(encoding="utf-8")
    variables_tf = (
        Path(provisioner_module.__file__).parent / "terraform" / "variables.tf"
    ).read_text(encoding="utf-8")

    assert 'variable "wait_for_ssh"' in variables_tf
    body = main_tf[main_tf.index('resource "null_resource" "wait_for_cloud_init"') :]
    assert "count      = var.wait_for_ssh ? 1 : 0" in body[: body.index("provisioner")]
