"""Unit tests for `npa soperator` deploy spec + tfvars rendering + CLI wiring.

These tests must not touch real infrastructure: they exercise pure spec/tfvars
logic and the Typer command surface (help + validation), mocking the terraform
lifecycle at the call site for the deploy path.
"""

from __future__ import annotations

import json
import textwrap

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.soperator import spec_from_mapping
from npa.soperator.spec import SoperatorSpec, SoperatorSpecError, WorkerPoolSpec, load_spec
from npa.soperator.tfvars import render_tfvars

runner = CliRunner()


def _base_spec_mapping() -> dict:
    return {
        "apiVersion": "npa.soperator/v0.0.1",
        "name": "npatest",
        "region": "us-central1",
        "tenant_id": "tenant-x",
        "project_id": "project-x",
        "ssh_public_keys": ["ssh-ed25519 AAAA me"],
        "workers": [
            {"name": "cpu", "platform": "cpu-d3", "preset": "8vcpu-32gb", "docker_cache": True},
            {
                "name": "gpu",
                "platform": "gpu-b200-sxm",
                "preset": "8gpu-160vcpu-1792gb",
                "size": 2,
                "fabric": "us-central1-b",
                "preemptible": True,
                "docker_cache": True,
            },
        ],
    }


def test_help() -> None:
    result = runner.invoke(app, ["soperator", "--help"])
    assert result.exit_code == 0
    assert "Slurm-on-Kubernetes" in result.output


def test_deploy_help_documents_spec_and_fixes() -> None:
    result = runner.invoke(app, ["soperator", "deploy", "--help"])
    assert result.exit_code == 0
    assert "--spec" in result.output
    assert "--apply-fixes" in result.output


def test_spec_multiple_presets_and_docker_cache() -> None:
    spec = spec_from_mapping(_base_spec_mapping())
    spec.validate()
    assert [w.name for w in spec.workers] == ["cpu", "gpu"]
    assert spec.workers[0].platform == "cpu-d3"
    assert spec.workers[1].platform == "gpu-b200-sxm"
    assert spec.workers[1].preemptible is True
    assert all(w.docker_cache for w in spec.workers)


def test_gpu_pool_requires_fabric() -> None:
    data = _base_spec_mapping()
    data["workers"][1]["fabric"] = ""
    spec = spec_from_mapping(data)
    with pytest.raises(SoperatorSpecError, match="requires a non-empty 'fabric'"):
        spec.validate()


def test_docker_cache_gib_must_be_divisible_by_93() -> None:
    pool = WorkerPoolSpec(name="cpu", docker_cache=True, docker_cache_gib=500)
    with pytest.raises(SoperatorSpecError, match="divisible by 93"):
        pool.validate()


def test_system_min_size_floor() -> None:
    spec = SoperatorSpec(name="c", system_min_size=1, workers=[WorkerPoolSpec(name="w")])
    with pytest.raises(SoperatorSpecError, match="system_min_size must be >= 3"):
        spec.validate()


def test_render_tfvars_emits_multi_preset_and_io_m3_cache() -> None:
    spec = spec_from_mapping(_base_spec_mapping())
    spec.validate()
    tf = render_tfvars(spec)
    # Both worker pools rendered with their distinct presets.
    assert 'platform = "cpu-d3"' in tf
    assert 'platform = "gpu-b200-sxm"' in tf
    assert 'preset   = "8gpu-160vcpu-1792gb"' in tf
    # GPU pool carries the fabric; CPU pool does not need it.
    assert 'infiniband_fabric = "us-central1-b"' in tf
    # Docker cache -> node_local_image_disk enabled with IO_M3 disk.
    assert "node_local_image_disk = {" in tf
    assert 'disk_type       = "NETWORK_SSD_IO_M3"' in tf
    assert "enabled = true" in tf
    # Preemptible GPU pool.
    assert "preemptible = {}" in tf
    # AppArmor default off (unconfined) and accounting/telemetry off.
    assert "use_default_apparmor_profile = false" in tf
    assert "accounting_enabled = false" in tf
    assert 'active_checks_scope    = "essential"' in tf
    assert "slurm_rest_enabled = false" in tf
    # Current upstream main requires an explicit node-group bundle version and
    # replaced the old root-level system_resources input with sizing tiers.
    assert 'k8s_version        = "1.34"' in tf
    assert 'node_group_version = "72"' in tf
    assert "system_resources =" not in tf


def test_gpu_bootstrap_checks_require_accounting_backed_rest_api() -> None:
    data = _base_spec_mapping()
    data["accounting"] = True

    tf = render_tfvars(spec_from_mapping(data))

    assert "accounting_enabled = true" in tf
    assert "slurm_rest_enabled = true" in tf
    assert 'active_checks_scope    = "dev"' in tf


def test_k8s_and_node_group_versions_parse_and_must_be_non_empty() -> None:
    data = _base_spec_mapping()
    data["k8s_version"] = "1.35"
    data["node_group_version"] = "80"
    spec = spec_from_mapping(data)
    spec.validate()
    assert spec.k8s_version == "1.35"
    assert spec.node_group_version == "80"
    assert 'k8s_version        = "1.35"' in render_tfvars(spec)
    assert 'node_group_version = "80"' in render_tfvars(spec)

    spec.node_group_version = ""
    with pytest.raises(SoperatorSpecError, match="must both be non-empty"):
        spec.validate()


def test_current_sizing_tier_control_plane_defaults() -> None:
    spec = SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="w")])
    assert spec.system_preset == "16vcpu-64gb"
    assert spec.controller_preset == "16vcpu-64gb"
    assert spec.login_preset == "16vcpu-64gb"


def test_resolve_login_ssh_key_from_operator_home(tmp_path) -> None:
    from npa.soperator.lifecycle import _with_resolved_ssh_public_keys

    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_ed25519.pub").write_text("ssh-ed25519 AAAA operator\n")
    original = SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="w")])

    resolved = _with_resolved_ssh_public_keys(original, home=tmp_path)

    assert resolved.ssh_public_keys == ["ssh-ed25519 AAAA operator"]
    assert original.ssh_public_keys == []


def test_explicit_login_ssh_key_wins_and_missing_fallback_fails(tmp_path) -> None:
    from npa.soperator.lifecycle import _with_resolved_ssh_public_keys

    explicit = SoperatorSpec(
        name="c",
        workers=[WorkerPoolSpec(name="w")],
        ssh_public_keys=["ssh-rsa AAAA explicit"],
    )
    assert _with_resolved_ssh_public_keys(explicit, home=tmp_path) is explicit

    missing = SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="w")])
    with pytest.raises(ValueError, match="requires at least one login SSH public key"):
        _with_resolved_ssh_public_keys(missing, home=tmp_path)


def test_render_tfvars_cpu_only_disables_image_disk() -> None:
    spec = SoperatorSpec(
        name="cpuonly",
        region="us-central1",
        ssh_public_keys=["ssh-ed25519 AAAA me"],
        workers=[WorkerPoolSpec(name="cpu", platform="cpu-d3", preset="8vcpu-32gb")],
    )
    spec.validate()
    tf = render_tfvars(spec)
    assert "enabled = false" in tf  # image disk disabled when docker_cache is off
    assert "NETWORK_SSD_IO_M3" in tf  # still present for the nfs_in_k8s PVC


def test_load_spec_from_yaml(tmp_path) -> None:
    path = tmp_path / "cluster.yaml"
    path.write_text(
        textwrap.dedent(
            """
            apiVersion: npa.soperator/v0.0.1
            name: fromyaml
            region: us-central1
            ssh_public_keys: ["ssh-ed25519 AAAA me"]
            workers:
              - name: cpu
                platform: cpu-d3
                preset: 8vcpu-32gb
                docker_cache: true
            """
        )
    )
    spec = load_spec(path)
    assert spec.name == "fromyaml"
    assert spec.workers[0].docker_cache is True


def test_destroy_reconstructs_tf_var_env_from_sidecar(tmp_path, monkeypatch) -> None:
    """destroy must rebuild the region/tenant/project/subnet/o11y TF_VARs.

    These are passed as env at apply time and never written to terraform.tfvars,
    so ``terraform destroy`` fails on "No value for required variable" unless the
    deploy-time env sidecar is replayed.
    """

    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    (recipe / "installations" / "example").mkdir(parents=True)
    install = recipe / "installations" / "npatest"
    install.mkdir(parents=True)
    lifecycle._write_env_sidecar(
        install,
        region="us-central1",
        tenant_id="tenant-abc",
        project_id="project-xyz",
        subnet_id="vpcsubnet-123",
        o11y_profile="npa-mk8s",
    )

    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    # _soperator_tf_env -> _terraform_env mints a real IAM token via the `nebius`
    # CLI; stub it so the destroy tests never touch real infra (CI has no nebius).
    monkeypatch.setattr(lifecycle, "_terraform_env", lambda nebius_bin: {})
    captured: dict[str, dict[str, str]] = {}

    class _Done:
        def __init__(self, stdout: str = "", returncode: int = 0) -> None:
            self.stdout = stdout
            self.returncode = returncode

    def fake_stream(cmd, *, cwd=None, env=None, timeout=None):
        return None  # terraform init

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        # The hardened destroy runs `terraform destroy` via _run_capture; record
        # its env. state pull -> empty (no cluster id); filesystem list -> none.
        if "destroy" in cmd:
            captured["env"] = dict(env or {})
        return _Done(stdout="")

    monkeypatch.setattr(lifecycle, "_run_stream", fake_stream)
    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    def fail_resolve(*args, **kwargs):  # sidecar present -> must not be called
        raise AssertionError("destroy fell back to resolve despite a sidecar")

    monkeypatch.setattr(lifecycle, "_resolve_subnet", fail_resolve)

    lifecycle.destroy_cluster("npatest", terraform_dir=recipe)

    env = captured["env"]
    assert env["TF_VAR_region"] == "us-central1"
    assert env["TF_VAR_iam_tenant_id"] == "tenant-abc"
    assert env["TF_VAR_iam_project_id"] == "project-xyz"
    assert env["TF_VAR_vpc_subnet_id"] == "vpcsubnet-123"
    assert env["TF_VAR_o11y_iam_tenant_id"] == "tenant-abc"
    assert env["TF_VAR_o11y_profile"] == "npa-mk8s"


def test_destroy_deletes_orphaned_vpc_allocation(tmp_path, monkeypatch) -> None:
    """destroy must delete a leftover ``soperator-<name>-*`` VPC allocation.

    The cloud-controller-manager can re-create the login LoadBalancer's static IP
    allocation mid-teardown after terraform deleted the in-state copy, leaving an
    orphan not in state. A later deploy then fails with "Allocation ... already
    exists", so destroy sweeps same-prefixed allocations after the cluster is gone.
    """

    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    (recipe / "installations" / "example").mkdir(parents=True)
    install = recipe / "installations" / "npasop"
    install.mkdir(parents=True)
    lifecycle._write_env_sidecar(
        install,
        region="us-central1",
        tenant_id="tenant-abc",
        project_id="project-xyz",
        subnet_id="vpcsubnet-123",
        o11y_profile="npa-mk8s",
    )

    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    # _soperator_tf_env -> _terraform_env mints a real IAM token via the `nebius`
    # CLI; stub it so the destroy tests never touch real infra (CI has no nebius).
    monkeypatch.setattr(lifecycle, "_terraform_env", lambda nebius_bin: {})
    deleted: list[str] = []

    class _Done:
        def __init__(self, stdout: str = "", returncode: int = 0) -> None:
            self.stdout = stdout
            self.returncode = returncode

    alloc_json = json.dumps(
        {
            "items": [
                {"metadata": {"id": "alloc-orphan", "name": "soperator-npasop-public-static-ip"}},
                {"metadata": {"id": "alloc-other", "name": "mk8snodegroup-abc-alias"}},
            ]
        }
    )

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        if "vpc" in cmd and "list" in cmd:
            return _Done(stdout=alloc_json)
        if "vpc" in cmd and "delete" in cmd:
            deleted.append(cmd[cmd.index("--id") + 1])
        return _Done(stdout="")

    monkeypatch.setattr(lifecycle, "_run_stream", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)
    monkeypatch.setattr(lifecycle, "_resolve_subnet", lambda *a, **k: "vpcsubnet-123")

    lifecycle.destroy_cluster("npasop", terraform_dir=recipe)

    # Only the soperator-npasop-* allocation is swept; node-group aliases are left.
    assert deleted == ["alloc-orphan"]


def test_destroy_deletes_orphaned_filesystems(tmp_path, monkeypatch) -> None:
    """destroy must delete leftover ``soperator-<name>-*`` filesystems.

    The recipe names the jail / controller-spool / accounting filesystems
    ``soperator-<name>-*``. If the destroy sweep matches only ``<name>-*`` they
    survive teardown and the next deploy fails with "filesystem ... already
    exists" (AlreadyExists). This locks in the full ``soperator-`` prefix.
    """

    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    (recipe / "installations" / "example").mkdir(parents=True)
    install = recipe / "installations" / "npasop"
    install.mkdir(parents=True)
    lifecycle._write_env_sidecar(
        install,
        region="us-central1",
        tenant_id="tenant-abc",
        project_id="project-xyz",
        subnet_id="vpcsubnet-123",
        o11y_profile="npa-mk8s",
    )

    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    # _soperator_tf_env -> _terraform_env mints a real IAM token via the `nebius`
    # CLI; stub it so the destroy tests never touch real infra (CI has no nebius).
    monkeypatch.setattr(lifecycle, "_terraform_env", lambda nebius_bin: {})
    deleted: list[str] = []

    class _Done:
        def __init__(self, stdout: str = "", returncode: int = 0) -> None:
            self.stdout = stdout
            self.returncode = returncode

    fs_json = json.dumps(
        {
            "items": [
                {"metadata": {"id": "fs-jail", "name": "soperator-npasop-jail"}},
                {"metadata": {"id": "fs-spool", "name": "soperator-npasop-controller-spool"}},
                # A same-project filesystem from another cluster must be left alone.
                {"metadata": {"id": "fs-other", "name": "soperator-npatest-jail"}},
            ]
        }
    )

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        if "filesystem" in cmd and "list" in cmd:
            return _Done(stdout=fs_json)
        if "filesystem" in cmd and "delete" in cmd:
            deleted.append(cmd[cmd.index("--id") + 1])
        return _Done(stdout="")

    monkeypatch.setattr(lifecycle, "_run_stream", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)
    monkeypatch.setattr(lifecycle, "_resolve_subnet", lambda *a, **k: "vpcsubnet-123")

    lifecycle.destroy_cluster("npasop", terraform_dir=recipe)

    # Only the soperator-npasop-* filesystems are swept; other clusters untouched.
    assert sorted(deleted) == ["fs-jail", "fs-spool"]


def test_nebius_cli_env_strips_stale_iam_token(monkeypatch) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "expired-token")
    monkeypatch.delenv("NPA_REUSE_IAM_TOKEN", raising=False)
    # Pre-flight ``nebius`` calls must drop a stale ambient token so the CLI falls
    # back to the auto-refreshing profile exec-plugin instead of failing 401.
    assert "NEBIUS_IAM_TOKEN" not in lifecycle._nebius_cli_env()


def test_nebius_cli_env_keeps_token_when_reuse_opt_in(monkeypatch) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "ci-injected-token")
    monkeypatch.setenv("NPA_REUSE_IAM_TOKEN", "1")
    # CI can intentionally inject a short-lived token; honor the opt-out.
    assert lifecycle._nebius_cli_env()["NEBIUS_IAM_TOKEN"] == "ci-injected-token"


class _Done:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_install_monitoring_crds_strips_token_and_verifies(monkeypatch) -> None:
    """Happy path: kubectl runs with the stale token stripped and the CRD is
    confirmed present before returning."""
    from npa.soperator import lifecycle

    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "expired-token")
    monkeypatch.delenv("NPA_REUSE_IAM_TOKEN", raising=False)
    seen_envs: list[dict[str, str]] = []

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        seen_envs.append(dict(env or {}))
        if "helmreleases" in cmd:
            return _Done(stdout='{"items": []}')
        if "get" in cmd and "crd" in cmd:
            return _Done(stdout="customresourcedefinition.apiextensions.k8s.io/"
                                "servicemonitors.monitoring.coreos.com\n")
        return _Done(stdout="serverside-applied")

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    lifecycle._install_monitoring_crds("kubectl", "ctx")

    # A stale ambient token shadows the kubeconfig exec-plugin; every kubectl
    # call must run without it so the plugin mints a fresh credential.
    assert seen_envs, "expected kubectl to be invoked"
    assert all("NEBIUS_IAM_TOKEN" not in e for e in seen_envs)


def test_monitoring_prerequisites_create_namespace_when_telemetry_is_off(
    monkeypatch,
) -> None:
    from npa.soperator import lifecycle

    calls: list[list[str]] = []

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        calls.append(cmd)
        if "helmreleases" in cmd:
            return _Done(stdout='{"items": []}')
        if "get" in cmd and "namespace" in cmd:
            return _Done(stderr="NotFound", returncode=1)
        if "get" in cmd and "crd" in cmd:
            return _Done(stdout="servicemonitors.monitoring.coreos.com")
        return _Done(stdout="created")

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    lifecycle._install_monitoring_crds("kubectl", "ctx")

    assert [
        "kubectl",
        "--context",
        "ctx",
        "create",
        "namespace",
        "monitoring-system",
    ] in calls


def test_monitoring_namespace_creation_failure_is_actionable(monkeypatch) -> None:
    from npa.soperator import lifecycle

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        if "get" in cmd and "namespace" in cmd:
            return _Done(stderr="NotFound", returncode=1)
        if "create" in cmd and "namespace" in cmd:
            return _Done(stderr="Forbidden", returncode=1)
        return _Done(stdout="")

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    with pytest.raises(RuntimeError, match="failed to ensure monitoring-system"):
        lifecycle._install_monitoring_crds("kubectl", "ctx")


def test_monitoring_namespace_creation_race_accepts_existing_namespace(
    monkeypatch,
) -> None:
    from npa.soperator import lifecycle

    namespace_reads = 0

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        nonlocal namespace_reads
        if "helmreleases" in cmd:
            return _Done(stdout='{"items": []}')
        if "get" in cmd and "namespace" in cmd:
            namespace_reads += 1
            if namespace_reads == 1:
                return _Done(stderr="NotFound", returncode=1)
            return _Done(stdout="namespace/monitoring-system")
        if "create" in cmd and "namespace" in cmd:
            return _Done(stderr="AlreadyExists", returncode=1)
        if "get" in cmd and "crd" in cmd:
            return _Done(stdout="servicemonitors.monitoring.coreos.com")
        return _Done(stdout="created")

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    lifecycle._install_monitoring_crds("kubectl", "ctx")
    assert namespace_reads == 2


def test_monitoring_prerequisites_reset_only_stalled_dashboard_release(
    monkeypatch,
) -> None:
    from npa.soperator import lifecycle

    calls: list[list[str]] = []
    releases = {
        "items": [
            {
                "metadata": {"name": "stack-monitoring-dashboards"},
                "status": {
                    "conditions": [
                        {
                            "type": "Ready",
                            "status": "False",
                            "reason": "RetriesExceeded",
                        }
                    ]
                },
            },
            {
                "metadata": {"name": "healthy-monitoring-dashboards"},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}]
                },
            },
            {
                "metadata": {"name": "progressing-monitoring-dashboards"},
                "status": {
                    "conditions": [
                        {
                            "type": "Ready",
                            "status": "Unknown",
                            "reason": "Progressing",
                        }
                    ]
                },
            },
            {"metadata": {"name": "unrelated"}},
        ]
    }

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        calls.append(cmd)
        if "namespace" in cmd:
            return _Done(stdout="namespace/monitoring-system")
        if "helmreleases" in cmd:
            return _Done(stdout=json.dumps(releases))
        if "get" in cmd and "crd" in cmd:
            return _Done(stdout="established")
        return _Done(stdout="ok")

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)
    monkeypatch.setattr(lifecycle.time, "time_ns", lambda: 1234)

    lifecycle._install_monitoring_crds("kubectl", "ctx")

    annotation_calls = [cmd for cmd in calls if "annotate" in cmd]
    assert annotation_calls == [
        [
            "kubectl",
            "--context",
            "ctx",
            "-n",
            "flux-system",
            "annotate",
            "helmrelease",
            "stack-monitoring-dashboards",
            "reconcile.fluxcd.io/requestedAt=1234",
            "reconcile.fluxcd.io/resetAt=1234",
            "--overwrite",
        ]
    ]


def test_monitoring_release_inspection_failure_is_actionable(monkeypatch) -> None:
    from npa.soperator import lifecycle

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        if "namespace" in cmd:
            return _Done(stdout="namespace/monitoring-system")
        if "helmreleases" in cmd:
            return _Done(stderr="Forbidden", returncode=1)
        if "get" in cmd and "crd" in cmd:
            return _Done(stdout="established")
        return _Done(stdout="ok")

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    with pytest.raises(RuntimeError, match="failed to inspect monitoring HelmReleases"):
        lifecycle._install_monitoring_crds("kubectl", "ctx")


def test_monitoring_release_inspection_allows_clean_pre_flux_install(monkeypatch) -> None:
    from npa.soperator import lifecycle

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        if "namespace" in cmd:
            return _Done(stdout="namespace/monitoring-system")
        if "helmreleases" in cmd:
            return _Done(
                stderr='the server doesn\'t have a resource type "helmreleases"',
                returncode=1,
            )
        if "get" in cmd and "crd" in cmd:
            return _Done(stdout="established")
        return _Done(stdout="ok")

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    lifecycle._install_monitoring_crds("kubectl", "ctx")


def test_install_monitoring_crds_raises_on_failure(monkeypatch) -> None:
    """A failed apply must raise (fail loud + fast), not be swallowed into a
    later operator HelmRelease timeout."""
    from npa.soperator import lifecycle

    monkeypatch.setattr(lifecycle.time, "sleep", lambda *a, **k: None)

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        if "apply" in cmd:
            return _Done(stderr="Unauthenticated: invalid token", returncode=1)
        return _Done(stdout="")

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    with pytest.raises(RuntimeError, match="prometheus-operator CRD"):
        lifecycle._install_monitoring_crds("kubectl", "ctx")


def test_install_monitoring_crds_raises_when_crd_absent(monkeypatch) -> None:
    """Apply reports success but the CRD never registers (wrong context / no-op):
    the post-install verification must catch it."""
    from npa.soperator import lifecycle

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        if "get" in cmd and "crd" in cmd:
            return _Done(stdout="")  # not registered
        return _Done(stdout="serverside-applied")

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    with pytest.raises(RuntimeError, match="ServiceMonitor CRD not present"):
        lifecycle._install_monitoring_crds("kubectl", "ctx")


def _write_recipe_locals(tmp_path, essential_body: str):
    """Write a minimal locals_active_checks.tf with an ``essential`` scope."""

    locals_tf = tmp_path / "modules" / "slurm" / "locals_active_checks.tf"
    locals_tf.parent.mkdir(parents=True, exist_ok=True)
    locals_tf.write_text(
        "locals {\n"
        "  active_checks_scopes = {\n"
        "    essential = {\n"
        f"{essential_body}"
        "    }\n"
        "  }\n"
        "}\n"
    )
    return locals_tf


def test_patch_active_checks_locals_adds_healthy_nodes_override(tmp_path) -> None:
    """The essential scope must skip ensure-healthy-nodes at creation, else
    wait-for-active-checks deadlocks on a CPU-only cluster (its GPU deps never
    run)."""
    from npa.soperator import lifecycle

    locals_tf = _write_recipe_locals(
        tmp_path,
        "      all-reduce-perf-nccl-in-docker = {\n"
        "        runAfterCreation = false\n"
        "      }\n",
    )

    assert lifecycle._patch_active_checks_locals(tmp_path) is True
    text = locals_tf.read_text()
    # The override lands inside the essential scope, before the first existing key.
    assert "ensure-healthy-nodes = {" in text
    assert "runAfterCreation = false" in text
    essential_idx = text.index("essential = {")
    assert text.index("ensure-healthy-nodes") > essential_idx


def test_patch_active_checks_locals_is_idempotent(tmp_path) -> None:
    from npa.soperator import lifecycle

    locals_tf = _write_recipe_locals(
        tmp_path,
        "      ssh-check = {\n"
        "        commentPrefix = null\n"
        "      }\n",
    )

    assert lifecycle._patch_active_checks_locals(tmp_path) is True
    once = locals_tf.read_text()
    assert lifecycle._patch_active_checks_locals(tmp_path) is False
    assert locals_tf.read_text() == once
    # Exactly one override block, not duplicated.
    assert once.count("ensure-healthy-nodes = {") == 1


def test_patch_active_checks_locals_missing_file(tmp_path) -> None:
    from npa.soperator import lifecycle

    # No modules/slurm/locals_active_checks.tf -> no-op, no crash.
    assert lifecycle._patch_active_checks_locals(tmp_path) is False


def test_patch_nodeconfigurator_allows_enroot_userns_and_is_idempotent(
    tmp_path,
) -> None:
    from npa.soperator import lifecycle

    template = (
        tmp_path
        / "modules"
        / "slurm"
        / "templates"
        / "helm_values"
        / "terraform_fluxcd_values.yaml.tftpl"
    )
    template.parent.mkdir(parents=True)
    template.write_text(
        "        fluxcd:\n"
        "          nodeConfigurator:\n"
        "            enabled: true\n"
        "            values:\n"
        "              resources: {}\n"
        "          anotherRelease:\n"
    )

    assert lifecycle._patch_nodeconfigurator_userns(tmp_path) is True
    patched = template.read_text()
    assert "sysctl -w kernel.unprivileged_userns_clone=1" in patched
    assert "sysctl -w kernel.apparmor_restrict_unprivileged_userns=0" in patched
    assert '[ "${apparmor_enabled}" = "false" ]' in patched
    assert "sysctl -w net.core.rmem_max=536870912" in patched
    assert patched.index("initContainers:") < patched.index("resources: {}")

    assert lifecycle._patch_nodeconfigurator_userns(tmp_path) is False
    assert template.read_text() == patched
    assert patched.count("# npa: allow Enroot user namespaces on Ubuntu hosts") == 1


def test_patch_nodeconfigurator_missing_template_is_safe(tmp_path) -> None:
    from npa.soperator import lifecycle

    assert lifecycle._patch_nodeconfigurator_userns(tmp_path) is False
