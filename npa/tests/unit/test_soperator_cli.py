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
from npa.soperator.spec import (
    DEFAULT_SOLUTIONS_LIBRARY_REF,
    SoperatorSpec,
    SoperatorSpecError,
    WorkerPoolSpec,
    load_spec,
    sizing_tier_for_worker_count,
)
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
    assert "--root-login-ssh-pub" in result.output
    assert DEFAULT_SOLUTIONS_LIBRARY_REF[:20] in result.output


def test_spec_multiple_presets_and_docker_cache() -> None:
    spec = spec_from_mapping(_base_spec_mapping())
    spec.validate()
    assert [w.name for w in spec.workers] == ["cpu", "gpu"]
    assert spec.workers[0].platform == "cpu-d3"
    assert spec.workers[1].platform == "gpu-b200-sxm"
    assert spec.workers[1].preemptible is True
    assert all(w.docker_cache for w in spec.workers)


def test_existing_positional_sdk_fields_keep_their_meaning() -> None:
    spec = SoperatorSpec(
        "c",
        "us-central1",
        "tenant",
        "project",
        "subnet",
        ["ssh-ed25519 AAAA legacy"],
        3,
        "16vcpu-64gb",
        "16vcpu-64gb",
        "16vcpu-64gb",
        [WorkerPoolSpec(name="w")],
    )
    spec.validate()
    assert spec.ssh_public_keys == ["ssh-ed25519 AAAA legacy"]
    assert spec.system_min_size == 3
    assert spec.workers[0].name == "w"


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
    # The pinned upstream contract requires an explicit node-group bundle and
    # resolves omitted control-plane presets from its worker-count sizing tier.
    assert 'k8s_version        = "1.34"' in tf
    assert 'node_group_version = "72"' in tf
    assert 'slurm_operator_version = "4.1.6"' in tf
    assert "max_size = 24" in tf
    assert tf.count("preset   = null") >= 2
    assert "system_resources =" not in tf


@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("accounting", [False, True])
def test_rest_and_accounting_are_independent_for_cpu_gpu_combinations(
    gpu: bool, accounting: bool
) -> None:
    data = _base_spec_mapping()
    data["accounting"] = accounting
    if not gpu:
        data["workers"] = [data["workers"][0]]

    spec = spec_from_mapping(data)
    spec.validate()
    tf = render_tfvars(spec)

    assert f"accounting_enabled = {str(accounting).lower()}" in tf
    # Omission is backward-compatible: REST follows accounting. GPU validation
    # remains mandatory through NPA's direct login-jail creation check when the
    # pinned operator cannot provide its REST-backed ActiveChecks.
    assert f"slurm_rest_enabled = {str(accounting).lower()}" in tf
    expected_scope = "dev" if gpu and accounting else "essential"
    assert f'active_checks_scope    = "{expected_scope}"' in tf


def test_gpu_cluster_allows_rest_disable_without_losing_direct_creation_check() -> None:
    data = _base_spec_mapping()
    data["accounting"] = False
    data["slurm_rest_enabled"] = False

    spec = spec_from_mapping(data)
    spec.validate()
    tf = render_tfvars(spec)
    assert "accounting_enabled = false" in tf
    assert "slurm_rest_enabled = false" in tf
    assert 'active_checks_scope    = "essential"' in tf


def test_explicit_rest_contract_rejects_operator_unsupported_no_accounting() -> None:
    data = _base_spec_mapping()
    data["accounting"] = False
    data["slurm_rest_enabled"] = True

    with pytest.raises(SoperatorSpecError, match="controller skips REST reconciliation"):
        spec_from_mapping(data).validate()

    # The toggles remain independent in the accepted direction: accounting can
    # be enabled while REST is explicitly disabled.
    data["accounting"] = True
    data["slurm_rest_enabled"] = False
    spec = spec_from_mapping(data)
    spec.validate()
    tf = render_tfvars(spec)
    assert "accounting_enabled = true" in tf
    assert "slurm_rest_enabled = false" in tf
    assert 'active_checks_scope    = "essential"' in tf


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
    assert spec.system_preset is None
    assert spec.controller_preset is None
    assert spec.accounting_preset is None
    assert spec.login_preset == "16vcpu-64gb"


@pytest.mark.parametrize(
    ("worker_count", "expected_tier"),
    [
        (9, "XS"),
        (10, "S"),
        (99, "S"),
        (100, "M"),
        (499, "M"),
        (500, "L"),
        (1999, "L"),
        (2000, "XL"),
    ],
)
def test_sizing_tier_threshold_boundaries(worker_count: int, expected_tier: str) -> None:
    assert sizing_tier_for_worker_count(worker_count) == expected_tier


@pytest.mark.parametrize(
    ("worker_count", "rejected", "accepted"),
    [
        (1, "8vcpu-32gb", "16vcpu-64gb"),
        (499, "8vcpu-32gb", "16vcpu-64gb"),
        (500, "16vcpu-64gb", "32vcpu-128gb"),
        (2000, "32vcpu-128gb", "64vcpu-256gb"),
    ],
)
def test_system_preset_rejected_and_accepted_at_every_capacity_boundary(
    worker_count: int, rejected: str, accepted: str
) -> None:
    workers = [WorkerPoolSpec(name="w", size=worker_count)]
    with pytest.raises(SoperatorSpecError, match="control_plane.system.preset"):
        SoperatorSpec(name="c", workers=workers, system_preset=rejected).validate()
    SoperatorSpec(name="c", workers=workers, system_preset=accepted).validate()


@pytest.mark.parametrize("role", ["controller", "login"])
def test_controller_and_login_actual_minimum_preset(role: str) -> None:
    kwargs = {f"{role}_preset": "8vcpu-32gb"}
    with pytest.raises(SoperatorSpecError, match=rf"control_plane.{role}.preset"):
        SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="w")], **kwargs).validate()
    kwargs[f"{role}_preset"] = "16vcpu-64gb"
    SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="w")], **kwargs).validate()


@pytest.mark.parametrize(
    ("worker_count", "rejected", "accepted"),
    [
        (499, "4vcpu-16gb", "8vcpu-32gb"),
        (500, "8vcpu-32gb", "16vcpu-64gb"),
        (2000, "16vcpu-64gb", "32vcpu-128gb"),
    ],
)
def test_accounting_preset_boundaries_when_accounting_enabled(
    worker_count: int, rejected: str, accepted: str
) -> None:
    workers = [WorkerPoolSpec(name="w", size=worker_count)]
    with pytest.raises(SoperatorSpecError, match="control_plane.accounting.preset"):
        SoperatorSpec(
            name="c",
            workers=workers,
            accounting=True,
            accounting_preset=rejected,
        ).validate()
    SoperatorSpec(
        name="c",
        workers=workers,
        accounting=True,
        accounting_preset=accepted,
    ).validate()


def test_system_max_size_preserves_upstream_autoscaling_and_validates_override() -> None:
    spec = SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="w")])
    assert "max_size = 24" in render_tfvars(spec)
    spec.system_min_size = 25
    assert "max_size = 25" in render_tfvars(spec)
    spec.system_max_size = 24
    with pytest.raises(SoperatorSpecError, match="max_size must be >= min_size"):
        spec.validate()


def test_resolve_login_ssh_key_from_operator_home(tmp_path) -> None:
    from npa.soperator.lifecycle import _with_resolved_ssh_public_keys

    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_ed25519.pub").write_text("ssh-ed25519 AAAA operator\n")
    original = SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="w")])

    resolved = _with_resolved_ssh_public_keys(original, home=tmp_path)

    assert resolved.root_login_ssh_public_key == "ssh-ed25519 AAAA operator"
    assert resolved.ssh_public_keys == []
    assert original.ssh_public_keys == []


def test_explicit_login_ssh_key_wins_and_missing_fallback_fails(tmp_path) -> None:
    from npa.soperator.lifecycle import _with_resolved_ssh_public_keys

    explicit = SoperatorSpec(
        name="c",
        workers=[WorkerPoolSpec(name="w")],
        ssh_public_keys=["ssh-rsa AAAA explicit"],
    )
    resolved = _with_resolved_ssh_public_keys(explicit, home=tmp_path)
    assert resolved.root_login_ssh_public_key == "ssh-rsa AAAA explicit"
    assert resolved.ssh_public_keys == []

    missing = SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="w")])
    with pytest.raises(ValueError, match="root access requires one SSH public key"):
        _with_resolved_ssh_public_keys(missing, home=tmp_path)


def test_root_login_ssh_key_precedence_and_nonstandard_ci_path(tmp_path) -> None:
    from npa.soperator.lifecycle import _resolve_root_login_ssh_public_key

    explicit_file = tmp_path / "ci" / "operator.pub"
    explicit_file.parent.mkdir()
    explicit_file.write_text("ssh-ed25519 AAAA explicit-file\n")
    env_file = tmp_path / "env.pub"
    env_file.write_text("ssh-ed25519 AAAA env-file\n")
    home_key = tmp_path / ".ssh" / "id_ed25519.pub"
    home_key.parent.mkdir()
    home_key.write_text("ssh-ed25519 AAAA home\n")
    env = {
        "NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY": "ssh-ed25519 AAAA env-inline",
        "NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY_FILE": str(env_file),
    }
    spec = SoperatorSpec(
        name="c",
        workers=[WorkerPoolSpec(name="w")],
        root_login_ssh_public_key="ssh-ed25519 AAAA spec",
    )

    resolved = _resolve_root_login_ssh_public_key(
        spec, explicit_file=explicit_file, environ=env, home=tmp_path
    )
    assert resolved.value.endswith("explicit-file")
    assert resolved.source == "explicit argument"
    assert resolved.fingerprint.startswith("SHA256:")

    resolved = _resolve_root_login_ssh_public_key(spec, environ=env, home=tmp_path)
    assert resolved.value.endswith("spec")
    assert resolved.source == "spec root_login_ssh_public_key"

    empty_spec = SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="w")])
    resolved = _resolve_root_login_ssh_public_key(empty_spec, environ=env, home=tmp_path)
    assert resolved.value.endswith("env-inline")
    assert resolved.source == "NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY"

    env.pop("NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY")
    resolved = _resolve_root_login_ssh_public_key(empty_spec, environ=env, home=tmp_path)
    assert resolved.value.endswith("env-file")
    assert resolved.source == "NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY_FILE"

    resolved = _resolve_root_login_ssh_public_key(empty_spec, environ={}, home=tmp_path)
    assert resolved.value.endswith("home")
    assert resolved.source == "operator default id_ed25519.pub"


@pytest.mark.parametrize(
    "bad_key",
    [
        "",
        "command=x ssh-ed25519 AAAA comment",
        "ssh-ed25519 not-base64! comment",
        "ssh-ed25519 AAAA one\nssh-ed25519 AAAA two",
    ],
)
def test_root_login_ssh_key_rejects_invalid_or_multiple_records(bad_key: str) -> None:
    spec = SoperatorSpec(
        name="c",
        workers=[WorkerPoolSpec(name="w")],
        root_login_ssh_public_key=bad_key,
    )
    if not bad_key:
        # Empty means omitted and is resolved later by the deploy environment.
        spec.validate()
    else:
        with pytest.raises(SoperatorSpecError, match="root login SSH key"):
            spec.validate()


def test_legacy_root_login_key_alias_accepts_one_record_only() -> None:
    spec = SoperatorSpec(
        name="c",
        workers=[WorkerPoolSpec(name="w")],
        ssh_public_keys=["ssh-ed25519 AAAA one", "ssh-ed25519 AAAA two"],
    )
    with pytest.raises(SoperatorSpecError, match="exactly one"):
        spec.validate()


def test_root_login_ssh_key_hcl_rendering_escapes_comment_safely() -> None:
    key = 'ssh-ed25519 AAAA ci-${var.bad}-%{if true}-"quoted"-\\path'
    spec = SoperatorSpec(
        name="c",
        workers=[WorkerPoolSpec(name="w")],
        root_login_ssh_public_key=key,
    )
    spec.validate()
    rendered = render_tfvars(spec)
    assert "${var.bad}" not in rendered.replace("$${var.bad}", "")
    assert "%{if true}" not in rendered.replace("%%{if true}", "")
    assert (
        '  "ssh-ed25519 AAAA ci-$${var.bad}-%%{if true}-\\"quoted\\"-\\\\path",'
        in rendered
    )


def test_operator_override_validation_and_verified_userns_contract() -> None:
    spec = SoperatorSpec(
        name="c",
        workers=[WorkerPoolSpec(name="w")],
        slurm_operator_version="latest",
    )
    with pytest.raises(SoperatorSpecError, match="semantic version"):
        spec.validate()

    spec.slurm_operator_version = "4.1.7"
    with pytest.raises(SoperatorSpecError, match="user-namespace override is verified only"):
        spec.validate()

    spec.use_default_apparmor_profile = True
    spec.validate()  # explicit newer chart is possible when the pinned sysctl override is unused

    spec.slurm_operator_version = "4.2.0"
    with pytest.raises(SoperatorSpecError, match="outside the pinned runtime contract"):
        spec.validate()


def test_rest_contract_rejects_string_boolean() -> None:
    data = _base_spec_mapping()
    data["slurm_rest_enabled"] = "false"

    with pytest.raises(SoperatorSpecError, match="must be a boolean"):
        spec_from_mapping(data)


def test_solutions_library_ref_requires_immutable_commit() -> None:
    from npa.soperator.lifecycle import _validate_immutable_solutions_library_ref

    assert _validate_immutable_solutions_library_ref(DEFAULT_SOLUTIONS_LIBRARY_REF) == (
        DEFAULT_SOLUTIONS_LIBRARY_REF
    )
    with pytest.raises(ValueError, match="immutable 40-character"):
        _validate_immutable_solutions_library_ref("main")


def test_solutions_library_resolution_preserves_legacy_install_state(tmp_path) -> None:
    from npa.soperator.lifecycle import _resolve_solutions_library

    recipe = tmp_path / "nebius-solutions-library" / "soperator"
    (recipe / "installations" / "example").mkdir(parents=True)

    assert (
        _resolve_solutions_library(None, tmp_path, DEFAULT_SOLUTIONS_LIBRARY_REF)
        == recipe
    )


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
    monkeypatch.setattr(lifecycle, "_assert_solutions_library_contract", lambda *a, **k: None)
    # _soperator_tf_env -> _terraform_env mints a real IAM token via the `nebius`
    # CLI; stub it so the destroy tests never touch real infra (CI has no nebius).
    monkeypatch.setattr(lifecycle, "_terraform_env", lambda nebius_bin, **kwargs: {})
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
    monkeypatch.setattr(lifecycle, "_assert_solutions_library_contract", lambda *a, **k: None)
    # _soperator_tf_env -> _terraform_env mints a real IAM token via the `nebius`
    # CLI; stub it so the destroy tests never touch real infra (CI has no nebius).
    monkeypatch.setattr(lifecycle, "_terraform_env", lambda nebius_bin, **kwargs: {})
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
    monkeypatch.setattr(lifecycle, "_assert_solutions_library_contract", lambda *a, **k: None)
    # _soperator_tf_env -> _terraform_env mints a real IAM token via the `nebius`
    # CLI; stub it so the destroy tests never touch real infra (CI has no nebius).
    monkeypatch.setattr(lifecycle, "_terraform_env", lambda nebius_bin, **kwargs: {})
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


def test_soperator_terraform_token_uses_explicit_nebius_profile(monkeypatch) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setenv("NPA_NEBIUS_PROFILE", "cross-tenant-profile")
    seen: dict[str, str] = {}

    def fake_tf_env(nebius_bin, *, profile=""):
        seen["profile"] = profile
        return {}

    monkeypatch.setattr(lifecycle, "_terraform_env", fake_tf_env)
    env = lifecycle._soperator_tf_env(
        "nebius",
        region="us-central1",
        tenant_id="tenant",
        project_id="project",
        subnet_id="subnet",
    )

    assert seen["profile"] == "cross-tenant-profile"
    assert env["TF_VAR_o11y_profile"] == "cross-tenant-profile"
    assert env["NEBIUS_PROFILE"] == "cross-tenant-profile"
    assert lifecycle._nebius_cli_env()["NEBIUS_PROFILE"] == "cross-tenant-profile"


def test_kube_credential_refresh_pins_selected_nebius_profile(monkeypatch) -> None:
    from npa.soperator import lifecycle

    calls: list[list[str]] = []
    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        lambda cmd, **kwargs: calls.append(cmd) or _Done(),
    )

    lifecycle._refresh_kube_credentials(
        "nebius",
        "cluster-id",
        "context",
        {"NPA_NEBIUS_PROFILE": "cross-tenant-profile"},
    )

    assert calls == [[
        "nebius",
        "--profile",
        "cross-tenant-profile",
        "mk8s",
        "cluster",
        "get-credentials",
        "--id",
        "cluster-id",
        "--external",
        "--force",
        "--context-name",
        "context",
    ]]


class _Done:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_direct_gpu_creation_check_uses_every_worker_and_gpu(monkeypatch) -> None:
    from npa.soperator import lifecycle

    calls: list[list[str]] = []

    def fake_capture(cmd, **kwargs):
        calls.append(cmd)
        return _Done(
            stdout=(
                "0: NPA_GPU_CREATION_CHECK_RESULT host=gpu-0 status=PASS command_rc=0\n"
                "1: NPA_GPU_CREATION_CHECK_RESULT host=gpu-1 status=PASS command_rc=0\n"
            )
        )

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)
    spec = SoperatorSpec(
        name="c",
        workers=[
            WorkerPoolSpec(
                name="gpu",
                platform="gpu-b200-sxm",
                preset="8gpu-160vcpu-1792gb",
                size=2,
                fabric="us-central1-b",
            )
        ],
    )

    checks = lifecycle._run_gpu_creation_checks(spec, "ctx", "kubectl")

    assert checks == [{
        "pool": "gpu",
        "nodes": 2,
        "gpus_per_node": 8,
        "tests": [
            "deviceQuery",
            "vectorAdd",
            "simpleMultiGPU",
            "p2pBandwidthLatencyTest",
        ],
        "status": "PASS",
    }]
    command = calls[0]
    assert "--nodes=2" in command
    assert "--ntasks=2" in command
    assert "--gpus-per-node=8" in command
    assert "--nodelist=gpu-0,gpu-1" in command
    task_script = command[-1]
    assert "health-checker run" in task_script
    assert "deviceQuery,vectorAdd,simpleMultiGPU,p2pBandwidthLatencyTest" in task_script


def test_direct_gpu_creation_check_fails_on_missing_worker_pass(monkeypatch) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        lambda *a, **k: _Done(
            stdout="0: NPA_GPU_CREATION_CHECK_RESULT host=gpu-0 status=PASS command_rc=0\n"
        ),
    )
    spec = SoperatorSpec(
        name="c",
        workers=[
            WorkerPoolSpec(
                name="gpu",
                platform="gpu-b200-sxm",
                preset="8gpu-160vcpu-1792gb",
                size=2,
                fabric="us-central1-b",
            )
        ],
    )

    with pytest.raises(RuntimeError, match="1/2 workers reported PASS"):
        lifecycle._run_gpu_creation_checks(spec, "ctx", "kubectl")


def test_direct_gpu_creation_check_is_noop_for_cpu_cluster(monkeypatch) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected kubectl")),
    )
    spec = SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="cpu")])

    assert lifecycle._run_gpu_creation_checks(spec, "ctx", "kubectl") == []


def test_superseded_activechecks_upgrade_aborts_only_old_hook(monkeypatch) -> None:
    from npa.soperator import lifecycle

    calls: list[list[str]] = []
    releases = {
        "items": [
            {
                "metadata": {
                    "name": "stack-soperator-activechecks",
                    "generation": 4,
                },
                "status": {
                    "lastAttemptedGeneration": 3,
                    "conditions": [
                        {
                            "type": "Reconciling",
                            "status": "True",
                            "reason": "Progressing",
                        }
                    ],
                },
            },
            {
                "metadata": {
                    "name": "current-soperator-activechecks",
                    "generation": 7,
                },
                "status": {
                    "lastAttemptedGeneration": 7,
                    "conditions": [
                        {
                            "type": "Reconciling",
                            "status": "True",
                            "reason": "Progressing",
                        }
                    ],
                },
            },
        ]
    }

    def fake_capture(cmd, **kwargs):
        calls.append(cmd)
        if "get" in cmd and "helmreleases" in cmd:
            return _Done(stdout=json.dumps(releases))
        return _Done(stdout="ok")

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)
    monkeypatch.setattr(lifecycle.time, "time_ns", lambda: 42)

    reset = lifecycle._abort_superseded_activechecks_upgrade("kubectl", "ctx")

    assert reset == ["stack-soperator-activechecks"]
    deletes = [cmd for cmd in calls if "delete" in cmd]
    assert len(deletes) == 1
    assert "wait-for-active-checks" in deletes[0]
    annotations = [cmd for cmd in calls if "annotate" in cmd]
    assert len(annotations) == 1
    assert "stack-soperator-activechecks" in annotations[0]
    assert "current-soperator-activechecks" not in annotations[0]


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


@pytest.mark.parametrize(
    "detail",
    [
        "failed to inspect monitoring HelmReleases: Forbidden",
        "failed to reset monitoring HelmRelease: transient server error",
    ],
)
def test_post_deploy_monitoring_repair_failure_is_best_effort(
    monkeypatch, detail: str
) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_install_monitoring_crds",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(detail)),
    )
    monkeypatch.setattr(lifecycle, "_patch_slurmcluster_crd", lambda *a, **k: True)
    monkeypatch.setattr(lifecycle, "_ensure_scripts_configmap", lambda *a, **k: True)
    monkeypatch.setattr(lifecycle, "_register_slurm_workers", lambda *a, **k: None)
    monkeypatch.setattr(
        lifecycle, "_abort_superseded_activechecks_upgrade", lambda *a, **k: []
    )
    messages: list[str] = []

    warnings = lifecycle.apply_post_deploy_fixes(
        "ctx", "kubectl", on_status=messages.append
    )

    assert warnings == [f"monitoring repair skipped after successful apply: {detail}"]
    assert any("post-deploy warning" in message for message in messages)


def test_post_deploy_monitoring_repair_clean_install_returns_no_warnings(
    monkeypatch,
) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(lifecycle, "_install_monitoring_crds", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_patch_slurmcluster_crd", lambda *a, **k: True)
    monkeypatch.setattr(lifecycle, "_ensure_scripts_configmap", lambda *a, **k: True)
    monkeypatch.setattr(lifecycle, "_register_slurm_workers", lambda *a, **k: None)
    monkeypatch.setattr(
        lifecycle, "_abort_superseded_activechecks_upgrade", lambda *a, **k: []
    )

    assert lifecycle.apply_post_deploy_fixes("ctx", "kubectl") == []


def test_slurmcluster_crd_patch_reports_kubernetes_write_failure(monkeypatch) -> None:
    from npa.soperator import lifecycle

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        if "get" in cmd:
            return _Done(stdout="slurmclusters.slurm.nebius.ai")
        assert "patch" in cmd
        return _Done(stderr="Forbidden", returncode=1)

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    assert lifecycle._patch_slurmcluster_crd("kubectl", "ctx") is False


def test_scripts_configmap_reports_kubernetes_write_failure(monkeypatch) -> None:
    from npa.soperator import lifecycle

    source = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "slurm-scripts"},
        "data": {"entrypoint.sh": "#!/bin/sh"},
    }

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        if "soperator-slurm-scripts" in cmd:
            return _Done(stderr="NotFound", returncode=1)
        return _Done(stdout=json.dumps(source))

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)
    monkeypatch.setattr(
        lifecycle.subprocess,
        "run",
        lambda *args, **kwargs: _Done(stderr="Forbidden", returncode=1),
    )

    assert lifecycle._ensure_scripts_configmap("kubectl", "ctx", "soperator") is False


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


def test_patch_nodeconfigurator_missing_own_values_refuses_sibling_chart(
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
    original = (
        "        fluxcd:\n"
        "          nodeConfigurator:\n"
        "            enabled: true\n"
        "          siblingChart:\n"
        "            enabled: true\n"
        "            values:\n"
        "              resources: {}\n"
    )
    template.write_text(original)

    with pytest.raises(lifecycle.UpstreamContractError, match="its own values"):
        lifecycle._patch_nodeconfigurator_userns(tmp_path)

    assert template.read_text() == original
    assert "privileged: true" not in template.read_text()


def test_contract_assertion_stops_before_installation_or_cloud_mutation(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from npa.soperator import lifecycle

    spec = SoperatorSpec(
        name="c",
        region="us-central1",
        tenant_id="tenant",
        project_id="project",
        workers=[WorkerPoolSpec(name="w")],
        root_login_ssh_public_key="ssh-ed25519 AAAA operator",
    )
    recipe = tmp_path / "soperator"
    (recipe / "installations" / "example").mkdir(parents=True)
    monkeypatch.setattr(lifecycle, "resolve_environment", lambda **k: SimpleNamespace(
        region="us-central1", tenant_id="tenant", project_id="project"
    ))
    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(lifecycle, "_resolve_solutions_library", lambda *a, **k: recipe)
    monkeypatch.setattr(
        lifecycle,
        "_assert_solutions_library_contract",
        lambda *a, **k: (_ for _ in ()).throw(
            lifecycle.UpstreamContractError("incompatible pinned contract")
        ),
    )
    prepared = False
    provider_read = False

    def prepare(*args, **kwargs):
        nonlocal prepared
        prepared = True

    def resolve_subnet(*args, **kwargs):
        nonlocal provider_read
        provider_read = True

    monkeypatch.setattr(lifecycle, "_prepare_installation", prepare)
    monkeypatch.setattr(lifecycle, "_resolve_subnet", resolve_subnet)

    with pytest.raises(lifecycle.UpstreamContractError, match="incompatible"):
        lifecycle.deploy_cluster(spec, work_root=tmp_path)

    assert prepared is False
    assert provider_read is False


def test_patch_nodeconfigurator_missing_template_is_safe(tmp_path) -> None:
    from npa.soperator import lifecycle

    assert lifecycle._patch_nodeconfigurator_userns(tmp_path) is False
