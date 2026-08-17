"""Unit tests for `npa soperator` deploy spec + tfvars rendering + CLI wiring.

These tests must not touch real infrastructure: they exercise pure spec/tfvars
logic and the Typer command surface (help + validation), mocking the terraform
lifecycle at the call site for the deploy path.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import textwrap
import threading

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
from npa.soperator.tfvars import _tfstr, render_tfvars

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
            {
                "name": "cpu",
                "platform": "cpu-d3",
                "preset": "8vcpu-32gb",
                "docker_cache": True,
            },
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
    # Rich truncates long option names to the available terminal column width.
    assert "--root-login-ss" in result.output
    assert DEFAULT_SOLUTIONS_LIBRARY_REF[:12] in result.output


@pytest.mark.parametrize(
    "message",
    [
        "Required executable not found: terraform",
        "Command failed (7): nebius project get: access_token=<redacted>",
    ],
)
def test_standalone_cli_backend_errors_are_clean_typer_failures(
    monkeypatch, message: str
) -> None:
    from npa.cluster_backends.process import BackendCommandError
    from npa.cluster_backends.soperator import SoperatorBackend

    def fail_status(*_args, **_kwargs):
        raise BackendCommandError(message)

    monkeypatch.setattr(SoperatorBackend, "status", fail_status)
    result = runner.invoke(
        app, ["soperator", "status", "--name", "c"], terminal_width=300
    )

    assert result.exit_code != 0
    assert "Soperator status failed" in result.output
    assert "Traceback" not in result.output
    assert "provider-secret" not in result.output


def test_cluster_status_redacts_provider_output(monkeypatch, tmp_path: Path) -> None:
    from npa.soperator import lifecycle

    kubectl = tmp_path / "kubectl"
    kubectl.write_text(
        "#!/bin/sh\nprintf 'secret_access_key: provider-secret\\n' >&2\nexit 9\n"
    )
    kubectl.chmod(0o700)
    monkeypatch.setenv("NPA_KUBECTL_BIN", str(kubectl))

    with pytest.raises(RuntimeError) as raised:
        lifecycle.cluster_status("c")
    assert "provider-secret" not in str(raised.value)
    assert "<redacted>" in str(raised.value)


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
    spec = SoperatorSpec(
        name="c", system_min_size=1, workers=[WorkerPoolSpec(name="w")]
    )
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

    with pytest.raises(
        SoperatorSpecError, match="controller skips REST reconciliation"
    ):
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
def test_sizing_tier_threshold_boundaries(
    worker_count: int, expected_tier: str
) -> None:
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


def test_system_max_size_preserves_upstream_autoscaling_and_validates_override() -> (
    None
):
    spec = SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="w")])
    assert spec.effective_system_max_size() == 24
    assert "max_size = 24" in render_tfvars(spec)
    spec.system_min_size = 25
    assert spec.effective_system_max_size() == 25
    assert "max_size = 25" in render_tfvars(spec)
    spec.system_max_size = 24
    with pytest.raises(SoperatorSpecError, match="max_size must be >= min_size"):
        spec.validate()


def test_resolve_login_ssh_key_from_operator_home(tmp_path) -> None:
    from npa.soperator.lifecycle import _resolve_root_login_ssh_public_key

    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_ed25519.pub").write_text("ssh-ed25519 AAAA operator\n")
    original = SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="w")])

    resolved = _resolve_root_login_ssh_public_key(original, home=tmp_path)

    assert resolved.value == "ssh-ed25519 AAAA operator"
    assert original.ssh_public_keys == []


def test_explicit_login_ssh_key_wins_and_missing_fallback_fails(tmp_path) -> None:
    from npa.soperator.lifecycle import _resolve_root_login_ssh_public_key

    explicit = SoperatorSpec(
        name="c",
        workers=[WorkerPoolSpec(name="w")],
        ssh_public_keys=["ssh-rsa AAAA explicit"],
    )
    resolved = _resolve_root_login_ssh_public_key(explicit, home=tmp_path)
    assert resolved.value == "ssh-rsa AAAA explicit"

    missing = SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="w")])
    with pytest.raises(ValueError, match="root access requires one SSH public key"):
        _resolve_root_login_ssh_public_key(missing, home=tmp_path)


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
    resolved = _resolve_root_login_ssh_public_key(
        empty_spec, environ=env, home=tmp_path
    )
    assert resolved.value.endswith("env-inline")
    assert resolved.source == "NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY"

    env.pop("NPA_SOPERATOR_ROOT_LOGIN_SSH_PUBLIC_KEY")
    resolved = _resolve_root_login_ssh_public_key(
        empty_spec, environ=env, home=tmp_path
    )
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


def test_tfstr_single_pass_preserves_literals_and_terraform_parses_value(
    tmp_path,
) -> None:
    value = (
        'mixed-$-${live}-$${literal}-%{if true}-%%{literal}-"quoted"-'
        "\\path-雪-ssh-comment"
    )
    rendered = _tfstr(value)
    assert "$${live}" in rendered
    assert "$${literal}" in rendered
    assert "$$${literal}" not in rendered
    assert "%%{if true}" in rendered
    assert "%%{literal}" in rendered
    assert "%%%{literal}" not in rendered

    terraform = shutil.which("terraform")
    if terraform is None:
        pytest.skip("terraform is required to validate generated HCL")
    parsed = subprocess.run(
        [terraform, f"-chdir={tmp_path}", "console"],
        input=f"jsonencode({rendered})\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert parsed.returncode == 0, parsed.stderr
    # terraform console returns a Terraform string containing JSON text.
    expected_value = value.replace("$${", "${").replace("%%{", "%{")
    assert json.loads(json.loads(parsed.stdout)) == expected_value


def test_root_login_ssh_key_hcl_rendering_escapes_comment_safely() -> None:
    key = 'ssh-ed25519 AAAA ci-${var.bad}-$${literal}-%{if true}-%%{literal}-"quoted"-\\path-雪'
    spec = SoperatorSpec(
        name="c",
        workers=[WorkerPoolSpec(name="w")],
        root_login_ssh_public_key=key,
    )
    spec.validate()
    rendered = render_tfvars(spec)
    assert "${var.bad}" not in rendered.replace("$${var.bad}", "")
    assert "%{if true}" not in rendered.replace("%%{if true}", "")
    assert "$${literal}" in rendered
    assert "$$${literal}" not in rendered
    assert "%%{literal}" in rendered
    assert "%%%{literal}" not in rendered


def test_operator_override_validation_and_verified_userns_contract() -> None:
    spec = SoperatorSpec(
        name="c",
        workers=[WorkerPoolSpec(name="w")],
        slurm_operator_version="latest",
    )
    with pytest.raises(SoperatorSpecError, match="semantic version"):
        spec.validate()

    spec.slurm_operator_version = "4.1.7"
    with pytest.raises(
        SoperatorSpecError, match="user-namespace override is verified only"
    ):
        spec.validate()

    spec.use_default_apparmor_profile = True
    spec.validate()  # explicit newer chart is possible when the pinned sysctl override is unused

    spec.slurm_operator_version = "4.2.0"
    with pytest.raises(SoperatorSpecError, match="use 4.1.6"):
        spec.validate()

    spec.slurm_operator_version = "4.1.0"
    with pytest.raises(
        SoperatorSpecError,
        match="4.1.0 is no longer verified; replace it with 4.1.6",
    ):
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


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def _legacy_shallow_checkout(tmp_path: Path) -> tuple[Path, str, Path, str]:
    """Create a moving-main shallow clone whose pinned commit is not local."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    remote = tmp_path / "remote"
    remote.mkdir()
    _git(remote, "init", "-b", "main")
    _git(remote, "config", "user.email", "tests@example.invalid")
    _git(remote, "config", "user.name", "NPA tests")
    example = remote / "soperator" / "installations" / "example"
    example.mkdir(parents=True)
    (example / "main.tf").write_text("# pinned\n")
    (remote / "README.md").write_text("pinned\n")
    (remote / "PINNED_ONLY.md").write_text("tracked at the pin\n")
    _git(remote, "add", ".")
    _git(remote, "commit", "-m", "pinned")
    pinned = _git(remote, "rev-parse", "HEAD").stdout.strip()
    (remote / "README.md").write_text("moving main\n")
    _git(remote, "rm", "PINNED_ONLY.md")
    _git(remote, "commit", "-am", "moving main")

    work_root = tmp_path / "work"
    work_root.mkdir()
    legacy = work_root / "nebius-solutions-library"
    subprocess.run(
        ["git", "clone", "--depth=1", f"file://{remote}", str(legacy)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    sentinel = legacy / "soperator" / "installations" / "live" / "terraform.tfstate"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b'{{"serial":7,"owner":"operator"}}\n')
    digest = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    missing = subprocess.run(
        ["git", "-C", str(legacy), "cat-file", "-e", f"{pinned}^{{commit}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert missing.returncode != 0
    return work_root, pinned, sentinel, digest


def test_solutions_library_reconciles_shallow_legacy_and_preserves_state(
    tmp_path,
) -> None:
    from npa.soperator.lifecycle import _resolve_solutions_library

    work_root, pinned, sentinel, digest = _legacy_shallow_checkout(tmp_path)
    recipe = _resolve_solutions_library(None, work_root, pinned)

    assert recipe == work_root / "nebius-solutions-library" / "soperator"
    assert _git(recipe.parent, "rev-parse", "HEAD").stdout.strip() == pinned
    detached = subprocess.run(
        ["git", "-C", str(recipe.parent), "symbolic-ref", "-q", "HEAD"],
        check=False,
    )
    assert detached.returncode != 0
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == digest


def test_solutions_library_detaches_attached_shallow_pin_and_preserves_state(
    tmp_path,
) -> None:
    from npa.soperator.lifecycle import _resolve_solutions_library

    work_root, _older_pin, sentinel, digest = _legacy_shallow_checkout(tmp_path)
    legacy = work_root / "nebius-solutions-library"
    attached_head = _git(legacy, "rev-parse", "HEAD").stdout.strip()
    assert (
        _git(legacy, "symbolic-ref", "-q", "HEAD").stdout.strip() == "refs/heads/main"
    )

    recipe = _resolve_solutions_library(None, work_root, attached_head)

    assert _git(recipe.parent, "rev-parse", "HEAD").stdout.strip() == attached_head
    detached = subprocess.run(
        ["git", "-C", str(recipe.parent), "symbolic-ref", "-q", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert detached.returncode != 0
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == digest


def test_solutions_library_cached_pin_is_idempotent_offline(tmp_path) -> None:
    from npa.soperator.lifecycle import _resolve_solutions_library

    work_root, pinned, sentinel, digest = _legacy_shallow_checkout(tmp_path)
    recipe = _resolve_solutions_library(None, work_root, pinned)
    _git(recipe.parent, "remote", "set-url", "origin", str(tmp_path / "offline"))

    assert _resolve_solutions_library(None, work_root, pinned) == recipe
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == digest


def test_solutions_library_detached_pin_cannot_race_to_another_ref(tmp_path) -> None:
    from npa.soperator.lifecycle import (
        SolutionsLibraryReconciliationError,
        _resolve_solutions_library,
    )

    work_root, pinned, sentinel, digest = _legacy_shallow_checkout(tmp_path)
    recipe = _resolve_solutions_library(None, work_root, pinned)
    moving_main = _git(
        recipe.parent, "ls-remote", "origin", "refs/heads/main"
    ).stdout.split()[0]

    with pytest.raises(SolutionsLibraryReconciliationError, match="already detached"):
        _resolve_solutions_library(None, work_root, moving_main)
    assert _git(recipe.parent, "rev-parse", "HEAD").stdout.strip() == pinned
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == digest


def test_solutions_library_missing_pin_offline_is_actionable_and_preserves_state(
    tmp_path,
) -> None:
    from npa.soperator.lifecycle import (
        SolutionsLibraryReconciliationError,
        _resolve_solutions_library,
    )

    work_root, pinned, sentinel, digest = _legacy_shallow_checkout(tmp_path)
    legacy = work_root / "nebius-solutions-library"
    _git(legacy, "remote", "set-url", "origin", str(tmp_path / "offline"))

    with pytest.raises(SolutionsLibraryReconciliationError, match="fetch origin"):
        _resolve_solutions_library(None, work_root, pinned)
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == digest
    assert _git(legacy, "rev-parse", "HEAD").stdout.strip() != pinned


def test_solutions_library_dirty_checkout_fails_without_touching_state(
    tmp_path,
) -> None:
    from npa.soperator.lifecycle import (
        SolutionsLibraryReconciliationError,
        _resolve_solutions_library,
    )

    work_root, pinned, sentinel, digest = _legacy_shallow_checkout(tmp_path)
    legacy = work_root / "nebius-solutions-library"
    (legacy / "README.md").write_text("operator edit\n")

    with pytest.raises(SolutionsLibraryReconciliationError, match="tracked changes"):
        _resolve_solutions_library(None, work_root, pinned)
    assert (legacy / "README.md").read_text() == "operator edit\n"
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == digest


def test_solutions_library_untracked_checkout_conflict_preserves_state(
    tmp_path,
) -> None:
    from npa.soperator.lifecycle import (
        SolutionsLibraryReconciliationError,
        _resolve_solutions_library,
    )

    work_root, pinned, sentinel, digest = _legacy_shallow_checkout(tmp_path)
    legacy = work_root / "nebius-solutions-library"
    conflict = legacy / "PINNED_ONLY.md"
    conflict.write_text("operator-owned untracked bytes\n")
    head_before = _git(legacy, "rev-parse", "HEAD").stdout.strip()

    with pytest.raises(
        SolutionsLibraryReconciliationError, match="untracked file conflicts"
    ):
        _resolve_solutions_library(None, work_root, pinned)

    assert _git(legacy, "rev-parse", "HEAD").stdout.strip() == head_before
    assert conflict.read_text() == "operator-owned untracked bytes\n"
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == digest


def test_solutions_library_concurrent_reconciliation_is_serial_and_idempotent(
    tmp_path,
) -> None:
    from npa.soperator.lifecycle import _resolve_solutions_library

    work_root, pinned, sentinel, digest = _legacy_shallow_checkout(tmp_path)
    barrier = threading.Barrier(4)
    recipes: list[Path] = []
    errors: list[BaseException] = []

    def resolve() -> None:
        try:
            barrier.wait()
            recipes.append(_resolve_solutions_library(None, work_root, pinned))
        except BaseException as exc:  # surfaced by the assertions below
            errors.append(exc)

    threads = [threading.Thread(target=resolve) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert len(recipes) == 4
    assert len(set(recipes)) == 1
    assert _git(recipes[0].parent, "rev-parse", "HEAD").stdout.strip() == pinned
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == digest


def test_fresh_solutions_library_clone_is_published_atomically(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    legacy_root, pinned, _sentinel, _digest = _legacy_shallow_checkout(
        tmp_path / "source"
    )
    origin = _git(
        legacy_root / "nebius-solutions-library", "remote", "get-url", "origin"
    ).stdout.strip()
    fresh_root = tmp_path / "fresh"
    monkeypatch.setattr(lifecycle, "_SOLUTIONS_LIBRARY_REPO", origin)

    recipe = lifecycle._resolve_solutions_library(None, fresh_root, pinned)

    assert (
        recipe == fresh_root / f"nebius-solutions-library-{pinned[:12]}" / "soperator"
    )
    assert _git(recipe.parent, "rev-parse", "HEAD").stdout.strip() == pinned
    assert not list(fresh_root.glob(".*.clone-*"))


def test_deploy_path_reconciles_legacy_source_before_provider_mutation(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from npa.soperator import lifecycle

    work_root, pinned, sentinel, digest = _legacy_shallow_checkout(tmp_path)
    seen: dict[str, Path] = {}

    def assert_contract(recipe: Path, *, ref: str) -> None:
        seen["recipe"] = recipe
        assert ref == pinned

    monkeypatch.setattr(
        lifecycle,
        "resolve_environment",
        lambda **kwargs: SimpleNamespace(
            region="us-central1", tenant_id="tenant", project_id="project"
        ),
    )
    required_bins: list[str] = []
    monkeypatch.setattr(
        lifecycle,
        "_require_bin",
        lambda name: required_bins.append(name) or name,
    )
    monkeypatch.setattr(
        lifecycle, "_assert_solutions_library_contract", assert_contract
    )
    monkeypatch.setattr(
        lifecycle,
        "_prepare_installation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("provider boundary crossed")
        ),
    )
    spec = SoperatorSpec(
        name="cluster",
        region="us-central1",
        tenant_id="tenant",
        project_id="project",
        root_login_ssh_public_key="ssh-ed25519 AAAA operator",
        workers=[WorkerPoolSpec(name="cpu")],
    )

    result = lifecycle.deploy_cluster(
        spec,
        work_root=work_root,
        solutions_library_ref=pinned,
        source_preflight_only=True,
    )

    assert seen["recipe"] == work_root / "nebius-solutions-library" / "soperator"
    assert _git(seen["recipe"].parent, "rev-parse", "HEAD").stdout.strip() == pinned
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == digest
    assert required_bins == ["git"]
    assert result["status"] == "source-preflight-passed"
    assert result["provider_mutation"] is False
    assert result["control_plane"]["system_max_size"] == 24


@pytest.mark.parametrize(
    "capture_failure", [None, "missing-id", "missing-name", "unreadable-auxiliary"]
)
def test_deploy_persists_exact_auxiliary_ids_before_reporting_success(
    tmp_path, monkeypatch, capture_failure
) -> None:
    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    install = recipe / "installations" / "owned"
    install.mkdir(parents=True)
    spec = SoperatorSpec(
        name="owned",
        region="us-central1",
        tenant_id="tenant-test",
        project_id="project-test",
        subnet_id="subnet-test",
        root_login_ssh_public_key="ssh-ed25519 AAAA operator",
        workers=[WorkerPoolSpec(name="cpu")],
    )
    monkeypatch.setattr(lifecycle, "_resolve_solutions_library", lambda *a, **k: recipe)
    monkeypatch.setattr(
        lifecycle, "_assert_solutions_library_contract", lambda *a, **k: None
    )
    monkeypatch.setattr(
        lifecycle,
        "_resolve_deploy_environment",
        lambda *a, **k: (
            "us-central1",
            "tenant-test",
            "project-test",
            "subnet-test",
            None,
        ),
    )
    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(
        lifecycle,
        "_resolve_reserved_worker_capacity",
        lambda desired, **kwargs: (desired, []),
    )
    monkeypatch.setattr(lifecycle, "_prepare_installation", lambda *a, **k: install)
    monkeypatch.setattr(
        lifecycle,
        "_soperator_tf_env",
        lambda *a, **k: {"TF_VAR_o11y_profile": "default"},
    )
    monkeypatch.setattr(lifecycle, "_run_terraform_command", lambda *a, **k: None)

    @contextmanager
    def safe_plan(*args, **kwargs):
        yield lifecycle.GuardedTerraformPlan(path=tmp_path / "plan")

    monkeypatch.setattr(
        lifecycle, "_terraform_plan_without_unsafe_replacements", safe_plan
    )
    monkeypatch.setattr(
        lifecycle,
        "_terraform_cluster_id",
        lambda *a, **k: "" if capture_failure == "missing-id" else "mk8scluster-owned",
    )
    monkeypatch.setattr(
        lifecycle,
        "_terraform_cluster_name",
        lambda *a, **k: "" if capture_failure == "missing-name" else "soperator-owned",
    )

    ownership_records = [
        {
            "kind": "filesystem",
            "provider_type": "nebius_compute_v1_filesystem",
            "id": "fs-jail",
            "name": "soperator-owned-jail",
        },
        {
            "kind": "filesystem",
            "provider_type": "nebius_compute_v1_filesystem",
            "id": "fs-spool",
            "name": "soperator-owned-controller-spool",
        },
        {
            "kind": "allocation",
            "provider_type": "nebius_vpc_v1_allocation",
            "id": "alloc-login",
            "name": "soperator-owned-public-static-ip",
        },
    ]

    def capture_ownership(*_args, **_kwargs):
        if capture_failure == "unreadable-auxiliary":
            raise RuntimeError("applied Terraform state could not be read")
        return ownership_records

    monkeypatch.setattr(
        lifecycle,
        "_terraform_owned_auxiliary_resources",
        capture_ownership,
    )

    if capture_failure:
        with pytest.raises(lifecycle.SoperatorStateCaptureError) as caught:
            lifecycle.deploy_cluster(spec, terraform_dir=recipe, apply_fixes=False)
        assert caught.value.result["deployment_status"] == "applied"
        assert caught.value.result["status"] == "deployed-state-capture-failed"
        retained = lifecycle._load_env_sidecar(install)
        assert retained is not None
        assert retained["cluster_id"] == ""

        # The same deploy is the recovery path: once state is readable, it
        # atomically promotes the retained pre-apply sidecar without cleanup.
        monkeypatch.setattr(
            lifecycle, "_terraform_cluster_id", lambda *a, **k: "mk8scluster-owned"
        )
        monkeypatch.setattr(
            lifecycle, "_terraform_cluster_name", lambda *a, **k: "soperator-owned"
        )
        monkeypatch.setattr(
            lifecycle,
            "_terraform_owned_auxiliary_resources",
            lambda *a, **k: ownership_records,
        )

    result = lifecycle.deploy_cluster(spec, terraform_dir=recipe, apply_fixes=False)

    assert result["status"] == "ready"
    sidecar = lifecycle._load_env_sidecar(install)
    assert sidecar is not None
    assert sidecar["cluster_id"] == "mk8scluster-owned"
    assert sidecar["provider_cluster_name"] == "soperator-owned"
    assert sidecar["owned_filesystem_ids"] == ["fs-jail", "fs-spool"]
    assert sidecar["owned_allocation_ids"] == ["alloc-login"]


def test_terraform_auxiliary_ownership_uses_only_exact_managed_ids(
    tmp_path, monkeypatch
) -> None:
    import subprocess

    from npa.soperator import lifecycle

    state = {
        "resources": [
            {
                "mode": "managed",
                "type": "nebius_compute_v1_filesystem",
                "instances": [
                    {"attributes": {"id": "fs-exact", "name": "soperator-c-jail"}}
                ],
            },
            {
                "mode": "managed",
                "type": "nebius_vpc_v1_allocation",
                "instances": [
                    {
                        "attributes": {
                            "id": "alloc-exact",
                            "name": "soperator-c-public-static-ip",
                        }
                    }
                ],
            },
            {
                "mode": "data",
                "type": "nebius_compute_v1_filesystem",
                "instances": [{"attributes": {"id": "fs-foreign", "name": "foreign"}}],
            },
        ]
    }
    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, json.dumps(state), ""),
    )

    assert lifecycle._terraform_owned_auxiliary_ids("terraform", tmp_path, {}) == (
        ["fs-exact"],
        ["alloc-exact"],
    )


def test_phase_one_failure_retains_typed_auxiliary_and_provider_name(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    install = recipe / "installations" / "owned"
    install.mkdir(parents=True)
    prior_records = [
        {
            "kind": "allocation",
            "provider_type": "nebius_vpc_v1_allocation",
            "id": "alloc-old",
            "name": "soperator-owned-public-static-ip",
        }
    ]
    lifecycle._write_env_sidecar(
        install,
        region="us-central1",
        tenant_id="tenant-test",
        project_id="project-test",
        subnet_id="subnet-test",
        o11y_profile="default",
        cluster_id="mk8scluster-old",
        cluster_name="owned",
        provider_cluster_name="soperator-owned",
        owned_allocation_ids=["alloc-old"],
        owned_auxiliary_resources=prior_records,
    )
    spec = SoperatorSpec(
        name="owned",
        region="us-central1",
        tenant_id="tenant-test",
        project_id="project-test",
        subnet_id="subnet-test",
        root_login_ssh_public_key="ssh-ed25519 AAAA operator",
        workers=[WorkerPoolSpec(name="cpu")],
    )
    monkeypatch.setattr(lifecycle, "_resolve_solutions_library", lambda *a, **k: recipe)
    monkeypatch.setattr(
        lifecycle, "_assert_solutions_library_contract", lambda *a, **k: None
    )
    monkeypatch.setattr(
        lifecycle,
        "_resolve_deploy_environment",
        lambda *a, **k: (
            "us-central1",
            "tenant-test",
            "project-test",
            "subnet-test",
            lifecycle._load_env_sidecar(install),
        ),
    )
    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(
        lifecycle,
        "_resolve_reserved_worker_capacity",
        lambda desired, **kwargs: (desired, []),
    )
    monkeypatch.setattr(lifecycle, "_prepare_installation", lambda *a, **k: install)
    monkeypatch.setattr(
        lifecycle,
        "_soperator_tf_env",
        lambda *a, **k: {"TF_VAR_o11y_profile": "default"},
    )
    monkeypatch.setattr(lifecycle, "_run_terraform_command", lambda *a, **k: None)

    @contextmanager
    def safe_plan(*args, **kwargs):
        yield lifecycle.GuardedTerraformPlan(path=tmp_path / "plan")

    monkeypatch.setattr(
        lifecycle, "_terraform_plan_without_unsafe_replacements", safe_plan
    )
    monkeypatch.setattr(
        lifecycle, "_terraform_cluster_id", lambda *a, **k: "mk8scluster-new"
    )
    monkeypatch.setattr(
        lifecycle, "_terraform_cluster_name", lambda *a, **k: "soperator-owned"
    )
    monkeypatch.setattr(lifecycle, "_refresh_kube_credentials", lambda *a, **k: None)
    monkeypatch.setattr(
        lifecycle,
        "_install_monitoring_crds",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("phase one stopped")),
    )

    with pytest.raises(RuntimeError, match="phase one stopped"):
        lifecycle.deploy_cluster(spec, terraform_dir=recipe, apply_fixes=True)

    retained = lifecycle._load_env_sidecar(install)
    assert retained is not None
    assert retained["cluster_id"] == "mk8scluster-new"
    assert retained["cluster_name"] == "owned"
    assert retained["provider_cluster_name"] == "soperator-owned"
    assert retained["owned_allocation_ids"] == ["alloc-old"]
    assert retained["owned_auxiliary_resources"] == prior_records


@pytest.mark.parametrize(
    "state",
    [
        {"resources": "invalid"},
        {
            "resources": [
                {
                    "type": "nebius_compute_v1_filesystem",
                    "instances": [{"attributes": {"id": ""}}],
                }
            ]
        },
    ],
)
def test_terraform_auxiliary_ownership_rejects_malformed_state_or_missing_ids(
    tmp_path, monkeypatch, state
) -> None:
    import subprocess

    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, json.dumps(state), ""),
    )

    with pytest.raises(RuntimeError, match="exact auxiliary ownership|exact ownership"):
        lifecycle._terraform_owned_auxiliary_ids("terraform", tmp_path, {})


def test_soperator_lifecycle_has_no_cluster_cli_private_dependency() -> None:
    from npa.soperator import lifecycle

    source = Path(lifecycle.__file__).read_text()
    assert "npa.cli.cluster.terraform_lifecycle" not in source
    assert lifecycle._run_capture.__module__ == "npa.cluster_backends.process"


def test_success_ownership_promotion_requires_exact_cluster_id(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(lifecycle, "_terraform_cluster_id", lambda *a, **k: "")
    with pytest.raises(RuntimeError, match="no exact Managed Kubernetes cluster ID"):
        lifecycle._require_applied_cluster_id("terraform", tmp_path, {})


def test_destroy_source_preflight_reconciles_without_provider_calls(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    work_root, pinned, sentinel, digest = _legacy_shallow_checkout(tmp_path)
    install = (
        work_root / "nebius-solutions-library" / "soperator" / "installations" / "live"
    )
    monkeypatch.setattr(
        lifecycle, "_assert_solutions_library_contract", lambda *a, **k: None
    )
    required_bins: list[str] = []
    monkeypatch.setattr(
        lifecycle,
        "_require_bin",
        lambda name: required_bins.append(name) or name,
    )

    result = lifecycle.destroy_cluster(
        "live",
        work_root=work_root,
        solutions_library_ref=pinned,
        source_preflight_only=True,
    )

    assert result == {
        "name": "live",
        "status": "source-preflight-passed",
        "install_dir": str(install),
        "solutions_library_ref": pinned,
        "provider_mutation": False,
    }
    assert required_bins == ["git"]
    assert _git(install.parents[2], "rev-parse", "HEAD").stdout.strip() == pinned
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == digest


def test_existing_deploy_uses_persisted_identity_not_changed_default(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    install = recipe / "installations" / "live"
    install.mkdir(parents=True)
    lifecycle._write_env_sidecar(
        install,
        region="us-central1",
        tenant_id="tenant-live",
        project_id="project-live",
        subnet_id="subnet-live",
        o11y_profile="live",
    )
    monkeypatch.setattr(
        lifecycle,
        "resolve_environment",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("ambient project resolution must not run")
        ),
    )
    spec = SoperatorSpec(
        name="live",
        region="us-central1",
        root_login_ssh_public_key="ssh-ed25519 AAAA operator",
        workers=[WorkerPoolSpec(name="cpu")],
    )

    resolved = lifecycle._resolve_deploy_environment(spec, recipe, project=None)

    assert resolved[:4] == (
        "us-central1",
        "tenant-live",
        "project-live",
        "subnet-live",
    )
    assert resolved[4] == json.loads((install / ".npa-soperator-env.json").read_text())
    assert (install / ".npa-soperator-env.json").stat().st_mode & 0o777 == 0o600
    assert not list(install.glob("..npa-soperator-env.json.*.tmp"))


@pytest.mark.parametrize("payload", ["not-json", "[]", '{"region":"us-central1"}'])
def test_existing_deploy_fails_closed_on_invalid_persisted_identity(
    tmp_path, monkeypatch, payload: str
) -> None:
    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    install = recipe / "installations" / "live"
    install.mkdir(parents=True)
    sidecar = install / ".npa-soperator-env.json"
    sidecar.write_text(payload)
    monkeypatch.setattr(
        lifecycle,
        "resolve_environment",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("ambient project resolution must not run")
        ),
    )
    spec = SoperatorSpec(
        name="live",
        region="us-central1",
        root_login_ssh_public_key="ssh-ed25519 AAAA operator",
        workers=[WorkerPoolSpec(name="cpu")],
    )

    with pytest.raises(ValueError, match="persisted.*identity|persisted identity"):
        lifecycle._resolve_deploy_environment(spec, recipe, project=None)

    assert sidecar.read_text() == payload


def test_env_sidecar_atomic_failure_preserves_previous_bytes(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    install = tmp_path / "install"
    install.mkdir()
    sidecar = install / ".npa-soperator-env.json"
    original = b'{"project_id":"original"}\n'
    sidecar.write_bytes(original)
    monkeypatch.setattr(
        lifecycle.os,
        "replace",
        lambda *args: (_ for _ in ()).throw(OSError("simulated replace failure")),
    )

    with pytest.raises(OSError, match="simulated replace failure"):
        lifecycle._write_env_sidecar(
            install,
            region="us-central1",
            tenant_id="tenant",
            project_id="project",
            subnet_id="subnet",
            o11y_profile="profile",
        )

    assert sidecar.read_bytes() == original
    assert not list(install.glob("..npa-soperator-env.json.*.tmp"))


def test_terraform_replacement_guard_blocks_plan_and_removes_private_plan(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    def fake_capture(command, **kwargs):
        if "plan" in command:
            plan_path = Path(
                next(
                    arg.removeprefix("-out=")
                    for arg in command
                    if arg.startswith("-out=")
                )
            )
            plan_path.write_bytes(b"private plan")
            return _Done(returncode=2)
        if "show" in command:
            return _Done(
                stdout=json.dumps(
                    {
                        "resource_changes": [
                            {
                                "address": "module.k8s.nebius_mk8s_v1_cluster.this",
                                "change": {"actions": ["delete", "create"]},
                            }
                        ]
                    }
                )
            )
        raise AssertionError(command)

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    with pytest.raises(
        lifecycle.ProviderReplacementPlanError,
        match="1 provider or unexpected destructive action.*refusing to apply",
    ) as caught:
        with lifecycle._terraform_plan_without_unsafe_replacements(
            "terraform", cwd=tmp_path, env={}, timeout=30
        ):
            raise AssertionError("replacement plan must never be yielded")

    assert caught.value.replacements == ["module.k8s.nebius_mk8s_v1_cluster.this"]
    assert not list(tmp_path.glob(".npa-plan-*"))


def test_terraform_replacement_guard_yields_exact_owner_only_saved_plan(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    calls: list[list[str]] = []

    def fake_capture(command, **kwargs):
        calls.append(command)
        if "plan" in command:
            plan_path = Path(
                next(
                    arg.removeprefix("-out=")
                    for arg in command
                    if arg.startswith("-out=")
                )
            )
            plan_path.write_bytes(b"private plan")
            return _Done(returncode=2)
        if "show" in command:
            return _Done(
                stdout=json.dumps(
                    {
                        "resource_changes": [
                            {
                                "address": "module.k8s.safe_update",
                                "change": {"actions": ["update"]},
                            }
                        ]
                    }
                )
            )
        raise AssertionError(command)

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    with lifecycle._terraform_plan_without_unsafe_replacements(
        "terraform",
        cwd=tmp_path,
        env={},
        timeout=30,
        targets=("module.k8s",),
    ) as guarded_plan:
        assert guarded_plan.path.is_file()
        assert guarded_plan.path.stat().st_mode & 0o777 == 0o600
        assert guarded_plan.safe_local_replacements == ()
        assert "-target=module.k8s" in calls[0]
        yielded_path = guarded_plan.path

    assert not yielded_path.exists()
    assert not list(tmp_path.glob(".npa-plan-*"))


def test_terraform_replacement_guard_allows_only_audited_local_refreshes(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    resource_changes = [
        {
            "address": address,
            "provider_name": provider,
            "change": {"actions": ["delete", "create"]},
        }
        for address, provider in lifecycle._SAFE_LOCAL_RECONCILIATION_REPLACEMENTS
    ]

    def fake_capture(command, **kwargs):
        if "plan" in command:
            plan_path = Path(
                next(
                    arg.removeprefix("-out=")
                    for arg in command
                    if arg.startswith("-out=")
                )
            )
            plan_path.write_bytes(b"private plan")
            return _Done(returncode=2)
        if "show" in command:
            return _Done(stdout=json.dumps({"resource_changes": resource_changes}))
        raise AssertionError(command)

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    with lifecycle._terraform_plan_without_unsafe_replacements(
        "terraform", cwd=tmp_path, env={}, timeout=30
    ) as guarded_plan:
        assert guarded_plan.safe_local_replacements == tuple(
            sorted(
                address
                for address, _provider in lifecycle._SAFE_LOCAL_RECONCILIATION_REPLACEMENTS
            )
        )


@pytest.mark.parametrize(
    ("address", "provider", "actions"),
    [
        (
            "module.k8s.terraform_data.kubectl_cluster_context",
            "registry.terraform.io/hashicorp/local",
            ["delete", "create"],
        ),
        (
            "module.login_script.local_file.this",
            "registry.terraform.io/hashicorp/local",
            ["delete"],
        ),
        (
            "module.k8s.nebius_mk8s_v1_cluster.this",
            "terraform-provider-nebius.storage.nebius.cloud/nebius/nebius",
            ["delete", "create"],
        ),
    ],
)
def test_terraform_replacement_guard_blocks_wrong_provider_or_delete(
    tmp_path, monkeypatch, address: str, provider: str, actions: list[str]
) -> None:
    from npa.soperator import lifecycle

    def fake_capture(command, **kwargs):
        if "plan" in command:
            plan_path = Path(
                next(
                    arg.removeprefix("-out=")
                    for arg in command
                    if arg.startswith("-out=")
                )
            )
            plan_path.write_bytes(b"private plan")
            return _Done(returncode=2)
        if "show" in command:
            return _Done(
                stdout=json.dumps(
                    {
                        "resource_changes": [
                            {
                                "address": address,
                                "provider_name": provider,
                                "change": {"actions": actions},
                            }
                        ]
                    }
                )
            )
        raise AssertionError(command)

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    with pytest.raises(lifecycle.ProviderReplacementPlanError) as caught:
        with lifecycle._terraform_plan_without_unsafe_replacements(
            "terraform", cwd=tmp_path, env={}, timeout=30
        ):
            raise AssertionError("unsafe plan must never be yielded")

    assert caught.value.replacements == [address]
    assert not list(tmp_path.glob(".npa-plan-*"))


def test_replacement_guard_stops_before_sidecar_overwrite_or_apply(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    (recipe / "installations" / "example").mkdir(parents=True)
    install = recipe / "installations" / "live"
    install.mkdir()
    lifecycle._write_env_sidecar(
        install,
        region="us-central1",
        tenant_id="tenant",
        project_id="project",
        subnet_id="subnet",
        o11y_profile="profile",
    )
    sidecar = install / ".npa-soperator-env.json"
    original = sidecar.read_bytes()
    commands: list[list[str]] = []
    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(
        lifecycle, "_assert_solutions_library_contract", lambda *a, **k: None
    )
    monkeypatch.setattr(lifecycle, "_prepare_installation", lambda *a, **k: install)
    monkeypatch.setattr(lifecycle, "_resolve_subnet", lambda *a, **k: "subnet")
    monkeypatch.setattr(
        lifecycle,
        "_soperator_tf_env",
        lambda *a, **k: {"TF_VAR_o11y_profile": "profile"},
    )
    monkeypatch.setattr(
        lifecycle,
        "_run_terraform_command",
        lambda command, **kwargs: commands.append(command),
    )

    @contextmanager
    def reject_plan(*args, **kwargs):
        raise lifecycle.ProviderReplacementPlanError(["module.k8s.cluster"])
        yield Path("unreachable")

    monkeypatch.setattr(
        lifecycle, "_terraform_plan_without_unsafe_replacements", reject_plan
    )
    spec = SoperatorSpec(
        name="live",
        region="us-central1",
        root_login_ssh_public_key="ssh-ed25519 AAAA operator",
        workers=[WorkerPoolSpec(name="cpu")],
    )

    with pytest.raises(lifecycle.ProviderReplacementPlanError):
        lifecycle.deploy_cluster(spec, terraform_dir=recipe, apply_fixes=False)

    assert commands == [["terraform", "init"]]
    assert sidecar.read_bytes() == original


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", "project-other"),
        ("tenant_id", "tenant-other"),
        ("subnet_id", "subnet-other"),
    ],
)
def test_existing_deploy_rejects_provider_identity_replacement(
    tmp_path, monkeypatch, field: str, value: str
) -> None:
    from types import SimpleNamespace

    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    install = recipe / "installations" / "live"
    install.mkdir(parents=True)
    lifecycle._write_env_sidecar(
        install,
        region="us-central1",
        tenant_id="tenant-live",
        project_id="project-live",
        subnet_id="subnet-live",
        o11y_profile="live",
    )
    monkeypatch.setattr(
        lifecycle,
        "resolve_environment",
        lambda **kwargs: SimpleNamespace(
            region="us-central1",
            tenant_id="tenant-live",
            project_id="project-live",
        ),
    )
    kwargs = {field: value}
    spec = SoperatorSpec(
        name="live",
        region="us-central1",
        root_login_ssh_public_key="ssh-ed25519 AAAA operator",
        workers=[WorkerPoolSpec(name="cpu")],
        **kwargs,
    )

    with pytest.raises(ValueError, match="refusing a provider-replacement deploy"):
        lifecycle._resolve_deploy_environment(spec, recipe, project=None)


def test_existing_deploy_rejects_mismatched_explicit_project_alias(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    install = recipe / "installations" / "live"
    install.mkdir(parents=True)
    lifecycle._write_env_sidecar(
        install,
        region="us-central1",
        tenant_id="tenant-live",
        project_id="project-live",
        subnet_id="subnet-live",
        o11y_profile="live",
    )
    monkeypatch.setattr(
        lifecycle,
        "resolve_environment",
        lambda **kwargs: SimpleNamespace(
            region="us-central1",
            tenant_id="tenant-other",
            project_id="project-other",
        ),
    )
    spec = SoperatorSpec(
        name="live",
        region="us-central1",
        root_login_ssh_public_key="ssh-ed25519 AAAA operator",
        workers=[WorkerPoolSpec(name="cpu")],
    )

    with pytest.raises(ValueError, match="does not match the persisted"):
        lifecycle._resolve_deploy_environment(spec, recipe, project="other")


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
        cluster_id="mk8scluster-exact",
        provider_cluster_name="soperator-npatest",
        owned_allocation_ids=["allocation-exact"],
    )

    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(
        lifecycle, "_assert_solutions_library_contract", lambda *a, **k: None
    )
    captured: dict[str, object] = {}
    command_timeouts: list[float | int | None] = []

    # _soperator_tf_env -> _terraform_env mints a real IAM token via the `nebius`
    # CLI; stub it so the destroy tests never touch real infra (CI has no nebius).
    def fake_terraform_env(_nebius_bin, **kwargs):
        command_timeouts.append(kwargs.get("timeout"))
        return {}

    monkeypatch.setattr(lifecycle, "_terraform_env", fake_terraform_env)
    deadlines: list[float] = []

    class _Done:
        def __init__(self, stdout: str = "", returncode: int = 0) -> None:
            self.stdout = stdout
            self.returncode = returncode

    def fake_stream(cmd, *, cwd=None, env=None, timeout=None):
        command_timeouts.append(timeout)
        return None  # terraform init

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        command_timeouts.append(timeout)
        # The hardened destroy runs `terraform destroy` via _run_capture; record
        # its env. state pull -> empty (no cluster id); filesystem list -> none.
        if "destroy" in cmd:
            captured["env"] = dict(env or {})
        if "mk8s" in cmd and "get" in cmd:
            return _Done(stdout="not found", returncode=1)
        return _Done(stdout="")

    monkeypatch.setattr(lifecycle, "_run_stream", fake_stream)
    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        lifecycle,
        "_resolve_exact_cluster_presence",
        lambda **kwargs: deadlines.append(kwargs["deadline"]) or (False, ""),
    )
    monkeypatch.setattr(
        lifecycle,
        "_cleanup_owned_provider_ids",
        lambda **kwargs: deadlines.append(kwargs["deadline"]) or [],
    )

    def fail_resolve(*args, **kwargs):  # sidecar present -> must not be called
        raise AssertionError("destroy fell back to resolve despite a sidecar")

    monkeypatch.setattr(lifecycle, "_resolve_subnet", fail_resolve)

    lifecycle.destroy_cluster("npatest", terraform_dir=recipe, timeout_minutes=1)

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["TF_VAR_region"] == "us-central1"
    assert env["TF_VAR_iam_tenant_id"] == "tenant-abc"
    assert env["TF_VAR_iam_project_id"] == "project-xyz"
    assert env["TF_VAR_vpc_subnet_id"] == "vpcsubnet-123"
    assert env["TF_VAR_o11y_iam_tenant_id"] == "tenant-abc"
    assert env["TF_VAR_o11y_profile"] == "npa-mk8s"
    assert len(deadlines) == 3
    assert len(set(deadlines)) == 1
    assert command_timeouts
    assert None not in command_timeouts, command_timeouts
    assert all(0 < timeout <= 60 for timeout in command_timeouts if timeout is not None)


def test_destroy_deletes_orphaned_vpc_allocation(tmp_path, monkeypatch) -> None:
    """An old sidecar retries exact-ID cleanup after cluster NotFound."""

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
        cluster_id="mk8scluster-exact",
        owned_allocation_ids=["alloc-orphan"],
    )

    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(
        lifecycle, "_assert_solutions_library_contract", lambda *a, **k: None
    )
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
                {
                    "metadata": {
                        "id": "alloc-orphan",
                        "name": "soperator-npasop-public-static-ip",
                    }
                },
                {"metadata": {"id": "alloc-other", "name": "mk8snodegroup-abc-alias"}},
            ]
        }
    )

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        if "mk8s" in cmd and "get" in cmd:
            return _Done(stdout="not found", returncode=1)
        if "vpc" in cmd and "list" in cmd:
            return _Done(stdout=alloc_json)
        if "vpc" in cmd and "delete" in cmd:
            deleted.append(cmd[cmd.index("--id") + 1])
            return _Done()
        if "vpc" in cmd and "get" in cmd:
            return _Done(stdout="not found", returncode=1)
        return _Done(stdout="")

    monkeypatch.setattr(lifecycle, "_run_stream", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)
    monkeypatch.setattr(lifecycle, "_resolve_subnet", lambda *a, **k: "vpcsubnet-123")

    lifecycle.destroy_cluster("npasop", terraform_dir=recipe)

    # Only the persisted exact allocation is deleted; unrelated inventory is left.
    assert deleted == ["alloc-orphan"]


def test_reconciles_ccm_recreated_allocation_by_typed_exact_name(
    monkeypatch,
) -> None:
    import subprocess
    from npa.soperator import lifecycle

    deleted = []
    get_calls = 0

    def capture(command, **_kwargs):
        nonlocal get_calls
        if "list" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {
                                    "id": "alloc-recreated",
                                    "name": "soperator-npasop-public-static-ip",
                                    "parent_id": "project-test",
                                }
                            },
                            {
                                "metadata": {
                                    "id": "alloc-unrelated",
                                    "name": "soperator-other-public-static-ip",
                                    "parent_id": "project-test",
                                }
                            },
                        ]
                    }
                ),
                "",
            )
        if "delete" in command:
            deleted.append(command[command.index("--id") + 1])
            return subprocess.CompletedProcess(command, 0, "", "")
        if "get" in command:
            get_calls += 1
            if get_calls == 1:
                return subprocess.CompletedProcess(
                    command, 2, "", "temporarily unavailable"
                )
            return subprocess.CompletedProcess(command, 1, "", "not found")
        raise AssertionError(command)

    monkeypatch.setattr(lifecycle, "_run_capture", capture)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)
    errors = lifecycle._reconcile_recreated_auxiliary_resources(
        nebius_bin="nebius",
        project_id="project-test",
        cluster_name="npasop",
        env={},
        records=[
            {
                "kind": "allocation",
                "provider_type": "nebius_vpc_v1_allocation",
                "id": "alloc-original",
                "name": "soperator-npasop-public-static-ip",
            }
        ],
        deadline=lifecycle.time.monotonic() + 60,
        on_status=None,
    )
    assert errors == []
    assert deleted == ["alloc-recreated"]
    assert get_calls == 2


def test_recreated_allocation_ambiguity_fails_closed(monkeypatch) -> None:
    import subprocess
    from npa.soperator import lifecycle

    payload = {
        "items": [
            {
                "metadata": {
                    "id": candidate,
                    "name": "soperator-npasop-public-static-ip",
                    "parent_id": "project-test",
                }
            }
            for candidate in ("alloc-a", "alloc-b")
        ]
    }
    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(payload), ""
        ),
    )
    errors = lifecycle._reconcile_recreated_auxiliary_resources(
        nebius_bin="nebius",
        project_id="project-test",
        cluster_name="npasop",
        env={},
        records=[
            {
                "kind": "allocation",
                "provider_type": "nebius_vpc_v1_allocation",
                "id": "alloc-original",
                "name": "soperator-npasop-public-static-ip",
            }
        ],
        deadline=lifecycle.time.monotonic() + 60,
        on_status=None,
    )
    assert errors == ["allocation exact-name ownership evidence is ambiguous"]


def test_exact_cluster_presence_retries_transient_read_then_proves_absence(
    monkeypatch,
) -> None:
    import subprocess
    from npa.soperator import lifecycle

    responses = iter(
        [
            subprocess.CompletedProcess([], 2, "", "temporarily unavailable"),
            subprocess.CompletedProcess([], 1, "", "not found"),
        ]
    )
    monkeypatch.setattr(
        lifecycle, "_run_capture", lambda *_args, **_kwargs: next(responses)
    )
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)

    present, error = lifecycle._resolve_exact_cluster_presence(
        nebius_bin="nebius",
        cluster_id="cluster-exact",
        cluster_name="sop-exact",
        project_id="project-exact",
        env={},
        deadline=lifecycle.time.monotonic() + 60,
    )

    assert present is False
    assert error == ""


def test_exact_cluster_presence_fails_closed_on_wrong_parent(monkeypatch) -> None:
    import subprocess
    from npa.soperator import lifecycle

    payload = {
        "metadata": {
            "id": "cluster-exact",
            "name": "sop-exact",
            "parent_id": "project-foreign",
        }
    }
    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(payload), ""
        ),
    )

    present, error = lifecycle._resolve_exact_cluster_presence(
        nebius_bin="nebius",
        cluster_id="cluster-exact",
        cluster_name="sop-exact",
        project_id="project-exact",
        env={},
        deadline=lifecycle.time.monotonic() + 60,
    )

    assert present is None
    assert error == "exact Managed Kubernetes ownership evidence is ambiguous"


def test_exact_cluster_presence_accepts_pinned_provider_name(monkeypatch) -> None:
    import subprocess
    from npa.soperator import lifecycle

    payload = {
        "metadata": {
            "id": "cluster-exact",
            "name": "soperator-npasop",
            "parent_id": "project-exact",
        }
    }
    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(payload), ""
        ),
    )

    present, error = lifecycle._resolve_exact_cluster_presence(
        nebius_bin="nebius",
        cluster_id="cluster-exact",
        cluster_name="soperator-npasop",
        project_id="project-exact",
        env={},
        deadline=lifecycle.time.monotonic() + 60,
    )

    assert present is True
    assert error == ""


def test_destroy_deletes_orphaned_filesystems(tmp_path, monkeypatch) -> None:
    """Destroy deletes only persisted exact filesystem IDs from ownership state."""

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
        cluster_id="mk8scluster-exact",
        provider_cluster_name="soperator-npasop",
        owned_filesystem_ids=["fs-jail", "fs-spool"],
    )

    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(
        lifecycle, "_assert_solutions_library_contract", lambda *a, **k: None
    )
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
                {
                    "metadata": {
                        "id": "fs-spool",
                        "name": "soperator-npasop-controller-spool",
                    }
                },
                # A same-project filesystem from another cluster must be left alone.
                {"metadata": {"id": "fs-other", "name": "soperator-npatest-jail"}},
            ]
        }
    )

    def fake_capture(cmd, *, cwd=None, env=None, timeout=None, check=True):
        if "mk8s" in cmd and "get" in cmd:
            return _Done(stdout="not found", returncode=1)
        if "filesystem" in cmd and "list" in cmd:
            return _Done(stdout=fs_json)
        if "filesystem" in cmd and "delete" in cmd:
            deleted.append(cmd[cmd.index("--id") + 1])
            return _Done()
        if "filesystem" in cmd and "get" in cmd:
            return _Done(stdout="not found", returncode=1)
        return _Done(stdout="")

    monkeypatch.setattr(lifecycle, "_run_stream", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)
    monkeypatch.setattr(lifecycle, "_resolve_subnet", lambda *a, **k: "vpcsubnet-123")

    lifecycle.destroy_cluster("npasop", terraform_dir=recipe)

    # Only the soperator-npasop-* filesystems are swept; other clusters untouched.
    assert sorted(deleted) == ["fs-jail", "fs-spool"]


@pytest.mark.parametrize("failure", ["invalid-list", "delete-failed", "still-present"])
def test_destroy_retains_state_when_exact_auxiliary_cleanup_is_uncertain(
    tmp_path, monkeypatch, failure: str
) -> None:
    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    (recipe / "installations" / "example").mkdir(parents=True)
    install = recipe / "installations" / "npasafe"
    install.mkdir(parents=True)
    state_path = install / "terraform.tfstate"
    state_path.write_text("durable-state")
    lifecycle._write_env_sidecar(
        install,
        region="us-central1",
        tenant_id="tenant-abc",
        project_id="project-xyz",
        subnet_id="vpcsubnet-123",
        o11y_profile="default",
        cluster_id="mk8scluster-exact",
        provider_cluster_name="soperator-npasafe",
        owned_filesystem_ids=["fs-exact"],
    )
    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(
        lifecycle, "_assert_solutions_library_contract", lambda *a, **k: None
    )
    monkeypatch.setattr(lifecycle, "_terraform_env", lambda *a, **k: {})
    monkeypatch.setattr(lifecycle, "_run_stream", lambda *a, **k: None)
    monkeypatch.setenv("NPA_KUBECTL_BIN", "missing-kubectl-for-test")

    class Done:
        def __init__(self, stdout: str = "", returncode: int = 0) -> None:
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    inventory = json.dumps(
        {"items": [{"metadata": {"id": "fs-exact", "name": "irrelevant"}}]}
    )

    def capture(command, **kwargs):
        if "state" in command and "pull" in command:
            return Done(json.dumps({"resources": []}))
        if "destroy" in command:
            return Done()
        if "mk8s" in command and "get" in command:
            return Done("not found", 1)
        if "filesystem" in command and "list" in command:
            return Done("not-json" if failure == "invalid-list" else inventory)
        if "filesystem" in command and "delete" in command:
            return (
                Done("permission denied", 2) if failure == "delete-failed" else Done()
            )
        if "filesystem" in command and "get" in command:
            return (
                Done(json.dumps({"metadata": {"id": "fs-exact"}}))
                if failure == "still-present"
                else Done("not found", 1)
            )
        return Done()

    monkeypatch.setattr(lifecycle, "_run_capture", capture)
    if failure == "still-present":
        monkeypatch.setattr(
            lifecycle,
            "_wait_for_provider_id_absence",
            lambda **_kwargs: "owned filesystem fs-exact is still present",
        )

    result = lifecycle.destroy_cluster("npasafe", terraform_dir=recipe)

    assert result and result["status"] == "destroy-incomplete"
    assert state_path.read_text() == "durable-state"
    assert (install / ".npa-soperator-env.json").is_file()


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

    assert calls == [
        [
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
        ]
    ]


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
        if "sinfo" in cmd:
            return _Done(stdout="gpu-7\ngpu-3\ncpu-0\n")
        if "squeue" in cmd:
            return _Done(stdout="")
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

    checks = lifecycle._run_gpu_creation_checks(
        spec, "ctx", "kubectl", timeout_seconds=123
    )

    assert checks == [
        {
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
        }
    ]
    command = next(call for call in calls if "srun" in call)
    assert "--nodes=2" in command
    assert "--ntasks=2" in command
    assert "--gpus-per-node=8" in command
    assert "--nodelist=gpu-3,gpu-7" in command
    immediate = int(
        next(arg.split("=", 1)[1] for arg in command if arg.startswith("--immediate="))
    )
    assert 1 <= immediate <= 123
    assert f"--time={lifecycle._slurm_time_limit(immediate)}" in command
    assert "--kill-on-bad-exit=1" in command
    task_script = command[-1]
    assert "health-checker run" in task_script
    assert "deviceQuery,vectorAdd,simpleMultiGPU,p2pBandwidthLatencyTest" in task_script
    assert 'if [ "$command_rc" -ne 0 ] || [ "$status" != PASS ]' in task_script


def test_direct_gpu_creation_check_fails_on_missing_worker_pass(monkeypatch) -> None:
    from npa.soperator import lifecycle

    def fake_capture(cmd, **kwargs):
        if "sinfo" in cmd:
            return _Done(stdout="gpu-0\ngpu-1\n")
        if "squeue" in cmd or "scancel" in cmd:
            return _Done(stdout="")
        return _Done(
            stdout="0: NPA_GPU_CREATION_CHECK_RESULT host=gpu-0 status=PASS command_rc=0\n"
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

    with pytest.raises(
        lifecycle.GPUCreationCheckError, match="1/2 workers reported PASS"
    ) as caught:
        lifecycle._run_gpu_creation_checks(spec, "ctx", "kubectl")
    assert caught.value.cleanup_confirmed is True


def test_direct_gpu_creation_check_is_noop_for_cpu_cluster(monkeypatch) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected kubectl")),
    )
    spec = SoperatorSpec(name="c", workers=[WorkerPoolSpec(name="cpu")])

    assert lifecycle._run_gpu_creation_checks(spec, "ctx", "kubectl") == []


def _gpu_creation_spec(*, size: int = 2) -> SoperatorSpec:
    return SoperatorSpec(
        name="c",
        workers=[
            WorkerPoolSpec(
                name="gpu",
                platform="gpu-b200-sxm",
                preset="8gpu-160vcpu-1792gb",
                size=size,
                fabric="us-central1-b",
            )
        ],
    )


def test_gpu_creation_check_rejects_invalid_timeout_before_slurm(monkeypatch) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected Slurm call")
        ),
    )
    with pytest.raises(ValueError, match="at least 1 second"):
        lifecycle._run_gpu_creation_checks(
            _gpu_creation_spec(), "ctx", "kubectl", timeout_seconds=0
        )


def test_deploy_rejects_invalid_gpu_timeout_before_source_or_provider(
    monkeypatch,
) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_resolve_root_login_ssh_public_key",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("SSH/source/provider preflight should not start")
        ),
    )
    spec = _gpu_creation_spec()
    with pytest.raises(ValueError, match="at least 1 second"):
        lifecycle.deploy_cluster(spec, gpu_creation_check_timeout_seconds=0)


def test_gpu_creation_check_requires_live_slurm_pool_cardinality(monkeypatch) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        lambda cmd, **kwargs: _Done(stdout="renamed-worker-a\nrenamed-worker-b\n"),
    )
    with pytest.raises(
        lifecycle.GPUCreationCheckError,
        match="cannot map pool 'gpu'.*expected 2.*found 0",
    ) as caught:
        lifecycle._run_gpu_creation_checks(_gpu_creation_spec(), "ctx", "kubectl")
    assert caught.value.phase == "node-mapping"


def test_gpu_creation_check_process_timeout_cancels_and_verifies_queue(
    monkeypatch,
) -> None:
    from npa.soperator import lifecycle

    calls: list[tuple[list[str], dict]] = []

    def fake_capture(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if "sinfo" in cmd:
            return _Done(stdout="gpu-0\ngpu-1\n")
        if "srun" in cmd:
            raise subprocess.TimeoutExpired(
                cmd,
                kwargs["timeout"],
                output="srun: job queued and waiting for resources\n",
            )
        if "scancel" in cmd or "squeue" in cmd:
            return _Done(stdout="")
        raise AssertionError(cmd)

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    with pytest.raises(
        lifecycle.GPUCreationCheckError,
        match="timed out after .* queued/running Slurm job was cancelled",
    ) as caught:
        lifecycle._run_gpu_creation_checks(
            _gpu_creation_spec(), "ctx", "kubectl", timeout_seconds=17
        )

    assert caught.value.phase == "process-timeout"
    assert caught.value.cleanup_confirmed is True
    srun_call, srun_kwargs = next(item for item in calls if "srun" in item[0])
    assert 0 < srun_kwargs["timeout"] <= 17
    assert any(arg.startswith("--immediate=") for arg in srun_call)
    assert any(arg.startswith("--time=") for arg in srun_call)
    assert any("scancel" in command for command, _ in calls)
    assert any("squeue" in command for command, _ in calls)


def test_gpu_creation_check_queue_failure_is_nonzero_and_cleaned(monkeypatch) -> None:
    from npa.soperator import lifecycle

    calls: list[list[str]] = []

    def fake_capture(cmd, **kwargs):
        calls.append(cmd)
        if "sinfo" in cmd:
            return _Done(stdout="gpu-0\ngpu-1\n")
        if "srun" in cmd:
            return _Done(
                stderr="srun: Requested nodes are unavailable; unable to allocate resources",
                returncode=1,
            )
        if "scancel" in cmd or "squeue" in cmd:
            return _Done(stdout="")
        raise AssertionError(cmd)

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)

    with pytest.raises(
        lifecycle.GPUCreationCheckError,
        match="0/2 workers reported PASS",
    ) as caught:
        lifecycle._run_gpu_creation_checks(
            _gpu_creation_spec(), "ctx", "kubectl", timeout_seconds=30
        )
    assert caught.value.cleanup_confirmed is True
    assert "Requested nodes are unavailable" in str(caught.value)
    assert any("scancel" in command for command in calls)


def test_gpu_creation_check_success_requires_no_remaining_slurm_job(
    monkeypatch,
) -> None:
    from npa.soperator import lifecycle

    def fake_capture(cmd, **kwargs):
        if "sinfo" in cmd:
            return _Done(stdout="gpu-0\ngpu-1\n")
        if "srun" in cmd:
            return _Done(
                stdout=(
                    "0: NPA_GPU_CREATION_CHECK_RESULT host=gpu-0 status=PASS command_rc=0\n"
                    "1: NPA_GPU_CREATION_CHECK_RESULT host=gpu-1 status=PASS command_rc=0\n"
                )
            )
        if "squeue" in cmd:
            return _Done(stdout="")
        raise AssertionError(cmd)

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)
    checks = lifecycle._run_gpu_creation_checks(
        _gpu_creation_spec(), "ctx", "kubectl", timeout_seconds=60
    )
    assert checks[0]["status"] == "PASS"


def test_gpu_creation_check_success_tolerates_slurm_completion_race(
    monkeypatch,
) -> None:
    from npa.soperator import lifecycle

    squeue_calls = 0

    def fake_capture(cmd, **kwargs):
        nonlocal squeue_calls
        if "sinfo" in cmd:
            return _Done(stdout="gpu-0\ngpu-1\n")
        if "srun" in cmd:
            return _Done(
                stdout=(
                    "0: NPA_GPU_CREATION_CHECK_RESULT host=gpu-0 status=PASS command_rc=0\n"
                    "1: NPA_GPU_CREATION_CHECK_RESULT host=gpu-1 status=PASS command_rc=0\n"
                )
            )
        if "squeue" in cmd:
            squeue_calls += 1
            return _Done(stdout="72\n" if squeue_calls == 1 else "")
        raise AssertionError(cmd)

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)

    checks = lifecycle._run_gpu_creation_checks(
        _gpu_creation_spec(), "ctx", "kubectl", timeout_seconds=60
    )

    assert checks[0]["status"] == "PASS"
    assert squeue_calls == 2


def _degraded_deployment_result(lifecycle) -> dict:
    failure = lifecycle.DeploymentValidationFailure(
        code="gpu_creation_check_failed",
        message="GPU creation check timed out after 17 seconds for pool 'gpu'",
        check="gpu-creation",
        pool="gpu",
        phase="process-timeout",
        cleanup_confirmed=True,
    )
    error = lifecycle.SoperatorDeploymentValidationError(
        {
            "name": "c",
            "region": "us-central1",
            "project_id": "project",
            "install_dir": "/safe/installations/c",
            "kube_context": "nebius-c-slurm",
            "worker_pools": ["gpu"],
            "docker_cache_pools": [],
            "control_plane": {"system_min_size": 3, "system_max_size": 24},
            "gpu_creation_checks": [],
        },
        failure,
    )
    return error.result


def test_sdk_typed_degraded_validation_retains_deploy_metadata(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from npa.sdk import soperator as sdk
    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    (recipe / "installations" / "example").mkdir(parents=True)
    install = recipe / "installations" / "c"
    install.mkdir()
    monkeypatch.setattr(
        lifecycle,
        "resolve_environment",
        lambda **kwargs: SimpleNamespace(
            region="us-central1", tenant_id="tenant", project_id="project"
        ),
    )
    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(lifecycle, "_resolve_solutions_library", lambda *a, **k: recipe)
    monkeypatch.setattr(
        lifecycle, "_assert_solutions_library_contract", lambda *a, **k: None
    )
    monkeypatch.setattr(lifecycle, "_prepare_installation", lambda *a, **k: install)
    monkeypatch.setattr(lifecycle, "_resolve_subnet", lambda *a, **k: "subnet")
    monkeypatch.setattr(
        lifecycle,
        "_soperator_tf_env",
        lambda *a, **k: {"TF_VAR_o11y_profile": "profile"},
    )
    terraform_commands: list[list[str]] = []
    monkeypatch.setattr(
        lifecycle,
        "_run_terraform_command",
        lambda command, **kwargs: terraform_commands.append(command),
    )
    monkeypatch.setattr(
        lifecycle,
        "_terraform_plan_without_unsafe_replacements",
        lambda *a, **k: nullcontext(
            lifecycle.GuardedTerraformPlan(install / "guarded.tfplan")
        ),
    )
    monkeypatch.setattr(
        lifecycle, "_terraform_cluster_id", lambda *a, **k: "mk8scluster-c"
    )
    monkeypatch.setattr(
        lifecycle, "_terraform_cluster_name", lambda *a, **k: "soperator-c"
    )
    monkeypatch.setattr(
        lifecycle, "_terraform_owned_auxiliary_resources", lambda *a, **k: []
    )
    monkeypatch.setattr(
        lifecycle,
        "_run_gpu_creation_checks",
        lambda *a, **k: (_ for _ in ()).throw(
            lifecycle.GPUCreationCheckError(
                "GPU creation check timed out after 17 seconds for pool 'gpu'",
                pool="gpu",
                phase="process-timeout",
                cleanup_confirmed=True,
            )
        ),
    )
    spec = _gpu_creation_spec(size=2)
    spec.region = "us-central1"
    spec.tenant_id = "tenant"
    spec.project_id = "project"
    spec.root_login_ssh_public_key = "ssh-ed25519 AAAA operator"

    with pytest.raises(lifecycle.SoperatorDeploymentValidationError) as caught:
        sdk.deploy(
            spec,
            terraform_dir=recipe,
            apply_fixes=False,
            gpu_creation_check_timeout_seconds=17,
        )

    result = caught.value.result
    assert terraform_commands == [
        ["terraform", "init"],
        ["terraform", "apply", str(install / "guarded.tfplan")],
    ]
    assert result["status"] == "degraded-validation"
    assert result["deployment_status"] == "applied"
    assert result["install_dir"] == str(install)
    assert result["kube_context"] == "nebius-c-slurm"
    assert result["worker_pools"] == ["gpu"]
    assert result["control_plane"]["system_max_size"] == 24
    assert result["replacement_guard"] == {
        "status": "passed",
        "plans_checked": 1,
        "replacement_count": 0,
        "safe_local_replacements_applied": 0,
    }
    assert result["validation"] == {
        "status": "failed",
        "code": "gpu_creation_check_failed",
        "check": "gpu-creation",
        "message": "GPU creation check timed out after 17 seconds for pool 'gpu'",
        "pool": "gpu",
        "phase": "process-timeout",
        "cleanup_confirmed": True,
    }


@pytest.mark.parametrize("output", ["text", "json"])
def test_cli_degraded_validation_exits_nonzero_with_metadata(
    tmp_path, monkeypatch, output: str
) -> None:
    from npa.soperator import lifecycle

    spec_path = tmp_path / "cluster.yaml"
    spec_path.write_text(
        textwrap.dedent(
            """
            apiVersion: npa.soperator/v0.0.1
            name: c
            region: us-central1
            tenant_id: tenant
            project_id: project
            root_login_ssh_public_key: ssh-ed25519 AAAA operator
            workers:
              - name: gpu
                platform: gpu-b200-sxm
                preset: 8gpu-160vcpu-1792gb
                size: 2
                fabric: us-central1-b
            """
        )
    )
    result_payload = _degraded_deployment_result(lifecycle)
    failure = lifecycle.DeploymentValidationFailure(
        code="gpu_creation_check_failed",
        message=result_payload["validation"]["message"],
        check="gpu-creation",
        pool="gpu",
        phase="process-timeout",
        cleanup_confirmed=True,
    )
    error = lifecycle.SoperatorDeploymentValidationError(result_payload, failure)
    seen_kwargs: dict = {}

    def fail_deploy(*args, **kwargs):
        seen_kwargs.update(kwargs)
        raise error

    monkeypatch.setattr(lifecycle, "deploy_cluster", fail_deploy)

    invoked = runner.invoke(
        app,
        ["soperator", "deploy", "--spec", str(spec_path), "--output", output],
    )

    assert invoked.exit_code == 1
    assert seen_kwargs["stream_terraform_output"] is (output != "json")
    if output == "json":
        parsed = json.loads(invoked.stdout)
        assert parsed["status"] == "degraded-validation"
        assert parsed["install_dir"] == "/safe/installations/c"
        assert parsed["validation"]["cleanup_confirmed"] is True
    else:
        assert (
            "was applied, but mandatory post-apply validation failed" in invoked.output
        )
        assert "nebius-c-slurm" in invoked.output
        assert "/safe/installations/c" in invoked.output
        assert "Deployed soperator cluster" not in invoked.output


def test_json_mode_terraform_runner_captures_child_output(
    monkeypatch, tmp_path, capsys
) -> None:
    from npa.soperator import lifecycle

    seen: dict = {}

    def fake_capture(args, **kwargs):
        seen["args"] = args
        seen.update(kwargs)
        return _Done(stdout="terraform progress that must not reach JSON stdout\n")

    monkeypatch.setattr(lifecycle, "_run_capture", fake_capture)
    lifecycle._run_terraform_command(
        ["terraform", "apply"],
        cwd=tmp_path,
        env={},
        timeout=42,
        stream_output=False,
    )

    assert capsys.readouterr().out == ""
    assert seen["timeout"] == 42
    assert seen["check"] is False


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
            return _Done(
                stdout="customresourcedefinition.apiextensions.k8s.io/"
                "servicemonitors.monitoring.coreos.com\n"
            )
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
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
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


def test_monitoring_release_inspection_allows_clean_pre_flux_install(
    monkeypatch,
) -> None:
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
        "      ssh-check = {\n        commentPrefix = null\n      }\n",
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


def _write_local_reconciliation_targets(tmp_path: Path) -> tuple[Path, Path]:
    kubectl_context = tmp_path / "modules" / "k8s" / "k8s_cluster.tf"
    login = tmp_path / "modules" / "login" / "main.tf"
    kubectl_context.parent.mkdir(parents=True)
    login.parent.mkdir(parents=True)
    kubectl_context.write_text(
        'resource "terraform_data" "kubectl_cluster_context" {\n'
        "  triggers_replace = [\n"
        "    nebius_mk8s_v1_cluster.this.id,\n"
        "    timestamp(),\n"
        "  ]\n"
        "}\n"
    )
    login.write_text(
        'resource "terraform_data" "lb_service_ip" {\n'
        "  triggers_replace = [\n"
        "    one(data.kubernetes_service_v1.slurm_login.metadata).resource_version\n"
        "  ]\n"
        "\n"
        "  input = one(one(one(data.kubernetes_service_v1.slurm_login.status)."
        "load_balancer).ingress).ip\n"
        "}\n"
    )
    return kubectl_context, login


def test_patch_local_reconciliation_triggers_is_stable_and_idempotent(tmp_path) -> None:
    from npa.soperator import lifecycle

    kubectl_context, login = _write_local_reconciliation_targets(tmp_path)

    assert lifecycle._patch_stable_local_reconciliation_triggers(tmp_path) is True
    kubectl_patched = kubectl_context.read_text()
    login_patched = login.read_text()
    assert "timestamp()" not in kubectl_patched
    assert "resource_version" not in login_patched
    assert "nebius_mk8s_v1_cluster.this.id" in kubectl_patched
    assert (
        "one(one(one(data.kubernetes_service_v1.slurm_login.status)."
        "load_balancer).ingress).ip" in login_patched
    )

    assert lifecycle._patch_stable_local_reconciliation_triggers(tmp_path) is False
    assert kubectl_context.read_text() == kubectl_patched
    assert login.read_text() == login_patched


def test_patch_local_reconciliation_triggers_fails_closed_before_any_write(
    tmp_path,
) -> None:
    from npa.soperator import lifecycle

    kubectl_context, login = _write_local_reconciliation_targets(tmp_path)
    kubectl_original = kubectl_context.read_bytes()
    login.write_text(login.read_text().replace("resource_version", "generation"))
    login_original = login.read_bytes()

    with pytest.raises(lifecycle.UpstreamContractError, match="login-script trigger"):
        lifecycle._patch_stable_local_reconciliation_triggers(tmp_path)

    assert kubectl_context.read_bytes() == kubectl_original
    assert login.read_bytes() == login_original


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
    monkeypatch.setattr(
        lifecycle,
        "resolve_environment",
        lambda **k: SimpleNamespace(
            region="us-central1", tenant_id="tenant", project_id="project"
        ),
    )
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


def _reserved_worker(**overrides) -> WorkerPoolSpec:
    values = {
        "name": "gpu",
        "platform": "gpu-b200-sxm",
        "preset": "8gpu-160vcpu-1792gb",
        "size": 2,
        "fabric": "us-central1-b",
        "capacity_block_group": "capacityblockgroup-test",
    }
    values.update(overrides)
    return WorkerPoolSpec(**values)


def _capacity_group(
    *,
    reservation_id: str = "capacityblockgroup-test",
    name: str = "reserved-b200-test",
    tenant_id: str = "tenant-test",
    state: str = "STATE_ACTIVE",
    region: str = "us-central1",
    platform: str = "gpu-b200-sxm",
    fabric: str = "us-central1-b",
    limit: int = 40,
    usage: int = 8,
) -> dict:
    return {
        "metadata": {
            "id": reservation_id,
            "name": name,
            "parent_id": tenant_id,
        },
        "status": {
            "state": state,
            "region": region,
            "current_limit": str(limit),
            "usage": str(usage),
            "resource_affinity": {
                "compute_v1": {"platform": platform, "fabric": fabric}
            },
        },
    }


def _reserved_provider_capture(
    groups: list[dict],
    *,
    project_tenant="tenant-test",
    project_region="us-central1",
    advice_available: int | None = None,
    advice_data_state: str = "DATA_STATE_FRESH",
):
    def capture(command, **kwargs):
        if command[1:4] == ["iam", "project", "get"]:
            return _Done(
                stdout=json.dumps(
                    {
                        "metadata": {
                            "id": "project-test",
                            "parent_id": project_tenant,
                        },
                        "status": {"region": project_region},
                    }
                )
            )
        if command[1:4] == ["capacity", "capacity-block-group", "list"]:
            return _Done(stdout=json.dumps({"items": groups}))
        if command[1:4] == ["capacity", "resource-advice", "list"]:
            available_gpus = 0
            if groups:
                status = groups[0]["status"]
                available_gpus = int(status["current_limit"]) - int(status["usage"])
            return _Done(
                stdout=json.dumps(
                    {
                        "items": [
                            {
                                "spec": {
                                    "region": "us-central1",
                                    "fabric": "us-central1-b",
                                    "compute_instance": {
                                        "platform": "gpu-b200-sxm",
                                        "preset": {
                                            "name": "8gpu-160vcpu-1792gb",
                                            "resources": {"gpu_count": 8},
                                        },
                                    },
                                },
                                "status": {
                                    "reserved": {
                                        "data_state": advice_data_state,
                                        "available": (
                                            available_gpus // 8
                                            if advice_available is None
                                            else advice_available
                                        ),
                                    }
                                },
                            }
                        ]
                    }
                )
            )
        if command[1:4] == ["capacity", "capacity-block-group", "get"]:
            return _Done(returncode=1, stderr="not found")
        raise AssertionError(command)

    return capture


def test_reserved_capacity_spec_modes_and_mutual_exclusion() -> None:
    reserved = _reserved_worker()
    assert reserved.capacity_mode() == "reserved"
    assert reserved.reservation_selector_kind() == "id"
    reserved.validate()
    assert WorkerPoolSpec(name="cpu").capacity_mode() == "on-demand"
    assert WorkerPoolSpec(name="cpu", preemptible=True).capacity_mode() == "preemptible"

    with pytest.raises(SoperatorSpecError, match="mutually exclusive"):
        _reserved_worker(preemptible=True).validate()
    with pytest.raises(SoperatorSpecError, match="set only one"):
        _reserved_worker(capacity_block_group_name="reserved-b200-test").validate()
    with pytest.raises(SoperatorSpecError, match="only for GPU"):
        WorkerPoolSpec(
            name="cpu", capacity_block_group="capacityblockgroup-test"
        ).validate()


def test_reserved_capacity_sdk_and_positional_compatibility() -> None:
    from npa.sdk import soperator as sdk
    from npa.soperator import lifecycle

    # Reservation fields were appended, so the historical positional SDK
    # constructor retains exactly the same field mapping.
    worker = WorkerPoolSpec(
        "gpu",
        "gpu-b200-sxm",
        "8gpu-160vcpu-1792gb",
        2,
        512,
        "us-central1-b",
        False,
        True,
        372,
        "NETWORK_SSD_IO_M3",
    )
    assert worker.capacity_mode() == "on-demand"
    assert worker.docker_cache is True
    assert worker.capacity_block_group == ""
    assert sdk.plan is lifecycle.plan_cluster
    assert sdk.SoperatorDeploymentValidationError is (
        lifecycle.SoperatorDeploymentValidationError
    )


def test_reserved_capacity_spec_parses_id_or_name_without_exposing_values() -> None:
    from npa.soperator.lifecycle import plan_cluster

    spec = spec_from_mapping(
        {
            "apiVersion": "npa.soperator/v0.0.1",
            "name": "c",
            "workers": [
                {
                    "name": "gpu",
                    "platform": "gpu-b200-sxm",
                    "preset": "8gpu-160vcpu-1792gb",
                    "fabric": "us-central1-b",
                    "capacity_block_group_name": " reserved-b200-test ",
                }
            ],
        }
    )
    spec.validate()
    assert spec.workers[0].capacity_block_group_name == "reserved-b200-test"
    plan = plan_cluster(spec)
    assert plan["workers"][0]["capacity_mode"] == "reserved"
    assert plan["workers"][0]["reservation_selector"] == "name"
    assert plan["workers"][0]["reservation_verified"] is False
    assert "reserved-b200-test" not in json.dumps(plan)


def test_reserved_capacity_tfvars_render_exact_strict_upstream_contract() -> None:
    spec = SoperatorSpec(name="c", workers=[_reserved_worker()])
    rendered = render_tfvars(spec)
    assert "preemptible = null" in rendered
    assert 'policy          = "STRICT"' in rendered
    assert 'reservation_ids = ["capacityblockgroup-test"]' in rendered
    assert "AUTO" not in rendered

    named = _reserved_worker(
        capacity_block_group="", capacity_block_group_name="reserved-b200-test"
    )
    with pytest.raises(ValueError, match="provider-resolved"):
        render_tfvars(SoperatorSpec(name="c", workers=[named]))


def test_soperator_plan_cli_distinguishes_capacity_modes_without_selectors(
    tmp_path,
) -> None:
    spec_path = tmp_path / "cluster.yaml"
    spec_path.write_text(
        textwrap.dedent(
            """
            apiVersion: npa.soperator/v0.0.1
            name: c
            workers:
              - name: ondemand
              - name: spot
                preemptible: true
              - name: reserved
                platform: gpu-b200-sxm
                preset: 8gpu-160vcpu-1792gb
                fabric: us-central1-b
                capacity_block_group_name: reserved-b200-test
            """
        )
    )
    result = runner.invoke(
        app,
        ["soperator", "plan", "--spec", str(spec_path), "--output", "json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [worker["capacity_mode"] for worker in payload["workers"]] == [
        "on-demand",
        "preemptible",
        "reserved",
    ]
    assert "reserved-b200-test" not in result.stdout


def test_reserved_capacity_name_preflight_resolves_and_verifies_without_echoing_id(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    group = _capacity_group(usage=8)
    monkeypatch.setattr(lifecycle, "_run_capture", _reserved_provider_capture([group]))
    worker = _reserved_worker(
        capacity_block_group="", capacity_block_group_name="reserved-b200-test"
    )
    spec = SoperatorSpec(name="c", workers=[worker])

    resolved, summary = lifecycle._resolve_reserved_worker_capacity(
        spec,
        install_dir=tmp_path,
        nebius_bin="nebius",
        tenant_id="tenant-test",
        project_id="project-test",
        region="us-central1",
        env={},
    )

    assert resolved.workers[0].resolved_capacity_block_group_id == (
        "capacityblockgroup-test"
    )
    assert summary[0]["capacity_mode"] == "reserved"
    assert summary[0]["reservation_verified"] is True
    assert "capacityblockgroup-test" not in json.dumps(summary)


@pytest.mark.parametrize(
    ("group_overrides", "message"),
    [
        ({"tenant_id": "tenant-other"}, "belongs to another tenant"),
        ({"state": "STATE_INACTIVE"}, "not active"),
        ({"region": "eu-north1"}, "region does not match"),
        ({"platform": "gpu-h200-sxm"}, "platform does not match"),
        ({"fabric": "us-central1-a"}, "fabric does not match"),
        ({"limit": 40, "usage": 32}, "insufficient"),
    ],
)
def test_reserved_capacity_preflight_fails_closed_on_incompatible_or_insufficient(
    tmp_path, monkeypatch, group_overrides: dict, message: str
) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        _reserved_provider_capture([_capacity_group(**group_overrides)]),
    )
    with pytest.raises(ValueError, match=message):
        lifecycle._resolve_reserved_worker_capacity(
            SoperatorSpec(name="c", workers=[_reserved_worker()]),
            install_dir=tmp_path,
            nebius_bin="nebius",
            tenant_id="tenant-test",
            project_id="project-test",
            region="us-central1",
            env={},
        )


def test_reserved_capacity_preflight_rejects_wrong_project_tenant(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        _reserved_provider_capture([_capacity_group()], project_tenant="tenant-other"),
    )
    with pytest.raises(ValueError, match="project does not belong"):
        lifecycle._resolve_reserved_worker_capacity(
            SoperatorSpec(name="c", workers=[_reserved_worker()]),
            install_dir=tmp_path,
            nebius_bin="nebius",
            tenant_id="tenant-test",
            project_id="project-test",
            region="us-central1",
            env={},
        )


def test_reserved_capacity_preflight_rejects_wrong_project_region(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        _reserved_provider_capture([_capacity_group()], project_region="eu-north1"),
    )
    with pytest.raises(ValueError, match="project region does not match"):
        lifecycle._resolve_reserved_worker_capacity(
            SoperatorSpec(name="c", workers=[_reserved_worker()]),
            install_dir=tmp_path,
            nebius_bin="nebius",
            tenant_id="tenant-test",
            project_id="project-test",
            region="us-central1",
            env={},
        )


@pytest.mark.parametrize(
    ("available", "data_state", "message"),
    [
        (1, "DATA_STATE_FRESH", "reserved preset capacity is insufficient"),
        (2, "DATA_STATE_STALE", "availability is stale or unavailable"),
    ],
)
def test_reserved_capacity_preflight_fails_closed_on_preset_availability(
    tmp_path,
    monkeypatch,
    available: int,
    data_state: str,
    message: str,
) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        _reserved_provider_capture(
            [_capacity_group(limit=40, usage=8)],
            advice_available=available,
            advice_data_state=data_state,
        ),
    )
    with pytest.raises(ValueError, match=message):
        lifecycle._resolve_reserved_worker_capacity(
            SoperatorSpec(name="c", workers=[_reserved_worker()]),
            install_dir=tmp_path,
            nebius_bin="nebius",
            tenant_id="tenant-test",
            project_id="project-test",
            region="us-central1",
            env={},
        )


def test_reserved_capacity_preflight_refuses_unreadable_authorization(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        lambda *args, **kwargs: _Done(returncode=1, stderr="permission denied"),
    )
    with pytest.raises(ValueError, match="identity preflight failed"):
        lifecycle._resolve_reserved_worker_capacity(
            SoperatorSpec(name="c", workers=[_reserved_worker()]),
            install_dir=tmp_path,
            nebius_bin="nebius",
            tenant_id="tenant-test",
            project_id="project-test",
            region="us-central1",
            env={},
        )


def test_reserved_capacity_name_preflight_rejects_missing_and_ambiguous(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    worker = _reserved_worker(
        capacity_block_group="", capacity_block_group_name="reserved-b200-test"
    )
    spec = SoperatorSpec(name="c", workers=[worker])
    monkeypatch.setattr(lifecycle, "_run_capture", _reserved_provider_capture([]))
    with pytest.raises(ValueError, match="name was not found"):
        lifecycle._resolve_reserved_worker_capacity(
            spec,
            install_dir=tmp_path,
            nebius_bin="nebius",
            tenant_id="tenant-test",
            project_id="project-test",
            region="us-central1",
            env={},
        )

    duplicate = _capacity_group(reservation_id="capacityblockgroup-test-2")
    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        _reserved_provider_capture([_capacity_group(), duplicate]),
    )
    with pytest.raises(ValueError, match="name is ambiguous"):
        lifecycle._resolve_reserved_worker_capacity(
            spec,
            install_dir=tmp_path,
            nebius_bin="nebius",
            tenant_id="tenant-test",
            project_id="project-test",
            region="us-central1",
            env={},
        )


def test_reserved_capacity_preflight_credits_already_applied_strict_workers(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    (tmp_path / "terraform.tfstate").write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "type": "nebius_mk8s_v1_node_group",
                        "name": "worker_v2",
                        "instances": [
                            {
                                "attributes": {
                                    "name": "gpu-0",
                                    "fixed_node_count": 2,
                                    "template": {
                                        "resources": {"preset": "8gpu-160vcpu-1792gb"},
                                        "reservation_policy": {
                                            "policy": "STRICT",
                                            "reservation_ids": [
                                                "capacityblockgroup-test"
                                            ],
                                        },
                                    },
                                }
                            }
                        ],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        lifecycle,
        "_run_capture",
        _reserved_provider_capture(
            [_capacity_group(limit=48, usage=40)],
            advice_data_state="DATA_STATE_UNAVAILABLE",
        ),
    )

    resolved, summary = lifecycle._resolve_reserved_worker_capacity(
        SoperatorSpec(name="c", workers=[_reserved_worker()]),
        install_dir=tmp_path,
        nebius_bin="nebius",
        tenant_id="tenant-test",
        project_id="project-test",
        region="us-central1",
        env={},
    )
    assert resolved.workers[0].capacity_mode() == "reserved"
    assert summary[0]["reservation_verified"] is True

    with pytest.raises(ValueError, match="availability is stale or unavailable"):
        lifecycle._resolve_reserved_worker_capacity(
            SoperatorSpec(name="c", workers=[_reserved_worker(size=3)]),
            install_dir=tmp_path,
            nebius_bin="nebius",
            tenant_id="tenant-test",
            project_id="project-test",
            region="us-central1",
            env={},
        )


def test_worker_capacity_status_reads_applied_modes_without_ids(tmp_path) -> None:
    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    install = recipe / "installations" / "c"
    install.mkdir(parents=True)
    instances = []
    for name, preemptible, reservation_policy in [
        ("ondemand-0", None, None),
        ("spot-0", {}, None),
        (
            "reserved-0",
            None,
            {
                "policy": "STRICT",
                "reservation_ids": ["capacityblockgroup-test"],
            },
        ),
    ]:
        instances.append(
            {
                "attributes": {
                    "name": name,
                    "fixed_node_count": 1,
                    "template": {
                        "preemptible": preemptible,
                        "reservation_policy": reservation_policy,
                    },
                }
            }
        )
    (install / "terraform.tfstate").write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "type": "nebius_mk8s_v1_node_group",
                        "name": "worker_v2",
                        "instances": instances,
                    }
                ]
            }
        )
    )

    status = lifecycle.worker_capacity_status("c", terraform_dir=recipe)
    assert {item["name"]: item["capacity_mode"] for item in status} == {
        "ondemand": "on-demand",
        "reserved": "reserved",
        "spot": "preemptible",
    }
    assert "capacityblockgroup-test" not in json.dumps(status)


def test_deploy_reservation_failure_stops_before_render_init_or_provider_mutation(
    tmp_path, monkeypatch
) -> None:
    from npa.soperator import lifecycle

    recipe = tmp_path / "soperator"
    (recipe / "installations" / "example").mkdir(parents=True)
    monkeypatch.setattr(lifecycle, "_require_bin", lambda name: name)
    monkeypatch.setattr(
        lifecycle, "_resolve_solutions_library", lambda *args, **kwargs: recipe
    )
    monkeypatch.setattr(
        lifecycle, "_assert_solutions_library_contract", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        lifecycle,
        "_resolve_deploy_environment",
        lambda *args, **kwargs: (
            "us-central1",
            "tenant-test",
            "project-test",
            "subnet-test",
            {},
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "_resolve_reserved_worker_capacity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("reserved capacity is insufficient")
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "_prepare_installation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tfvars rendering must not start")
        ),
    )
    monkeypatch.setattr(
        lifecycle,
        "_run_terraform_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("terraform must not start")
        ),
    )
    spec = SoperatorSpec(
        name="c",
        region="us-central1",
        root_login_ssh_public_key="ssh-ed25519 AAAA operator",
        workers=[_reserved_worker()],
    )

    with pytest.raises(ValueError, match="reserved capacity is insufficient"):
        lifecycle.deploy_cluster(spec, terraform_dir=recipe)
