from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.cluster import app
from npa.cli.cluster import terraform_lifecycle as tf_mod


def test_cluster_terraform_defaults_all_gpu_pools_to_managed_driver_image() -> None:
    main_tf = (
        Path(__file__).resolve().parents[3] / "deploy" / "cluster" / "main.tf"
    ).read_text()

    assert '["auto", "managed-image"]' in main_tf
    assert "var.gpu_nodes_count > 0" in main_tf
    assert 'tonumber(regex("^([0-9]+)gpu-"' in main_tf
    assert "local.gpus_per_node > 1" in main_tf
    assert "gpu_nodes_driverfull_image      = local.gpu_nodes_driverfull_image" in main_tf
    assert "gpu_nodes_driver_preset         = var.managed_driver_preset" in main_tf
    assert "gpu_nodes_driverfull_image      = false" not in main_tf


runner = CliRunner()
_REAL_WHOLE_PATH_PREFLIGHT = tf_mod._preflight_whole_path_capacity


def _completed(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


@pytest.fixture(autouse=True)
def _node_group_ssh_key(tmp_path_factory, monkeypatch) -> Path:
    """The vendored module rejects a node-group key path that does not exist.

    The isolated test HOME has no ~/.ssh, so give every up/down run a real key.
    """
    key = tmp_path_factory.mktemp("ssh") / "id_ed25519.pub"
    key.write_text("ssh-ed25519 AAAAC3Nz test@example\n")
    monkeypatch.setenv("NPA_SSH_PUBLIC_KEY", str(key))
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
    # Tests in this module exercise Terraform sequencing and the legacy
    # filestore/GPU diagnostics. The shared cumulative gate has focused tests
    # below and in test_provisioning_preflight.py.
    monkeypatch.setattr(
        tf_mod, "_preflight_whole_path_capacity", lambda *_args, **_kwargs: None
    )
    return key


def _find_call(stream_calls: list[list[str]], *prefix: str) -> list[str] | None:
    for call in stream_calls:
        if call[: len(prefix)] == list(prefix):
            return call
    return None


def test_explicit_context_is_the_terraform_resource_name() -> None:
    tfvars = {"cluster_name": "npa-cluster"}
    context = "k8s-live-unique"

    assert tf_mod._apply_context_cluster_name(tfvars, context) == context
    assert tfvars["cluster_name"] == context
    with pytest.raises(Exception, match="conflicts with the resolved cluster name"):
        tf_mod._apply_context_cluster_name(
            tfvars, "different-context", inherited_name=context
        )


def test_saved_cluster_identity_uses_resolved_environment_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = []
    monkeypatch.setattr(
        tf_mod, "save_cluster_state", lambda state, **_kwargs: captured.append(state)
    )

    tf_mod._save_terraform_cluster_state(
        {"cluster_name": "exact", "cpu_nodes_count": 1},
        {"id": "cluster-id"},
        "exact",
        tmp_path / "kubeconfig",
        env={
            "TF_VAR_parent_id": "project-exact",
            "TF_VAR_region": "us-central1",
        },
    )

    assert captured[0].project_id == "project-exact"
    assert captured[0].region == "us-central1"


def _successful_stream(tf_dir: Path, calls: list[list[str]]):
    """Mock Terraform and materialize the state a successful apply guarantees."""

    def run(args, **_kwargs):
        calls.append(args)
        if args[:2] == ["terraform", "apply"]:
            (tf_dir / "terraform.tfstate").write_text(
                json.dumps({"version": 4, "resources": []})
            )
        return _completed()

    return run


def test_up_runs_terraform_writes_kubeconfig_and_validates(
    monkeypatch, tmp_path: Path
) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        "\n".join(
            [
                'parent_id = "project-a"',
                'tenant_id = "tenant-a"',
                'region = "region-a"',
                'cluster_name = "cluster-a"',
                "cpu_nodes_count = 0",
                "gpu_nodes_count = 2",
                'gpu_nodes_preset = "8gpu-192vcpu-1744gb"',
                "enable_filestore = true",
                'subnet_id = "subnet-a"',
            ]
        )
        + "\n"
    )
    stream_calls: list[list[str]] = []
    stream_envs: list[dict[str, str]] = []

    def fake_require_bin(binary: str) -> str:
        return binary

    def fake_stream(args, **kwargs):
        stream_calls.append(args)
        stream_envs.append(kwargs.get("env", {}))
        if args[:2] == ["terraform", "apply"]:
            (tf_dir / "terraform.tfstate").write_text(
                json.dumps({"version": 4, "resources": []})
            )
        return _completed()

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        if args[:4] == ["nebius", "quotas", "quota-allowance", "get-by-name"]:
            return _completed(json.dumps({"spec": {"limit": str(2 * 1024**4)}}))
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        if args[:3] == ["terraform", "output", "-json"]:
            return _completed(
                json.dumps(
                    {
                        "kube_cluster": {
                            "value": {
                                "id": "mk8scluster-a",
                                "name": "cluster-a",
                                "endpoints": {
                                    "public_endpoint": "https://cluster.example"
                                },
                            }
                        }
                    }
                )
            )
        if args[:3] == ["kubectl", "get", "nodes"]:
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {
                                    "name": "gpu-0",
                                    "labels": {
                                        "node.kubernetes.io/instance-type": "gpu-b200-sxm"
                                    },
                                },
                                "status": {
                                    "conditions": [{"type": "Ready", "status": "True"}],
                                    "allocatable": {"nvidia.com/gpu": "8"},
                                    "nodeInfo": {"bootID": "boot-0"},
                                }
                            },
                            {
                                "metadata": {
                                    "name": "gpu-1",
                                    "labels": {
                                        "node.kubernetes.io/instance-type": "gpu-b200-sxm"
                                    },
                                },
                                "status": {
                                    "conditions": [{"type": "Ready", "status": "True"}],
                                    "allocatable": {"nvidia.com/gpu": "8"},
                                    "nodeInfo": {"bootID": "boot-1"},
                                }
                            },
                        ]
                    }
                )
            )
        if args[:4] == ["kubectl", "get", "pods", "-n"]:
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {"name": "device-plugin"},
                                "status": {
                                    "phase": "Running",
                                    "containerStatuses": [{"ready": True}],
                                },
                            }
                        ]
                    }
                )
            )
        if args[:3] == ["kubectl", "get", "storageclass"]:
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {
                                    "name": "csi-mounted-fs-path-sc",
                                    "annotations": {
                                        "storageclass.kubernetes.io/is-default-class": "true"
                                    },
                                }
                            }
                        ]
                    }
                )
            )
        raise AssertionError(args)

    saved = []
    monkeypatch.setattr(tf_mod, "_require_bin", fake_require_bin)
    monkeypatch.setattr(tf_mod, "_run_stream", fake_stream)
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)
    monkeypatch.setattr(
        tf_mod, "save_cluster_state", lambda state, metadata=None: saved.append(state)
    )

    result = runner.invoke(
        app,
        [
            "up",
            "--terraform-dir",
            str(tf_dir),
            "--capacity-block-group",
            "capacityblockgroup-test",
            "--gpu-health-stabilization-seconds",
            "0",
            "--skip-gpu-cuda-smoke",
            "--skip-sky-smoke",
        ],
    )

    assert result.exit_code == 0, result.output
    assert ["terraform", "init", "-lockfile=readonly"] in stream_calls
    init_index = stream_calls.index(["terraform", "init", "-lockfile=readonly"])
    isolated_data = Path(stream_envs[init_index]["TF_DATA_DIR"])
    assert isolated_data != tf_dir / ".terraform"
    assert not isolated_data.exists()
    assert not (tf_dir / ".terraform").exists()
    # -var beats terraform.tfvars; TF_VAR_* does not, so the flag has to be passed
    # explicitly as well as exported.
    apply_call = _find_call(stream_calls, "terraform", "apply", "-auto-approve")
    assert apply_call is not None
    assert "capacity_block_group=capacityblockgroup-test" in apply_call
    apply_env = stream_envs[stream_calls.index(apply_call)]
    assert apply_env["TF_VAR_capacity_block_group"] == "capacityblockgroup-test"
    assert any(
        call[:4] == ["nebius", "mk8s", "cluster", "get-credentials"]
        for call in stream_calls
    )
    assert [state.last_seen_state for state in saved] == ["VALIDATING", "RUNNING"]
    assert saved[-1].cluster_id == "mk8scluster-a"
    assert "16 allocatable GPUs" in result.output


def test_inherited_topology_overrides_every_effective_terraform_input() -> None:
    from npa.provisioning_preflight import WholePathPreflightPlan, resolve_topology

    tfvars = {
        "cluster_name": "checked-in",
        "parent_id": "wrong-project",
        "tenant_id": "wrong-tenant",
        "region": "wrong-region",
        "cpu_nodes_count": 99,
        "gpu_nodes_count": 99,
        "cpu_nodes_platform": "wrong-cpu",
        "cpu_nodes_preset": "wrong-cpu-preset",
        "gpu_nodes_platform": "wrong-gpu",
        "gpu_nodes_preset": "wrong-gpu-preset",
        "gpu_nodes_preemptible": False,
    }
    plan = WholePathPreflightPlan(
        project_alias="demo",
        project_id="project-a",
        tenant_id="tenant-a",
        region="region-a",
        topology=resolve_topology(
            cluster_name="canonical",
            cpu_nodes=1,
            gpu_nodes=2,
            cpu_platform="cpu-d3",
            cpu_preset="8vcpu-32gb",
            gpu_platform="gpu-rtx6000",
            gpu_preset="1gpu-24vcpu-218gb",
            preemptible=True,
        ),
        decision="ready",
    )

    tf_mod._apply_inherited_plan_tfvars(tfvars, plan)

    assert tfvars == {
        "cluster_name": "canonical",
        "parent_id": "project-a",
        "tenant_id": "tenant-a",
        "region": "region-a",
        "cpu_nodes_count": 1,
        "gpu_nodes_count": 2,
        "cpu_nodes_platform": "cpu-d3",
        "cpu_nodes_preset": "8vcpu-32gb",
        "gpu_nodes_platform": "gpu-rtx6000",
        "gpu_nodes_preset": "1gpu-24vcpu-218gb",
        "gpu_nodes_preemptible": True,
    }
    assert tf_mod._string_var_args("cluster_name", tfvars["cluster_name"]) == [
        "-var",
        "cluster_name=canonical",
    ]


def test_direct_cluster_preflight_keeps_supplied_project_alias(monkeypatch) -> None:
    from npa import provisioning_preflight
    from npa.provisioning_preflight import ExistingCapacity, WholePathPreflightPlan

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        tf_mod, "_preflight_whole_path_capacity", _REAL_WHOLE_PATH_PREFLIGHT
    )
    monkeypatch.setattr(
        provisioning_preflight,
        "discover_existing_capacity",
        lambda **_kwargs: ExistingCapacity(),
    )

    def build(**kwargs):
        captured.update(kwargs)
        return WholePathPreflightPlan(
            project_alias=str(kwargs["project_alias"]),
            project_id=str(kwargs["project_id"]),
            tenant_id=str(kwargs["tenant_id"]),
            region=str(kwargs["region"]),
            topology=kwargs["topology"],
            decision="ready",
        )

    monkeypatch.setattr(provisioning_preflight, "build_whole_path_plan", build)
    tf_mod._preflight_whole_path_capacity(
        {
            "parent_id": "project-a",
            "tenant_id": "tenant-a",
            "region": "region-a",
            "cluster_name": "cluster-a",
            "cpu_nodes_count": 0,
            "gpu_nodes_count": 0,
        },
        {},
        context="cluster-a",
        project_alias="demo",
    )

    assert captured["project_alias"] == "demo"


@pytest.mark.parametrize(
    "blocking_quota", ["compute.instance.count", "compute.disk.count"]
)
def test_up_default_shape_quota_blocker_runs_before_terraform_apply(
    monkeypatch, tmp_path: Path, blocking_quota: str
) -> None:
    from npa import provisioning_preflight
    from npa.provisioning_preflight import (
        ExistingCapacity,
        QuotaObservation,
    )

    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'parent_id = "project-a"\ntenant_id = "tenant-a"\n'
        'region = "eu-north1"\ncluster_name = "cluster-a"\n'
    )
    monkeypatch.setattr(
        tf_mod, "_preflight_whole_path_capacity", _REAL_WHOLE_PATH_PREFLIGHT
    )
    monkeypatch.setattr(
        provisioning_preflight,
        "discover_existing_capacity",
        lambda **_kwargs: ExistingCapacity(),
    )
    monkeypatch.setattr(
        provisioning_preflight,
        "read_provider_quotas",
        lambda _tenant, _region, names: {
            name: QuotaObservation(
                name=name,
                used=10 if name == blocking_quota else 0,
                limit=10 if name == blocking_quota else 100,
                state="known",
            )
            for name in names
        },
    )
    terraform_calls: list[list[str]] = []
    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(
        tf_mod,
        "_run_stream",
        lambda args, **_kwargs: terraform_calls.append(args),
    )
    monkeypatch.setattr(
        tf_mod,
        "_run_capture",
        lambda args, **_kwargs: _completed(json.dumps({"terraform_version": "1.12.2"})),
    )

    result = runner.invoke(
        app,
        [
            "up",
            "--terraform-dir",
            str(tf_dir),
            "--skip-validate",
            "--skip-sky-smoke",
        ],
    )

    assert result.exit_code != 0
    assert blocking_quota in result.output
    assert terraform_calls == []


def test_failed_fresh_apply_rolls_back_only_its_new_terraform_state(
    monkeypatch, tmp_path: Path, mocker
) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'parent_id = "project-a"\ntenant_id = "tenant-a"\n'
        'region = "eu-north1"\ncluster_name = "cluster-a"\n'
        "cpu_nodes_count = 0\ngpu_nodes_count = 0\n"
    )

    def capture(args, **_kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        raise AssertionError(args)

    def stream(args, **_kwargs):
        if args[:2] == ["terraform", "apply"]:
            (tf_dir / "terraform.tfstate").write_text(
                json.dumps(
                    {
                        "version": 4,
                        "resources": [{"type": "nebius_vpc_v1_network", "name": "new"}],
                    }
                )
            )
            raise RuntimeError("provider race after network create")
        return _completed()

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_capture", capture)
    monkeypatch.setattr(tf_mod, "_run_stream", stream)
    down = mocker.patch.object(tf_mod, "down_cmd")

    result = runner.invoke(
        app,
        [
            "up",
            "--terraform-dir",
            str(tf_dir),
            "--skip-validate",
            "--skip-sky-smoke",
        ],
    )

    assert result.exit_code != 0
    assert "provider race after network create" in result.output
    down.assert_called_once()
    assert down.call_args.kwargs["context_name"] == "cluster-a"


def test_validate_cluster_accepts_compute_csi_when_filestore_is_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    responses = {
        ("kubectl", "get", "nodes", "-o", "json"): {
            "items": [{
                "metadata": {
                    "name": "gpu-0",
                    "labels": {"node.kubernetes.io/instance-type": "gpu-rtx6000"},
                },
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "allocatable": {"nvidia.com/gpu": "1"},
                    "nodeInfo": {"bootID": "boot-a"},
                }
            }]
        },
        ("kubectl", "get", "pods", "-n", "nvidia-device-plugin", "-o", "json"): {
            "items": [{
                "metadata": {"name": "device-plugin"},
                "status": {
                    "phase": "Running",
                    "containerStatuses": [{"ready": True}],
                },
            }]
        },
        ("kubectl", "get", "storageclass", "-o", "json"): {
            "items": [{
                "metadata": {
                    "name": "compute-csi-default-sc",
                    "annotations": {"storageclass.kubernetes.io/is-default-class": "true"},
                }
            }]
        },
    }

    monkeypatch.setattr(
        tf_mod,
        "_run_capture",
        lambda args, **_kwargs: _completed(json.dumps(responses[tuple(args)])),
    )

    result = tf_mod._validate_cluster_once(
        "kubectl",
        tmp_path / "kubeconfig",
        {
            "cpu_nodes_count": 0,
            "gpu_nodes_count": 1,
            "gpu_nodes_platform": "gpu-rtx6000",
            "gpu_nodes_preset": "1gpu-24vcpu-218gb",
            "gpu_driver_mode": "auto",
            "enable_filestore": False,
        },
    )

    assert result == {
        "ready_nodes": 1,
        "gpu_nodes": 1,
        "total_gpus": 1,
        "default_storage_class": "compute-csi-default-sc",
    }


def test_up_stops_on_unmanaged_duplicate(monkeypatch, tmp_path: Path) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'parent_id = "project-a"\ncluster_name = "cluster-a"\n'
    )

    def fake_require_bin(binary: str) -> str:
        return binary

    def fake_stream(args, **kwargs):
        return _completed()

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {"metadata": {"id": "mk8scluster-a", "name": "cluster-a"}}
                        ]
                    }
                )
            )
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", fake_require_bin)
    monkeypatch.setattr(tf_mod, "_run_stream", fake_stream)
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code != 0
    assert "outside this Terraform" in result.output
    assert "mk8scluster-a" in result.output
    assert "npa cluster destroy" in result.output
    assert "--project-id" in result.output
    assert "nebius mk8s cluster delete" not in result.output


def test_up_allows_duplicate_managed_by_terraform_state(
    monkeypatch, tmp_path: Path
) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        "\n".join(
            [
                'parent_id = "project-a"',
                'tenant_id = "tenant-a"',
                'region = "region-a"',
                'cluster_name = "cluster-a"',
                'existing_filestore = "computefilesystem-a"',
            ]
        )
        + "\n"
    )
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {"metadata": {"id": "mk8scluster-a", "name": "cluster-a"}}
                        ]
                    }
                )
            )
        if args[:2] == ["terraform", "state"]:
            return _completed(
                json.dumps(
                    {
                        "resources": [
                            {
                                "type": "nebius_mk8s_v1_cluster",
                                "instances": [{"attributes": {"id": "mk8scluster-a"}}],
                            }
                        ]
                    }
                )
            )
        if args[:3] == ["terraform", "output", "-json"]:
            return _completed(
                json.dumps(
                    {
                        "kube_cluster": {
                            "value": {
                                "id": "mk8scluster-a",
                                "name": "cluster-a",
                                "endpoints": {},
                            }
                        }
                    }
                )
            )
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", _successful_stream(tf_dir, stream_calls))
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)
    monkeypatch.setattr(tf_mod, "save_cluster_state", lambda state, metadata=None: None)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code == 0, result.output
    assert _find_call(stream_calls, "terraform", "apply", "-auto-approve")


def test_up_stops_when_filestore_quota_is_too_small(
    monkeypatch, tmp_path: Path
) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        "\n".join(
            [
                'parent_id = "project-a"',
                'tenant_id = "tenant-a"',
                'region = "region-a"',
                'cluster_name = "cluster-a"',
                "enable_filestore = true",
                "filestore_disk_size_gibibytes = 1024",
            ]
        )
        + "\n"
    )
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        if args[:4] == ["nebius", "quotas", "quota-allowance", "get-by-name"]:
            return _completed(json.dumps({"spec": {"limit": "0"}}))
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", _successful_stream(tf_dir, stream_calls))
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code != 0
    assert "Shared filesystem quota is insufficient" in result.output
    assert _find_call(stream_calls, "terraform", "apply", "-auto-approve") is None


def test_up_skips_filestore_quota_when_disabled_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    """The default FTUE shape (no enable_filestore) applies with zero SFS quota.

    The Shared Filesystem quota preflight must NOT run when filestore is not
    opted into, so `npa cluster up` / `provision-if-absent` succeed with zero
    Shared Filesystem SSD quota.
    """
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        "\n".join(
            [
                'parent_id = "project-a"',
                'tenant_id = "tenant-a"',
                'region = "region-a"',
                'cluster_name = "cluster-a"',
            ]
        )
        + "\n"
    )
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        if args[:4] == ["nebius", "quotas", "quota-allowance", "get-by-name"]:
            raise AssertionError(
                "filestore quota must not be checked when filestore is off"
            )
        if args[:3] == ["terraform", "output", "-json"]:
            return _completed(
                json.dumps(
                    {
                        "kube_cluster": {
                            "value": {
                                "id": "mk8scluster-a",
                                "name": "cluster-a",
                                "endpoints": {},
                            }
                        }
                    }
                )
            )
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", _successful_stream(tf_dir, stream_calls))
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)
    monkeypatch.setattr(tf_mod, "save_cluster_state", lambda state, metadata=None: None)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code == 0, result.output
    assert _find_call(stream_calls, "terraform", "apply", "-auto-approve")


def test_up_validation_accepts_block_default_sc_when_filestore_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    """With filestore off, validation does not require the filesystem CSI SC.

    The filesystem CSI (and its `csi-mounted-fs-path-sc` default) is only
    installed when the shared filesystem is enabled, so validation must accept
    the platform block-storage default StorageClass instead of failing.
    """
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        "\n".join(
            [
                'parent_id = "project-a"',
                'tenant_id = "tenant-a"',
                'region = "region-a"',
                'cluster_name = "cluster-a"',
                "cpu_nodes_count = 0",
                "gpu_nodes_count = 1",
                'gpu_nodes_preset = "1gpu-24vcpu-218gb"',
            ]
        )
        + "\n"
    )
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        if args[:4] == ["nebius", "quotas", "quota-allowance", "get-by-name"]:
            return _completed(
                json.dumps({"spec": {"limit": "8"}, "status": {"usage": "0"}})
            )
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        if args[:3] == ["terraform", "output", "-json"]:
            return _completed(
                json.dumps(
                    {
                        "kube_cluster": {
                            "value": {
                                "id": "mk8scluster-a",
                                "name": "cluster-a",
                                "endpoints": {
                                    "public_endpoint": "https://cluster.example"
                                },
                            }
                        }
                    }
                )
            )
        if args[:3] == ["kubectl", "get", "nodes"]:
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {
                                    "name": "gpu-0",
                                    "labels": {
                                        "node.kubernetes.io/instance-type": "gpu-rtx6000"
                                    },
                                },
                                "status": {
                                    "conditions": [{"type": "Ready", "status": "True"}],
                                    "allocatable": {"nvidia.com/gpu": "1"},
                                    "nodeInfo": {"bootID": "boot-0"},
                                }
                            }
                        ]
                    }
                )
            )
        if args[:4] == ["kubectl", "get", "pods", "-n"]:
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {"name": "device-plugin"},
                                "status": {
                                    "phase": "Running",
                                    "containerStatuses": [{"ready": True}],
                                },
                            }
                        ]
                    }
                )
            )
        if args[:3] == ["kubectl", "get", "storageclass"]:
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {
                                    "name": "compute-csi-default-sc",
                                    "annotations": {
                                        "storageclass.kubernetes.io/is-default-class": "true"
                                    },
                                }
                            }
                        ]
                    }
                )
            )
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", _successful_stream(tf_dir, stream_calls))
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)
    monkeypatch.setattr(tf_mod, "save_cluster_state", lambda state, metadata=None: None)

    result = runner.invoke(
        app,
        [
            "up",
            "--terraform-dir",
            str(tf_dir),
            "--gpu-health-stabilization-seconds",
            "0",
            "--skip-gpu-cuda-smoke",
            "--skip-sky-smoke",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "default StorageClass compute-csi-default-sc" in result.output


def test_preflight_instance_count_quota_refuses_when_insufficient(monkeypatch) -> None:
    """Predict a compute.instance.count shortfall before any apply (the agent-vs-GPUs case)."""
    from npa.clients import nebius as nebius_module

    # An agent VM is already running (usage 1) against a limit-2 tenant.
    monkeypatch.setattr(
        nebius_module, "get_compute_instance_quota", lambda _t, _r: (1, 2)
    )
    tfvars = {
        "gpu_nodes_count": 2,
        "cpu_nodes_count": 1,
        "tenant_id": "tenant-x",
        "region": "us-central1",
    }

    with pytest.raises(Exception) as excinfo:  # typer.BadParameter
        tf_mod._preflight_instance_count_quota(tfvars, {})

    message = str(excinfo.value)
    assert "needs 3 compute instance" in message
    assert "only 1 free" in message
    assert "compute.instance.count" in message


def test_preflight_instance_count_quota_passes_with_headroom(monkeypatch) -> None:
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        nebius_module, "get_compute_instance_quota", lambda _t, _r: (0, 5)
    )
    tf_mod._preflight_instance_count_quota(
        {"gpu_nodes_count": 2, "cpu_nodes_count": 1, "tenant_id": "t", "region": "r"},
        {},
    )


def test_preflight_instance_count_quota_counts_no_tfvars_defaults(monkeypatch) -> None:
    from npa.clients import nebius as nebius_module

    monkeypatch.setattr(
        nebius_module, "get_compute_instance_quota", lambda _t, _r: (0, 1)
    )

    with pytest.raises(Exception) as excinfo:  # typer.BadParameter
        tf_mod._preflight_instance_count_quota(
            {"tenant_id": "tenant-x", "region": "us-central1"}, {}
        )

    assert "needs 2 compute instance" in str(excinfo.value)
    assert "1 GPU + 1 CPU" in str(excinfo.value)


def test_preflight_instance_count_quota_noop_when_unreadable(monkeypatch) -> None:
    from npa.clients import nebius as nebius_module

    def _boom(_t, _r):
        return (None, None)

    monkeypatch.setattr(nebius_module, "get_compute_instance_quota", _boom)
    # Must not raise even though nodes are requested.
    tf_mod._preflight_instance_count_quota(
        {"gpu_nodes_count": 8, "cpu_nodes_count": 4, "tenant_id": "t", "region": "r"},
        {},
    )


def test_preflight_instance_count_quota_noop_without_tenant_or_region(
    monkeypatch,
) -> None:
    from npa.clients import nebius as nebius_module

    def _boom(_t, _r):  # pragma: no cover - must not be reached
        raise AssertionError("quota lookup should be skipped without tenant/region")

    monkeypatch.setattr(nebius_module, "get_compute_instance_quota", _boom)
    tf_mod._preflight_instance_count_quota({"gpu_nodes_count": 2}, {})


def test_node_count_flag_overrides_tfvars_and_beats_it_with_var() -> None:
    tfvars: dict = {"gpu_nodes_count": 2}
    # -1 keeps the configured value; a real value overrides tfvars and adds -var.
    tf_mod._apply_node_count_override(tfvars, "gpu_nodes_count", -1)
    assert tfvars["gpu_nodes_count"] == 2
    assert tf_mod._node_count_var_args(tfvars, "gpu_nodes_count", -1) == []

    tf_mod._apply_node_count_override(tfvars, "gpu_nodes_count", 0)
    assert tfvars["gpu_nodes_count"] == 0
    assert tf_mod._node_count_var_args(tfvars, "gpu_nodes_count", 0) == [
        "-var",
        "gpu_nodes_count=0",
    ]


def test_node_shape_flags_override_tfvars_and_emit_explicit_vars() -> None:
    tfvars = {
        "cpu_nodes_platform": "old-cpu",
        "gpu_nodes_preset": "old-gpu-preset",
    }

    tf_mod._apply_string_override(tfvars, "cpu_nodes_platform", "cpu-d3")
    tf_mod._apply_string_override(tfvars, "gpu_nodes_preset", "1gpu-24vcpu-218gb")

    assert tfvars["cpu_nodes_platform"] == "cpu-d3"
    assert tfvars["gpu_nodes_preset"] == "1gpu-24vcpu-218gb"
    assert tf_mod._string_var_args("gpu_nodes_platform", "gpu-rtx6000") == [
        "-var",
        "gpu_nodes_platform=gpu-rtx6000",
    ]
    assert tf_mod._string_var_args("gpu_nodes_platform", "") == []


def test_down_runs_terraform_destroy(monkeypatch, tmp_path: Path) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfstate").write_text("{}")
    stream_calls: list[list[str]] = []

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(
        tf_mod, "_run_capture", lambda *args, **kwargs: _completed("token-a\n")
    )

    def fake_stream(args, **kwargs):
        stream_calls.append(args)
        return _completed()

    monkeypatch.setattr(tf_mod, "_run_stream", fake_stream)

    result = runner.invoke(app, ["down", "--terraform-dir", str(tf_dir), "--force"])

    assert result.exit_code == 0, result.output
    assert ["terraform", "init", "-lockfile=readonly"] in stream_calls
    assert _find_call(stream_calls, "terraform", "destroy", "-auto-approve")


def test_down_preview_uses_the_selected_npa_cluster_kubeconfig(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cluster import state as state_module

    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'parent_id = "project-a"\ncluster_name = "selected-cluster"\n'
    )
    saved_kubeconfig = tmp_path / "clusters" / "selected-cluster" / "kubeconfig"
    saved_kubeconfig.parent.mkdir(parents=True)
    saved_kubeconfig.write_text("apiVersion: v1\n")
    observed: list[tuple[Path | None, str]] = []

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(
        tf_mod, "_run_capture", lambda *args, **kwargs: _completed("token-a\n")
    )
    monkeypatch.setattr(tf_mod, "_run_stream", lambda *args, **kwargs: _completed())
    monkeypatch.setattr(
        state_module,
        "existing_kubeconfig",
        lambda name: saved_kubeconfig if name == "selected-cluster" else None,
    )
    monkeypatch.setattr(
        tf_mod,
        "_report_drain_blockers",
        lambda path, *, context="": observed.append((path, context)),
    )

    result = runner.invoke(
        app,
        [
            "down",
            "--terraform-dir",
            str(tf_dir),
            "--kubeconfig",
            str(saved_kubeconfig),
            "--force",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == [(saved_kubeconfig, "selected-cluster")]


def test_down_with_no_cluster_is_a_true_local_noop(monkeypatch, tmp_path: Path) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text('cluster_name = "never-created"\n')

    def unexpected(*_args, **_kwargs):
        raise AssertionError("no-cluster teardown crossed an external boundary")

    monkeypatch.setattr(tf_mod, "_require_bin", unexpected)
    monkeypatch.setattr(tf_mod, "_terraform_env", unexpected)
    monkeypatch.setattr(tf_mod, "_report_drain_blockers", unexpected)
    monkeypatch.setattr(tf_mod, "_run_stream", unexpected)
    monkeypatch.setattr(tf_mod, "_run_capture", unexpected)

    result = runner.invoke(app, ["down", "--terraform-dir", str(tf_dir), "--force"])

    assert result.exit_code == 0, result.output
    assert "Nothing to do" in result.output
    assert "authentication" in result.output
    assert "Kubernetes/RBAC" in result.output
    assert not (tf_dir / ".terraform").exists()


def test_down_checksum_mismatch_is_actionable_and_keeps_lock_immutable(
    monkeypatch, tmp_path: Path
) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfstate").write_text("{}")
    lock_file = tf_dir / ".terraform.lock.hcl"
    original_lock = 'provider "example.invalid/test" {}\n'
    lock_file.write_text(original_lock)
    calls: list[list[str]] = []

    def fake_stream(args, **kwargs):
        calls.append(args)
        if args[:2] == ["terraform", "init"]:
            raise tf_mod.typer.BadParameter(
                "the local package doesn't match any of the checksums previously recorded"
            )
        raise AssertionError(f"destroy must not run after checksum failure: {args}")

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_preflight_terraform_version", lambda *_args: None)
    monkeypatch.setattr(tf_mod, "_terraform_env", lambda _binary: {})
    monkeypatch.setattr(
        tf_mod, "_report_drain_blockers", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(tf_mod, "_run_stream", fake_stream)

    result = runner.invoke(
        app,
        ["down", "--terraform-dir", str(tf_dir), "--force"],
        terminal_width=80,
    )

    assert result.exit_code != 0
    # Rich may wrap between words according to the runner's terminal width;
    # assert the operator message rather than its presentation whitespace.
    output = " ".join(result.output.replace("│", " ").split())
    assert "checksum verification failed" in output
    assert "did not modify the lock file" in output
    assert "Checksum bypass is forbidden" in output
    assert "providers lock" in output
    assert lock_file.read_text() == original_lock
    assert not (tf_dir / ".terraform").exists()
    assert all(call[:2] != ["terraform", "destroy"] for call in calls)


def test_up_rejects_terraform_older_than_the_vendored_modules(
    monkeypatch, tmp_path: Path
) -> None:
    """The vendored modules need >= 1.12; an old binary must fail before init."""
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text('parent_id = "project-test"\n')
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.9.8"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", _successful_stream(tf_dir, stream_calls))
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code != 0
    assert "1.9.8 is too old" in result.output
    assert "1.12.0" in result.output
    assert stream_calls == []


def test_up_rejects_an_iam_token_pinned_in_tfvars(monkeypatch, tmp_path: Path) -> None:
    """Terraform prefers tfvars over TF_VAR_*, so a pinned token shadows the fresh one."""
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'parent_id = "project-a"\niam_token = "<nebius-iam-token>"\n'
    )
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", _successful_stream(tf_dir, stream_calls))
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code != 0
    assert "iam_token" in result.output
    assert stream_calls == []


def test_up_stops_before_apply_when_the_gpu_quota_is_zero(
    monkeypatch, tmp_path: Path
) -> None:
    """A GPU quota of 0 becomes QuotaFailure + a silent Terraform retry loop."""
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        "\n".join(
            [
                'parent_id = "project-a"',
                'tenant_id = "tenant-a"',
                'region = "us-central1"',
                'cluster_name = "cluster-a"',
                "gpu_nodes_count = 1",
                'gpu_nodes_platform = "gpu-rtx6000"',
                'gpu_nodes_preset = "1gpu-24vcpu-218gb"',
            ]
        )
        + "\n"
    )
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        if args[:4] == ["nebius", "quotas", "quota-allowance", "get-by-name"]:
            return _completed(
                json.dumps({"spec": {"limit": "0"}, "status": {"usage": "0"}})
            )
        if args[:4] == ["nebius", "capacity", "resource-advice", "list"]:
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {
                                "spec": {
                                    "region": "us-central1",
                                    "compute_instance": {
                                        "platform": "gpu-rtx6000",
                                        "preset": {"name": "1gpu-24vcpu-218gb"},
                                    },
                                },
                                "status": {
                                    "on_demand": {
                                        "availability_level": "AVAILABILITY_LEVEL_LIMIT_REACHED"
                                    },
                                    "preemptible": {
                                        "availability_level": "AVAILABILITY_LEVEL_HIGH",
                                        "available": 44,
                                    },
                                },
                            }
                        ]
                    }
                )
            )
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", _successful_stream(tf_dir, stream_calls))
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code != 0
    assert "GPU quota is insufficient" in result.output
    assert "gpu_nodes_preemptible" in result.output
    assert _find_call(stream_calls, "terraform", "apply", "-auto-approve") is None


def test_up_skips_the_gpu_quota_gate_for_preemptible_nodes(
    monkeypatch, tmp_path: Path
) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        "\n".join(
            [
                'parent_id = "project-a"',
                'tenant_id = "tenant-a"',
                'region = "us-central1"',
                "gpu_nodes_count = 1",
                'gpu_nodes_preset = "1gpu-24vcpu-218gb"',
                "gpu_nodes_preemptible = true",
            ]
        )
        + "\n"
    )
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        if args[:3] == ["terraform", "output", "-json"]:
            return _completed(
                json.dumps(
                    {"kube_cluster": {"value": {"id": "mk8scluster-a", "name": "c"}}}
                )
            )
        if args[:4] == ["nebius", "quotas", "quota-allowance", "get-by-name"]:
            raise AssertionError(
                "preemptible nodes must not consult the on-demand quota"
            )
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", _successful_stream(tf_dir, stream_calls))
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)
    monkeypatch.setattr(tf_mod, "save_cluster_state", lambda state, metadata=None: None)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code == 0, result.output
    assert _find_call(stream_calls, "terraform", "apply", "-auto-approve")


def test_up_explains_what_may_exist_after_an_interrupt(
    monkeypatch, tmp_path: Path
) -> None:
    """Ctrl-C used to leave a running cluster with no kubeconfig and no guidance."""
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'parent_id = "project-a"\ntenant_id = "tenant-a"\nregion = "us-central1"\n'
        'cluster_name = "npa-cluster"\n'
    )

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        raise AssertionError(args)

    def fake_stream(args, **kwargs):
        if args[:2] == ["terraform", "apply"]:
            raise KeyboardInterrupt
        return _completed()

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", fake_stream)
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code != 0
    assert "was interrupted" in result.output
    assert "npa cluster down" in result.output
    assert "npa cluster up" in result.output


def test_kubeconfig_cmd_adopts_a_running_cluster(monkeypatch, tmp_path: Path) -> None:
    """An interrupted `up` leaves a cluster running with no kubeconfig; adopt it."""
    stream_calls: list[list[str]] = []
    saved: list[object] = []

    def fake_capture(args, **kwargs):
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {"metadata": {"name": "other", "id": "mk8scluster-other"}},
                            {
                                "metadata": {
                                    "name": "npa-cluster",
                                    "id": "mk8scluster-live",
                                }
                            },
                        ]
                    }
                )
            )
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(
        tf_mod,
        "_run_stream",
        lambda args, **kwargs: stream_calls.append(args) or _completed(),
    )
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)
    monkeypatch.setattr(
        tf_mod, "save_cluster_state", lambda state, metadata=None: saved.append(state)
    )

    kubeconfig = tmp_path / "kubeconfig"
    result = runner.invoke(
        app,
        [
            "kubeconfig",
            "--cluster-name",
            "npa-cluster",
            "--project-id",
            "project-a",
            "--kubeconfig",
            str(kubeconfig),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "mk8scluster-live" in result.output
    assert str(kubeconfig) in result.output
    assert "--infra k8s/npa-cluster" in result.output
    credentials = _find_call(
        stream_calls, "nebius", "mk8s", "cluster", "get-credentials"
    )
    assert credentials is not None
    assert "mk8scluster-live" in credentials
    assert str(kubeconfig) in credentials
    assert saved and saved[-1].cluster_id == "mk8scluster-live"


def test_kubeconfig_cmd_names_what_exists_when_the_cluster_is_absent(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_capture(args, **kwargs):
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app,
        ["kubeconfig", "--cluster-name", "npa-cluster", "--project-id", "project-a"],
    )

    assert result.exit_code != 0
    assert "No Managed Kubernetes cluster named" in result.output
    assert "nebius mk8s cluster list" in result.output


def test_duplicate_cluster_guard_offers_adoption(monkeypatch, tmp_path: Path) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'parent_id = "project-a"\ncluster_name = "npa-cluster"\n'
    )

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {
                                    "name": "npa-cluster",
                                    "id": "mk8scluster-live",
                                }
                            }
                        ]
                    }
                )
            )
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", lambda args, **kwargs: _completed())
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code != 0
    assert "npa cluster kubeconfig" in result.output
    assert "mk8scluster-live" in result.output


def test_terminal_node_group_failure_reads_nested_events() -> None:
    """Nebius reports QuotaFailure under status.events[].last_occurrence.

    Regression: only top-level status keys were scanned, so the real reason never
    printed and the apply was never cancelled — the watcher looked like it worked
    while missing every actual refusal.
    """
    status = {
        "state": "PROVISIONING",
        "target_node_count": "1",
        "ready_node_count": "0",
        "events": [
            {
                "type": "QuotaFailure",
                "last_occurrence": (
                    "QuotaFailure: quota exceeded for compute.instance.gpu.rtx6000 in us-central1"
                ),
                "count": 12,
            }
        ],
    }

    assert "quota exceeded" in tf_mod.terminal_node_group_failure(status)
    line = tf_mod._format_node_group_status(status)
    assert "PROVISIONING (0/1 ready)" in line
    assert "QuotaFailure: quota exceeded" in line


def test_node_group_status_ignores_benign_nested_fields() -> None:
    status = {
        "state": "RUNNING",
        "target_node_count": "2",
        "ready_node_count": "2",
        "events": [{"type": "Scaled", "last_occurrence": "node group scaled to 2"}],
    }

    assert tf_mod.terminal_node_group_failure(status) == ""
    assert "scaled to 2" in tf_mod._format_node_group_status(status)


def test_node_group_instance_deletion_not_found_is_idempotent_progress() -> None:
    status = {
        "state": "DELETING",
        "target_node_count": "0",
        "ready_node_count": "0",
        "events": [
            {
                "type": "ComputeInstanceDeletionFailed",
                "last_occurrence": (
                    "rpc error: code = NotFound desc = compute instance was not found"
                ),
                "count": 1,
            }
        ],
    }

    line = tf_mod._format_node_group_status(status)

    assert "DELETING (0/0 ready)" in line
    assert "already absent" in line
    assert "idempotent" in line
    assert "ComputeInstanceDeletionFailed" not in line


def test_node_group_genuine_instance_deletion_failure_stays_visible() -> None:
    status = {
        "state": "DELETING",
        "target_node_count": "0",
        "ready_node_count": "1",
        "events": [
            {
                "type": "ComputeInstanceDeletionFailed",
                "last_occurrence": (
                    "ComputeInstanceDeletionFailed: PermissionDenied deleting compute instance"
                ),
                "count": 3,
            }
        ],
    }

    line = tf_mod._format_node_group_status(status)

    assert "ComputeInstanceDeletionFailed" in line
    assert "PermissionDenied" in line
    assert "already absent" not in line


def test_terminal_node_group_failure_detects_a_refusal() -> None:
    """QuotaFailure/no-capacity cannot be waited out; slow provisioning can."""
    assert "QuotaFailure" in tf_mod.terminal_node_group_failure(
        {
            "state": "PROVISIONING",
            "error_message": "QuotaFailure: rtx6000 limit reached",
        }
    )
    assert tf_mod.terminal_node_group_failure(
        {
            "state": "ERROR",
            "conditions": "InsufficientCapacity: no capacity in this zone",
        }
    )
    # A group that is merely slow, or failing for another reason, is left alone.
    assert (
        tf_mod.terminal_node_group_failure(
            {"state": "PROVISIONING", "ready_node_count": "0"}
        )
        == ""
    )
    assert tf_mod.terminal_node_group_failure({"state": "RUNNING"}) == ""
    assert (
        tf_mod.terminal_node_group_failure({"error_message": "node not ready yet"})
        == ""
    )
    assert tf_mod.terminal_node_group_failure({}) == ""


def test_run_stream_cancels_the_command_when_the_watcher_reports_a_refusal() -> None:
    """The apply is stopped instead of retrying to the Terraform timeout."""
    import pytest

    reasons = iter(["", "", "node group gpu: QuotaFailure"])

    with pytest.raises(Exception, match="QuotaFailure"):
        tf_mod._run_stream(
            ["sleep", "60"],
            cancel=lambda: next(reasons, "node group gpu: QuotaFailure"),
        )


def test_run_stream_returns_normally_when_nothing_cancels() -> None:
    result = tf_mod._run_stream(["true"], cancel=lambda: "")

    assert result.returncode == 0


def test_up_cancels_the_apply_on_a_refused_node_group(
    monkeypatch, tmp_path: Path
) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        "\n".join(
            [
                'parent_id = "project-a"',
                'tenant_id = "tenant-a"',
                'region = "us-central1"',
                'cluster_name = "npa-cluster"',
                "gpu_nodes_count = 1",
                'gpu_nodes_preset = "1gpu-24vcpu-218gb"',
            ]
        )
        + "\n"
    )

    cluster_lists: list[int] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            # Empty for the pre-apply duplicate guard; present once apply is
            # creating it, which is when the watcher looks it up.
            cluster_lists.append(1)
            if len(cluster_lists) == 1:
                return _completed('{"items":[]}')
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {"metadata": {"name": "npa-cluster", "id": "mk8scluster-a"}}
                        ]
                    }
                )
            )
        if args[:4] == ["nebius", "mk8s", "node-group", "list"]:
            return _completed(
                json.dumps(
                    {
                        "items": [
                            {
                                "metadata": {"name": "npa-cluster-ng-gpu-0"},
                                "status": {
                                    "state": "PROVISIONING",
                                    "target_node_count": "1",
                                    "ready_node_count": "0",
                                    "error_message": "QuotaFailure: compute.instance.gpu.rtx6000",
                                },
                            }
                        ]
                    }
                )
            )
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        if args[:4] == ["nebius", "quotas", "quota-allowance", "get-by-name"]:
            # Quota reads fine and has headroom, so the preflight passes and the
            # refusal only shows up once the node group is being created.
            return _completed(
                json.dumps({"spec": {"limit": "8"}, "status": {"usage": "0"}})
            )
        raise AssertionError(args)

    def fake_stream(args, **kwargs):
        cancel = kwargs.get("cancel")
        if args[:2] == ["terraform", "apply"] and cancel is not None:
            # Simulate the watcher observing the refusal mid-apply.
            watcher = _watchers[-1]
            watcher._poll()
            reason = cancel()
            assert reason, "the watcher should have reported the refusal"
            raise tf_mod.typer.BadParameter(f"Cancelled `terraform apply`: {reason}")
        return _completed()

    _watchers: list[object] = []
    original_watcher = tf_mod._NodeGroupWatcher

    class RecordingWatcher(original_watcher):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _watchers.append(self)

        def start(self) -> None:  # no background thread in the test
            return

    monkeypatch.setattr(tf_mod, "_NodeGroupWatcher", RecordingWatcher)
    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", fake_stream)
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code != 0
    assert "QuotaFailure" in result.output
    # And the operator is told what exists and how to clean it up.
    assert "npa cluster down" in result.output


def test_up_warns_when_the_gpu_quota_cannot_be_read(
    monkeypatch, tmp_path: Path
) -> None:
    """Skipping the check silently is what let the hang be a surprise."""
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        "\n".join(
            [
                'parent_id = "project-a"',
                'tenant_id = "tenant-a"',
                'region = "us-central1"',
                "gpu_nodes_count = 1",
                'gpu_nodes_preset = "1gpu-24vcpu-218gb"',
            ]
        )
        + "\n"
    )
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        if args[:4] == ["nebius", "quotas", "quota-allowance", "get-by-name"]:
            return _completed("", returncode=1)  # e.g. PermissionDenied
        if args[:3] == ["terraform", "output", "-json"]:
            return _completed(
                json.dumps(
                    {"kube_cluster": {"value": {"id": "mk8scluster-a", "name": "c"}}}
                )
            )
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", _successful_stream(tf_dir, stream_calls))
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)
    monkeypatch.setattr(tf_mod, "save_cluster_state", lambda state, metadata=None: None)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    # Unreadable quota never blocks a provision, but it is no longer silent.
    assert result.exit_code == 0, result.output
    assert "not quota-checked" in result.output
    assert _find_call(stream_calls, "terraform", "apply", "-auto-approve")


def test_node_group_status_keeps_failure_text() -> None:
    """QuotaFailure is not in the documented schema, so pass it through verbatim."""
    line = tf_mod._format_node_group_status(
        {
            "state": "PROVISIONING",
            "target_node_count": "1",
            "ready_node_count": "0",
            "error_message": "QuotaFailure: compute.instance.gpu.rtx6000 limit reached",
        }
    )

    assert "PROVISIONING (0/1 ready)" in line
    assert "QuotaFailure" in line


def test_up_pins_an_existing_ssh_public_key(monkeypatch, tmp_path: Path) -> None:
    """The module rejects a key path that does not exist; ~/.ssh/id_rsa.pub often does not."""
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text('parent_id = "project-test"\n')
    key = tmp_path / "keys" / "id_ed25519.pub"
    key.parent.mkdir()
    key.write_text("ssh-ed25519 AAAAC3Nz test@example\n")
    monkeypatch.setenv("NPA_SSH_PUBLIC_KEY", str(key))
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        if args[:3] == ["terraform", "output", "-json"]:
            return _completed(
                json.dumps(
                    {"kube_cluster": {"value": {"id": "mk8scluster-a", "name": "c"}}}
                )
            )
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", _successful_stream(tf_dir, stream_calls))
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)
    monkeypatch.setattr(tf_mod, "save_cluster_state", lambda state, metadata=None: None)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code == 0, result.output
    apply_call = _find_call(stream_calls, "terraform", "apply", "-auto-approve")
    assert apply_call is not None
    assert f'ssh_public_key={{path="{key}"}}' in apply_call


def test_up_keeps_an_explicit_ssh_public_key_from_tfvars(
    monkeypatch, tmp_path: Path
) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        'parent_id = "project-test"\n'
        'ssh_public_key = { path = "~/.ssh/custom.pub" }\n'
    )
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        if args[:3] == ["terraform", "output", "-json"]:
            return _completed(
                json.dumps(
                    {"kube_cluster": {"value": {"id": "mk8scluster-a", "name": "c"}}}
                )
            )
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", _successful_stream(tf_dir, stream_calls))
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)
    monkeypatch.setattr(tf_mod, "save_cluster_state", lambda state, metadata=None: None)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code == 0, result.output
    apply_call = _find_call(stream_calls, "terraform", "apply", "-auto-approve")
    assert apply_call is not None
    assert not any(arg.startswith("ssh_public_key=") for arg in apply_call)


def test_read_tfvars_keeps_complete_multiline_object_assignment(tmp_path: Path) -> None:
    tf_dir = tmp_path / "terraform"
    tf_dir.mkdir()
    (tf_dir / "terraform.tfvars").write_text(
        """
ssh_public_key = {
  path = "~/.ssh/custom.pub"
  metadata = {
    owner = "operator"
  }
}
region = "eu-north1"
""",
        encoding="utf-8",
    )

    values = tf_mod._read_tfvars(tf_dir)

    assert values["ssh_public_key"].startswith("{")
    assert 'owner = "operator"' in values["ssh_public_key"]
    assert values["ssh_public_key"].rstrip().endswith("}")
    assert values["region"] == "eu-north1"


def test_up_explains_a_missing_ssh_public_key(monkeypatch, tmp_path: Path) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text('parent_id = "project-test"\n')
    monkeypatch.setenv("NPA_SSH_PUBLIC_KEY", str(tmp_path / "absent.pub"))
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
        if args[:2] == ["terraform", "version"]:
            return _completed(json.dumps({"terraform_version": "1.12.2"}))
        if args[:3] == ["nebius", "iam", "get-access-token"]:
            return _completed("token-a\n")
        if args[:4] == ["nebius", "mk8s", "cluster", "list"]:
            return _completed('{"items":[]}')
        if args[:2] == ["terraform", "state"]:
            return _completed("", returncode=1)
        raise AssertionError(args)

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_stream", _successful_stream(tf_dir, stream_calls))
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code != 0
    assert "No SSH public key found" in result.output
    assert "ssh-keygen -t ed25519" in result.output
    assert _find_call(stream_calls, "terraform", "apply", "-auto-approve") is None


def test_tfvar_bool_reads_false_strings_from_the_environment() -> None:
    """TF_VAR_* values are strings and bool("false") is True."""
    env = {"TF_VAR_enable_filestore": "false"}
    assert tf_mod._tfvar_bool({}, env, "enable_filestore", True) is False
    assert (
        tf_mod._tfvar_bool(
            {}, {"TF_VAR_enable_filestore": "1"}, "enable_filestore", False
        )
        is True
    )
    assert (
        tf_mod._tfvar_bool({"enable_filestore": True}, {}, "enable_filestore", False)
        is True
    )
    assert tf_mod._tfvar_bool({}, {}, "enable_filestore", False) is False


def test_shared_filesystem_requested_covers_existing_filestore() -> None:
    """existing_filestore implies enable_filestore in deploy/cluster/main.tf."""
    assert (
        tf_mod._shared_filesystem_requested(
            {"existing_filestore": "computefilesystem-a"}, {}
        )
        is True
    )
    assert tf_mod._shared_filesystem_requested({"existing_filestore": ""}, {}) is False
    assert (
        tf_mod._shared_filesystem_requested(
            {},
            {"TF_VAR_enable_filestore": "false"},
        )
        is False
    )


def test_terraform_env_refreshes_stale_token_by_default(monkeypatch) -> None:
    # A stale ambient token must NOT shadow the freshly minted profile token.
    monkeypatch.setenv("TF_VAR_iam_token", "stale-cloud-env-token")
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "stale-cloud-env-token")
    monkeypatch.delenv("NPA_REUSE_IAM_TOKEN", raising=False)

    captured_env: dict[str, str] = {}

    def fake_capture(args, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        return _completed("fresh-token\n")

    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    env = tf_mod._terraform_env("nebius")

    assert env["TF_VAR_iam_token"] == "fresh-token"
    assert env["NEBIUS_IAM_TOKEN"] == "fresh-token"
    # The stale token is cleared before minting so it cannot leak into the mint call.
    assert captured_env.get("TF_VAR_iam_token") in (None, "")
    assert captured_env.get("NEBIUS_IAM_TOKEN") in (None, "")


def test_terraform_env_reuses_token_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("TF_VAR_iam_token", "intentional-ci-token")
    monkeypatch.setenv("NPA_REUSE_IAM_TOKEN", "1")

    def fail_capture(args, **kwargs):
        raise AssertionError("must not mint a new token when reuse is opted in")

    monkeypatch.setattr(tf_mod, "_run_capture", fail_capture)

    env = tf_mod._terraform_env("nebius")

    assert env["TF_VAR_iam_token"] == "intentional-ci-token"


def test_terraform_env_mints_when_no_token_present(monkeypatch) -> None:
    monkeypatch.delenv("TF_VAR_iam_token", raising=False)
    monkeypatch.delenv("NEBIUS_IAM_TOKEN", raising=False)
    monkeypatch.setenv("NPA_REUSE_IAM_TOKEN", "1")

    monkeypatch.setattr(
        tf_mod, "_run_capture", lambda *a, **k: _completed("minted-token\n")
    )

    env = tf_mod._terraform_env("nebius")

    assert env["TF_VAR_iam_token"] == "minted-token"
