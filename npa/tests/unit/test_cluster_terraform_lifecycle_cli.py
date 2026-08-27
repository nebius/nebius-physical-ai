from __future__ import annotations

import json
import subprocess
import sys
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
    assert (
        "gpu_nodes_driverfull_image      = local.gpu_nodes_driverfull_image" in main_tf
    )
    assert "gpu_nodes_driver_preset         = var.managed_driver_preset" in main_tf
    assert "gpu_nodes_driverfull_image      = false" not in main_tf


def test_cluster_terraform_wires_operator_filesystem_csi_repository() -> None:
    cluster_dir = Path(__file__).resolve().parents[3] / "deploy" / "cluster"

    assert (
        "chart_repository                    = var.filesystem_csi_chart_repository"
        in (cluster_dir / "main.tf").read_text()
    )
    assert 'variable "filesystem_csi_chart_repository"' in (
        cluster_dir / "variables.tf"
    ).read_text()


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


def test_run_stream_capture_output_is_visible_and_retained(capsys) -> None:
    result = tf_mod._run_stream(
        [
            sys.executable,
            "-c",
            "import sys; print('check-visible'); print('check-detail', file=sys.stderr)",
        ],
        capture_output=True,
    )

    visible = capsys.readouterr()
    assert "check-visible" in visible.out
    assert "check-detail" in visible.err
    assert "check-visible" in result.stdout
    assert "check-detail" in result.stderr


def test_cluster_failure_message_redacts_provider_secret() -> None:
    message = tf_mod._redacted_exception_message(
        "cluster up failed", RuntimeError("iam_token=provider-secret")
    )
    assert "provider-secret" not in message
    assert "<redacted>" in message


def test_skypilot_smoke_scopes_check_and_uses_explicit_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    streams: list[tuple[list[str], dict[str, str], Path]] = []
    monkeypatch.setattr(tf_mod, "_require_bin", lambda value: value)

    def stream(cmd, **kwargs):  # noqa: ANN001
        streams.append((cmd, kwargs["env"], kwargs["cwd"]))
        output = "Kubernetes: enabled [compute]\n" if cmd[1] == "check" else ""
        return _completed(output)

    monkeypatch.setattr(tf_mod, "_run_stream", stream)
    monkeypatch.setattr(tf_mod, "_wait_for_sky_down", lambda *_args, **_kwargs: None)

    tf_mod._run_skypilot_smoke(
        kubeconfig,
        "fleet-exact",
        "provider-cluster",
        "RTXPRO6000:1",
        sky_bin="/opt/npa/sky",
    )

    assert streams[0][0] == [
        "/opt/npa/sky",
        "check",
        "--config",
        'kubernetes.allowed_contexts=["fleet-exact"]',
        "kubernetes",
    ]
    assert streams[0][1]["KUBECONFIG"] == str(kubeconfig)
    assert streams[0][2] == kubeconfig.parent
    launch = streams[1][0]
    assert launch[0:2] == ["/opt/npa/sky", "launch"]
    assert launch[launch.index("--config") + 1] == (
        'kubernetes.allowed_contexts=["fleet-exact"]'
    )
    assert launch[launch.index("--gpus") + 1] == "RTXPRO6000:1"
    assert streams[1][2] == kubeconfig.parent
    down = streams[2][0]
    assert down[0:2] == ["/opt/npa/sky", "down"]
    assert down[down.index("--config") + 1] == (
        'kubernetes.allowed_contexts=["fleet-exact"]'
    )
    assert streams[2][2] == kubeconfig.parent


def test_skypilot_auto_detection_uses_exact_context_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[list[str], Path | None]] = []

    def capture(cmd, **kwargs):  # noqa: ANN001
        seen.append((cmd, kwargs.get("cwd")))
        return _completed("RTXPRO-6000-BLACKWELL-SERVER-EDITION  1  1 of 1 free\n")

    monkeypatch.setattr(tf_mod, "_run_capture", capture)
    accelerator = tf_mod._detect_skypilot_gpu(
        "/opt/npa/sky",
        "k8s/fleet-exact",
        {},
        config_override='kubernetes.allowed_contexts=["fleet-exact"]',
        cwd=Path("/durable/sky"),
    )

    assert accelerator == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    assert seen == [
        (
            [
                "/opt/npa/sky",
                "show-gpus",
                "--config",
                'kubernetes.allowed_contexts=["fleet-exact"]',
                "--infra",
                "k8s/fleet-exact",
                "--all",
            ],
            Path("/durable/sky"),
        )
    ]


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
                'filesystem_csi_chart_repository = "oci://charts.example.invalid/nebius"',
                'subnet_id = "subnet-a"',
            ]
        )
        + "\n"
    )
    stream_calls: list[list[str]] = []
    stream_envs: list[dict[str, str]] = []
    gpu_events: list[tuple[str, object]] = []

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
                                },
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
                                },
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
        "npa.orchestration.skypilot.k8s_gpu_catalog.wait_for_kubernetes_accelerators",
        lambda accelerators, **kwargs: (
            gpu_events.append(("readiness", (accelerators, kwargs))) or {}
        ),
    )
    monkeypatch.setattr(
        tf_mod,
        "_run_skypilot_smoke",
        lambda *args, **kwargs: gpu_events.append(("smoke", (args, kwargs))),
    )
    monkeypatch.setattr(
        tf_mod,
        "_check_skypilot_kubernetes",
        lambda *args, **kwargs: gpu_events.append(("check", (args, kwargs))),
    )
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
            "--sky-smoke",
            "--sky-gpus",
            "RTXPRO6000:1",
            "--sky-bin",
            "/opt/npa/sky",
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
    assert [event[0] for event in gpu_events] == ["check", "readiness", "smoke"]
    check_args, check_kwargs = gpu_events[0][1]
    assert check_args[1] == "cluster-a"
    assert check_kwargs["sky_bin"] == "/opt/npa/sky"
    readiness_args, readiness_kwargs = gpu_events[1][1]
    assert readiness_args == ["RTXPRO6000:1"]
    assert readiness_kwargs["label_known_gpus"] is True
    assert readiness_kwargs["sky_bin"] == "/opt/npa/sky"
    _smoke_args, smoke_kwargs = gpu_events[2][1]
    assert smoke_kwargs["sky_bin"] == "/opt/npa/sky"
    assert smoke_kwargs["credentials_checked"] is True


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
    assert down.call_args.kwargs["operation_id"] == ""


def test_validate_cluster_accepts_compute_csi_when_filestore_is_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    responses = {
        ("kubectl", "get", "nodes", "-o", "json"): {
            "items": [
                {
                    "metadata": {
                        "name": "gpu-0",
                        "labels": {"node.kubernetes.io/instance-type": "gpu-rtx6000"},
                    },
                    "status": {
                        "conditions": [{"type": "Ready", "status": "True"}],
                        "allocatable": {"nvidia.com/gpu": "1"},
                        "nodeInfo": {"bootID": "boot-a"},
                    },
                }
            ]
        },
        ("kubectl", "get", "pods", "-n", "nvidia-device-plugin", "-o", "json"): {
            "items": [
                {
                    "metadata": {"name": "device-plugin"},
                    "status": {
                        "phase": "Running",
                        "containerStatuses": [{"ready": True}],
                    },
                }
            ]
        },
        ("kubectl", "get", "storageclass", "-o", "json"): {
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
                'filesystem_csi_chart_repository = "oci://charts.example.invalid/nebius"',
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
                'filesystem_csi_chart_repository = "oci://charts.example.invalid/nebius"',
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


@pytest.mark.parametrize(
    "tfvars",
    [
        {"enable_filestore": True},
        {"existing_filestore": "filesystem-a"},
    ],
)
def test_filestore_preflight_requires_csi_repository(
    tfvars: dict[str, object],
) -> None:
    with pytest.raises(
        tf_mod.typer.BadParameter,
        match="filesystem_csi_chart_repository",
    ):
        tf_mod._preflight_filestore_quota("nebius", tfvars, {})


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
                                },
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


def _save_legacy_cluster_ownership(
    monkeypatch, tmp_path: Path, *, metadata: object
) -> None:
    from npa.cluster import state as state_module

    monkeypatch.setattr(state_module, "CLUSTERS_DIR", tmp_path / "clusters")
    state_module.save_cluster_state(
        state_module.ClusterState(
            name="selected-cluster",
            cluster_id="cluster-a",
            project_id="project-a",
            region="us-central1",
            node_count=1,
            node_platform="cpu-d3",
            node_preset="8vcpu-32gb",
            k8s_version="1.34",
            subnet_id="subnet-a",
            created_at="2026-01-01T00:00:00Z",
        )
    )
    state_module.metadata_file("selected-cluster").write_text(json.dumps(metadata))


def test_down_fails_closed_on_unreadable_present_ownership_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfstate").write_text("{}")
    (tf_dir / "terraform.tfvars").write_text(
        'parent_id = "project-a"\ncluster_name = "selected-cluster"\n'
    )
    _save_legacy_cluster_ownership(monkeypatch, tmp_path, metadata={})
    from npa.cluster import state as state_module

    state_module.metadata_file("selected-cluster").write_text("not-json")
    monkeypatch.setattr(
        tf_mod,
        "_run_stream",
        lambda *_args, **_kwargs: pytest.fail("Terraform must not run"),
    )

    result = runner.invoke(app, ["down", "--terraform-dir", str(tf_dir), "--force"])

    assert result.exit_code != 0
    assert "ownership metadata is unreadable" in result.output


def test_down_requires_legacy_terraform_state_to_match_exact_cluster_id(
    monkeypatch, tmp_path: Path
) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfstate").write_text("{}")
    (tf_dir / "terraform.tfvars").write_text(
        'parent_id = "project-a"\ncluster_name = "selected-cluster"\n'
    )
    _save_legacy_cluster_ownership(
        monkeypatch,
        tmp_path,
        metadata={"managed_by": "npa cluster terraform"},
    )
    stream_calls: list[list[str]] = []
    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_preflight_terraform_version", lambda *_args: None)
    monkeypatch.setattr(tf_mod, "_terraform_env", lambda _binary: {})

    def fake_capture(args, **_kwargs):
        if args[:3] == ["terraform", "state", "pull"]:
            return _completed(
                json.dumps(
                    {"outputs": {"kube_cluster": {"value": {"id": "cluster-b"}}}}
                )
            )
        return _completed()

    def fake_stream(args, **_kwargs):
        stream_calls.append(args)
        if args[:2] == ["terraform", "destroy"]:
            pytest.fail("mismatched Terraform state must not be destroyed")
        return _completed()

    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)
    monkeypatch.setattr(tf_mod, "_run_stream", fake_stream)

    result = runner.invoke(
        app,
        ["down", "--terraform-dir", str(tf_dir), "--force"],
        terminal_width=200,
    )

    assert result.exit_code != 0
    output = " ".join(result.output.replace("│", " ").split())
    assert "does not own exactly the persisted cluster ID" in output
    assert not _find_call(stream_calls, "terraform", "destroy")


@pytest.mark.parametrize(
    "state_text, expected",
    [
        ("", "requires readable retained Terraform state"),
        ("not-json", "Terraform state is malformed"),
        (json.dumps({"resources": []}), "lacks valid lineage/resource ownership"),
    ],
)
def test_residual_recovery_rejects_missing_or_malformed_state(
    monkeypatch, tmp_path: Path, state_text: str, expected: str
) -> None:
    monkeypatch.setattr(
        tf_mod,
        "_run_capture",
        lambda *_args, **_kwargs: _completed(state_text),
    )

    with pytest.raises(tf_mod.typer.BadParameter, match=expected):
        tf_mod._verify_residual_terraform_ownership(
            "terraform",
            tmp_path,
            {},
            project_id="project-a",
            cluster_id="cluster-a",
        )


def test_residual_recovery_rejects_foreign_project_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    state = {
        "lineage": "lineage-a",
        "serial": 4,
        "resources": [
            {
                "mode": "managed",
                "type": "nebius_vpc_v1_subnet",
                "instances": [
                    {"attributes": {"id": "subnet-a", "parent_id": "project-b"}}
                ],
            }
        ],
    }
    monkeypatch.setattr(
        tf_mod,
        "_run_capture",
        lambda *_args, **_kwargs: _completed(json.dumps(state)),
    )

    with pytest.raises(
        tf_mod.typer.BadParameter, match="do not match the requested project"
    ):
        tf_mod._verify_residual_terraform_ownership(
            "terraform",
            tmp_path,
            {},
            project_id="project-a",
            cluster_id="cluster-a",
        )


def test_residual_recovery_accepts_exact_cluster_scoped_node_group(
    monkeypatch, tmp_path: Path
) -> None:
    state = {
        "lineage": "lineage-a",
        "serial": 5,
        "resources": [
            {
                "mode": "managed",
                "type": "nebius_mk8s_v1_node_group",
                "instances": [
                    {
                        "attributes": {
                            "id": "node-group-a",
                            "parent_id": "cluster-a",
                        }
                    }
                ],
            },
            {
                "mode": "managed",
                "type": "nebius_applications_v1alpha1_k8s_release",
                "instances": [
                    {
                        "attributes": {
                            "id": "release-a",
                            "parent_id": "project-a",
                            "cluster_id": "cluster-a",
                        }
                    }
                ],
            },
        ],
    }
    monkeypatch.setattr(
        tf_mod,
        "_run_capture",
        lambda *_args, **_kwargs: _completed(json.dumps(state)),
    )

    assert tf_mod._verify_residual_terraform_ownership(
        "terraform",
        tmp_path,
        {},
        project_id="project-a",
        cluster_id="cluster-a",
    ) == [
        "nebius_applications_v1alpha1_k8s_release",
        "nebius_mk8s_v1_node_group",
    ]


def test_residual_recovery_ignores_zero_instance_cluster_block(
    monkeypatch, tmp_path: Path
) -> None:
    state = {
        "lineage": "lineage-a",
        "serial": 6,
        "resources": [
            {
                "mode": "managed",
                "type": "nebius_mk8s_v1_cluster",
                "instances": [],
            },
            {
                "mode": "managed",
                "type": "nebius_mk8s_v1_node_group",
                "instances": [
                    {
                        "attributes": {
                            "id": "node-group-a",
                            "parent_id": "cluster-a",
                        }
                    }
                ],
            },
        ],
    }
    monkeypatch.setattr(
        tf_mod,
        "_run_capture",
        lambda *_args, **_kwargs: _completed(json.dumps(state)),
    )

    assert tf_mod._verify_residual_terraform_ownership(
        "terraform",
        tmp_path,
        {},
        project_id="project-a",
        cluster_id="cluster-a",
    ) == ["nebius_mk8s_v1_node_group"]


@pytest.mark.parametrize(
    "resource_type, attributes, expected",
    [
        (
            "nebius_mk8s_v1_node_group",
            {"id": "node-group-a", "parent_id": "cluster-other"},
            "exact persisted cluster",
        ),
        (
            "nebius_vpc_v1_network",
            {"id": "network-a"},
            "requested project",
        ),
        (
            "nebius_unknown_v1_resource",
            {"id": "unknown-a", "parent_id": "project-a"},
            "does not recognize Nebius residual type",
        ),
        (
            "nebius_applications_v1alpha1_k8s_release",
            {
                "id": "release-a",
                "parent_id": "project-a",
                "cluster_id": "cluster-other",
            },
            "application-release ownership",
        ),
    ],
)
def test_residual_recovery_rejects_wrong_missing_or_unknown_nebius_evidence(
    monkeypatch,
    tmp_path: Path,
    resource_type: str,
    attributes: dict[str, str],
    expected: str,
) -> None:
    state = {
        "lineage": "lineage-a",
        "serial": 6,
        "resources": [
            {
                "mode": "managed",
                "type": resource_type,
                "instances": [{"attributes": attributes}],
            }
        ],
    }
    monkeypatch.setattr(
        tf_mod,
        "_run_capture",
        lambda *_args, **_kwargs: _completed(json.dumps(state)),
    )

    with pytest.raises(tf_mod.typer.BadParameter, match=expected):
        tf_mod._verify_residual_terraform_ownership(
            "terraform",
            tmp_path,
            {},
            project_id="project-a",
            cluster_id="cluster-a",
        )


@pytest.mark.parametrize("destroy_fails", [False, True])
def test_down_recovers_provider_absent_cluster_residuals_and_retains_on_failure(
    monkeypatch, tmp_path: Path, destroy_fails: bool
) -> None:
    from npa.cluster import api as api_module
    from npa.cluster import state as state_module
    from npa.cluster.exceptions import ClusterNotFoundError

    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfstate").write_text("{}")
    (tf_dir / "terraform.tfvars").write_text(
        'parent_id = "project-a"\ncluster_name = "selected-cluster"\n'
    )
    _save_legacy_cluster_ownership(
        monkeypatch,
        tmp_path,
        metadata={"managed_by": "npa cluster terraform"},
    )
    residual_state = {
        "lineage": "lineage-a",
        "serial": 8,
        "resources": [
            {
                "mode": "managed",
                "type": "nebius_vpc_v1_network",
                "instances": [
                    {"attributes": {"id": "network-a", "parent_id": "project-a"}}
                ],
            },
            {
                "mode": "managed",
                "type": "nebius_vpc_v1_subnet",
                "instances": [
                    {"attributes": {"id": "subnet-a", "parent_id": "project-a"}}
                ],
            },
            {
                "mode": "managed",
                "type": "nebius_mk8s_v1_node_group",
                "instances": [
                    {
                        "attributes": {
                            "id": "node-group-a",
                            "parent_id": "cluster-a",
                        }
                    }
                ],
            },
        ],
    }
    calls: list[list[str]] = []

    class MissingClusterClient:
        def __init__(self, **_kwargs):
            pass

        def get_cluster(self, *_args, **_kwargs):
            raise ClusterNotFoundError("absent")

    monkeypatch.setattr(api_module, "MK8sClient", MissingClusterClient)
    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_preflight_terraform_version", lambda *_args: None)
    monkeypatch.setattr(tf_mod, "_terraform_env", lambda _binary: {})
    monkeypatch.setattr(
        tf_mod,
        "_run_capture",
        lambda args, **_kwargs: (
            _completed(json.dumps(residual_state))
            if args[:3] == ["terraform", "state", "pull"]
            else _completed()
        ),
    )

    def stream(args, **_kwargs):
        calls.append(args)
        if destroy_fails and args[:2] == ["terraform", "destroy"]:
            raise RuntimeError("retained residual destroy failed")
        return _completed()

    monkeypatch.setattr(tf_mod, "_run_stream", stream)
    result = runner.invoke(
        app,
        ["down", "--terraform-dir", str(tf_dir), "--force"],
        terminal_width=200,
    )

    if destroy_fails:
        assert result.exit_code != 0
        assert state_module.cluster_dir("selected-cluster").exists()
    else:
        assert result.exit_code == 0, result.output
        assert not state_module.cluster_dir("selected-cluster").exists()
    assert _find_call(calls, "terraform", "destroy", "-auto-approve")


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


def test_down_terminalizes_pre_mutation_operation_without_provider_calls(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.provisioning_journal import ProvisioningOperation

    journal_dir = tmp_path / "operations"
    monkeypatch.setenv("NPA_OPERATION_JOURNAL_DIR", str(journal_dir))
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    operation = ProvisioningOperation.prepare(
        command="npa cluster up",
        project_alias="demo",
        project_id="project-demo",
        tenant_id="tenant-demo",
        region="eu-test1",
        backend={"kind": "local-state", "terraform_dir": str(tf_dir)},
        resource_type="cluster",
        requested_name="never-mutated",
        ownership_source="cluster-terraform",
        resume_command="npa cluster up --project demo --context never-mutated",
        destroy_command="npa cluster down --project demo --context never-mutated --force",
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("pre-mutation recovery crossed an external boundary")

    monkeypatch.setattr(tf_mod, "_require_bin", unexpected)
    monkeypatch.setattr(tf_mod, "_terraform_env", unexpected)
    monkeypatch.setattr(tf_mod, "_report_drain_blockers", unexpected)
    monkeypatch.setattr(tf_mod, "_run_stream", unexpected)
    monkeypatch.setattr(tf_mod, "_run_capture", unexpected)

    result = runner.invoke(
        app,
        [
            "down",
            "--terraform-dir",
            str(tf_dir),
            "--project-id",
            "project-demo",
            "--tenant-id",
            "tenant-demo",
            "--region",
            "eu-test1",
            "--context",
            "never-mutated",
            "--operation-id",
            operation.operation_id,
            "--force",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["outcome"] == "already_absent"
    assert payload["no_op"] is True
    assert payload["resources_removed"] == []
    assert operation.read()["phase"] == "destroyed"
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
    assert saved[-1].name == "npa-cluster"
    assert saved[-1].provider_name == "npa-cluster"


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
        'parent_id = "project-test"\nssh_public_key = { path = "~/.ssh/custom.pub" }\n'
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


def test_fresh_shared_up_resolves_subnet_and_uses_id_backed_project(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cluster_backends.base import MaterializedPlan
    from npa.fleet import lifecycle

    tf_dir = tmp_path / "deploy" / "cluster"
    recipe = tf_dir / "vendor" / "nebius-solutions-library" / "k8s-training"
    recipe.mkdir(parents=True)
    (recipe / "variables.tf").write_text('variable "cluster_name" { type = string }\n')
    (recipe.parent / "modules").mkdir()
    (tf_dir / "terraform.tfvars").write_text(
        "\n".join(
            [
                'tenant_id = "tenant-test"',
                'parent_id = "project-test"',
                'region = "region-test"',
                'cluster_name = "fresh"',
                "cpu_nodes_count = 1",
                "gpu_nodes_count = 0",
            ]
        )
    )
    kubeconfig = tmp_path / "fresh-kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")
    applied = {}
    apply_requests = []
    preflight_requests = []
    network_calls = []
    events = []

    class Adapter:
        def preflight(self, desired, request):
            events.append("preflight")
            preflight_requests.append(request)
            return {"backend": "mk8s"}

        def materialize(self, desired, request):
            return MaterializedPlan("mk8s", {}, {})

        def apply(self, desired, request):
            events.append("apply")
            applied["desired"] = desired
            applied["request"] = request
            apply_requests.append(request)
            return {
                "status": "deployed",
                "cluster_id": "cluster-test",
                "kubeconfig": str(kubeconfig),
            }

    monkeypatch.setattr(tf_mod, "get_backend", lambda _name: Adapter())
    monkeypatch.setattr(tf_mod, "_require_bin", lambda value: value)
    monkeypatch.setattr(tf_mod, "_preflight_terraform_version", lambda *_args: None)
    monkeypatch.setattr(
        tf_mod,
        "_terraform_env",
        lambda _bin: {
            "TF_VAR_tenant_id": "tenant-test",
            "TF_VAR_parent_id": "project-test",
            "TF_VAR_region": "region-test",
            "TF_VAR_enable_filestore": "true",
            "TF_VAR_filesystem_csi_chart_repository": (
                "oci://charts.example.invalid/nebius"
            ),
        },
    )

    def ensure(*_args, **kwargs):
        events.append("ensure-subnet")
        network_calls.append(kwargs)
        return "subnet-created", "network-created"

    monkeypatch.setattr(lifecycle, "ensure_subnet", ensure)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-sky-smoke"],
    )

    assert result.exit_code == 0, result.output
    assert applied["desired"].subnet_id == "subnet-created"
    assert applied["desired"].filesystem_csi_chart_repository == (
        "oci://charts.example.invalid/nebius"
    )
    assert applied["request"].subnet_id == "subnet-created"
    assert applied["request"].project.name == ""
    assert applied["request"].project.project_id == "project-test"
    assert applied["request"].post_deploy_validation == "standalone-full"
    assert preflight_requests[0].provider_preflight is True
    assert network_calls[0]["network_state_path"].name == ".npa-fleet-network.json"
    assert events == ["preflight", "ensure-subnet", "apply"]

    skipped = runner.invoke(
        app,
        [
            "up",
            "--terraform-dir",
            str(tf_dir),
            "--skip-validate",
            "--skip-sky-smoke",
        ],
    )
    assert skipped.exit_code == 0, skipped.output
    assert apply_requests[-1].post_deploy_validation == "skip"
    assert "Post-deploy validation skipped" in skipped.output


def test_fresh_shared_up_does_not_create_network_when_capacity_preflight_fails(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.cluster_backends.base import MaterializedPlan
    from npa.fleet import lifecycle

    tf_dir = tmp_path / "deploy" / "cluster"
    recipe = tf_dir / "vendor" / "nebius-solutions-library" / "k8s-training"
    recipe.mkdir(parents=True)
    (recipe / "variables.tf").write_text('variable "cluster_name" { type = string }\n')
    (recipe.parent / "modules").mkdir()
    (tf_dir / "terraform.tfvars").write_text(
        'tenant_id = "tenant-test"\n'
        'parent_id = "project-test"\n'
        'region = "region-test"\n'
        'cluster_name = "fresh"\n'
        "cpu_nodes_count = 1\n"
        "gpu_nodes_count = 0\n"
    )

    class Adapter:
        def preflight(self, desired, request):
            return {"backend": "mk8s"}

        def materialize(self, desired, request):
            return MaterializedPlan("mk8s", {}, {})

    monkeypatch.setattr(tf_mod, "get_backend", lambda _name: Adapter())
    monkeypatch.setattr(tf_mod, "_require_bin", lambda value: value)
    monkeypatch.setattr(tf_mod, "_preflight_terraform_version", lambda *_args: None)
    monkeypatch.setattr(
        tf_mod,
        "_terraform_env",
        lambda _bin: {
            "TF_VAR_tenant_id": "tenant-test",
            "TF_VAR_parent_id": "project-test",
            "TF_VAR_region": "region-test",
        },
    )
    monkeypatch.setattr(
        tf_mod,
        "_preflight_whole_path_capacity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("quota blocked")),
    )
    monkeypatch.setattr(
        lifecycle,
        "ensure_subnet",
        lambda *_args, **_kwargs: pytest.fail(
            "network mutation must follow successful capacity preflight"
        ),
    )

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-sky-smoke"],
    )

    assert result.exit_code != 0
    assert "quota blocked" in result.output
