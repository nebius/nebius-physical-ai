from __future__ import annotations

import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from npa.cli.cluster import app
from npa.cli.cluster import terraform_lifecycle as tf_mod


runner = CliRunner()


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_up_runs_terraform_writes_kubeconfig_and_validates(monkeypatch, tmp_path: Path) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        "\n".join(
            [
                'parent_id = "project-a"',
                'tenant_id = "tenant-a"',
                'region = "region-a"',
                'cluster_name = "cluster-a"',
                'gpu_nodes_count = 2',
                'gpu_nodes_preset = "8gpu-192vcpu-1744gb"',
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
        return _completed()

    def fake_capture(args, **kwargs):
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
                                "endpoints": {"public_endpoint": "https://cluster.example"},
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
                                "status": {
                                    "conditions": [{"type": "Ready", "status": "True"}],
                                    "allocatable": {"nvidia.com/gpu": "8"},
                                }
                            },
                            {
                                "status": {
                                    "conditions": [{"type": "Ready", "status": "True"}],
                                    "allocatable": {"nvidia.com/gpu": "8"},
                                }
                            },
                        ]
                    }
                )
            )
        if args[:4] == ["kubectl", "get", "pods", "-n"]:
            return _completed(json.dumps({"items": [{"status": {"phase": "Running"}}]}))
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
    monkeypatch.setattr(tf_mod, "save_cluster_state", lambda state, metadata=None: saved.append(state))

    result = runner.invoke(
        app,
        [
            "up",
            "--terraform-dir",
            str(tf_dir),
            "--capacity-block-group",
            "capacityblockgroup-test",
            "--skip-sky-smoke",
        ],
    )

    assert result.exit_code == 0, result.output
    assert ["terraform", "init"] in stream_calls
    assert ["terraform", "apply", "-auto-approve"] in stream_calls
    apply_env = stream_envs[stream_calls.index(["terraform", "apply", "-auto-approve"])]
    assert apply_env["TF_VAR_capacity_block_group"] == "capacityblockgroup-test"
    assert any(call[:4] == ["nebius", "mk8s", "cluster", "get-credentials"] for call in stream_calls)
    assert saved[-1].cluster_id == "mk8scluster-a"
    assert "16 allocatable GPUs" in result.output


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


def test_up_allows_duplicate_managed_by_terraform_state(monkeypatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(tf_mod, "_run_stream", lambda args, **kwargs: stream_calls.append(args) or _completed())
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)
    monkeypatch.setattr(tf_mod, "save_cluster_state", lambda state, metadata=None: None)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code == 0, result.output
    assert ["terraform", "apply", "-auto-approve"] in stream_calls


def test_up_stops_when_filestore_quota_is_too_small(monkeypatch, tmp_path: Path) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    (tf_dir / "terraform.tfvars").write_text(
        "\n".join(
            [
                'parent_id = "project-a"',
                'tenant_id = "tenant-a"',
                'region = "region-a"',
                'cluster_name = "cluster-a"',
                'filestore_disk_size_gibibytes = 1024',
            ]
        )
        + "\n"
    )
    stream_calls: list[list[str]] = []

    def fake_capture(args, **kwargs):
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
    monkeypatch.setattr(tf_mod, "_run_stream", lambda args, **kwargs: stream_calls.append(args) or _completed())
    monkeypatch.setattr(tf_mod, "_run_capture", fake_capture)

    result = runner.invoke(
        app,
        ["up", "--terraform-dir", str(tf_dir), "--skip-validate", "--skip-sky-smoke"],
    )

    assert result.exit_code != 0
    assert "Shared filesystem quota is insufficient" in result.output
    assert ["terraform", "apply", "-auto-approve"] not in stream_calls


def test_down_runs_terraform_destroy(monkeypatch, tmp_path: Path) -> None:
    tf_dir = tmp_path / "deploy" / "cluster"
    tf_dir.mkdir(parents=True)
    stream_calls: list[list[str]] = []

    monkeypatch.setattr(tf_mod, "_require_bin", lambda binary: binary)
    monkeypatch.setattr(tf_mod, "_run_capture", lambda *args, **kwargs: _completed("token-a\n"))

    def fake_stream(args, **kwargs):
        stream_calls.append(args)
        return _completed()

    monkeypatch.setattr(tf_mod, "_run_stream", fake_stream)

    result = runner.invoke(app, ["down", "--terraform-dir", str(tf_dir), "--force"])

    assert result.exit_code == 0, result.output
    assert ["terraform", "init"] in stream_calls
    assert ["terraform", "destroy", "-auto-approve"] in stream_calls


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
