"""Detector deploy must prove the exact executing target before Kubernetes writes."""

import json
import subprocess
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import EnvironmentConfig, StorageConfig
from npa.clients.credentials import CredentialsConfig
from npa.orchestration.skypilot.k8s_gpu_catalog import discover_kubernetes_gpu_inventory as discover_actual_inventory


@pytest.fixture
def deployment(monkeypatch, tmp_path):
    from npa.cli.workbench import detection_training as cli
    from npa.cluster import identity, state
    from npa.clients import config, nebius, storage_validation
    from npa.orchestration.npa_workflow import submit_credentials
    from npa.orchestration.skypilot import k8s_gpu_catalog

    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("synthetic")
    selected = SimpleNamespace(project_alias="project-fixture", project_id="project-test", context="context-fixture", kubeconfig=kubeconfig, cluster_absent=False)
    calls = {"mutations": [], "probes": [], "secret": None, "identity": []}
    def cluster(**kwargs):
        calls["identity"].append(kwargs)
        return selected
    monkeypatch.setattr(identity, "resolve_verified_cluster_identity", cluster)
    monkeypatch.setattr(state, "load_cluster_state", lambda context: SimpleNamespace(project_id="project-test"))
    monkeypatch.setattr(config, "resolve_environment", lambda project: EnvironmentConfig(project_id="project-test", tenant_id="tenant-test", region="eu-north1"))
    monkeypatch.setattr(nebius, "get_project_identity", lambda *args, **kwargs: SimpleNamespace(project_id="project-test", tenant_id="tenant-test", region="eu-north1"))
    monkeypatch.setattr(nebius, "get_bucket_by_name", lambda project_id, bucket: {"metadata": {"name": bucket, "parent_id": project_id}})
    monkeypatch.setattr(submit_credentials, "resolve_project_storage", lambda project: StorageConfig(checkpoint_bucket="synthetic", endpoint_url="https://storage.eu-north1.nebius.cloud", aws_access_key_id="project-access", aws_secret_access_key="project-secret"))
    monkeypatch.setattr(submit_credentials, "load_credentials", lambda **kwargs: CredentialsConfig(s3_access_key_id="global-access", s3_secret_access_key="global-secret"))
    def probe(**kwargs):
        calls["probes"].append(kwargs)
        return SimpleNamespace(ok=True, retained_object=False)
    monkeypatch.setattr(storage_validation, "probe_storage_write", probe)
    inventory = SimpleNamespace(error="", unbound_pending_gpu_pods=0, nodes=[SimpleNamespace(name="node-fixture", labels=(("gpu", "B200"),), ready=True, schedulable=True, capacity=8, allocatable=8, committed=0, free=8, products=("B200",))])
    def discover(**kwargs):
        kwargs["runner"](["kubectl", "get", "pods", "--all-namespaces"], capture_output=True, text=True)
        return inventory
    monkeypatch.setattr(k8s_gpu_catalog, "discover_kubernetes_gpu_inventory", discover)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout='{"items":[]}', stderr=""))
    def secret(**kwargs):
        calls["mutations"].append("secret")
        calls["secret"] = kwargs
    monkeypatch.setattr(cli, "_provision_service_secret", secret)
    monkeypatch.setattr(cli, "_kubectl", lambda *args, **kwargs: calls["mutations"].append("kubectl") or "")
    monkeypatch.setenv("DETECTION_TRAINING_TOKEN", "synthetic-bearer")
    args = ["workbench", "detection-training", "deploy", "--project", "project-fixture", "--cluster-name", "context-fixture", "--kubeconfig", str(kubeconfig), "--output-path", "s3://synthetic/task/detection", "--gpu-type", "b200", "--node-selector-key", "gpu", "--node-selector-value", "B200", "--image", "example/detector:test", "--output", "json"]
    return SimpleNamespace(args=args, calls=calls, inventory=inventory, selected=selected)


def test_deploy_probes_and_injects_one_project_scoped_principal(deployment):
    result = CliRunner().invoke(app, deployment.args)
    assert result.exit_code == 0, str(result.exception)
    assert deployment.calls["mutations"] == ["secret", "kubectl", "kubectl"]
    assert len(deployment.calls["probes"]) == 1
    probe = deployment.calls["probes"][0]
    assert probe["prefix"] == "task/detection"  # A directory even without a trailing slash.
    assert probe["access_key_id"] == "project-access"
    assert probe["secret_access_key"] == "project-secret"
    env = deployment.calls["secret"]["env"]
    assert env["AWS_ACCESS_KEY_ID"] == probe["access_key_id"]
    assert env["AWS_SECRET_ACCESS_KEY"] == probe["secret_access_key"]
    assert env["AWS_ENDPOINT_URL_S3"] == env["AWS_ENDPOINT_URL"] == probe["endpoint_url"]
    assert env["AWS_REGION"] == "eu-north1"
    assert deployment.calls["secret"]["kubeconfig"] == str(deployment.selected.kubeconfig)
    assert "project-secret" not in result.output and "synthetic-bearer" not in result.output


@pytest.mark.parametrize("case", ["cluster", "owner", "prefix", "gpu_product", "gpu_mismatch", "gpu_unknown", "credential_pair", "endpoint"])
def test_deploy_scope_access_and_gpu_failures_precede_any_mutation(deployment, monkeypatch, case):
    from npa.cluster import identity
    from npa.clients import nebius, storage_validation
    if case == "cluster":
        def wrong_cluster(**kwargs):
            raise identity.ClusterIdentityError("synthetic private provider detail")
        monkeypatch.setattr(identity, "resolve_verified_cluster_identity", wrong_cluster)
    elif case == "owner":
        monkeypatch.setattr(nebius, "get_bucket_by_name", lambda *args: {"metadata": {"name": "synthetic", "parent_id": "foreign-project"}})
    elif case == "prefix":
        monkeypatch.setattr(storage_validation, "probe_storage_write", lambda **kwargs: SimpleNamespace(ok=False, phase="put", error=SimpleNamespace(kind=SimpleNamespace(value="permission_denied"))))
    elif case == "gpu_product":
        deployment.inventory.nodes = []
    elif case == "gpu_mismatch":
        deployment.inventory.nodes[0].products = ("RTXPRO6000",)
    elif case == "gpu_unknown":
        deployment.inventory.error = "private provider diagnostic"
    elif case == "credential_pair":
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "incomplete-access")
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    else:
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.eu-west1.nebius.cloud")
    result = CliRunner().invoke(app, deployment.args)
    assert result.exit_code != 0
    assert deployment.calls["mutations"] == []
    if case != "prefix":
        assert deployment.calls["probes"] == []
    assert "private provider" not in result.output
    assert "project-secret" not in result.output


@pytest.mark.parametrize("case", ["unrelated_busy", "owned_replacement", "spoofed_labels", "foreign_deployment_uid", "foreign_namespace", "unknown_pods", "incomplete_pods", "unknown_ownership", "pending_unbound", "rolling_update"])
def test_busy_gpu_requires_exact_owned_recreate_allocation_with_real_inventory(deployment, monkeypatch, case):
    from npa.orchestration.skypilot import k8s_gpu_catalog

    # Exercise real inventory commitment/free accounting, replacing only kubectl IO.
    monkeypatch.setattr(k8s_gpu_catalog, "discover_kubernetes_gpu_inventory", discover_actual_inventory)
    owner = {"kind": "ReplicaSet", "uid": "replicaset-current", "controller": True}
    pod = {
        "metadata": {"name": "pod-fixture", "uid": "pod-current", "namespace": "default", "labels": {"app.kubernetes.io/instance": "npa-detection-training"}, "ownerReferences": [owner]},
        "spec": {"nodeName": "node-fixture", "containers": [{"resources": {"requests": {"nvidia.com/gpu": "1"}}}]},
        "status": {"phase": "Running"},
    }
    node = {
        "metadata": {"name": "node-fixture", "labels": {"gpu": "B200", "nvidia.com/gpu.product": "NVIDIA-B200"}},
        "spec": {},
        "status": {"conditions": [{"type": "Ready", "status": "True"}], "capacity": {"nvidia.com/gpu": "1"}, "allocatable": {"nvidia.com/gpu": "1", "cpu": "8", "memory": "32Gi", "pods": "110"}},
    }
    existing = {"metadata": {"name": "npa-detection-training", "namespace": "default", "uid": "deployment-current"}, "spec": {"strategy": {"type": "Recreate"}}}
    replica_set = {"metadata": {"uid": "replicaset-current", "namespace": "default", "ownerReferences": [{"kind": "Deployment", "uid": "deployment-current", "controller": True}]}}
    if case == "unrelated_busy":
        existing = None
    elif case == "spoofed_labels":
        pod["metadata"]["ownerReferences"][0]["uid"] = "replicaset-foreign"
    elif case == "foreign_deployment_uid":
        replica_set["metadata"]["ownerReferences"][0]["uid"] = "deployment-foreign"
    elif case == "foreign_namespace":
        pod["metadata"]["namespace"] = "foreign-namespace"
    elif case == "pending_unbound":
        pod["spec"].pop("nodeName")
        pod["status"]["phase"] = "Pending"
    elif case == "rolling_update":
        existing["spec"]["strategy"]["type"] = "RollingUpdate"
    observations = []
    def kubectl(command, **kwargs):
        assert command[0] == "kubectl"
        assert command[command.index("--context") + 1] == deployment.selected.context
        assert command[command.index("--kubeconfig") + 1] == str(deployment.selected.kubeconfig)
        observations.append(command)
        if "nodes" in command:
            payload = {"items": [node]}
        elif "pods" in command:
            if case == "unknown_pods":
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="private provider evidence")
            payload = {} if case == "incomplete_pods" else {"items": [pod]}
        elif "deployment" in command:
            if case == "unknown_ownership":
                raise subprocess.CalledProcessError(1, command, stderr="private provider evidence")
            if existing is None:
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            payload = existing
        elif "replicasets" in command:
            payload = {"items": [replica_set]}
        else:
            raise AssertionError("unexpected kubectl operation")
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr(subprocess, "run", kubectl)
    result = CliRunner().invoke(app, deployment.args)
    assert any("pods" in command for command in observations)
    if case == "owned_replacement":
        assert result.exit_code == 0, str(result.exception)
        assert deployment.calls["mutations"] == ["secret", "kubectl", "kubectl"]
        assert any("replicasets" in command for command in observations)
    else:
        assert result.exit_code != 0
        assert deployment.calls["mutations"] == []
        assert deployment.calls["probes"] == []
    assert "private provider evidence" not in result.output
