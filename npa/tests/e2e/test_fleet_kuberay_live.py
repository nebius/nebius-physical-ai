"""Native Ray worker execution on an operator-selected, already deployed fleet."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import uuid

import pytest

from npa.cluster_backends.kuberay import RAY_IMAGE, RAY_VERSION
from npa.fleet.spec import load_spec

pytestmark = pytest.mark.e2e


def _record_command(evidence, name, argv, stdin=None):
    result = subprocess.run(argv, input=stdin, text=True, capture_output=True)
    path = evidence / (name + ".json")
    with open(path, "w", opener=lambda p, f: os.open(p, f, 0o600)) as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump({"argv": argv, "exit_code": result.returncode, "stdout": result.stdout,
                   "stderr": result.stderr}, stream)
    assert result.returncode == 0, f"{name} failed; inspect private evidence"
    return result.stdout


@contextmanager
def _job_cleanup(run, execute, directory, identity):
    errors = []
    try:
        yield
    except Exception as exc:
        errors.append(exc)
    finally:
        for name, args in [
            ("stop", ["ray", "job", "stop", "--address", "http://127.0.0.1:8265", identity]),
            ("remove-source", ["python", "-c", "import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); p.exists() and shutil.rmtree(p)", directory]),
        ]:
            try:
                run(name, execute + args)
            except Exception as exc:
                errors.append(exc)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise RuntimeError("Native proof and cleanup failures", errors) from errors[0]


def _configured_target():
    config_path = os.environ.get("NPA_FLEET_KUBERAY_LIVE_CONFIG")
    if not os.environ.get("NPA_FLEET_KUBERAY_LIVE_CONFIG"):
        pytest.skip("requires an owner-private spec, exact kubeconfig and evidence directory")
    config = json.loads(Path(config_path).read_text())
    spec = load_spec(config["spec"])
    targets = spec.cluster_targets()
    assert len(targets) == 1, "provide the exact one-target verification spec"
    _, cluster = targets[0]
    policy = cluster.kuberay
    assert policy and policy.enabled
    evidence = Path(config["evidence_dir"])
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert evidence.stat().st_mode & 0o077 == 0
    kubeconfig = Path(config["kubeconfig"])
    assert kubeconfig.is_file()
    return cluster, evidence, kubeconfig


def _verify_cluster_resource(run, policy):
    ray = json.loads(run("raycluster", ["get", "rayclusters", "-o", "json"]))["items"]
    assert len(ray) == 1
    resource = ray[0]
    assert resource["spec"]["rayVersion"] == RAY_VERSION
    assert resource["spec"]["enableInTreeAutoscaling"] is False
    head = resource["spec"]["headGroupSpec"]
    assert head["serviceType"] == "ClusterIP"
    assert head["rayStartParams"]["num-cpus"] == "0"
    groups = resource["spec"]["workerGroupSpecs"]
    assert len(groups) == 1
    group = groups[0]
    assert [group[k] for k in ("replicas", "minReplicas", "maxReplicas")] == [policy.worker_replicas] * 3
    assert group["rayStartParams"]["num-cpus"] == str(policy.worker_cpus)
    expected_resources = {"cpu": str(policy.worker_cpus), "memory": f"{policy.worker_memory_gib}Gi"}
    assert group["template"]["spec"]["containers"][0]["resources"] == {
        "requests": expected_resources, "limits": expected_resources,
    }
    return resource, expected_resources


def _verify_runtime_pods(run, cluster, resource, expected_resources):
    policy = cluster.kuberay
    pods = json.loads(run("pods", ["get", "pods", "-o", "json"]))["items"]
    ray_pods = [p for p in pods if p["metadata"]["labels"].get("ray.io/cluster") == resource["metadata"]["name"]]
    assert len(ray_pods) == policy.worker_replicas + 1
    head_pod = next(p for p in ray_pods if p["metadata"]["labels"]["ray.io/node-type"] == "head")
    for pod in ray_pods:
        assert pod["spec"]["automountServiceAccountToken"] is False
        assert pod["spec"]["nodeSelector"] == {"node.kubernetes.io/instance-type": cluster.cpu_nodes.platform}
        security = pod["spec"]["securityContext"]
        assert security["runAsNonRoot"] is True
        assert all(security[key] == 1000 for key in ("runAsUser", "runAsGroup", "fsGroup"))
        assert security["seccompProfile"] == {"type": "RuntimeDefault"}
        runtime = pod["spec"]["containers"][0]
        assert runtime["image"] == RAY_IMAGE
        assert runtime["securityContext"]["allowPrivilegeEscalation"] is False
        assert runtime["securityContext"]["capabilities"]["drop"] == ["ALL"]
        resources = {"cpu": "1", "memory": "4Gi"} if pod is head_pod else expected_resources
        assert runtime["resources"] == {"requests": resources, "limits": resources}
        assert all(s["ready"] for s in pod["status"]["containerStatuses"])
        assert RAY_IMAGE.split("@", 1)[1] in pod["status"]["containerStatuses"][0]["imageID"]
    return head_pod


def _verify_network_resources(run):
    policies = json.loads(run("networkpolicy", ["get", "networkpolicy", "ray-cluster-ingress", "-o", "json"]))
    assert policies["spec"]["ingress"] == [{"from": [{"podSelector": {}}]}]
    assert policies["spec"]["podSelector"] == {}
    assert policies["spec"]["policyTypes"] == ["Ingress"]
    services = json.loads(run("services", ["get", "services", "-o", "json"]))["items"]
    assert all(s["spec"]["type"] == "ClusterIP" for s in services)
    assert json.loads(run("pvcs", ["get", "pvc", "-o", "json"]))["items"] == []


def _verify_native_worker_job(run, head_pod, policy):
    pod_name = head_pod["metadata"]["name"]
    execute = ["exec", pod_name, "-c", "ray-head", "--"]
    run("ray-status", execute + ["ray", "status"])
    identity = "npa-cpu-proof-" + uuid.uuid4().hex
    directory = "/tmp/" + identity
    source = Path(__file__).resolve().parents[2] / "examples/fleet/kuberay/verify_workers.py"
    prepare = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'verify_workers.py').write_text(sys.stdin.read())"
    with _job_cleanup(run, execute, directory, identity):
        run("stage", ["exec", "-i", pod_name, "-c", "ray-head", "--", "python", "-c", prepare, directory], source.read_text())
        run("submit", execute + ["ray", "job", "submit", "--address", "http://127.0.0.1:8265", "--submission-id", identity,
                                 "--working-dir", directory, "--", "python", "verify_workers.py"])
        status = run("status", execute + ["ray", "job", "status", "--address", "http://127.0.0.1:8265", identity])
        assert f"Job '{identity}' succeeded" in status
        logs = run("logs", execute + ["ray", "job", "logs", "--address", "http://127.0.0.1:8265", identity])
        result = json.loads(next(line.split("KUBERAY_RESULT=", 1)[1] for line in logs.splitlines() if "KUBERAY_RESULT=" in line))
        assert result["ray_version"] == RAY_VERSION
        assert result["worker_count"] == policy.worker_replicas
        assert len({r["node_id"] for r in result["results"]}) == policy.worker_replicas
        assert all(r["node_id"] != result["head_node_id"] for r in result["results"])
        for index, shard in enumerate(result["results"]):
            assert shard["sum"] == sum(i * i for i in range(index * 10000, (index + 1) * 10000))


def test_fleet_kuberay_resources_and_native_worker_job():
    """Verify actual fleet resources and native work on every fixed CPU worker.

    Args:
        None. NPA_FLEET_KUBERAY_LIVE_CONFIG selects the private validation inputs.
    Returns:
        None.
    Raises:
        AssertionError: Deployed resources or native job results violate the contract.
        Exception: Private inputs, native commands or job cleanup fail.
    """
    cluster, evidence, kubeconfig = _configured_target()
    prefix = ["kubectl", "--kubeconfig", str(kubeconfig), "-n", "ray-cluster"]

    def run(name, arguments, stdin=None):
        return _record_command(evidence, name, prefix + arguments, stdin)

    resource, worker_resources = _verify_cluster_resource(run, cluster.kuberay)
    head_pod = _verify_runtime_pods(run, cluster, resource, worker_resources)
    _verify_network_resources(run)
    _verify_native_worker_job(run, head_pod, cluster.kuberay)
