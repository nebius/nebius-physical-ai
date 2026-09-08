"""Keep Nano's public image separate from operator-fetched serving payloads."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "npa/docker/workbench/cosmos3-nano-video"


def _pins(path):
    return dict(re.findall(r"(?m)^([a-z0-9-]+)==([^\s]+)", path.read_text()))


def test_nano_runtime_lock_preserves_parent_dependency_versions():
    lock = IMAGE_DIR / "requirements.lock"
    serving = IMAGE_DIR.parent / "cosmos3-serving/requirements.lock"
    extras, parent = _pins(lock), _pins(serving)
    assert extras["ray"] == "2.56.0"
    # Historical vendor-image overrides are not Ray Serve dependencies.
    assert "nltk" not in extras
    assert all(extras[name] == parent[name] for name in extras.keys() & parent.keys())
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    assert f"NPA_COSMOS3_NANO_CLOSURE_SHA256={digest}" in (IMAGE_DIR / "Dockerfile").read_text()


def test_nano_public_layer_closure_contains_only_bootstrap_and_adapters():
    docker = (IMAGE_DIR / "Dockerfile").read_text()
    assert "npa-cosmos3-serving@sha256:3342bbe44bd1c00ebf05ab4c9d7286058a94bb5ce90b49b164b23604d3acf180" in docker
    assert "vllm/vllm-omni:" not in docker
    assert "-r /opt/npa-cosmos3-serving/packaging-requirements.txt" in docker
    assert not re.search(r"pip install[^\n]*(torch|vllm|cuda|ray)", docker)
    assert 'npa.redistribution="public"' in docker
    assert "USER 10001:10001" in docker
    assert "cosmos3-super-benchmark/prepare_guardrail_runtime.py" in docker
    bootstrap = (IMAGE_DIR / "runtime_bootstrap.sh").read_text()
    assert "flock 9" in bootstrap
    assert "--require-hashes" in bootstrap
    assert "sha256sum -c -" in bootstrap
    assert "NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES" not in docker


def test_cluster_stages_and_shares_runtime_without_public_pull_credentials():
    deploy = ROOT / "npa/deploy/cosmos3-nano-video"
    job = yaml.safe_load((deploy / "weights-job.yaml").read_text())["spec"]["template"]["spec"]
    service = yaml.safe_load((deploy / "rayservice.yaml").read_text())
    cluster = service["spec"]["rayClusterConfig"]
    pods = [job, cluster["headGroupSpec"]["template"]["spec"]]
    pods += [item["template"]["spec"] for item in cluster["workerGroupSpecs"]]
    assert job["containers"][0]["command"][0] == "/usr/local/bin/npa-cosmos3-nano-bootstrap"
    for pod in pods:
        assert "imagePullSecrets" not in pod
        container = pod["containers"][0]
        mount = next(item for item in container["volumeMounts"] if item["mountPath"] == "/opt/npa-cosmos3-serving/runtime")
        assert mount["subPath"] == "runtime"
        acceptance = next(item for item in container["env"] if item["name"] == "NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE")
        assert acceptance["valueFrom"]["secretKeyRef"] == {
            "name": "cosmos3-nano-video-runtime", "key": "license-acceptance",
            "optional": True,
        }
