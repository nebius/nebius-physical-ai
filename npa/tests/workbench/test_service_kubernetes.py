"""The in-cluster LanceDB service.

`deploy` could reach a local docker daemon, a managed VM, or LanceDB Cloud — none of which a
workflow stage can use, because a stage runs in a pod and a pod cannot resolve an operator's
localhost. Two templates could not retire for exactly that reason: both `bdd100k-pipeline` and
`dataset-ingest-curate` have a stage that writes to LanceDB, and it failed live with
`[Errno -2] Name or service not known` (EVIDENCE §R16).
"""

from __future__ import annotations

import json
import subprocess

import pytest

from npa.workbench import service_kubernetes as k8s


def _manifests(**overrides):
    kwargs = {
        "name": "npa-lancedb",
        "port": 8686,
        "image": "registry.example/npa-lancedb:1.2.3",
        "storage_path": "s3://bucket/lancedb/",
        "service_env": {"LANCEDB_STORAGE_PATH": "s3://bucket/lancedb/"},
        "secret_name": "npa-lancedb-storage",
        "storage_endpoint_url": "https://storage.example.com",
    }
    kwargs.update(overrides)
    return k8s.build_manifests(**kwargs)


def test_the_endpoint_is_resolvable_from_any_pod_in_the_cluster() -> None:
    assert (
        k8s.service_endpoint("npa-lancedb", "default", 8686)
        == "http://npa-lancedb.default.svc.cluster.local:8686"
    )


def test_a_deployment_and_a_service_are_produced() -> None:
    deployment, service = _manifests()

    assert deployment["kind"] == "Deployment"
    assert service["kind"] == "Service"
    # The Service selects the Deployment's pods; a typo here yields an endpoint that resolves
    # and answers nothing.
    assert service["spec"]["selector"] == deployment["spec"]["selector"]["matchLabels"]
    assert service["spec"]["type"] == "ClusterIP"


def test_credentials_come_from_a_secret_not_the_manifest() -> None:
    """`kubectl get deploy -o yaml` is readable by anyone with namespace read access."""

    deployment, _ = _manifests()
    env = deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    by_name = {entry["name"]: entry for entry in env}

    for key in k8s.STORAGE_SECRET_ENVS:
        assert "value" not in by_name[key], (
            f"{key} must not be a literal in the manifest"
        )
        assert (
            by_name[key]["valueFrom"]["secretKeyRef"]["name"] == "npa-lancedb-storage"
        )


def test_no_secret_env_when_storage_is_local() -> None:
    deployment, _ = _manifests(storage_path="/var/lib/lancedb", secret_name="")
    env = deployment["spec"]["template"]["spec"]["containers"][0]["env"]

    assert not any(entry["name"] in k8s.STORAGE_SECRET_ENVS for entry in env)


def test_readiness_gates_the_service_endpoints() -> None:
    """A stage must never resolve the name to a pod that is still opening its storage."""

    deployment, _ = _manifests()
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["port"] == 8686


def test_everything_created_is_labelled_so_destroy_can_find_only_its_own() -> None:
    for manifest in _manifests():
        labels = manifest["metadata"]["labels"]
        assert labels["app.kubernetes.io/managed-by"] == "npa"


def test_image_pull_secrets_are_wired_when_given() -> None:
    deployment, _ = _manifests(image_pull_secrets=("private-ghcr",))

    assert deployment["spec"]["template"]["spec"]["imagePullSecrets"] == [
        {"name": "private-ghcr"}
    ]


def test_build_rejects_a_missing_image_or_storage_path() -> None:
    with pytest.raises(k8s.ServiceKubernetesError, match="image reference"):
        k8s.build_manifests(name="x", port=1, image="", service_env={})
    with pytest.raises(k8s.ServiceKubernetesError, match="image reference"):
        k8s.build_manifests(name="x", port=1, image="   ", service_env={})


def test_apply_sends_one_list_document_and_surfaces_failure() -> None:
    seen: dict[str, object] = {}

    def ok(args, stdin=None, timeout=300):
        seen["args"] = args
        seen["stdin"] = stdin
        return subprocess.CompletedProcess(args, 0, stdout="created\n", stderr="")

    assert k8s.apply(_manifests(), runner=ok) == "created"
    assert seen["args"] == ["apply", "-f", "-"]
    payload = json.loads(str(seen["stdin"]))
    assert payload["kind"] == "List" and len(payload["items"]) == 2

    def broken(args, stdin=None, timeout=300):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="nope")

    with pytest.raises(k8s.ServiceKubernetesError, match="nope"):
        k8s.apply(_manifests(), runner=broken)


def test_wait_uses_rollout_status_not_condition_available() -> None:
    """Live: a deploy shipped a broken image and reported "running".

    `--for=condition=Available` is satisfied by the OLD ReplicaSet's healthy pod during a
    rolling update, so the new pod crash-looped unnoticed and the next run failed against the
    old code. `rollout status` waits for the new ReplicaSet.
    """

    seen: dict[str, object] = {}

    def ok(args, stdin=None, timeout=300):
        seen["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="rolled out", stderr="")

    k8s.wait_available("npa-lancedb", "default", runner=ok)

    assert seen["args"][:2] == ["rollout", "status"]  # type: ignore[index]
    assert "--for=condition=Available" not in seen["args"]  # type: ignore[operator]


def test_wait_reports_the_timeout_it_used() -> None:
    def never(args, stdin=None, timeout=300):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="timed out")

    with pytest.raises(k8s.ServiceKubernetesError, match="within 30s"):
        k8s.wait_available("npa-lancedb", "default", timeout_seconds=30, runner=never)


def test_destroy_targets_only_npa_managed_objects() -> None:
    seen: dict[str, object] = {}

    def ok(args, stdin=None, timeout=300):
        seen["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="deleted", stderr="")

    k8s.destroy("npa-lancedb", "default", runner=ok)

    assert "app=npa-lancedb,app.kubernetes.io/managed-by=npa" in seen["args"]  # type: ignore[operator]
    assert "--ignore-not-found" in seen["args"]  # type: ignore[operator]


def test_the_storage_secret_refuses_to_be_created_empty() -> None:
    with pytest.raises(k8s.ServiceKubernetesError, match="AWS_SECRET_ACCESS_KEY"):
        k8s.ensure_storage_secret(
            "npa-lancedb-storage",
            "default",
            {"AWS_ACCESS_KEY_ID": "key"},
            runner=lambda *a, **k: subprocess.CompletedProcess(
                [], 0, stdout="", stderr=""
            ),
        )


def test_registry_host_is_detected_for_a_private_image() -> None:
    assert (
        k8s.registry_host("registry-us.example/ns/npa-lancedb:1")
        == "registry-us.example"
    )
    assert k8s.registry_host("localhost:5000/npa-lancedb:1") == "localhost:5000"
    # A bare name is Docker Hub, which needs no pull secret here.
    assert k8s.registry_host("npa-lancedb:1") == ""
    assert k8s.registry_host("library/npa-lancedb:1") == ""


def test_auto_auth_is_none_when_no_token_is_configured(monkeypatch) -> None:
    """Live: `auto` meant `token`, no token existed, and /health answered 500 forever.

    The readiness probe could never pass, so the Deployment never became Available and the
    deploy timed out with no hint about why. A ClusterIP Service is not reachable off-cluster,
    so `auto` means "token if the operator supplied one".
    """

    from npa.cli.workbench.lancedb.deploy import _auth_mode_for
    from npa.cli.workbench.lancedb.helpers import DEFAULT_TOKEN_ENV, LanceDBRuntime

    monkeypatch.delenv(DEFAULT_TOKEN_ENV, raising=False)
    assert _auth_mode_for(LanceDBRuntime.kubernetes, "auto") == "none"

    monkeypatch.setenv(DEFAULT_TOKEN_ENV, "s3cr3t")
    assert _auth_mode_for(LanceDBRuntime.kubernetes, "auto") == "token"


def test_explicit_token_mode_is_still_honoured(monkeypatch) -> None:
    from npa.cli.workbench.lancedb.deploy import _auth_mode_for
    from npa.cli.workbench.lancedb.helpers import DEFAULT_TOKEN_ENV, LanceDBRuntime

    monkeypatch.delenv(DEFAULT_TOKEN_ENV, raising=False)
    assert _auth_mode_for(LanceDBRuntime.kubernetes, "token") == "token"
