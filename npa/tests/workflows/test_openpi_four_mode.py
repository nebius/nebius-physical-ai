from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.skypilot_render import secret_env_hints_for_plan
from npa.orchestration.npa_workflow.submit import load_spec_for_submit
from npa.workflows.byof import openpi_pipeline as pipeline
from npa.workflows.byof import openpi_service as service
from npa.workflows.byof import openpi_live
from npa.workflows.byof import openpi_service_rbac as service_rbac

ROOT = Path(__file__).resolve().parents[3]
SPEC = (
    ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "openpi-pi05-four-mode.yaml"
)
DIGEST_IMAGE = "registry.example.invalid/openpi@sha256:" + "a" * 64


def test_live_deployer_is_installed_publicly_and_reuses_tls_material() -> None:
    pyproject = tomllib.loads((ROOT / "npa/pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["npa-openpi-live-deploy"] == (
        "npa.workflows.byof.openpi_live:main"
    )
    source = Path(openpi_live.__file__).read_text(encoding="utf-8")
    assert 'encoded_tls["ca.crt"]' in source
    assert "if existing_auth is None:" in source
    assert "hmac.compare_digest" in openpi_live.gateway_program()
    assert 'state["total_connections"]' in openpi_live.gateway_program()


def test_terms_gate_exits_before_openpi_import_or_checkpoint_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    output = tmp_path / "success.json"
    diagnostics = tmp_path / "attempt-diagnostics"
    monkeypatch.delenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", raising=False)
    before = {
        name for name in sys.modules if name == "openpi" or name.startswith("openpi.")
    }

    with pytest.raises(SystemExit) as raised:
        pipeline.main(
            [
                "direct",
                "--output-uri",
                str(output),
                "--terms-diagnostic-root-uri",
                str(diagnostics),
                "--runtime-image",
                DIGEST_IMAGE,
            ]
        )

    assert raised.value.code == 64
    assert not output.exists()
    refusal_paths = list(diagnostics.glob("*.json"))
    assert len(refusal_paths) == 1
    refusal = json.loads(refusal_paths[0].read_text(encoding="utf-8"))
    assert all(
        refusal[key] == value for key, value in pipeline._terms_refusal().items()
    )
    assert refusal["declared_success_output_uri"] == str(output)
    assert refusal["stage"] == "direct"
    after = {
        name for name in sys.modules if name == "openpi" or name.startswith("openpi.")
    }
    assert after == before

    # An accepted retry under the same logical output prefix is not poisoned by
    # the refusal. Its declared success object remains independently writable.
    monkeypatch.setenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", "YES")
    pipeline._gate_or_exit(
        str(output), diagnostic_root_uri=str(diagnostics), stage="direct"
    )
    success = {"schema": "test.success.v1", "status": "passed"}
    pipeline._write_json_uri(str(output), success)
    assert json.loads(output.read_text(encoding="utf-8")) == success
    assert json.loads(refusal_paths[0].read_text(encoding="utf-8")) == refusal


def test_service_terms_refusal_writes_only_attempt_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    success = tmp_path / "service-success.json"
    diagnostic_root = tmp_path / "service-diagnostics"
    monkeypatch.delenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", raising=False)

    assert (
        service._run(
            SimpleNamespace(
                output_uri=str(success),
                terms_diagnostic_root_uri=str(diagnostic_root),
            )
        )
        == 64
    )

    assert not success.exists()
    diagnostics = list(diagnostic_root.glob("serve-*.json"))
    assert len(diagnostics) == 1
    refusal = json.loads(diagnostics[0].read_text(encoding="utf-8"))
    assert refusal["declared_success_output_uri"] == str(success)
    assert refusal["stage"] == "serve"


def test_negative_gate_persists_refusal_then_writes_accepted_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    success = tmp_path / "reports" / "negative-gate.json"
    diagnostics = tmp_path / "diagnostics" / "terms-refusals"
    monkeypatch.setenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", "YES")

    assert (
        pipeline.main(
            [
                "negative-gate",
                "--output-uri",
                str(success),
                "--terms-diagnostic-root-uri",
                str(diagnostics),
                "--runtime-image",
                DIGEST_IMAGE,
            ]
        )
        == 0
    )

    result = json.loads(success.read_text(encoding="utf-8"))
    refusal = result["tested_child_refusal"]
    assert result["accepted_retry_same_logical_output_uri"] is True
    assert refusal["declared_success_output_uri"] == str(success)
    assert refusal["declared_success_output_uri_untouched"] is True
    assert refusal["diagnostic_persistence"] == "separate_attempt_scoped_uri"
    diagnostic = Path(refusal["diagnostic_uri"])
    assert diagnostic.parent == diagnostics
    persisted = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert persisted["attempt_id"] == refusal["attempt_id"]
    assert persisted["status"] == "refused"


def test_mini_dataset_is_byte_reproducible_and_split_disjoint(tmp_path: Path) -> None:
    first_arrays, first_manifest = pipeline.build_mini_dataset(
        train_samples=4, heldout_samples=2, seed=20260815
    )
    second_arrays, second_manifest = pipeline.build_mini_dataset(
        train_samples=4, heldout_samples=2, seed=20260815
    )
    first = pipeline.deterministic_npz(first_arrays)
    second = pipeline.deterministic_npz(second_arrays)
    assert first == second
    assert first_manifest == second_manifest
    assert first_manifest["split_isolation"] == {
        "sample_id_intersection": [],
        "sample_hash_intersection": [],
        "disjoint": True,
    }

    archive = tmp_path / "dataset.npz"
    manifest = tmp_path / "manifest.json"
    archive.write_bytes(first)
    first_manifest["archive_sha256"] = pipeline._sha256_bytes(first)
    manifest.write_text(json.dumps(first_manifest), encoding="utf-8")
    loaded, loaded_manifest = pipeline._load_dataset(str(archive), str(manifest))
    assert loaded_manifest["archive_sha256"] == pipeline._sha256_bytes(first)
    assert np.asarray(loaded["train_actions"]).shape == (4, 15, 8)
    assert np.asarray(loaded["heldout_actions"]).shape == (2, 15, 8)
    assert np.asarray(loaded["train_exterior_image"]).dtype == np.uint8
    loaded["heldout_joint_position"][0, 0] += np.float32(0.25)
    with pytest.raises(pipeline.OpenPIPipelineError, match="content hashes"):
        pipeline._validate_dataset_arrays(loaded, loaded_manifest)


def test_dataset_missing_array_has_schema_context() -> None:
    arrays, manifest = pipeline.build_mini_dataset(
        train_samples=2, heldout_samples=1, seed=20260815
    )
    del arrays["heldout_actions"]

    with pytest.raises(
        pipeline.OpenPIPipelineError,
        match="missing required array 'heldout_actions'.*'heldout' split",
    ):
        pipeline._validate_dataset_arrays(arrays, manifest)


def test_optimizer_rng_is_folded_by_step_and_deterministic() -> None:
    class FakeRandom:
        @staticmethod
        def fold_in(key: str, step: int) -> tuple[str, int]:
            return key, step

    first = [
        pipeline._optimizer_step_rng(FakeRandom, "base-key", step) for step in range(3)
    ]
    second = [
        pipeline._optimizer_step_rng(FakeRandom, "base-key", step) for step in range(3)
    ]
    assert (
        first
        == second
        == [
            ("base-key", 0),
            ("base-key", 1),
            ("base-key", 2),
        ]
    )
    assert len(set(first)) == 3


def test_action_contract_requires_finite_float64_t_by_eight() -> None:
    valid = np.zeros((15, 8), dtype=np.float64)
    assert pipeline.validate_actions(valid, label="test") == {
        "shape": [15, 8],
        "dtype": "float64",
        "finite": True,
        "minimum_horizon_satisfied": True,
    }
    with pytest.raises(pipeline.OpenPIPipelineError, match="float64"):
        pipeline.validate_actions(valid.astype(np.float32), label="test")
    with pytest.raises(pipeline.OpenPIPipelineError, match="T>=5"):
        pipeline.validate_actions(np.zeros((4, 8), dtype=np.float64), label="test")
    invalid = valid.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(pipeline.OpenPIPipelineError, match="non-finite"):
        pipeline.validate_actions(invalid, label="test")


def test_redistribution_boundary_keeps_images_and_artifacts_private() -> None:
    assert pipeline._redistribution_evidence(trained_checkpoint=True) == {
        "runtime_image": "restricted_private_operator_registry",
        "openpi_source": "Apache-2.0",
        "base_checkpoint": "runtime_only_not_redistributed",
        "dataset": "private_operator_object_storage_only",
        "trained_checkpoint": "private_operator_object_storage_only",
    }


def _service_manifests() -> dict[str, dict]:
    return service.build_manifests(
        run_id="openpi-four-mode-contract",
        namespace="default",
        runtime_image=DIGEST_IMAGE,
        checkpoint_uri=pipeline.DEFAULT_CHECKPOINT_URI,
        config_name=pipeline.DEFAULT_CONFIG_NAME,
        gpu_count=1,
        expected_gpu_type="B200",
        expected_compute_capability="10.0",
        server_cpu="16",
        server_memory="96Gi",
        client_cpu="2",
        client_memory="8Gi",
        pull_secret="operator-byof-pull-secret",
        liveness_initial_delay_seconds=600,
        gpu_node_selector_key="nebius.com/gpu-name",
        gpu_node_selector_value="B200",
        cache_size="40Gi",
    )


def test_service_is_clusterip_digest_pinned_probed_and_cross_pod() -> None:
    manifests = _service_manifests()
    deployment = manifests["deployment"]
    service_manifest = manifests["service"]
    client = manifests["client_job"]
    server_container = deployment["spec"]["template"]["spec"]["containers"][0]
    client_container = client["spec"]["template"]["spec"]["containers"][0]

    assert set(manifests) == {"secret", "deployment", "service", "client_job"}
    assert service_manifest["spec"]["type"] == "ClusterIP"
    assert "Ingress" not in {item["kind"] for item in manifests.values()}
    assert server_container["image"] == DIGEST_IMAGE
    assert client_container["image"] == DIGEST_IMAGE
    assert server_container["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert "nvidia.com/gpu" not in client_container["resources"]["requests"]
    assert server_container["readinessProbe"]["httpGet"]["path"] == "/healthz"
    assert server_container["livenessProbe"]["httpGet"]["path"] == "/healthz"
    assert deployment["spec"]["progressDeadlineSeconds"] == 1200
    assert client["spec"]["activeDeadlineSeconds"] == 600
    assert client["spec"]["backoffLimit"] == 0
    assert deployment["spec"]["template"]["spec"]["nodeSelector"] == {
        "nebius.com/gpu-name": "B200"
    }
    assert client_container["command"][0] == "/opt/venv/bin/python"
    assert "range(2)" in client_container["command"][2]
    assert "/dev/termination-log" in client_container["command"][2]
    assert "NPA_OPENPI_SERVER_HARDWARE=" in server_container["command"][2]
    assert server_container["command"][2].startswith("set -euo pipefail;")
    assert "from npa." not in server_container["command"][2]
    compile(service._server_hardware_program(), "openpi-server-hardware", "exec")
    server_shell = server_container["command"][2]
    assert (
        server_shell.index('test "$NPA_OPENPI_ACCEPT_GEMMA_TERMS"')
        < (server_shell.index("NPA_OPENPI_SERVER_HARDWARE="))
        < server_shell.index("download.maybe_download")
        < server_shell.index("serve_policy.py")
    )
    server_env = {item["name"]: item for item in server_container["env"]}
    assert server_env["OPENPI_EXPECTED_GPU_TYPE"]["value"] == "B200"
    assert server_env["OPENPI_EXPECTED_GPU_COUNT"]["value"] == "1"
    assert server_env["OPENPI_EXPECTED_COMPUTE_CAPABILITY"]["value"] == "10.0"
    secret_ref = server_container["env"][0]["valueFrom"]["secretKeyRef"]
    assert secret_ref["key"] == "NPA_OPENPI_ACCEPT_GEMMA_TERMS"
    assert "value" not in server_container["env"][0]
    assert service.service_resource_names("openpi-four-mode-contract") == {
        key: manifest["metadata"]["name"] for key, manifest in manifests.items()
    }


def test_live_service_is_persistent_authenticated_and_cache_backed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "NPA_MODEL_CACHE_PVC",
        "NPA_MODEL_CACHE_HOST_PATH",
        "NPA_MODEL_CACHE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    manifests = openpi_live.build_live_manifests(
        run_id="live-contract",
        namespace="openpi-live",
        runtime_image=DIGEST_IMAGE,
        checkpoint_uri=pipeline.DEFAULT_CHECKPOINT_URI,
        config_name=pipeline.DEFAULT_CONFIG_NAME,
        expected_gpu_type="B200",
        expected_compute_capability="10.0",
        cache_pvc="runtime-weight-cache",
        auth_secret="run-auth",
        tls_secret="run-tls",
        kubelet_source_cidrs=("192.0.2.20/32", "2001:db8::20/128"),
        source_ranges=("192.0.2.10/32",),
    )

    assert set(manifests) == {
        "terms_secret",
        "deployment",
        "service",
        "network_policy",
    }
    deployment = manifests["deployment"]
    assert deployment["spec"]["strategy"] == {"type": "Recreate"}
    assert deployment["spec"]["replicas"] == 1
    assert (
        deployment["spec"]["template"]["metadata"]["labels"][
            "app.kubernetes.io/managed-by"
        ]
        == openpi_live.LIVE_MANAGED_BY
    )
    pod = deployment["spec"]["template"]["spec"]
    assert {
        "name": "openpi-cache",
        "persistentVolumeClaim": {"claimName": "runtime-weight-cache"},
    } in pod["volumes"]
    assert pod["securityContext"]["fsGroup"] == 1000
    assert pod["initContainers"][0]["securityContext"]["runAsUser"] == 0
    assert pod["initContainers"][0]["volumeMounts"] == [
        {"name": "openpi-cache", "mountPath": "/cache"}
    ]
    containers = {item["name"]: item for item in pod["containers"]}
    assert set(containers) == {"openpi-policy", "authenticated-gateway"}
    assert all(
        item["securityContext"]["runAsNonRoot"] is True for item in containers.values()
    )
    assert all(
        item["securityContext"]["runAsUser"] == 1000
        and item["securityContext"]["runAsGroup"] == 1000
        for item in containers.values()
    )
    gateway = containers["authenticated-gateway"]
    assert gateway["readinessProbe"]["httpGet"]["port"] == 8002
    assert gateway["livenessProbe"]["httpGet"]["port"] == 8002
    assert gateway["command"][:2] == ["/opt/venv/bin/python", "-c"]
    compile(gateway["command"][2], "openpi-authenticated-gateway", "exec")
    assert '("0.0.0.0", 8002)' in gateway["command"][2]
    assert "Authorization" in gateway["command"][2]
    assert "MAX_MESSAGE" in gateway["command"][2]
    assert "REQUEST_TIMEOUT" in gateway["command"][2]

    exposed = manifests["service"]["spec"]
    assert exposed["type"] == "LoadBalancer"
    assert exposed["externalTrafficPolicy"] == "Cluster"
    assert exposed["ports"] == [
        {"name": "wss", "protocol": "TCP", "port": 443, "targetPort": 8443}
    ]
    assert exposed["loadBalancerSourceRanges"] == ["192.0.2.10/32"]
    ingress = manifests["network_policy"]["spec"]["ingress"]
    assert ingress[0] == {"ports": [{"protocol": "TCP", "port": 8443}]}
    assert ingress[1]["from"] == [
        {"ipBlock": {"cidr": "192.0.2.20/32"}},
        {"ipBlock": {"cidr": "2001:db8::20/128"}},
    ]
    probe_ports = {
        int(container[probe]["httpGet"]["port"])
        for container in pod["containers"]
        for probe in ("readinessProbe", "livenessProbe")
    }
    assert probe_ports == {8000, 8002}
    assert {entry["port"] for entry in ingress[1]["ports"]} == probe_ports
    assert all(entry["protocol"] == "TCP" for entry in ingress[1]["ports"])
    assert manifests["network_policy"]["metadata"]["annotations"] == {
        "npa.nebius.ai/kubelet-probe-source-contract": "cilium-node-ip-ipblock",
        "npa.nebius.ai/kubelet-probe-host-network-tradeoff": "documented",
    }
    assert all("from" not in rule for rule in ingress[:1])
    assert all(
        rule.get("from")
        for rule in ingress
        if any(port["port"] in probe_ports for port in rule["ports"])
        and rule is not ingress[0]
    )
    rendered = openpi_live.public_contract(manifests)
    assert "api-key" in rendered  # mounted key name, never the key value
    assert '"stringData"' not in rendered
    assert '"name":"run-auth","namespace"' not in rendered
    assert '"name":"run-tls","namespace"' not in rendered


def test_live_kubelet_sources_are_exact_internal_node_addresses() -> None:
    nodes = [
        SimpleNamespace(
            status=SimpleNamespace(
                addresses=[
                    SimpleNamespace(type="Hostname", address="worker-a"),
                    SimpleNamespace(type="InternalIP", address="192.0.2.30"),
                    SimpleNamespace(type="InternalIP", address="2001:db8::30"),
                ]
            )
        )
    ]
    assert openpi_live._kubelet_source_cidrs(nodes) == (
        "192.0.2.30/32",
        "2001:db8::30/128",
    )
    with pytest.raises(openpi_live.OpenPIServiceError, match="empty"):
        openpi_live._kubelet_source_cidrs([])


def test_live_cache_uses_single_writer_block_storage_contract() -> None:
    pvc = openpi_live._cache_pvc_manifest(
        name="runtime-weight-cache",
        namespace="openpi-live",
        size="64Gi",
        storage_class="compute-csi-default-sc",
        labels={"app.kubernetes.io/managed-by": openpi_live.LIVE_MANAGED_BY},
    )

    assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]
    assert pvc["spec"]["storageClassName"] == "compute-csi-default-sc"
    assert pvc["spec"]["resources"]["requests"]["storage"] == "64Gi"


def test_live_client_bundle_is_private_and_certificate_matches_endpoint(
    tmp_path: Path,
) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    ca, certificate, private_key = openpi_live._certificate("192.0.2.22")
    assert b"BEGIN CERTIFICATE" in ca
    assert b"BEGIN CERTIFICATE" in certificate
    assert serialization.load_pem_private_key(private_key, password=None) is not None
    issuer = x509.load_pem_x509_certificate(ca)
    leaf = x509.load_pem_x509_certificate(certificate)
    issuer.public_key().verify(
        leaf.signature,
        leaf.tbs_certificate_bytes,
        padding.PKCS1v15(),
        leaf.signature_hash_algorithm,
    )
    assert leaf.fingerprint(hashes.SHA256())
    bundle = tmp_path / "client"
    openpi_live._write_client_bundle(
        bundle,
        endpoint="192.0.2.22",
        ca=ca,
        token="x" * 48,
    )
    assert bundle.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in bundle.iterdir())
    endpoint = json.loads((bundle / "endpoint.json").read_text(encoding="utf-8"))
    assert endpoint == {"scheme": "wss", "host": "192.0.2.22", "port": 443}
    assert (bundle / "api-key").read_text(encoding="utf-8").strip() == "x" * 48


def test_live_tls_rollout_digest_changes_with_rotated_material() -> None:
    first = openpi_live._certificate("192.0.2.22")
    second = openpi_live._certificate("192.0.2.22")

    assert openpi_live._tls_rollout_digest(*first) == openpi_live._tls_rollout_digest(
        *first
    )
    assert openpi_live._tls_rollout_digest(*first) != openpi_live._tls_rollout_digest(
        *second
    )


def test_live_apply_refuses_foreign_preexisting_object() -> None:
    foreign = SimpleNamespace(metadata=SimpleNamespace(labels={}))

    with pytest.raises(service.OpenPIServiceError, match="unowned"):
        openpi_live._apply_owned(
            read=lambda **_kwargs: foreign,
            create=lambda **_kwargs: None,
            patch=lambda **_kwargs: None,
            name="live",
            namespace="default",
            body={},
            run_id="live-contract",
        )


def test_service_server_hardware_log_is_machine_readable() -> None:
    value = service._prefixed_json(
        'startup\nNPA_OPENPI_SERVER_HARDWARE={"gpu_count_allocated": 1}\nready',
        "NPA_OPENPI_SERVER_HARDWARE=",
    )
    assert value == {"gpu_count_allocated": 1}
    client = service._prefixed_json(
        'NPA_OPENPI_CLIENT_RESULT={"request_count": 2, "status": "passed"}',
        "NPA_OPENPI_CLIENT_RESULT=",
    )
    assert client == {"request_count": 2, "status": "passed"}


def test_service_client_job_is_created_only_after_server_phase() -> None:
    calls: list[str] = []

    class Core:
        def create_namespaced_secret(self, _namespace: str, _body: dict) -> None:
            calls.append("secret")

        def create_namespaced_service(self, _namespace: str, _body: dict) -> None:
            calls.append("service")

        def read_namespaced_secret(self, _name: str, _namespace: str) -> None:
            raise _Gone()

        def read_namespaced_service(self, _name: str, _namespace: str) -> None:
            raise _Gone()

    class Apps:
        def create_namespaced_deployment(self, _namespace: str, _body: dict) -> None:
            calls.append("deployment")

        def read_namespaced_deployment(self, _name: str, _namespace: str) -> None:
            raise _Gone()

    class Batch:
        def create_namespaced_job(self, _namespace: str, _body: dict) -> None:
            calls.append("client_job")

        def read_namespaced_job(self, _name: str, _namespace: str) -> None:
            raise _Gone()

    manifests = _service_manifests()
    service._create_server_objects(Core(), Apps(), manifests)
    assert calls == ["secret", "deployment", "service"]
    service._create_client_job(Batch(), manifests)
    assert calls == ["secret", "deployment", "service", "client_job"]


def test_service_controller_rbac_is_name_scoped_with_qualified_residuals() -> None:
    run_id = "openpi-four-mode-contract"
    service_account = service.controller_service_account_name(run_id)
    manifests = service.build_controller_rbac_manifests(
        run_id=run_id,
        namespace="openpi",
        service_account=service_account,
    )

    assert set(manifests) == {"service_account", "role", "role_binding"}
    for manifest in manifests.values():
        metadata = manifest["metadata"]
        assert metadata["name"] == service_account
        assert metadata["namespace"] == "openpi"
        assert metadata["annotations"]["npa.nebius.ai/cleanup-owner"] == run_id
    assert manifests["role_binding"]["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": service_account,
            "namespace": "openpi",
        }
    ]
    rules = manifests["role"]["rules"]
    assert {resource for rule in rules for resource in rule["resources"]} == {
        "deployments",
        "jobs",
        "pods",
        "secrets",
        "services",
    }
    assert {verb for rule in rules for verb in rule["verbs"]} <= {
        "create",
        "delete",
        "get",
        "list",
    }
    assert "update" not in {verb for rule in rules for verb in rule["verbs"]}
    assert all("pods/log" not in rule["resources"] for rule in rules)
    assert all(
        "delete" not in rule["verbs"] for rule in rules if "pods" in rule["resources"]
    )
    exact = {
        resource: rule["resourceNames"]
        for rule in rules
        for resource in rule["resources"]
        if "resourceNames" in rule
    }
    base = service._safe_name(run_id)
    assert exact == {
        "services": [f"{base}-policy"],
        "secrets": [f"{base}-terms"],
        "deployments": [base],
        "jobs": [f"{base}-client"],
    }
    create_rules = [rule for rule in rules if rule["verbs"] == ["create"]]
    assert create_rules
    assert all("resourceNames" not in rule for rule in create_rules)


@pytest.mark.parametrize("name", ["Agent-SA", "bad_name", "-leading"])
def test_service_controller_rbac_rejects_invalid_identity(name: str) -> None:
    with pytest.raises(service.OpenPIServiceError, match="DNS label"):
        service.build_controller_rbac_manifests(
            run_id="openpi-four-mode-contract",
            namespace="default",
            service_account=name,
        )


class _RbacMissing(Exception):
    status = 404


def _api_object(manifest: dict) -> SimpleNamespace:
    metadata = manifest["metadata"]
    values: dict[str, object] = {
        "metadata": SimpleNamespace(
            annotations=metadata.get("annotations", {}),
            labels=metadata.get("labels", {}),
        )
    }
    if "rules" in manifest:
        values["rules"] = [
            SimpleNamespace(
                api_groups=rule["apiGroups"],
                resources=rule["resources"],
                resource_names=rule.get("resourceNames", []),
                verbs=rule["verbs"],
            )
            for rule in manifest["rules"]
        ]
    if "subjects" in manifest:
        values["subjects"] = [
            SimpleNamespace(**subject) for subject in manifest["subjects"]
        ]
        values["role_ref"] = SimpleNamespace(
            api_group=manifest["roleRef"]["apiGroup"],
            kind=manifest["roleRef"]["kind"],
            name=manifest["roleRef"]["name"],
        )
    return SimpleNamespace(**values)


class _FakeCoreRbac:
    def __init__(self) -> None:
        self.service_accounts: dict[str, object] = {}

    def read_namespaced_service_account(self, name: str, _namespace: str) -> object:
        if name not in self.service_accounts:
            raise _RbacMissing()
        return self.service_accounts[name]

    def create_namespaced_service_account(self, _namespace: str, body: dict) -> None:
        self.service_accounts[body["metadata"]["name"]] = _api_object(body)

    def delete_namespaced_service_account(self, name: str, _namespace: str) -> None:
        self.service_accounts.pop(name, None)


class _FakeRbacApi:
    def __init__(self) -> None:
        self.roles: dict[str, object] = {}
        self.bindings: dict[str, object] = {}

    def read_namespaced_role(self, name: str, _namespace: str) -> object:
        if name not in self.roles:
            raise _RbacMissing()
        return self.roles[name]

    def read_namespaced_role_binding(self, name: str, _namespace: str) -> object:
        if name not in self.bindings:
            raise _RbacMissing()
        return self.bindings[name]

    def create_namespaced_role(self, _namespace: str, body: dict) -> None:
        self.roles[body["metadata"]["name"]] = _api_object(body)

    def create_namespaced_role_binding(self, _namespace: str, body: dict) -> None:
        self.bindings[body["metadata"]["name"]] = _api_object(body)

    def delete_namespaced_role(self, name: str, _namespace: str) -> None:
        self.roles.pop(name, None)

    def delete_namespaced_role_binding(self, name: str, _namespace: str) -> None:
        self.bindings.pop(name, None)


def test_service_controller_rbac_apply_reuse_delete_is_exact() -> None:
    core = _FakeCoreRbac()
    rbac = _FakeRbacApi()
    run_id = "openpi-four-mode-contract"
    name = service.controller_service_account_name(run_id)
    args = {
        "run_id": run_id,
        "namespace": "openpi",
        "service_account": name,
    }

    applied = service_rbac.apply_controller_rbac(core, rbac, **args)
    assert applied["created"] == ["service_account", "role", "role_binding"]
    scope = applied["permission_scope"]
    assert scope["classification"] == "name_scoped_with_kubernetes_residuals"
    assert scope["foreign_secret_contents_readable"] is False
    assert scope["pod_logs_readable"] is False
    assert "least_privilege" not in applied
    reused = service_rbac.apply_controller_rbac(core, rbac, **args)
    assert reused["created"] == []
    assert reused["reused_exact_owned"] == [
        "service_account",
        "role",
        "role_binding",
    ]
    deleted = service_rbac.delete_controller_rbac(core, rbac, **args)
    assert deleted["all_exact_resources_absent"] is True
    assert not core.service_accounts and not rbac.roles and not rbac.bindings


def test_service_controller_rbac_refuses_foreign_preexisting_identity() -> None:
    core = _FakeCoreRbac()
    rbac = _FakeRbacApi()
    run_id = "openpi-four-mode-contract"
    name = service.controller_service_account_name(run_id)
    core.service_accounts[name] = SimpleNamespace(
        metadata=SimpleNamespace(
            annotations={"npa.nebius.ai/cleanup-owner": "another-run"},
            labels={"app.kubernetes.io/managed-by": service.CONTROLLER_MANAGED_BY},
        )
    )
    with pytest.raises(service.OpenPIServiceError, match="ownership is not proven"):
        service_rbac.apply_controller_rbac(
            core,
            rbac,
            run_id=run_id,
            namespace="openpi",
            service_account=name,
        )


def test_service_controller_rbac_delete_timeout_has_finalizer_diagnostics() -> None:
    run_id = "openpi-four-mode-contract"
    name = service.controller_service_account_name(run_id)
    desired = service.build_controller_rbac_manifests(
        run_id=run_id, namespace="openpi", service_account=name
    )["service_account"]

    class StuckCore(_FakeCoreRbac):
        def __init__(self) -> None:
            super().__init__()
            obj = _api_object(desired)
            obj.metadata.deletion_timestamp = "2026-08-16T00:00:00Z"
            obj.metadata.finalizers = ["foreign.example/finalizer"]
            self.service_accounts[name] = obj

        def delete_namespaced_service_account(
            self, _name: str, _namespace: str
        ) -> None:
            return None

    clock = _FakeClock()
    with pytest.raises(
        service.OpenPIServiceError, match="RBAC cleanup incomplete.*finalizer"
    ):
        service_rbac._delete_controller_rbac(
            StuckCore(),
            _FakeRbacApi(),
            run_id=run_id,
            namespace="openpi",
            service_account=name,
            only={"service_account"},
            timeout=2,
            poll_interval=1,
            clock=clock,
            sleep=clock.sleep,
        )


class _Gone(Exception):
    status = 404


class _FakeApi:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []
        self.pods: list[object] = []

    def delete_namespaced_service(self, name, namespace, body=None, **_kwargs):
        self.deleted.append(("service", name))

    def delete_namespaced_secret(self, name, namespace, body=None, **_kwargs):
        self.deleted.append(("secret", name))

    def read_namespaced_service(self, _name, _namespace):
        raise _Gone()

    def read_namespaced_secret(self, _name, _namespace):
        raise _Gone()

    def list_namespaced_pod(self, _namespace, label_selector=None):
        return SimpleNamespace(items=self.pods)

    def delete_namespaced_pod(self, name, _namespace):
        self.deleted.append(("pod", name))
        self.pods = [pod for pod in self.pods if pod.metadata.name != name]


class _FakeApps:
    def __init__(self, deleted: list[tuple[str, str]]) -> None:
        self.deleted = deleted

    def delete_namespaced_deployment(self, name, namespace, body=None, **_kwargs):
        self.deleted.append(("deployment", name))

    def read_namespaced_deployment(self, _name, _namespace):
        raise _Gone()


class _FakeBatch:
    def __init__(self, deleted: list[tuple[str, str]]) -> None:
        self.deleted = deleted

    def delete_namespaced_job(self, name, namespace, body=None, **_kwargs):
        self.deleted.append(("client_job", name))

    def read_namespaced_job(self, _name, _namespace):
        raise _Gone()


def test_service_cleanup_uses_only_exact_names_and_verifies_absence() -> None:
    names = {
        key: value["metadata"]["name"] for key, value in _service_manifests().items()
    }
    api = _FakeApi()
    apps = _FakeApps(api.deleted)
    batch = _FakeBatch(api.deleted)
    verified = service._delete_and_verify(
        api, apps, batch, namespace="default", names=names
    )
    assert verified == {
        "client_job": True,
        "deployment": True,
        "pods": True,
        "service": True,
        "secret": True,
    }
    assert api.deleted == [
        ("client_job", names["client_job"]),
        ("deployment", names["deployment"]),
        ("service", names["service"]),
        ("secret", names["secret"]),
    ]


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def _condition(
    condition_type: str, status: str, *, reason: str = "", message: str = ""
) -> SimpleNamespace:
    return SimpleNamespace(
        type=condition_type,
        status=status,
        reason=reason,
        message=message,
    )


def _pod(
    *,
    name: str = "pod-1",
    phase: str = "Pending",
    ready: bool = False,
    waiting_reason: str = "",
) -> SimpleNamespace:
    waiting = (
        SimpleNamespace(reason=waiting_reason, message="container cannot start")
        if waiting_reason
        else None
    )
    state = SimpleNamespace(waiting=waiting, terminated=None)
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, uid=f"uid-{name}", labels={}),
        status=SimpleNamespace(
            phase=phase,
            conditions=[_condition("Ready", "True" if ready else "False")],
            container_statuses=[
                SimpleNamespace(
                    name="openpi-server",
                    ready=ready,
                    restart_count=0,
                    state=state,
                )
            ],
        ),
    )


class _WaitCore:
    def __init__(self, pods: list[object]) -> None:
        self.pods = pods

    def list_namespaced_pod(self, _namespace: str, label_selector: str = ""):
        return SimpleNamespace(items=self.pods)


class _WaitApps:
    def __init__(self, deployment: object) -> None:
        self.deployment = deployment

    def read_namespaced_deployment(self, _name: str, _namespace: str) -> object:
        return self.deployment


def _deployment(*, available: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        status=SimpleNamespace(
            available_replicas=available,
            ready_replicas=available,
            unavailable_replicas=0 if available else 1,
            conditions=[],
        )
    )


def test_service_server_wait_success() -> None:
    pod = _pod(phase="Running", ready=True)
    assert (
        service._wait_server_ready(
            _WaitCore([pod]), _WaitApps(_deployment(available=1)), "default", "server"
        )
        is pod
    )


@pytest.mark.parametrize("reason", ["ImagePullBackOff", "ErrImagePull"])
def test_service_server_wait_terminal_container_failure(reason: str) -> None:
    with pytest.raises(service.OpenPIServiceError, match=reason):
        service._wait_server_ready(
            _WaitCore([_pod(waiting_reason=reason)]),
            _WaitApps(_deployment()),
            "default",
            "server",
        )


def test_service_server_wait_fails_on_unschedulable_missing_gpu_label() -> None:
    pod = _pod()
    pod.status.conditions.append(
        _condition(
            "PodScheduled",
            "False",
            reason="Unschedulable",
            message="node(s) didn't match nebius.com/gpu-name=B200",
        )
    )
    with pytest.raises(service.OpenPIServiceError, match="Unschedulable.*gpu-name"):
        service._wait_server_ready(
            _WaitCore([pod]), _WaitApps(_deployment()), "default", "server"
        )


def test_service_server_wait_fails_on_health_probe_progress_deadline() -> None:
    deployment = _deployment()
    deployment.status.conditions = [
        _condition(
            "Progressing",
            "False",
            reason="ProgressDeadlineExceeded",
            message="readiness probe never passed",
        )
    ]
    with pytest.raises(
        service.OpenPIServiceError, match="progress deadline.*readiness probe"
    ):
        service._wait_server_ready(
            _WaitCore([_pod()]), _WaitApps(deployment), "default", "server"
        )


def test_service_server_wait_timeout_has_last_state_without_real_sleep() -> None:
    clock = _FakeClock()
    with pytest.raises(service.OpenPIServiceError, match="timed out.*Pending"):
        service._wait_server_ready(
            _WaitCore([_pod()]),
            _WaitApps(_deployment()),
            "default",
            "server",
            timeout=2,
            poll_interval=1,
            clock=clock,
            sleep=clock.sleep,
        )


def test_service_wait_api_uncertainty_fails_closed() -> None:
    class UncertainApps:
        def read_namespaced_deployment(self, _name: str, _namespace: str) -> object:
            raise RuntimeError("API response uncertain")

    with pytest.raises(RuntimeError, match="uncertain"):
        service._wait_server_ready(_WaitCore([]), UncertainApps(), "default", "server")


def test_service_api_uncertainty_after_creation_triggers_exact_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kubernetes import client, config

    monkeypatch.setenv("NPA_OPENPI_ACCEPT_GEMMA_TERMS", "YES")
    monkeypatch.setattr(config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(client, "CoreV1Api", object)
    monkeypatch.setattr(client, "AppsV1Api", object)
    monkeypatch.setattr(client, "BatchV1Api", object)
    monkeypatch.setattr(
        service, "_assert_targets_absent", lambda *_args, **_kwargs: None
    )

    def create_server(
        _api: object,
        _apps: object,
        _manifests: object,
        *,
        created: set[str],
        **_kwargs: object,
    ) -> None:
        created.update({"secret", "deployment", "service"})

    monkeypatch.setattr(service, "_create_server_objects", create_server)
    monkeypatch.setattr(
        service,
        "_wait_server_ready",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("Kubernetes API response uncertain")
        ),
    )
    cleanup_calls: list[set[str]] = []

    def cleanup(
        *_args: object, created: set[str], **_kwargs: object
    ) -> dict[str, bool]:
        cleanup_calls.append(set(created))
        return {
            "secret": True,
            "deployment": True,
            "service": True,
            "client_job": True,
            "pods": True,
        }

    monkeypatch.setattr(service, "_delete_and_verify", cleanup)
    args = service.build_parser().parse_args(
        [
            "--run-id",
            "api-uncertain",
            "--output-uri",
            "file:///tmp/api-uncertain-service.json",
            "--cleanup-output-uri",
            "file:///tmp/api-uncertain-cleanup.json",
            "--runtime-image",
            DIGEST_IMAGE,
            "--checkpoint-uri",
            "file:///tmp/runtime-only-checkpoint",
            "--expected-gpu-type",
            "B200",
            "--expected-compute-capability",
            "10.0",
            "--controller-service-account",
            service.controller_service_account_name("api-uncertain"),
        ]
    )

    with pytest.raises(RuntimeError, match="API response uncertain"):
        service._run(args)

    assert cleanup_calls == [{"secret", "deployment", "service"}]


class _WaitBatch:
    def __init__(self, *, succeeded: int = 0, failed: int = 0) -> None:
        self.job = SimpleNamespace(
            status=SimpleNamespace(
                active=0 if succeeded or failed else 1,
                succeeded=succeeded,
                failed=failed,
                conditions=[],
            )
        )

    def read_namespaced_job(self, _name: str, _namespace: str) -> object:
        return self.job


def test_service_client_wait_success_and_terminal_failure() -> None:
    pod = _pod(name="client", phase="Succeeded")
    assert (
        service._wait_client(
            _WaitCore([pod]), _WaitBatch(succeeded=1), "default", "client"
        )
        is pod
    )
    with pytest.raises(service.OpenPIServiceError, match="client job failed"):
        service._wait_client(
            _WaitCore([_pod(name="client", phase="Failed")]),
            _WaitBatch(failed=1),
            "default",
            "client",
        )


def test_service_client_wait_timeout_has_last_state_without_real_sleep() -> None:
    clock = _FakeClock()
    with pytest.raises(service.OpenPIServiceError, match="timed out.*active"):
        service._wait_client(
            _WaitCore([_pod(name="client")]),
            _WaitBatch(),
            "default",
            "client",
            timeout=2,
            poll_interval=1,
            clock=clock,
            sleep=clock.sleep,
        )


def test_service_cleanup_times_out_on_stuck_orphan_pod_without_pod_delete() -> None:
    names = {
        key: value["metadata"]["name"] for key, value in _service_manifests().items()
    }
    api = _FakeApi()
    api.pods = [
        SimpleNamespace(
            metadata=SimpleNamespace(
                name="exact-client-pod",
                labels={
                    "app.kubernetes.io/managed-by": "npa",
                    "app.kubernetes.io/part-of": "openpi-four-mode",
                    "npa.nebius.ai/cleanup-owner": names["deployment"],
                    "app": names["client_job"],
                },
            )
        )
    ]
    apps = _FakeApps(api.deleted)
    batch = _FakeBatch(api.deleted)

    clock = _FakeClock()
    with pytest.raises(service.OpenPIServiceError, match="pod deletion timed out"):
        service._delete_and_verify(
            api,
            apps,
            batch,
            namespace="default",
            names=names,
            timeout=2,
            poll_interval=1,
            clock=clock,
            sleep=clock.sleep,
        )

    assert all(kind != "pod" for kind, _name in api.deleted)


def test_service_preflight_refuses_preexisting_exact_name_without_cleanup() -> None:
    names = {
        key: value["metadata"]["name"] for key, value in _service_manifests().items()
    }
    api = _FakeApi()

    class ExistingApps(_FakeApps):
        def read_namespaced_deployment(self, _name, _namespace):
            return SimpleNamespace(metadata=SimpleNamespace(name=names["deployment"]))

    with pytest.raises(service.OpenPIServiceError, match="ownership is not proven"):
        service._assert_targets_absent(
            api,
            ExistingApps(api.deleted),
            _FakeBatch(api.deleted),
            namespace="default",
            names=names,
        )
    assert api.deleted == []


def test_service_partial_create_tracks_uncertain_identity_for_cleanup() -> None:
    manifests = _service_manifests()
    created: set[str] = set()

    class Core:
        def create_namespaced_secret(self, _namespace: str, _body: dict) -> None:
            pass

        def create_namespaced_service(self, _namespace: str, _body: dict) -> None:
            raise AssertionError("service must not be reached")

        def read_namespaced_secret(self, _name: str, _namespace: str) -> None:
            raise _Gone()

        def read_namespaced_service(self, _name: str, _namespace: str) -> None:
            raise _Gone()

    class Apps:
        def create_namespaced_deployment(self, _namespace: str, _body: dict) -> None:
            raise RuntimeError("deployment create failed")

        def read_namespaced_deployment(self, _name: str, _namespace: str) -> None:
            raise _Gone()

    with pytest.raises(RuntimeError, match="deployment create failed"):
        service._create_server_objects(Core(), Apps(), manifests, created=created)
    # Deployment identity is deliberately tracked before its create request:
    # the API response may be lost after a successful server-side create.
    assert created == {"secret", "deployment"}
    names = {key: value["metadata"]["name"] for key, value in manifests.items()}
    cleanup_api = _FakeApi()
    verified = service._delete_and_verify(
        cleanup_api,
        _FakeApps(cleanup_api.deleted),
        _FakeBatch(cleanup_api.deleted),
        namespace="default",
        names=names,
        created=created,
        manifests=manifests,
    )
    assert all(verified.values())
    assert cleanup_api.deleted == [
        ("deployment", names["deployment"]),
        ("secret", names["secret"]),
    ]


def test_service_cleanup_refuses_foreign_pod_before_deleting_anything() -> None:
    names = {
        key: value["metadata"]["name"] for key, value in _service_manifests().items()
    }
    api = _FakeApi()
    api.pods = [
        SimpleNamespace(
            metadata=SimpleNamespace(
                name="foreign-pod",
                labels={
                    "app.kubernetes.io/managed-by": "someone-else",
                    "npa.nebius.ai/cleanup-owner": names["deployment"],
                },
            )
        )
    ]
    with pytest.raises(service.OpenPIServiceError, match="exact OpenPI"):
        service._delete_and_verify(
            api,
            _FakeApps(api.deleted),
            _FakeBatch(api.deleted),
            namespace="default",
            names=names,
        )
    assert api.deleted == []


def test_four_mode_spec_plans_complete_lineage_and_configurable_resources() -> None:
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="openpi-four-mode-plan")
    assert [step.tool_ref for step in plan.steps] == [
        "workbench.openpi.negative_terms_gate",
        "workbench.openpi.prepare_data",
        "workbench.openpi.direct",
        "workbench.openpi.serve",
        "workbench.openpi.train",
        "workbench.openpi.evaluate",
    ]
    assert secret_env_hints_for_plan(plan.steps) == ("NPA_OPENPI_ACCEPT_GEMMA_TERMS",)
    gpu = plan.steps[0].resources_profile
    assert gpu["accelerators"] == "B200:1"
    assert gpu["image"] == "registry.example.invalid/openpi@sha256:" + "0" * 64
    train = plan.steps[4]
    evaluate = plan.steps[5]
    assert any(
        item["uri"].endswith("/checkpoints/trained/manifest.json")
        for item in train.outputs
    )
    assert any(
        item["uri"].endswith("/checkpoints/trained/manifest.json")
        for item in evaluate.inputs
    )
    assert any(
        item["uri"].endswith("/reports/training.json") for item in evaluate.inputs
    )
    spec_text = SPEC.read_text(encoding="utf-8")
    for configurable in (
        "gpu_type",
        "gpu_count",
        "gpu_cpus",
        "gpu_memory",
        "gpu_disk_size",
        "gpu_scratch_size",
        "gpu_num_nodes",
        "service_namespace",
        "service_server_ready_timeout_seconds",
        "service_cleanup_timeout_seconds",
        "serve_artifact_root_uri",
        "terms_diagnostic_root_uri",
        "trained_checkpoint_uri",
    ):
        assert f"{configurable}:" in spec_text


def test_openpi_serving_output_root_override_keeps_declared_outputs_in_sync() -> None:
    shared_root = "s3://example-bucket/custom-serving-attempt"
    spec = load_spec_for_submit(
        SPEC,
        config_overrides={"serve_artifact_root_uri": shared_root},
    )
    plan = build_plan(spec, run_id="openpi-output-override")
    serve = next(
        step for step in plan.steps if step.tool_ref == "workbench.openpi.serve"
    )
    assert [item["uri"] for item in serve.outputs] == [
        f"{shared_root}/service.json",
        f"{shared_root}/cleanup.json",
    ]
    argv = list(serve.argv)
    assert argv[argv.index("--output-uri") + 1] == f"{shared_root}/service.json"
    assert argv[argv.index("--cleanup-output-uri") + 1] == (
        f"{shared_root}/cleanup.json"
    )
    assert argv[argv.index("--server-ready-timeout-seconds") + 1] == "1200"
    assert argv[argv.index("--client-timeout-seconds") + 1] == "600"


def test_openpi_gpu_stages_use_pinned_vendor_python_only() -> None:
    gpu_stages = (
        "negative_terms_gate",
        "direct",
        "train",
        "evaluate",
    )
    for stage in gpu_stages:
        argv = TOOL_CATALOG[f"workbench.openpi.{stage}"].argv_template
        assert argv[:3] == [
            "/opt/venv/bin/python",
            "-m",
            "npa.workflows.byof.openpi_pipeline",
        ]

    assert TOOL_CATALOG["workbench.openpi.prepare_data"].argv_template[0] == "python3"
    assert TOOL_CATALOG["workbench.openpi.serve"].argv_template[0] == "python3"


def test_policy_checkpoint_lands_on_the_durable_cache_when_configured(
    monkeypatch,
) -> None:
    """A Deployment replaces its pod on every rollout, drain and image change.

    With the checkpoint on an emptyDir, each of those re-downloaded the gated
    Gemma-derived weights onto a GPU that is already running.
    """

    monkeypatch.setenv("NPA_MODEL_CACHE_PVC", "npa-model-cache")

    pod = _service_manifests()["deployment"]["spec"]["template"]["spec"]

    assert {
        "name": "npa-model-cache",
        "persistentVolumeClaim": {"claimName": "npa-model-cache"},
    } in pod["volumes"]
    server = next(c for c in pod["containers"] if c["name"] == "openpi-policy")
    assert {
        "name": "npa-model-cache",
        "mountPath": "/opt/npa-model-cache",
    } in server["volumeMounts"]
    env = {item["name"]: item["value"] for item in server["env"] if "value" in item}
    assert env["OPENPI_DATA_HOME"] == "/opt/npa-model-cache/openpi"
    assert env["HF_HOME"] == "/opt/npa-model-cache/huggingface"


def test_policy_checkpoint_keeps_its_ephemeral_volume_without_a_cache(
    monkeypatch,
) -> None:
    for name in (
        "NPA_MODEL_CACHE_PVC",
        "NPA_MODEL_CACHE_HOST_PATH",
        "NPA_MODEL_CACHE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    pod = _service_manifests()["deployment"]["spec"]["template"]["spec"]

    assert [volume["name"] for volume in pod["volumes"]] == ["openpi-cache"]
    server = next(c for c in pod["containers"] if c["name"] == "openpi-policy")
    env = {item["name"]: item["value"] for item in server["env"] if "value" in item}
    assert env["OPENPI_DATA_HOME"] == "/workspace/openpi-server-cache"
