from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.skypilot_render import secret_env_hints_for_plan
from npa.workflows.byof import openpi_pipeline as pipeline
from npa.workflows.byof import openpi_service as service
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


def test_terms_gate_exits_before_openpi_import_or_checkpoint_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    output = tmp_path / "refusal.json"
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
                "--runtime-image",
                DIGEST_IMAGE,
            ]
        )

    assert raised.value.code == 64
    assert json.loads(output.read_text(encoding="utf-8")) == pipeline._terms_refusal()
    after = {
        name for name in sys.modules if name == "openpi" or name.startswith("openpi.")
    }
    assert after == before


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
        pull_secret="npa-nebius-registry",
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
    assert deployment["spec"]["template"]["spec"]["nodeSelector"] == {
        "nebius.com/gpu-name": "B200"
    }
    assert client_container["command"][0] == "/opt/venv/bin/python"
    assert "range(2)" in client_container["command"][2]
    assert "NPA_OPENPI_CLIENT_RESULT=" in client_container["command"][2]
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

    class Apps:
        def create_namespaced_deployment(self, _namespace: str, _body: dict) -> None:
            calls.append("deployment")

    class Batch:
        def create_namespaced_job(self, _namespace: str, _body: dict) -> None:
            calls.append("client_job")

    manifests = _service_manifests()
    service._create_server_objects(Core(), Apps(), manifests)
    assert calls == ["secret", "deployment", "service"]
    service._create_client_job(Batch(), manifests)
    assert calls == ["secret", "deployment", "service", "client_job"]


def test_service_controller_rbac_is_run_owned_and_least_privilege() -> None:
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
        "pods/log",
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


def test_service_cleanup_explicitly_removes_exact_orphan_pod() -> None:
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

    verified = service._delete_and_verify(
        api, apps, batch, namespace="default", names=names
    )

    assert verified["pods"] is True
    assert api.deleted[-1] == ("pod", "exact-client-pod")
    assert not api.pods


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


def test_service_partial_create_tracks_only_successful_objects() -> None:
    manifests = _service_manifests()
    created: set[str] = set()

    class Core:
        def create_namespaced_secret(self, _namespace: str, _body: dict) -> None:
            pass

        def create_namespaced_service(self, _namespace: str, _body: dict) -> None:
            raise AssertionError("service must not be reached")

    class Apps:
        def create_namespaced_deployment(self, _namespace: str, _body: dict) -> None:
            raise RuntimeError("deployment create failed")

    with pytest.raises(RuntimeError, match="deployment create failed"):
        service._create_server_objects(Core(), Apps(), manifests, created=created)
    assert created == {"secret"}


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
        "trained_checkpoint_uri",
    ):
        assert f"{configurable}:" in spec_text


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
