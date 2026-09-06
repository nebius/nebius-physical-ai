"""Exercise target resolution and real probe behavior at provider boundaries."""

from io import BytesIO
from types import SimpleNamespace

from botocore.exceptions import ClientError
import pytest

from npa.execution_preflight import (
    ExecutionPreflightError,
    resolve_execution_target,
    verify_execution_target,
    verify_serverless_execution,
    verify_serverless_gpu,
)
from npa.orchestration.npa_workflow.submit_credentials import (
    SubmitCredentialContext,
    resolve_submit_credentials,
)


class ProbeS3:
    def __init__(self):
        self.objects = {}
        self.calls = []
        self.denied_prefix = ""
        self.corrupt = False
        self.deny_read = False

    def put_object(self, **kwargs):
        self.calls.append(("put", kwargs["Bucket"], kwargs["Key"]))
        if self.denied_prefix and kwargs["Key"].startswith(self.denied_prefix):
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "private-provider-text"}}, "PutObject")
        self.objects[kwargs["Key"]] = kwargs["Body"]

    def get_object(self, **kwargs):
        self.calls.append(("get", kwargs["Bucket"], kwargs["Key"]))
        if self.deny_read:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
        return {"Body": BytesIO(b"wrong" if self.corrupt else self.objects[kwargs["Key"]])}

    def delete_object(self, **kwargs):
        self.calls.append(("delete", kwargs["Bucket"], kwargs["Key"]))
        self.objects.pop(kwargs["Key"], None)

    def head_object(self, **kwargs):
        if kwargs["Key"] not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "HeadObject")
        return {"ContentLength": len(self.objects[kwargs["Key"]])}


@pytest.fixture
def provider(monkeypatch):
    env = SimpleNamespace(project_id="project-unit", tenant_id="tenant-unit", region="eu-west1")
    monkeypatch.setattr("npa.clients.config.resolve_environment", lambda project: env)
    monkeypatch.setattr("npa.clients.nebius.get_project_identity", lambda *args, **kwargs: env)
    monkeypatch.setattr("npa.clients.nebius.get_bucket_by_name", lambda parent, name: {"metadata": {"parent_id": parent, "name": name}})
    monkeypatch.setattr("npa.cluster.identity.resolve_verified_cluster_identity", lambda **kwargs: SimpleNamespace(cluster_absent=False, project_id=env.project_id, context=kwargs["context"]))
    monkeypatch.setattr("npa.cluster.state.load_cluster_state", lambda context: None)
    client = ProbeS3()
    captured = []

    def storage_client(**kwargs):
        captured.append(kwargs)
        return client

    monkeypatch.setattr("npa.clients.storage_validation._storage_client", storage_client)
    return SimpleNamespace(env=env, s3=client, connections=captured)


def target(**kwargs):
    return resolve_execution_target(
        project="unit", context="unit-context", output_uris=["s3://unit-output/task/results/"],
        credentials=SubmitCredentialContext(
            endpoint_url="https://storage.eu-west1.nebius.cloud",
            access_key_id="unit-executing-access", secret_access_key="unit-executing-secret",
            provenance={"credentials": "environment", "endpoint": "cli"},
        ), **kwargs,
    )


def test_exact_prefix_roundtrip_uses_executing_principal_and_safe_report(provider):
    selected = target()
    report = verify_execution_target(selected)
    assert report["checks"]["storage_write_read"] == "pass"
    assert report["destination_count"] == 1
    assert report["provenance"]["credentials"] == "environment"
    assert provider.connections[0]["access"] == selected.credentials.access_key_id
    assert provider.connections[0]["secret"] == selected.credentials.secret_access_key
    assert provider.connections[0]["region"] == selected.region
    assert [call[0] for call in provider.s3.calls] == ["put", "get", "delete"]
    assert all(call[2].startswith("task/results/.npa-probes/") for call in provider.s3.calls)
    assert not provider.s3.objects
    for private in (selected.project_id, selected.tenant_id, selected.context, selected.credentials.access_key_id, selected.credentials.secret_access_key, "unit-output"):
        assert private not in repr(selected)
        assert private not in str(report)


@pytest.mark.parametrize("wrong", ["project_id", "tenant_id", "region"])
def test_provider_scope_mismatch_never_writes(provider, monkeypatch, wrong):
    selected = target()
    values = vars(provider.env).copy()
    values[wrong] = "wrong"
    monkeypatch.setattr("npa.clients.nebius.get_project_identity", lambda *args, **kwargs: SimpleNamespace(**values))
    with pytest.raises(ExecutionPreflightError, match="scope"):
        verify_execution_target(selected)
    assert not provider.s3.calls


@pytest.mark.parametrize("record", [None, {"metadata": {"name": "unit-output", "parent_id": "other-project"}}, {"metadata": {"name": "unit-output"}}])
def test_bucket_owner_missing_or_wrong_never_writes(provider, monkeypatch, record):
    monkeypatch.setattr("npa.clients.nebius.get_bucket_by_name", lambda *args: record)
    with pytest.raises(ExecutionPreflightError, match="storage_owner"):
        verify_execution_target(target())
    assert not provider.s3.calls


@pytest.mark.parametrize("error", [TimeoutError("private-url"), PermissionError("private-token")])
def test_unknown_or_denied_provider_evidence_fails_closed(provider, monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr("npa.clients.nebius.get_project_identity", unavailable)
    with pytest.raises(ExecutionPreflightError) as caught:
        verify_execution_target(target())
    assert caught.value.status == "unknown"
    assert "private" not in str(caught.value)
    assert not provider.s3.calls


@pytest.mark.parametrize("identity", [SimpleNamespace(cluster_absent=False, project_id="other", context="unit-context"), SimpleNamespace(cluster_absent=False, project_id="project-unit", context="wrong"), SimpleNamespace(cluster_absent=True, project_id="project-unit", context="unit-context")])
def test_cluster_mismatch_never_writes_even_with_isolated_controller(provider, monkeypatch, identity):
    monkeypatch.setattr("npa.cluster.identity.resolve_verified_cluster_identity", lambda **kwargs: identity)
    with pytest.raises(ExecutionPreflightError, match="cluster_owner"):
        verify_execution_target(target())
    assert not provider.s3.calls


@pytest.mark.parametrize("failure", ["prefix", "read", "integrity"])
def test_exact_prefix_denial_and_readback_failure(provider, failure):
    if failure == "prefix":
        provider.s3.denied_prefix = "task/"
    elif failure == "read":
        provider.s3.deny_read = True
    else:
        provider.s3.corrupt = True
    with pytest.raises(ExecutionPreflightError, match="storage_access") as caught:
        verify_execution_target(target())
    assert "private-provider-text" not in str(caught.value)
    assert "unit-output" not in str(caught.value)
    assert all(call[2].startswith("task/") for call in provider.s3.calls)


def test_explicit_region_or_identity_cannot_retarget_saved_scope(provider):
    with pytest.raises(ExecutionPreflightError, match="region"):
        target(region="eu-north1")
    with pytest.raises(ExecutionPreflightError, match="scope"):
        target(project_id="other-project")


@pytest.mark.parametrize("endpoint", ["https://storage.eu-north1.nebius.cloud", "https://user:secret@storage.eu-west1.nebius.cloud", "https://storage.eu-west1.nebius.cloud/?token=secret"])
def test_endpoint_must_match_effective_region_without_credentials(provider, endpoint):
    with pytest.raises(ExecutionPreflightError, match="endpoint") as caught:
        resolve_execution_target(project="unit", output_uris=["s3://unit-output/task/"], credentials=SubmitCredentialContext(endpoint_url=endpoint, access_key_id="unit", secret_access_key="unit"))
    assert "secret" not in str(caught.value)


def catalog(gpu="gpu-b200-sxm", parent="project-unit"):
    return {"items": [{"metadata": {"id": "platform-unit", "name": gpu, "parent_id": parent}, "spec": {"presets": [{"name": "1gpu-20vcpu-224gb", "resources": {"gpu_count": 1}}]}}]}


@pytest.mark.parametrize("gpu", ["gpu-b200-sxm", "gpu-rtx6000"])
def test_provider_owned_product_requires_exact_project_scoped_lookup(provider, monkeypatch, gpu):
    payload = catalog(gpu, parent="catalog-project")
    calls = []

    def lookup(args):
        calls.append(args)
        return payload if args[2] == "list" else payload["items"][0]

    monkeypatch.setattr("npa.clients.nebius._run_json", lookup)
    verify_serverless_gpu(project_id="project-unit", gpu_type=gpu, gpu_count=1, preset="1gpu-20vcpu-224gb")
    assert calls == [
        ["compute", "platform", "list", "--parent-id", "project-unit", "--all"],
        ["compute", "platform", "get-by-name", "--parent-id", "project-unit", "--name", gpu],
    ]


@pytest.mark.parametrize("field", ["id", "name", "parent_id", "presets"])
def test_disagreeing_project_product_lookup_is_unknown_before_writes(provider, monkeypatch, field):
    import copy

    payload = catalog(parent="catalog-project")
    corroboration = copy.deepcopy(payload["items"][0])
    if field == "presets":
        corroboration["spec"][field] = []
    else:
        corroboration["metadata"][field] = "other-value"
    monkeypatch.setattr("npa.clients.nebius._run_json", lambda args: payload if args[2] == "list" else corroboration)
    with pytest.raises(ExecutionPreflightError) as caught:
        verify_execution_target(target(), gpu_check=lambda: verify_serverless_gpu(project_id="project-unit", gpu_type="gpu-b200-sxm", gpu_count=1, preset="1gpu-20vcpu-224gb"))
    assert caught.value.status == "unknown"
    assert not provider.s3.calls


@pytest.mark.parametrize("gpu", ["gpu-b200-sxm", "gpu-rtx6000"])
def test_project_regional_supported_gpu_product_and_preset(provider, monkeypatch, gpu):
    calls = []
    monkeypatch.setattr("npa.clients.nebius._run_json", lambda args: calls.append(args) or catalog(gpu))
    verify_serverless_gpu(project_id="project-unit", gpu_type=gpu, gpu_count=1, preset="1gpu-20vcpu-224gb")
    assert calls == [["compute", "platform", "list", "--parent-id", "project-unit", "--all"]]


@pytest.mark.parametrize("payload,status", [(catalog("gpu-other"), "fail"), (catalog(parent="other-project"), "unknown"), ({}, "unknown"), ({"items": [{"metadata": {"name": "gpu-b200-sxm", "parent_id": "project-unit"}}]}, "unknown")])
def test_unsupported_and_unknown_gpu_evidence_prevents_storage(provider, monkeypatch, payload, status):
    monkeypatch.setattr("npa.clients.nebius._run_json", lambda args: payload)
    with pytest.raises(ExecutionPreflightError) as caught:
        verify_execution_target(target(), gpu_check=lambda: verify_serverless_gpu(project_id="project-unit", gpu_type="gpu-b200-sxm", gpu_count=1, preset="1gpu-20vcpu-224gb"))
    assert caught.value.status == status
    assert not provider.s3.calls


def test_gpu_preset_count_mismatch(provider, monkeypatch):
    monkeypatch.setattr("npa.clients.nebius._run_json", lambda args: catalog())
    with pytest.raises(ExecutionPreflightError, match="count"):
        verify_serverless_gpu(project_id="project-unit", gpu_type="gpu-b200-sxm", gpu_count=2, preset="1gpu-20vcpu-224gb")


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr("npa.orchestration.npa_workflow.submit_credentials.resolve_project_storage", lambda project: SimpleNamespace(checkpoint_bucket="project-output", endpoint_url="https://storage.eu-west1.nebius.cloud", aws_access_key_id="project-access", aws_secret_access_key="project-secret"))
    monkeypatch.setattr("npa.orchestration.npa_workflow.submit_credentials.load_credentials", lambda **kwargs: SimpleNamespace(tokens={}, hf_token="", s3_access_key_id="shared-access", s3_secret_access_key="shared-secret", s3_endpoint="https://storage.eu-north1.nebius.cloud", s3_bucket="shared-output"))


def test_atomic_credentials_and_endpoint_precedence(configured):
    context = resolve_submit_credentials(project="unit", explicit_endpoint="https://storage.eu-west1.nebius.cloud", workflow_env={"AWS_ACCESS_KEY_ID": "workflow-access", "AWS_SECRET_ACCESS_KEY": "workflow-secret", "AWS_ENDPOINT_URL": "https://workflow.invalid"}, environ={"AWS_ACCESS_KEY_ID": "process-access", "AWS_SECRET_ACCESS_KEY": "process-secret", "AWS_ENDPOINT_URL": "https://process.invalid"}, requested=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])
    assert context.access_key_id == "process-access"
    assert context.secret_access_key == "process-secret"
    assert context.secret_values["AWS_ACCESS_KEY_ID"] == context.access_key_id
    assert context.endpoint_url == "https://storage.eu-west1.nebius.cloud"
    assert context.provenance["credentials"] == "environment"
    assert context.provenance["endpoint"] == "cli"


@pytest.mark.parametrize("partial", [{"AWS_ACCESS_KEY_ID": "different-access"}, {"AWS_SECRET_ACCESS_KEY": "different-secret"}])
def test_partial_explicit_credentials_never_mix_saved_principals(configured, partial):
    with pytest.raises(ValueError, match="Incomplete S3 credential pair") as caught:
        resolve_submit_credentials(environ=partial)
    assert "different" not in str(caught.value)


def test_workflow_credentials_precede_project_and_shared(configured):
    context = resolve_submit_credentials(environ={}, workflow_env={"AWS_ACCESS_KEY_ID": "workflow-access", "AWS_SECRET_ACCESS_KEY": "workflow-secret"})
    assert context.access_key_id == "workflow-access"
    assert context.secret_access_key == "workflow-secret"
    assert context.provenance["credentials"] == "workflow.env"


def test_requested_endpoint_secret_cannot_override_explicit_endpoint(configured):
    context = resolve_submit_credentials(explicit_endpoint="https://storage.eu-west1.nebius.cloud", environ={"AWS_ENDPOINT_URL": "https://other.invalid"}, requested=["AWS_ENDPOINT_URL"])
    assert context.secret_values["AWS_ENDPOINT_URL"] == context.endpoint_url


@pytest.mark.parametrize("explicit", ["", "https://explicit.invalid"])
def test_endpoint_aliases_resolve_by_layer_and_normalize_worker_values(configured, explicit):
    from npa.orchestration.npa_workflow.submit_credentials import STORAGE_ENDPOINT_ENV_NAMES

    selected = resolve_submit_credentials(
        explicit_endpoint=explicit,
        environ={"AWS_ENDPOINT_URL": "https://process.invalid"},
        workflow_env={"AWS_ENDPOINT_URL_S3": "https://workflow.invalid"},
        requested=STORAGE_ENDPOINT_ENV_NAMES,
    )
    assert selected.endpoint_url == (explicit or "https://process.invalid")
    assert set(selected.secret_values.values()) == {selected.endpoint_url}
    assert selected.provenance["endpoint"] == ("cli" if explicit else "environment")


def test_service_specific_endpoint_is_the_effective_endpoint(configured):
    selected = resolve_submit_credentials(environ={"AWS_ENDPOINT_URL_S3": "https://service.invalid", "AWS_ENDPOINT_URL": "https://generic.invalid"})
    assert selected.endpoint_url == "https://service.invalid"


@pytest.mark.parametrize("override", [
    {"envs": {"AWS_ENDPOINT_URL": "https://wrong.invalid"}},
    {"envs": {"AWS_ENDPOINT_URL_S3": "https://wrong.invalid"}},
    {"envs": {"NPA_STORAGE_ENDPOINT": "https://wrong.invalid"}},
    {"resources": {"kubernetes": {"pod_config": {"spec": {"containers": [{"env": [{"name": "AWS_ACCESS_KEY_ID", "valueFrom": {"secretKeyRef": {"name": "other", "key": "access"}}}]}]}}}}},
    {"resources": {"kubernetes": {"pod_config": {"spec": {"containers": [{"envFrom": [{"secretRef": {"name": "other"}}]}]}}}}},
])
def test_worker_overrides_cannot_change_checked_principal(provider, override):
    from npa.execution_preflight import verify_worker_environment

    with pytest.raises(ExecutionPreflightError, match="worker_environment"):
        verify_worker_environment(target(), [override])


def test_missing_worker_keys_never_probe_saved_credentials(provider, configured):
    with pytest.raises(ExecutionPreflightError, match="credentials"):
        verify_serverless_execution(project="unit", project_id="project-unit", gpu_type="gpu-b200-sxm", gpu_count=1, preset="1gpu-20vcpu-224gb", output_uri="s3://unit-output/task/", extra_env={"AWS_ENDPOINT_URL": "https://storage.eu-west1.nebius.cloud"})
    assert not provider.s3.calls


def test_serverless_wrapper_probes_exact_env_and_output(provider, configured, monkeypatch):
    monkeypatch.setattr("npa.clients.nebius._run_json", lambda args: catalog())
    report = verify_serverless_execution(project="unit", project_id="project-unit", gpu_type="gpu-b200-sxm", gpu_count=1, preset="1gpu-20vcpu-224gb", output_uri="s3://unit-output/task/results/", extra_env={"AWS_ACCESS_KEY_ID": "executing-access", "AWS_SECRET_ACCESS_KEY": "executing-secret", "AWS_ENDPOINT_URL": "https://storage.eu-west1.nebius.cloud"})
    assert report["execution_readiness"] == "pass"
    assert provider.connections[0]["access"] == "executing-access"


def test_workflow_adapter_checks_planned_output_not_config_bucket_root(provider, monkeypatch):
    from npa.cli.workbench.workflow import _execution_target_preflight
    from npa.orchestration.npa_workflow.submit import load_spec_for_submit
    from pathlib import Path

    # Use the real planner on an existing shipped spec. Only external provider
    # calls are replaced; all resolved output locations flow into actual probes.
    path = Path(__file__).parents[3] / "workflows" / "testing" / "cosmos-fetch.yaml"
    spec = load_spec_for_submit(path, config_overrides={"bucket": "unit-output", "prefix": "task"})
    selected, report = _execution_target_preflight(spec, project="unit", context="unit-context", region="eu-west1", run_id="unit-run", assume_decision="", credentials=target().credentials)
    assert report["destination_count"] >= 1
    assert selected.output_uris
    assert all(call[2] for call in provider.s3.calls)


def test_actual_submit_cli_rejects_denied_effective_prefix_before_create(provider, configured, monkeypatch, tmp_path):
    from npa.cli.main import app
    from npa.cli.workbench import workflow as cli
    from typer.testing import CliRunner
    from unittest.mock import Mock

    spec = tmp_path / "target.yaml"
    spec.write_text("""apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata: {name: target}
config: {bucket: spec-output, prefix: spec-prefix}
resources: {cpu: {cloud: kubernetes, cpus: 1, memory: 1Gi}}
initial: execute
states:
  execute:
    resources: cpu
    run: {shell: 'true'}
    terminal: true
""")
    monkeypatch.setattr(cli, "_adopt_npa_kubeconfig", lambda context: True)
    monkeypatch.setattr("npa.orchestration.npa_workflow.model_cache_preflight.adopt_model_cache_claim", lambda **kwargs: "")
    launch = Mock()
    monkeypatch.setattr("npa.orchestration.skypilot.workflow.submit_workflow", launch)
    monkeypatch.setenv("NPA_S3_BUCKET", "env-output")
    monkeypatch.setenv("NPA_S3_PREFIX", "env-prefix")
    provider.s3.denied_prefix = "cli-prefix"
    result = CliRunner().invoke(app, ["workbench", "workflow", "submit", str(spec),
        "--project", "unit", "--run-id", "unit-run", "--infra", "k8s/unit-context",
        "--s3-bucket", "cli-output", "--s3-prefix", "cli-prefix", "--var", "bucket=var-output",
        "--image", "ghcr.io/example/unit:dev", "--no-stage-src", "--skip-preflight", "--no-preflight-images", "--no-deploy-if-absent"])
    assert result.exit_code == 1, result.output
    assert "storage_access" in result.output
    assert provider.s3.calls[0][1] == "cli-output"
    assert provider.s3.calls[0][2].startswith("cli-prefix/")
    launch.assert_not_called()
    assert "private-provider-text" not in result.output


def test_single_node_gpu_preflight_rejects_wrong_product(provider, monkeypatch):
    from npa.cli.workbench.workflow import _preflight_submit_gang_capacity
    from npa.orchestration.skypilot.k8s_gpu_catalog import (
        KubernetesGpuInventory, KubernetesGpuNode, UnsatisfiableAcceleratorError,
    )
    inventory = KubernetesGpuInventory(
        "unit-context", 1, 1, 1, 1, ("NVIDIA-B200",), {}, nodes=(
            KubernetesGpuNode("unit-node", True, True, ("NVIDIA-B200",), 1, 1, 0, 1, free_pod_slots=1),
        ),
    )
    monkeypatch.setattr("npa.orchestration.skypilot.k8s_gpu_catalog.discover_kubernetes_gpu_inventory", lambda **kwargs: inventory)
    spec = SimpleNamespace(
        states={"train": SimpleNamespace(name="train", resources="gpu")},
        resources={"gpu": {"accelerators": "RTXPRO6000:1"}}, config={},
    )
    with pytest.raises(UnsatisfiableAcceleratorError):
        _preflight_submit_gang_capacity(spec, context="unit-context")
    spec.resources["gpu"]["accelerators"] = "B200:1"
    assert _preflight_submit_gang_capacity(spec, context="unit-context")[0]["node_count"] == 1


def test_bdd_planner_directory_roles_probe_exact_denied_training_child(provider):
    from pathlib import Path
    from npa.cli.workbench.workflow import _execution_target_preflight
    from npa.execution_preflight import workflow_output_destinations
    from npa.orchestration.npa_workflow.submit import load_spec_for_submit

    spec = load_spec_for_submit(Path(__file__).parents[3] / "workflows/testing/bdd100k-pipeline.yaml",
                                config_overrides={"bucket": "unit-output", "prefix": "task"})
    destinations = workflow_output_destinations(spec, run_id="unit-run")
    training = {uri: kind for uri, kind in destinations.items() if "/training/" in uri}
    assert len(training) == 3
    assert all(kind == "directory" and not uri.endswith("/") for uri, kind in training.items())
    provider.s3.denied_prefix = "task/training/bdd100k_rider_train/"
    with pytest.raises(ExecutionPreflightError, match="storage_access"):
        _execution_target_preflight(spec, project="unit", context="unit-context", region="eu-west1",
            run_id="unit-run", assume_decision="", credentials=target().credentials)
    assert any(call[2].startswith(provider.s3.denied_prefix) for call in provider.s3.calls)
    assert not any(call[2].startswith("task/training/.npa-probes/") for call in provider.s3.calls)


def test_render_preserves_directory_roles_for_sdk_gate(provider):
    import json
    from pathlib import Path
    from npa.orchestration.npa_workflow.submit import load_spec_for_submit
    from npa.orchestration.npa_workflow.runtime import plan_preview
    from npa.orchestration.npa_workflow.skypilot_render import render_skypilot_steps_yaml, SkypilotRenderOptions
    import yaml

    spec = load_spec_for_submit(Path(__file__).parents[3] / "workflows/testing/bdd100k-pipeline.yaml",
                                config_overrides={"bucket": "unit-output", "prefix": "task"})
    steps = [step for step in plan_preview(spec, run_id="unit-run").steps if step.state == "train-rider"]
    rendered = render_skypilot_steps_yaml(spec, steps, run_id="unit-run",
        options=SkypilotRenderOptions(image_overrides={"*": "ghcr.io/example/unit:dev"}, materialize_registry_secrets=False))
    task = next(doc for doc in yaml.safe_load_all(rendered) if "envs" in doc)
    declaration = json.loads(task["envs"]["NPA_EXECUTION_OUTPUTS"])
    assert declaration == [{"uri": "s3://unit-output/task/training/bdd100k_rider_train", "kind": "directory"}]


def raw_task(**env):
    return {"name": "unit", "resources": {"cloud": "kubernetes", "region": "unit-context"},
            "envs": {"NPA_OUTPUT_PATH": "s3://unit-output/task/raw-output",
                     "AWS_ACCESS_KEY_ID": "yaml-access", "AWS_SECRET_ACCESS_KEY": "yaml-secret",
                     "AWS_ENDPOINT_URL": "https://storage.eu-west1.nebius.cloud", **env}, "run": "true"}


def test_raw_production_environment_supplies_exact_principal(provider, configured):
    from npa.execution_preflight import preflight_skypilot_submission

    document = raw_task()
    selected, report, injected = preflight_skypilot_submission([document], project="unit", infra="k8s/unit-context")
    assert selected.credentials.provenance["credentials"] == "workflow.env"
    assert provider.connections[0]["access"] == "yaml-access"
    assert report["execution_readiness"] == "pass"
    assert injected["AWS_ACCESS_KEY_ID"] == document["envs"]["AWS_ACCESS_KEY_ID"]
    assert provider.s3.calls[0][2].startswith("task/raw-output/.npa-probes/")


def test_multi_document_execution_header_is_not_mutated(provider, configured):
    from npa.execution_preflight import preflight_skypilot_submission

    header = {"name": "workflow", "execution": "serial"}
    preflight_skypilot_submission([header, raw_task()], project="unit", infra="k8s/unit-context")
    assert header == {"name": "workflow", "execution": "serial"}


def test_controller_and_unspecified_task_are_pinned_to_the_verified_context(provider, configured):
    from npa.execution_preflight import preflight_skypilot_submission

    document = raw_task()
    document["resources"].pop("region")
    config = {"jobs": {"controller": {"resources": {"cloud": "kubernetes"}}},
              "kubernetes": {"allowed_contexts": ["stale-context"]}}
    preflight_skypilot_submission([document], project="unit", infra="k8s/unit-context", global_config=config)
    assert document["resources"]["region"] == "unit-context"
    assert config["jobs"]["controller"]["resources"]["region"] == "unit-context"
    assert config["kubernetes"]["allowed_contexts"] == ["unit-context"]


def test_mismatched_controller_context_prevents_storage(provider, configured):
    from npa.execution_preflight import preflight_skypilot_submission

    config = {"jobs": {"controller": {"resources": {"cloud": "kubernetes", "region": "wrong-context"}}}}
    with pytest.raises(ExecutionPreflightError, match="cluster_owner"):
        preflight_skypilot_submission([raw_task()], project="unit", infra="k8s/unit-context", global_config=config)
    assert not provider.s3.calls


def test_internal_skypilot_override_prevents_storage(provider, configured, monkeypatch):
    from npa.execution_preflight import preflight_skypilot_submission

    monkeypatch.setenv("SKYPILOT_CONFIG", '{"kubernetes":{"pod_config":{"secret":"private-value"}}}')
    with pytest.raises(ExecutionPreflightError, match="worker_environment") as caught:
        preflight_skypilot_submission([raw_task()], project="unit", infra="k8s/unit-context")
    assert "private-value" not in str(caught.value)
    assert not provider.s3.calls


@pytest.mark.parametrize("explicit_path", [False, True])
def test_implicit_project_override_prevents_storage(provider, configured, monkeypatch, tmp_path, explicit_path):
    from npa.execution_preflight import preflight_skypilot_submission

    path = tmp_path / ("other.yaml" if explicit_path else ".sky.yaml")
    path.write_text("kubernetes: {pod_config: {private: value}}")
    if explicit_path:
        monkeypatch.setenv("SKYPILOT_PROJECT_CONFIG", str(path))
    with pytest.raises(ExecutionPreflightError, match="--config-path"):
        preflight_skypilot_submission([raw_task()], project="unit", infra="k8s/unit-context", cwd=str(tmp_path))
    assert not provider.s3.calls


def test_workflow_ledger_prefix_uses_interpreter_resolved_run_tokens(provider, tmp_path):
    from npa.execution_preflight import workflow_output_destinations
    from npa.orchestration.npa_workflow.submit import load_spec_for_submit

    source = tmp_path / "spec.yaml"
    source.write_text("""apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata: {name: token-prefix}
config: {bucket: unit-output, prefix: 'task/{{run.id}}'}
initial: execute
states:
  execute:
    run: {shell: 'true'}
    terminal: true
""")
    assert workflow_output_destinations(load_spec_for_submit(source), run_id="unit-run") == {
        "s3://unit-output/task/unit-run/": "directory",
    }


@pytest.mark.parametrize("prefix", ["explicit-prefix", "", "separate-ledger-prefix"])
def test_runtime_resolved_spec_overrides_ambient_storage_through_actual_sdk(provider, configured, monkeypatch, tmp_path, prefix):
    import yaml
    from unittest.mock import Mock
    from npa.orchestration.npa_workflow.runtime import RuntimeOptions, SkyPilotWaveExecutor, WaveAttempt
    from npa.orchestration.npa_workflow.submit import load_spec_for_submit
    from npa.orchestration.skypilot import workflow

    source = tmp_path / "spec.yaml"
    source.write_text("""apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata: {name: override}
config: {bucket: initial-output, prefix: initial-prefix}
initial: execute
states:
  execute:
    run: {shell: 'true'}
    terminal: true
""")
    spec = load_spec_for_submit(source, config_overrides={"bucket": "unit-output", "prefix": prefix})
    monkeypatch.setenv("NPA_S3_BUCKET", "ambient-output")
    monkeypatch.setenv("NPA_S3_PREFIX", "ambient-prefix")
    document = raw_task(NPA_OUTPUT_PATH=f"s3://unit-output/{prefix + '/' if prefix else ''}outputs")
    if prefix == "separate-ledger-prefix":
        document["envs"].pop("NPA_OUTPUT_PATH")
        document["envs"]["NPA_EXECUTION_OUTPUTS"] = '[{"uri":"s3://unit-output/sibling-evaluation/metrics.json","kind":"file"}]'
    rendered = tmp_path / "wave.yaml"
    rendered.write_text(yaml.safe_dump(document))
    runtime = SimpleNamespace(isolated_config_dir=tmp_path, sky_bin=tmp_path / "sky", global_config_path=None)
    monkeypatch.setattr(workflow, "resolve_config", lambda **kwargs: runtime)
    monkeypatch.setattr(workflow, "ensure_skypilot_version", lambda path: path)
    monkeypatch.setattr(workflow, "sky_environment", lambda path: {})
    # Stop only at the external controller boundary, after the actual shared
    # SDK preflight has verified the selected configuration and exact writes.
    controller = Mock(side_effect=RuntimeError("verified-controller-boundary"))
    monkeypatch.setattr(workflow, "_ensure_local_api_daemon_cwd_locked", controller)
    executor = SkyPilotWaveExecutor(spec, run_id="unit", options=RuntimeOptions(project="unit", infra="k8s/unit-context"), output_checker=lambda uri: True)
    with pytest.raises(RuntimeError, match="verified-controller-boundary"):
        executor._submit(rendered, "unit", WaveAttempt(key="unit", states=["execute"], kind="serial", group="", attempt=1))
    controller.assert_called_once()
    assert provider.s3.calls
    assert {call[1] for call in provider.s3.calls} == {"unit-output"}
    assert all("ambient-prefix" not in call[2] for call in provider.s3.calls)
    if prefix == "separate-ledger-prefix":
        assert all(call[2].startswith("sibling-evaluation/") for call in provider.s3.calls)


def test_raw_explicit_pair_overrides_yaml_and_saved_pair_consistently(provider, configured):
    from npa.execution_preflight import preflight_skypilot_submission

    document = raw_task()
    preflight_skypilot_submission([document], project="unit", infra="k8s/unit-context",
        extra_env={"AWS_ACCESS_KEY_ID": "explicit-access", "AWS_SECRET_ACCESS_KEY": "explicit-secret"})
    assert provider.connections[0]["access"] == "explicit-access"
    assert document["envs"]["AWS_ACCESS_KEY_ID"] == "explicit-access"


@pytest.mark.parametrize("change,check", [
    ({"resources": {"cloud": "kubernetes", "region": "other-context"}}, "cluster_owner"),
    ({"envs": {"UNKNOWN_WRITER": "s3://unit-output/ambiguous"}}, "storage_target"),
    ({"resources": [{"cloud": "kubernetes"}]}, "gpu"),
])
def test_raw_target_uncertainty_never_probes(provider, configured, change, check):
    from npa.execution_preflight import preflight_skypilot_submission

    document = raw_task()
    document.update(change)
    with pytest.raises(ExecutionPreflightError, match=check):
        preflight_skypilot_submission([document], project="unit", infra="k8s/unit-context")
    assert not provider.s3.calls


def test_actual_raw_cli_prefix_denial_prevents_submit(provider, configured, monkeypatch, tmp_path):
    import yaml
    from unittest.mock import Mock
    from typer.testing import CliRunner
    from npa.cli.main import app

    path = tmp_path / "raw.yaml"
    path.write_text(yaml.safe_dump(raw_task()))
    runtime = SimpleNamespace(isolated_config_dir=tmp_path, sky_bin=tmp_path / "sky", global_config_path=None)
    monkeypatch.setattr("npa.orchestration.skypilot._bin.resolve_config", lambda **kwargs: runtime)
    monkeypatch.setattr("npa.orchestration.skypilot._bin.ensure_skypilot_version", lambda path: path)
    provider.s3.denied_prefix = "task/raw-output/"
    launch = Mock()
    monkeypatch.setattr("npa.orchestration.skypilot.workflow.submit_workflow", launch)
    result = CliRunner().invoke(app, ["workbench", "workflow", "submit", str(path), "--project", "unit",
        "--infra", "k8s/unit-context", "--run-id", "raw-unit", "--skip-preflight"])
    assert result.exit_code == 1, result.output
    assert "storage_access" in result.output
    assert provider.connections[0]["access"] == "yaml-access"
    launch.assert_not_called()


def test_actual_sdk_prefix_denial_prevents_controller_and_job_create(provider, configured, monkeypatch, tmp_path):
    import yaml
    from unittest.mock import Mock
    from npa.orchestration.skypilot import workflow

    path = tmp_path / "raw.yaml"
    path.write_text(yaml.safe_dump(raw_task()))
    runtime = SimpleNamespace(isolated_config_dir=tmp_path, sky_bin=tmp_path / "sky", global_config_path=None)
    monkeypatch.setattr(workflow, "resolve_config", lambda **kwargs: runtime)
    monkeypatch.setattr(workflow, "ensure_skypilot_version", lambda path: path)
    monkeypatch.setattr(workflow, "sky_environment", lambda path: {})
    controller = Mock()
    launch = Mock()
    monkeypatch.setattr(workflow, "_ensure_local_api_daemon_cwd_locked", controller)
    monkeypatch.setattr(workflow, "_run_launch", launch)
    provider.s3.denied_prefix = "task/raw-output/"
    with pytest.raises(workflow.SkyPilotSubmitError, match="storage_access"):
        workflow.submit_workflow(path, "raw-unit", project="unit", infra="k8s/unit-context")
    controller.assert_not_called()
    launch.assert_not_called()


@pytest.mark.parametrize("matching", [False, True])
@pytest.mark.parametrize("mount_kind", ["writable", "scalar-input"])
def test_actual_sdk_nebius_mount_profile_verified_before_storage_and_create(provider, configured, monkeypatch, tmp_path, matching, mount_kind):
    import os
    import shlex
    import sys
    import yaml
    from unittest.mock import Mock
    from npa.orchestration.skypilot import workflow

    aws = tmp_path / ".aws"
    aws.mkdir()
    access, secret = ("yaml-access", "yaml-secret") if matching else ("different-access", "different-secret")
    (aws / "credentials").write_text(f"[nebius]\naws_access_key_id = {access}\naws_secret_access_key = {secret}\n")
    (aws / "config").write_text("[profile nebius]\nregion = eu-west1\nendpoint_url = https://storage.eu-west1.nebius.cloud\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    interpreter = bin_dir / "python"
    interpreter.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    interpreter.chmod(0o755)
    document = raw_task()
    document["file_mounts"] = {"/mnt/output": {"source": "nebius://unit-output/task/mounted", "store": "NEBIUS", "mode": "MOUNT"}}
    if mount_kind == "scalar-input":
        document["file_mounts"] = {"/mnt/input": "nebius://public-input/data"}
        document["envs"].pop("NPA_OUTPUT_PATH")
        document["envs"]["NPA_EXECUTION_OUTPUTS"] = "[]"
    path = tmp_path / "raw.yaml"
    path.write_text(yaml.safe_dump(document))
    runtime = SimpleNamespace(isolated_config_dir=tmp_path, sky_bin=bin_dir / "sky", global_config_path=None)
    monkeypatch.setattr(workflow, "resolve_config", lambda **kwargs: runtime)
    monkeypatch.setattr(workflow, "ensure_skypilot_version", lambda path: path)
    env = {key: value for key, value in os.environ.items() if not key.startswith("AWS_")}
    env.update(HOME=str(tmp_path), AWS_EC2_METADATA_DISABLED="true")
    monkeypatch.setattr(workflow, "sky_environment", lambda path: env)
    controller = Mock(side_effect=RuntimeError("reached verified controller boundary"))
    launch = Mock()
    monkeypatch.setattr(workflow, "_ensure_local_api_daemon_cwd_locked", controller)
    monkeypatch.setattr(workflow, "_run_launch", launch)
    if matching:
        with pytest.raises(RuntimeError, match="reached verified controller boundary"):
            workflow.submit_workflow(path, "raw-unit", project="unit", infra="k8s/unit-context")
        controller.assert_called_once()
        if mount_kind == "scalar-input":
            assert not provider.s3.calls
        else:
            assert {call[2].split("/.npa-probes/")[0] for call in provider.s3.calls} == {"task/raw-output", "task/mounted"}
    else:
        with pytest.raises(workflow.SkyPilotSubmitError, match="AWS profile disagrees") as caught:
            workflow.submit_workflow(path, "raw-unit", project="unit", infra="k8s/unit-context")
        assert access not in str(caught.value) and secret not in str(caught.value)
        assert not provider.s3.calls
        controller.assert_not_called()
    launch.assert_not_called()


@pytest.mark.parametrize("resources,infra", [
    ({"cloud": "nebius", "region": "eu-west1", "accelerators": "B200:1"}, "nebius"),
    ({"region": "eu-west1", "accelerators": "B200:1"}, "nebius"),
    ({"infra": "nebius/eu-west1", "accelerators": "B200:1"}, ""),
])
def test_actual_native_sdk_persists_verified_project_and_resource_shape_before_controller(provider, configured, monkeypatch, tmp_path, resources, infra):
    import json
    import yaml
    from pathlib import Path
    from unittest.mock import Mock
    from npa.orchestration.skypilot import workflow, native_preflight

    document = raw_task()
    document["resources"] = resources
    source = tmp_path / "native.yaml"
    source.write_text(yaml.safe_dump(document))
    runtime = SimpleNamespace(isolated_config_dir=tmp_path, sky_bin=tmp_path / "sky", global_config_path=None)
    monkeypatch.setattr(workflow, "resolve_config", lambda **kwargs: runtime)
    monkeypatch.setattr(workflow, "ensure_skypilot_version", lambda path: path)
    monkeypatch.setattr(workflow, "sky_environment", lambda path: {})
    requests = []

    def managed(args, **kwargs):
        request = json.loads(Path(args[2]).read_text())
        requests.append(request)
        assert request["project"] == "project-unit"
        selections = [{**shape["resources"], "instance_type": "gpu-unit_preset-unit"} for shape in request["shapes"]]
        Path(args[3]).write_text(json.dumps({"status": "pass", "resources": selections}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(native_preflight.subprocess, "run", managed)
    controller = Mock(side_effect=RuntimeError("verified-native-controller-boundary"))
    monkeypatch.setattr(workflow, "_ensure_local_api_daemon_cwd_locked", controller)
    with pytest.raises(RuntimeError, match="verified-native-controller-boundary"):
        workflow.submit_workflow(source, "native-unit", project="unit", infra=infra, controller_backend="nebius")
    controller.assert_called_once()
    assert len(requests) == 1
    assert len(requests[0]["shapes"]) == 2  # Native task plus native controller.
    assert requests[0]["shapes"][0]["resources"]["accelerators"] == "B200:1"
    assert requests[0]["shapes"][0]["resources"]["cloud"] == "nebius"
    assert requests[0]["shapes"][0]["resources"]["region"] == "eu-west1"
    config = yaml.safe_load(Path(controller.call_args.kwargs["env"]["SKYPILOT_GLOBAL_CONFIG"]).read_text())
    assert config["nebius"]["region_configs"]["eu-west1"]["project_id"] == "project-unit"
    assert config["jobs"]["controller"]["resources"]["region"] == "eu-west1"
    assert config["jobs"]["controller"]["resources"]["instance_type"] == "gpu-unit_preset-unit"
    assert provider.s3.calls
    prepared = next(tmp_path.rglob("workflow.yaml"))
    assert yaml.safe_load(prepared.read_text())["resources"]["instance_type"] == "gpu-unit_preset-unit"


@pytest.mark.parametrize("entry", [
    {"name": "AWS_SESSION_TOKEN", "value": "different-principal-token"},
    {"name": "AWS_SESSION_TOKEN", "valueFrom": {"secretKeyRef": {"name": "other", "key": "session"}}},
    {"name": "AWS_SESSION_TOKEN", "value": "", "valueFrom": {"secretKeyRef": {"name": "other", "key": "session"}}},
])
def test_pod_session_token_rejected_before_raw_storage_probe(provider, configured, entry):
    from npa.execution_preflight import preflight_skypilot_submission

    document = raw_task()
    document["resources"]["kubernetes"] = {"pod_config": {"spec": {"containers": [{"env": [entry]}]}}}
    with pytest.raises(ExecutionPreflightError, match="worker_environment"):
        preflight_skypilot_submission([document], project="unit", infra="k8s/unit-context")
    assert not provider.s3.calls
