"""Native preflight follows Sky's selected credentials and optimizer before writes."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import yaml

from npa.execution_preflight import ExecutionPreflightError, ExecutionTarget
from npa.orchestration.skypilot import native_preflight as native


@pytest.fixture
def target():
    return ExecutionTarget("fixture", "project-fixture", "tenant-fixture", "region-fixture")


@pytest.fixture
def managed(monkeypatch):
    """Replace only managed Sky/provider dependencies; execute our real probe."""
    state = NS(project="project-fixture", tenant="tenant-fixture", region="region-fixture",
               owner="tenant-fixture", gpu_count=1, version="0.12.2", events=[],
               selected="gpu-fixture_1gpu", missing_preset=False, provider_error=None)

    class Dag:
        def __enter__(self):
            state.dag = self
            return self

        def __exit__(self, *args):
            pass

    class Task:
        @staticmethod
        def from_yaml_config(config):
            assert set(config) <= {"resources", "num_nodes", "config"}, "eager storage must not be constructed"
            state.events.append(("task", deepcopy(config)))
            state.dag.task = NS(config=config)
            return state.dag.task

    class Optimizer:
        @staticmethod
        def optimize(dag, *, quiet):
            state.events.append(("optimize",))
            resource = deepcopy(dag.task.config["resources"])
            gpu_count = 1 if resource.get("accelerators") else 0
            instance = state.selected if gpu_count else "cpu-fixture_4vcpu"
            resource.pop("any_of", None)
            resource.pop("ordered", None)
            resource["instance_type"] = instance
            # Pinned upstream to_yaml_config produces infra, not cloud/region.
            resource.pop("cloud", None)
            resource.pop("region", None)
            resource["infra"] = "nebius/region-fixture"
            if "config" in dag.task.config:
                resource["_cluster_config_overrides"] = dag.task.config["config"]
            dag.task.best_resources = NS(cloud="Nebius", region=state.region, zone=None,
                                         instance_type=instance,
                                         accelerators={"fixture": 1} if gpu_count else None,
                                         to_yaml_config=lambda: resource)

    class ProjectService:
        def __init__(self, sdk):
            assert sdk == "executing-sky-sdk"

        def get(self, request, *, timeout):
            state.events.append(("project-get", request.id))
            if state.provider_error:
                raise state.provider_error
            return NS(metadata=NS(id=state.project, parent_id=state.owner),
                      status=NS(region=state.region))

    class PlatformService:
        def __init__(self, sdk):
            assert sdk == "executing-sky-sdk"

        def get_by_name(self, request, *, timeout):
            state.events.append(("platform-get", request.parent_id, request.name))
            gpu = request.name.startswith("gpu-")
            presets = [] if state.missing_preset else [NS(name="1gpu" if gpu else "4vcpu",
                resources=NS(gpu_count=state.gpu_count if gpu else 0))]
            return NS(metadata=NS(id="catalog-product", name=request.name, parent_id="provider-catalog"),
                      spec=NS(presets=presets))

    def get_project(region):
        state.events.append(("sky-project-lookup", region))
        return state.project

    adaptor = NS(get_tenant_id=lambda: state.tenant, sdk=lambda: "executing-sky-sdk",
                 iam=lambda: NS(ProjectServiceClient=ProjectService, GetProjectRequest=NS),
                 compute=lambda: NS(PlatformServiceClient=PlatformService),
                 nebius_common=lambda: NS(GetByNameRequest=NS), READ_TIMEOUT=30,
                 sync_call=lambda result: result)
    sky = NS(__version__=state.version, Dag=Dag, Task=Task,
             skypilot_config=NS(get_effective_region_config=lambda **kwargs: state.project))
    for name, module in {"sky": sky, "sky.adaptors": NS(nebius=adaptor),
                         "sky.optimizer": NS(Optimizer=Optimizer),
                         "sky.provision.nebius": NS(utils=NS(get_project_by_region=get_project))}.items():
        monkeypatch.setitem(sys.modules, name, module)
    state.sky = sky
    return state


def request():
    return {"project": "project-fixture", "tenant": "tenant-fixture", "region": "region-fixture",
            "shapes": [{"resources": {"cloud": "nebius", "region": "region-fixture", "accelerators": "fixture:1"}}]}


def test_managed_probe_verifies_actual_sky_identity_then_optimizer_then_project_product(managed):
    result = native._managed_probe(request())
    assert result["status"] == "pass"
    assert result["resources"][0]["instance_type"] == "gpu-fixture_1gpu"
    assert "infra" not in result["resources"][0]
    assert [event[0] for event in managed.events] == ["sky-project-lookup", "project-get", "task", "optimize", "platform-get"]
    assert managed.events[-1] == ("platform-get", "project-fixture", "gpu-fixture")


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("tenant", "other-tenant", "tenant"), ("project", "other-project", "project"),
    ("owner", "other-tenant", "scope"), ("region", "other-region", "scope"),
    ("owner", "", "identity"),
])
def test_wrong_actual_credential_scope_never_optimizes_or_queries_products(managed, field, value, reason):
    setattr(managed, field, value)
    with pytest.raises(native._NativeFailure) as raised:
        native._managed_probe(request())
    assert raised.value.reason == reason
    assert not any(event[0] in {"optimize", "platform-get"} for event in managed.events)


@pytest.mark.parametrize(("field", "value", "reason", "status"), [
    ("missing_preset", True, "product", "fail"),
    ("gpu_count", 8, "shape", "fail"),
    ("selected", "not-a-provider-instance", "catalog", "unknown"),
    ("provider_error", RuntimeError("credential-secret provider-private-url"), "provider", "unknown"),
])
def test_provider_shape_and_uncertainty_fail_closed(managed, field, value, reason, status):
    setattr(managed, field, value)
    with pytest.raises(native._NativeFailure) as raised:
        native._managed_probe(request())
    assert (raised.value.reason, raised.value.status) == (reason, status)
    assert "credential-secret" not in str(raised.value)


def test_wrong_managed_version_fails_before_provider(managed):
    managed.sky.__version__ = "0.11.0"
    with pytest.raises(native._NativeFailure, match="runtime"):
        native._managed_probe(request())
    assert managed.events == []


@pytest.fixture
def subprocess_probe(monkeypatch, managed):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        req, result = Path(argv[-2]), Path(argv[-1])
        config = Path(kwargs["env"]["SKYPILOT_GLOBAL_CONFIG"])
        assert req.stat().st_mode & 0o777 == 0o600
        assert config.stat().st_mode & 0o777 == 0o600
        assert req.parent.stat().st_mode & 0o777 == 0o700
        parsed_config = yaml.safe_load(config.read_text())
        assert parsed_config["nebius"]["region_configs"]["region-fixture"]["project_id"] == "project-fixture"
        try:
            payload = native._managed_probe(json.loads(req.read_text()))
        except native._NativeFailure as exc:
            payload = {"status": exc.status, "reason": exc.reason, "check": exc.check}
        result.write_text(json.dumps(payload))
        return subprocess.CompletedProcess(argv, 0, stdout="provider-secret", stderr="private-url")

    monkeypatch.setattr(native.subprocess, "run", run)
    return calls


def test_native_gpu_task_and_cpu_controller_pinned_without_payload_or_storage(target, subprocess_probe, tmp_path):
    documents = [{"resources": {"cloud": "nebius", "accelerators": "fixture:1"},
                  "run": "training", "file_mounts": {"/data": "s3://private/data"},
                  "config": {"docker": {"run_options": ["--ipc=host"]}}}]
    config = {"jobs": {"controller": {"resources": {"cloud": "nebius", "cpus": "4+", "autostop": False}}}}
    original = deepcopy(documents)
    native.verify_native_nebius_submission(documents=documents, target=target,
        global_config=config, sky_bin="/managed/bin/sky", extra_env={"HOME": "/isolated", "EXECUTION_SECRET": "kept"}, cwd=tmp_path)
    assert documents[0]["resources"]["instance_type"] == "gpu-fixture_1gpu"
    assert "_cluster_config_overrides" not in documents[0]["resources"]
    assert documents[0]["file_mounts"] == original[0]["file_mounts"]
    assert config["jobs"]["controller"]["resources"]["instance_type"] == "cpu-fixture_4vcpu"
    assert config["jobs"]["controller"]["resources"]["autostop"] is False
    argv, kwargs = subprocess_probe[0]
    assert argv[0] == "/managed/bin/python"
    assert kwargs["cwd"] == tmp_path
    assert kwargs["env"]["HOME"] == "/isolated"
    assert kwargs["env"]["EXECUTION_SECRET"] == "kept"
    assert not Path(argv[-2]).exists()
    assert "private" not in " ".join(argv)


@pytest.mark.parametrize("config", [
    {"nebius": {"project_id": "other-project"}},
    {"nebius": {"tenant_id": "other-tenant"}},
    {"nebius": {"region_configs": {"region-fixture": {"project_id": "other-project"}}}},
])
def test_explicit_scope_mismatch_rejected_without_subprocess(target, subprocess_probe, config):
    documents = [{"resources": {"cloud": "nebius"}}]
    before = deepcopy(config)
    with pytest.raises(ExecutionPreflightError):
        native.verify_native_nebius_submission(documents=documents, target=target,
            global_config=config, sky_bin="/managed/bin/sky", extra_env={})
    assert subprocess_probe == []
    assert config == before


@pytest.mark.parametrize("resources", [
    {"cloud": "nebius", "region": "other-region"},
    {"infra": "nebius/other-region"},
    {"cloud": "nebius", "any_of": [{"region": "other-region"}]},
])
def test_different_resource_destination_rejected_before_provider(target, subprocess_probe, resources):
    with pytest.raises(ExecutionPreflightError):
        native.verify_native_nebius_submission(documents=[{"resources": resources}], target=target,
            global_config={}, sky_bin="/managed/bin/sky", extra_env={})
    assert subprocess_probe == []


def test_provider_failure_keeps_input_copies_and_raw_errors_private(target, managed, subprocess_probe):
    managed.provider_error = RuntimeError("private-credential and URL")
    documents = [{"resources": {"cloud": "nebius"}}]
    config = {}
    before = deepcopy(documents)
    with pytest.raises(ExecutionPreflightError) as raised:
        native.verify_native_nebius_submission(documents=documents, target=target,
            global_config=config, sky_bin="/managed/bin/sky", extra_env={})
    assert raised.value.status == "unknown"
    assert "private" not in str(raised.value)
    assert documents == before
    assert config == {}


def test_task_identity_override_is_not_silently_ignored(target, subprocess_probe):
    with pytest.raises(ExecutionPreflightError, match="overrides"):
        native.verify_native_nebius_submission(documents=[{"resources": {"cloud": "nebius"},
            "config": {"nebius": {"tenant_id": "other-tenant"}}}], target=target,
            global_config={}, sky_bin="/managed/bin/sky", extra_env={})
    assert not subprocess_probe


def test_managed_entrypoint_error_protocol_never_serializes_exception(monkeypatch, tmp_path, capsys):
    req, result = tmp_path / "request.json", tmp_path / "result.json"
    req.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["native_preflight.py", str(req), str(result)])
    def fail(_request):
        raise RuntimeError("secret credential private URL")
    monkeypatch.setattr(native, "_managed_probe", fail)
    native._main()
    assert json.loads(result.read_text()) == {"check": "native_scope", "reason": "runtime", "status": "unknown"}
    assert result.stat().st_mode & 0o777 == 0o600
    assert capsys.readouterr().out == ""


def test_no_executing_runtime_cannot_be_treated_as_ready(monkeypatch, target):
    def fail(*args, **kwargs):
        raise OSError("private interpreter path")
    monkeypatch.setattr(native.subprocess, "run", fail)
    with pytest.raises(ExecutionPreflightError) as raised:
        native.verify_native_nebius_submission(documents=[{"resources": {"cloud": "nebius"}}],
            target=target, global_config={}, sky_bin="/missing/sky", extra_env={})
    assert raised.value.status == "unknown"
    assert "/missing" not in str(raised.value)


def test_nested_alternative_cannot_override_provider_identity(target, subprocess_probe):
    with pytest.raises(ExecutionPreflightError, match="overrides"):
        native.verify_native_nebius_submission(documents=[{"resources": {"cloud": "nebius",
            "any_of": [{"_cluster_config_overrides": {"nebius": {"tenant_id": "other-tenant"}}}]}}],
            target=target, global_config={}, sky_bin="/managed/bin/sky", extra_env={})
    assert not subprocess_probe


def test_internal_resource_overrides_are_not_duplicated_in_task_config(target, managed, subprocess_probe):
    documents = [{"resources": {"cloud": "nebius",
                  "_cluster_config_overrides": {"docker": {"run_options": ["--ipc=host"]}}}}]
    native.verify_native_nebius_submission(documents=documents, target=target,
        global_config={}, sky_bin="/managed/bin/sky", extra_env={})
    parsed_shape = next(event[1] for event in managed.events if event[0] == "task")
    assert "config" in parsed_shape
    assert "_cluster_config_overrides" not in parsed_shape["resources"]
    assert documents[0]["resources"]["_cluster_config_overrides"] == parsed_shape["config"]
