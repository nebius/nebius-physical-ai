"""Contract tests for the immutable, authorization-gated robomimic BYOF path."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.workbench import workflow as workflow_cli
from npa.cli.main import app
from npa.deploy import images
from npa.orchestration.npa_workflow import build_plan, load_spec


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows" / "testing" / "byof-robomimic.yaml"
READINESS = ROOT / "workflows" / "testing" / "byof-robomimic.readiness.json"
DOC = ROOT / "docs" / "workbench" / "byof-robomimic.md"
IMAGE_ROOT = ROOT / "npa" / "docker" / "workbench" / "robomimic"
SMOKE = IMAGE_ROOT / "smoke.py"
BAKED_LOCK = IMAGE_ROOT / "baked-requirements.lock"
PROFILE = (
    ROOT
    / "npa"
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "byof-solution-smoke-robomimic-b200-gpu.yaml"
)
SOURCE_REVISION = "d309eaecc18acf4152a830a895a6984b8ac71b05"
DATASET_REVISION = "74fa018461f479cd9fd15b924a16103012096203"
DATASET_SHA256 = "2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540"
BASE_DIGEST = "sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84"
BUILD_COMMAND_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
DEPENDENCY_LOCK_SHA256 = (
    "65efcf0065ad4662b348e54e3f2d86996d934a518fcad0e89ecf012399ce1504"
)


def _workflow_config() -> dict[str, object]:
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    config = payload["config"]
    assert isinstance(config, dict)
    return config


def _smoke_python() -> str:
    return SMOKE.read_text(encoding="utf-8")


def _live_e2e_module():
    """Import the live module so its pure contracts run in the default suite."""

    tests_root = str(ROOT / "npa")
    sys.path.insert(0, tests_root)
    try:
        return importlib.import_module("tests.e2e.test_byof_onboarding_live_e2e")
    finally:
        sys.path.remove(tests_root)


def _byof_runner_module():
    path = ROOT / "npa" / "scripts" / "run_byof_repo.py"
    specification = importlib.util.spec_from_file_location(
        "robomimic_byof_runner_contract", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _fake_robomimic_kubectl(
    *,
    create_failure: bool = False,
    foreign_owner_after_create: bool = False,
    delete_failure: str = "",
    delete_oserror: str = "",
    ownership_get_oserror: str = "",
    absence_get_oserror: str = "",
    replace_before_delete: str = "",
    extra_permission: bool = False,
) -> tuple[list[list[str]], Callable[..., subprocess.CompletedProcess[str]]]:
    calls: list[list[str]] = []
    objects: dict[str, dict[str, object]] = {}
    cleanup_started = False
    replacement_done = False

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal cleanup_started, replacement_done
        calls.append(command)
        if "auth" in command:
            verb = command[command.index("can-i") + 1]
            resource = command[command.index("can-i") + 2]
            allowed = (verb, resource) == ("get", "pods")
            return subprocess.CompletedProcess(
                command, 0 if allowed else 1, "yes\n" if allowed else "no\n", ""
            )
        if "selfsubjectrulesreviews" in " ".join(command):
            payload = {
                "status": {
                    "resourceRules": [
                        {
                            "apiGroups": [""],
                            "resources": ["pods"],
                            "verbs": ["get"],
                        },
                        {
                            "apiGroups": ["authorization.k8s.io"],
                            "resources": [
                                "selfsubjectaccessreviews",
                                "selfsubjectrulesreviews",
                            ],
                            "verbs": ["create"],
                        },
                        {
                            "apiGroups": ["authentication.k8s.io"],
                            "resources": ["selfsubjectreviews"],
                            "verbs": ["create"],
                        },
                    ]
                }
            }
            if extra_permission:
                payload["status"]["resourceRules"].append(
                    {
                        "apiGroups": [""],
                        "resources": ["secrets"],
                        "verbs": ["get"],
                    }
                )
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if "create" in command:
            manifest = yaml.safe_load(str(kwargs.get("input") or ""))
            resource = f"{str(manifest['kind']).lower()}/{manifest['metadata']['name']}"
            manifest["metadata"]["uid"] = f"uid-{len(objects)}"
            manifest["metadata"]["resourceVersion"] = f"rv-{len(objects)}"
            objects[resource] = manifest
            if create_failure:
                if foreign_owner_after_create:
                    manifest["metadata"]["annotations"]["npa.nebius.ai/owner-token"] = (
                        "another-invocation"
                    )
                return subprocess.CompletedProcess(command, 1, "", "response lost")
            return subprocess.CompletedProcess(command, 0, "", "")
        if "delete" in command:
            raw_path = command[command.index("--raw") + 1]
            plural, name = raw_path.rsplit("/", 2)[-2:]
            kind = {
                "serviceaccounts": "serviceaccount",
                "roles": "role",
                "rolebindings": "rolebinding",
            }[plural]
            resource = f"{kind}/{name}"
            if delete_oserror and resource.startswith(delete_oserror):
                raise FileNotFoundError("simulated kubectl launch failure")
            if delete_failure and resource.startswith(delete_failure):
                return subprocess.CompletedProcess(
                    command, 1, "", "simulated delete error"
                )
            manifest = objects.get(resource)
            if (
                manifest is not None
                and replace_before_delete
                and resource.startswith(replace_before_delete)
                and not replacement_done
            ):
                replacement_done = True
                manifest["metadata"]["uid"] = "foreign-uid"
                manifest["metadata"]["resourceVersion"] = "foreign-rv"
                manifest["metadata"]["annotations"]["npa.nebius.ai/owner-token"] = (
                    "another-invocation"
                )
            delete_options = json.loads(str(kwargs.get("input") or "{}"))
            preconditions = delete_options.get("preconditions", {})
            if manifest is not None and (
                preconditions.get("uid") != manifest["metadata"].get("uid")
                or preconditions.get("resourceVersion")
                != manifest["metadata"].get("resourceVersion")
            ):
                return subprocess.CompletedProcess(
                    command, 1, "", "delete precondition failed"
                )
            objects.pop(resource, None)
            return subprocess.CompletedProcess(command, 0, "", "")
        if "get" in command:
            resource = command[command.index("--namespace") + 2]
            manifest = objects.get(resource)
            output_format = command[command.index("-o") + 1]
            if output_format == "json":
                cleanup_started = True
                if ownership_get_oserror and resource.startswith(ownership_get_oserror):
                    raise FileNotFoundError("simulated ownership lookup failure")
            elif (
                cleanup_started
                and absence_get_oserror
                and resource.startswith(absence_get_oserror)
            ):
                raise FileNotFoundError("simulated absence lookup failure")
            if manifest is None:
                return subprocess.CompletedProcess(command, 0, "", "")
            stdout = (
                json.dumps(manifest) if output_format == "json" else f"{resource}\n"
            )
            return subprocess.CompletedProcess(command, 0, stdout, "")
        return subprocess.CompletedProcess(command, 0, "", "")

    return calls, fake_run


@pytest.mark.parametrize(
    "missing_variable",
    (
        "NPA_E2E_PROJECT",
        "NPA_BYOF_ROBOMIMIC_REGISTRY",
        "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY",
        "NPA_BYOF_KUBECONFIG",
        "NPA_BYOF_K8S_CONTEXT",
        "NPA_BYOF_K8S_NAMESPACE",
        "NPA_E2E_S3_BUCKET",
        "NPA_E2E_MK8S_RESERVED_CAPACITY",
        "NPA_BYOF_LIVE_GPU",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_PVC",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256",
        "AWS_ENDPOINT_URL",
    ),
)
def test_robomimic_gate_refuses_missing_manager_context_in_default_suite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_variable: str,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    selectors = {
        "NPA_E2E_PROJECT": "manager-project",
        "NPA_BYOF_ROBOMIMIC_REGISTRY": "private.invalid/robomimic",
        "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY": "private",
        "NPA_BYOF_KUBECONFIG": str(kubeconfig),
        "NPA_BYOF_K8S_CONTEXT": "manager-context",
        "NPA_BYOF_K8S_NAMESPACE": "robomimic-validation",
        "NPA_E2E_S3_BUCKET": "manager-bucket",
        "NPA_E2E_MK8S_RESERVED_CAPACITY": "1",
        "NPA_BYOF_LIVE_GPU": "1",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200": "1",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_PVC": "robomimic-runtime-exact",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256": "a" * 64,
        "AWS_ENDPOINT_URL": "https://storage.test-region.nebius.cloud",
    }
    for variable, value in selectors.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.delenv(missing_variable, raising=False)

    with pytest.raises(AssertionError):
        _live_e2e_module()._robomimic_live_selectors("manager-project")


@pytest.mark.parametrize(
    "endpoint",
    (
        "http://storage.test-region.nebius.cloud",
        "https://user:secret@storage.test-region.nebius.cloud",
        "https://storage.test-region.nebius.cloud.attacker.invalid",
        "https://127.0.0.1",
        "https://storage.test-region.nebius.cloud:8443",
        "https://storage.test-region.nebius.cloud/path",
        "https://storage.test-region.nebius.cloud?target=other",
        "https://storage.test-region.nebius.cloud#fragment",
        "https://storage.test-region.nebius.cloud:invalid",
    ),
)
def test_robomimic_storage_endpoint_refuses_credential_exfiltration(
    endpoint: str,
) -> None:
    with pytest.raises(AssertionError):
        _live_e2e_module()._robomimic_storage_endpoint(endpoint)


def test_robomimic_storage_endpoint_returns_one_canonical_https_origin() -> None:
    endpoint = "https://storage.test-region.nebius.cloud"
    assert _live_e2e_module()._robomimic_storage_endpoint(endpoint) == endpoint


def test_robomimic_strict_attestation_is_resolved_only_in_run_local_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_E2E_MK8S_RESERVED_CAPACITY", "1")
    module = _live_e2e_module()
    rendered = module._materialize_robomimic_attested_profile(
        tmp_path / "robomimic-attested.yaml",
        namespace="robomimic-validation",
        service_account="npa-robomimic-run-scoped",
        runtime_pvc="robomimic-runtime-exact",
        runtime_inventory_sha256="a" * 64,
    )

    source_task = list(yaml.safe_load_all(PROFILE.read_text(encoding="utf-8")))[1]
    rendered_task = list(yaml.safe_load_all(rendered.read_text(encoding="utf-8")))[1]
    assert source_task["envs"]["NPA_ROBOMIMIC_STRICT_B200_ATTESTED"] == ""
    assert rendered_task["envs"]["NPA_ROBOMIMIC_STRICT_B200_ATTESTED"] == "1"
    assert (
        rendered_task["envs"]["NPA_ROBOMIMIC_EXPECTED_NAMESPACE"]
        == "robomimic-validation"
    )
    assert (
        rendered_task["envs"]["NPA_ROBOMIMIC_EXPECTED_SERVICE_ACCOUNT"]
        == "npa-robomimic-run-scoped"
    )
    assert rendered_task["envs"]["NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256"] == "a" * 64
    volumes = rendered_task["config"]["kubernetes"]["pod_config"]["spec"]["volumes"]
    runtime_volume = next(
        item for item in volumes if item["name"] == "robomimic-runtime"
    )
    assert (
        runtime_volume["persistentVolumeClaim"]["claimName"]
        == "robomimic-runtime-exact"
    )
    assert (
        rendered_task["config"]["kubernetes"]["pod_config"]["spec"][
            "serviceAccountName"
        ]
        == "npa-robomimic-run-scoped"
    )


def test_robomimic_observer_manifest_is_run_scoped_and_least_privilege() -> None:
    module = _live_e2e_module()
    run_id = "robomimic-contract-test"
    name = module._robomimic_observer_name(run_id)
    manifests = module._robomimic_observer_manifests(
        run_id=run_id,
        namespace="robomimic-validation",
        service_account=name,
        owner_token="unit-owner-token",
    )

    assert len(manifests) == 3
    assert {manifest["kind"] for manifest in manifests} == {
        "ServiceAccount",
        "Role",
        "RoleBinding",
    }
    assert all(manifest["metadata"]["name"] == name for manifest in manifests)
    assert all(
        manifest["metadata"]["annotations"]["npa.nebius.ai/owner-token"]
        == "unit-owner-token"
        for manifest in manifests
    )
    manifests[0]["metadata"]["annotations"]["npa.nebius.ai/owner-token"] = (
        "mutated-service-account-token"
    )
    assert all(
        manifest["metadata"]["annotations"]["npa.nebius.ai/owner-token"]
        == "unit-owner-token"
        for manifest in manifests[1:]
    )
    role = next(manifest for manifest in manifests if manifest["kind"] == "Role")
    assert role["rules"] == [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}
    ]
    binding = next(
        manifest for manifest in manifests if manifest["kind"] == "RoleBinding"
    )
    assert binding["roleRef"]["name"] == name
    assert binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": name,
            "namespace": "robomimic-validation",
        }
    ]


def test_robomimic_observer_refuses_any_extra_resource_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _live_e2e_module()
    calls, fake_run = _fake_robomimic_kubectl(extra_permission=True)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    selectors = {
        "kubeconfig": "/owner-only/kubeconfig",
        "context": "manager-context",
        "namespace": "robomimic-validation",
    }

    with pytest.raises(AssertionError, match="unexpected"):
        with module._robomimic_observer_rbac(
            run_id="extra-permission-contract", selectors=selectors, env={}
        ):
            pytest.fail("an overprivileged observer must not reach the workload")

    assert sum("delete" in command for command in calls) == 3


@pytest.mark.parametrize("observed_namespace", ("default", "other-validation"))
def test_robomimic_target_refuses_context_namespace_mismatch(
    monkeypatch: pytest.MonkeyPatch, observed_namespace: str
) -> None:
    module = _live_e2e_module()
    target = SimpleNamespace(
        kubeconfig="/owner-only/kubeconfig",
        context="manager-context",
        namespace="robomimic-validation",
    )
    monkeypatch.setattr(module, "skypilot_config_for_project", lambda _: "")
    monkeypatch.setattr(module, "resolve_byof_kubernetes_target", lambda _: target)
    monkeypatch.setattr(module, "resolve_skypilot_bin", lambda: None)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, observed_namespace, ""
        ),
    )
    selectors = {
        "kubeconfig": target.kubeconfig,
        "context": target.context,
        "namespace": target.namespace,
        "storage_endpoint": "https://storage.test-region.nebius.cloud",
    }

    with pytest.raises(AssertionError, match="namespace does not match"):
        module._robomimic_target_env("manager-project", selectors, [])


def test_robomimic_observer_cleanup_runs_after_gate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _live_e2e_module()
    calls, fake_run = _fake_robomimic_kubectl()
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    selectors = {
        "kubeconfig": "/owner-only/kubeconfig",
        "context": "manager-context",
        "namespace": "robomimic-validation",
    }

    with pytest.raises(RuntimeError, match="simulated gate failure"):
        with module._robomimic_observer_rbac(
            run_id="cleanup-contract", selectors=selectors, env={}
        ):
            raise RuntimeError("simulated gate failure")

    assert any("create" in command for command in calls)
    assert any("delete" in command for command in calls)
    assert sum("-o" in command and "name" in command for command in calls) == 6


def test_robomimic_observer_cleanup_checks_every_resource_after_delete_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _live_e2e_module()
    calls, fake_run = _fake_robomimic_kubectl(delete_failure="rolebinding/")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    selectors = {
        "kubeconfig": "/owner-only/kubeconfig",
        "context": "manager-context",
        "namespace": "robomimic-validation",
    }

    with pytest.raises(RuntimeError, match="simulated gate failure") as raised:
        with module._robomimic_observer_rbac(
            run_id="cleanup-error-contract", selectors=selectors, env={}
        ):
            raise RuntimeError("simulated gate failure")

    assert any("cleanup failed" in note for note in raised.value.__notes__)
    assert sum("delete" in command for command in calls) == 3
    assert sum("-o" in command and "name" in command for command in calls) == 6


def test_robomimic_observer_cleanup_covers_ambiguous_create_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _live_e2e_module()
    calls, fake_run = _fake_robomimic_kubectl(create_failure=True)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    selectors = {
        "kubeconfig": "/owner-only/kubeconfig",
        "context": "manager-context",
        "namespace": "robomimic-validation",
    }

    with pytest.raises(RuntimeError, match="response lost"):
        with module._robomimic_observer_rbac(
            run_id="ambiguous-create-contract", selectors=selectors, env={}
        ):
            pytest.fail("the context must not yield after create failure")

    deleted = [command for command in calls if "delete" in command]
    assert len(deleted) == 1
    assert "/serviceaccounts/" in " ".join(deleted[0])


def test_robomimic_observer_cleanup_never_deletes_foreign_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _live_e2e_module()
    calls, fake_run = _fake_robomimic_kubectl(
        create_failure=True, foreign_owner_after_create=True
    )
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    selectors = {
        "kubeconfig": "/owner-only/kubeconfig",
        "context": "manager-context",
        "namespace": "robomimic-validation",
    }

    with pytest.raises(RuntimeError, match="response lost") as raised:
        with module._robomimic_observer_rbac(
            run_id="foreign-owner-contract", selectors=selectors, env={}
        ):
            pytest.fail("the context must not yield after create failure")

    assert not any("delete" in command for command in calls)
    assert any("owned by another invocation" in note for note in raised.value.__notes__)


def test_robomimic_observer_cleanup_continues_after_kubectl_launch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _live_e2e_module()
    calls, fake_run = _fake_robomimic_kubectl(delete_oserror="rolebinding/")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    selectors = {
        "kubeconfig": "/owner-only/kubeconfig",
        "context": "manager-context",
        "namespace": "robomimic-validation",
    }

    with pytest.raises(RuntimeError, match="simulated gate failure") as raised:
        with module._robomimic_observer_rbac(
            run_id="cleanup-oserror-contract", selectors=selectors, env={}
        ):
            raise RuntimeError("simulated gate failure")

    assert any("FileNotFoundError" in note for note in raised.value.__notes__)
    assert sum("delete" in command for command in calls) == 3
    assert sum("-o" in command and "name" in command for command in calls) == 6


@pytest.mark.parametrize(
    ("failure_kind", "expected_note"),
    (
        ("ownership", "inspect rolebinding/"),
        ("absence", "verify rolebinding/"),
    ),
)
def test_robomimic_observer_cleanup_contains_every_lookup_oserror(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    expected_note: str,
) -> None:
    module = _live_e2e_module()
    options = (
        {"ownership_get_oserror": "rolebinding/"}
        if failure_kind == "ownership"
        else {
            "delete_failure": "rolebinding/",
            "absence_get_oserror": "rolebinding/",
        }
    )
    calls, fake_run = _fake_robomimic_kubectl(**options)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    selectors = {
        "kubeconfig": "/owner-only/kubeconfig",
        "context": "manager-context",
        "namespace": "robomimic-validation",
    }

    with pytest.raises(RuntimeError, match="simulated gate failure") as raised:
        with module._robomimic_observer_rbac(
            run_id=f"{failure_kind}-oserror-contract", selectors=selectors, env={}
        ):
            raise RuntimeError("simulated gate failure")

    assert any(expected_note in note for note in raised.value.__notes__)
    assert sum("-o" in command and "name" in command for command in calls) == 6


def test_robomimic_observer_delete_uses_object_identity_preconditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _live_e2e_module()
    calls, fake_run = _fake_robomimic_kubectl(replace_before_delete="rolebinding/")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    selectors = {
        "kubeconfig": "/owner-only/kubeconfig",
        "context": "manager-context",
        "namespace": "robomimic-validation",
    }

    with pytest.raises(AssertionError, match="cleanup failed") as raised:
        with module._robomimic_observer_rbac(
            run_id="delete-precondition-contract", selectors=selectors, env={}
        ):
            pass

    assert "delete precondition failed" in str(raised.value)
    raw_deletes = [command for command in calls if "delete" in command]
    assert raw_deletes and all("--raw" in command for command in raw_deletes)


def test_robomimic_observer_generates_a_fresh_owner_token_per_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _live_e2e_module()
    calls, fake_run = _fake_robomimic_kubectl()
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    observed_tokens: list[str] = []
    original = module._robomimic_observer_manifests

    def capture_token(**kwargs: object):
        observed_tokens.append(str(kwargs["owner_token"]))
        return original(**kwargs)

    monkeypatch.setattr(module, "_robomimic_observer_manifests", capture_token)
    selectors = {
        "kubeconfig": "/owner-only/kubeconfig",
        "context": "manager-context",
        "namespace": "robomimic-validation",
    }

    for run_id in ("token-contract-one", "token-contract-two"):
        with module._robomimic_observer_rbac(
            run_id=run_id, selectors=selectors, env={}
        ):
            pass

    assert len(observed_tokens) == 2
    assert observed_tokens[0] != observed_tokens[1]
    assert all(len(token) == 64 for token in observed_tokens)
    assert sum("create" in command for command in calls) == 8


def test_robomimic_observer_preserves_primary_error_without_add_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Python310Error(RuntimeError):
        add_note = None

    module = _live_e2e_module()
    _, fake_run = _fake_robomimic_kubectl(delete_failure="rolebinding/")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    selectors = {
        "kubeconfig": "/owner-only/kubeconfig",
        "context": "manager-context",
        "namespace": "robomimic-validation",
    }

    with pytest.raises(Python310Error, match="simulated Python 3.10 failure") as raised:
        with module._robomimic_observer_rbac(
            run_id="python310-contract", selectors=selectors, env={}
        ):
            raise Python310Error("simulated Python 3.10 failure")

    assert any("cleanup failed" in str(arg) for arg in raised.value.args)


@pytest.mark.parametrize(
    "public_registry",
    ("docker.io/example", "quay.io/example", "public.ecr.aws/example"),
)
def test_robomimic_gate_rejects_known_public_registry_hosts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    public_registry: str,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    values = {
        "NPA_E2E_PROJECT": "manager-project",
        "NPA_BYOF_ROBOMIMIC_REGISTRY": public_registry,
        "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY": "private",
        "NPA_BYOF_KUBECONFIG": str(kubeconfig),
        "NPA_BYOF_K8S_CONTEXT": "manager-context",
        "NPA_BYOF_K8S_NAMESPACE": "robomimic-validation",
        "NPA_E2E_S3_BUCKET": "manager-bucket",
        "NPA_E2E_MK8S_RESERVED_CAPACITY": "1",
        "NPA_BYOF_LIVE_GPU": "1",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200": "1",
    }
    for variable, value in values.items():
        monkeypatch.setenv(variable, value)

    with pytest.raises(AssertionError):
        _live_e2e_module()._robomimic_live_selectors("manager-project")


def test_robomimic_publication_quarantine_covers_explicit_dev_tags() -> None:
    source_sha = "a" * 40
    with pytest.raises(ValueError, match="publication-quarantined"):
        images.development_image_for_tool("robomimic", git_sha=source_sha)
    with pytest.raises(ValueError, match="publication-quarantined"):
        images.development_image_for_tool(
            "robomimic",
            git_sha=source_sha,
            registry="ghcr.io/example/public",
        )
    with pytest.raises(ValueError, match="publication-quarantined"):
        images.container_image_for_tool(
            "robomimic",
            tag=f"dev-{source_sha}",
        )
    assert (
        images.container_image_for_tool(
            "robomimic",
            registry="private.invalid/robomimic",
            tag=f"dev-{source_sha}",
        )
        == f"private.invalid/robomimic/npa-robomimic:dev-{source_sha}"
    )


def test_robomimic_runner_refuses_before_build_without_manager_context() -> None:
    env = dict(os.environ)
    for variable in (
        "NPA_E2E_PROJECT",
        "NPA_BYOF_ROBOMIMIC_REGISTRY",
        "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY",
        "NPA_BYOF_KUBECONFIG",
        "NPA_BYOF_K8S_CONTEXT",
        "NPA_BYOF_K8S_NAMESPACE",
        "NPA_E2E_S3_BUCKET",
        "NPA_E2E_MK8S_RESERVED_CAPACITY",
        "NPA_BYOF_LIVE_GPU",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200",
    ):
        env.pop(variable, None)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "npa" / "scripts" / "run_byof_repo.py"),
            "--repo-url",
            "https://github.com/ARISE-Initiative/robomimic.git",
            "--repo-ref",
            SOURCE_REVISION,
            "--solution-name",
            "robomimic",
            "--skip-build",
            "--skip-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 64
    assert payload["status"] == "refused"
    assert payload["build_started"] is False
    assert payload["push_started"] is False
    assert payload["run_started"] is False


def test_robomimic_runner_cannot_disguise_registered_base_image(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = _byof_runner_module()
    monkeypatch.setenv(
        "NPA_BYOF_ROBOMIMIC_IMAGE",
        "private.invalid/robomimic/npa-robomimic@sha256:" + "a" * 64,
    )
    for variable in (
        "NPA_E2E_PROJECT",
        "NPA_BYOF_ROBOMIMIC_REGISTRY",
        "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY",
        "NPA_BYOF_KUBECONFIG",
        "NPA_BYOF_K8S_CONTEXT",
        "NPA_BYOF_K8S_NAMESPACE",
        "NPA_E2E_S3_BUCKET",
        "NPA_E2E_MK8S_RESERVED_CAPACITY",
        "NPA_BYOF_LIVE_GPU",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_PVC",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(
        runner,
        "_base_image_candidates",
        lambda **_: pytest.fail("disguised robomimic request reached image resolution"),
    )

    status = runner.main(
        [
            "--repo-url",
            "https://github.com/example/unrelated.git",
            "--solution-name",
            "unrelated",
            "--registry",
            "private.invalid/robomimic",
            "--base-profile",
            "prebuilt",
            "--base-image",
            "tool://robomimic",
            "--skip-build",
            "--skip-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 64
    assert payload["status"] == "refused"
    assert payload["build_started"] is False
    assert payload["push_started"] is False
    assert payload["run_started"] is False


@pytest.mark.parametrize(
    ("option", "image_ref"),
    (
        ("--base-image", "registry.invalid/team/npa-robomimic@sha256:" + "a" * 64),
        ("--image", "registry.invalid/team/robomimic:candidate"),
        ("--base-image", "docker:registry.invalid/team/npa-robomimic:candidate"),
    ),
)
def test_robomimic_runner_recognizes_direct_image_references(
    option: str, image_ref: str
) -> None:
    runner = _byof_runner_module()
    args = runner._parse_args(
        [
            "--repo-url",
            "https://github.com/example/unrelated.git",
            "--solution-name",
            "unrelated",
            option,
            image_ref,
        ]
    )
    assert runner._is_robomimic_request(args) is True


def test_robomimic_runner_recognizes_exact_source_ref_through_renamed_mirror() -> None:
    runner = _byof_runner_module()
    args = runner._parse_args(
        [
            "--repo-url",
            "https://github.com/example/renamed-mirror.git",
            "--repo-ref",
            SOURCE_REVISION,
            "--solution-name",
            "renamed",
            "--capability-name",
            "renamed",
            "--smoke-artifact-name",
            "renamed.json",
            "--smoke-command",
            "python train.py",
        ]
    )
    assert runner._is_robomimic_request(args) is True


def test_workflow_refusal_recognizes_exact_source_ref_through_renaming() -> None:
    spec = SimpleNamespace(
        name="renamed",
        config={
            "repo_url": "https://github.com/example/renamed-mirror.git",
            "repo_ref": SOURCE_REVISION,
            "solution_name": "renamed",
            "execution_policy": "direct",
        },
    )
    assert workflow_cli._is_dedicated_live_gate_spec(spec) is True


@pytest.mark.parametrize(
    (
        "registry",
        "manager_registry",
        "visibility",
        "image",
        "output_root",
        "extra_args",
        "expected_error",
    ),
    (
        (
            "private.invalid/robomimic",
            "private.invalid/robomimic",
            "private",
            "quay.io/example/robomimic:latest",
            "s3://manager-bucket/oss-solutions/robomimic",
            (),
            "image does not target the manager-issued private registry",
        ),
        (
            "quay.io:443/example",
            "quay.io:443/example",
            "private",
            "",
            "s3://manager-bucket/oss-solutions/robomimic",
            (),
            "requires an operator-private registry",
        ),
        (
            "private.invalid/robomimic",
            "private.invalid/robomimic",
            "private",
            "",
            "s3://other-bucket/oss-solutions/robomimic",
            (),
            "manager-issued bucket and solution prefix",
        ),
        (
            "private.invalid/actual",
            "private.invalid/manager",
            "private",
            "",
            "s3://manager-bucket/oss-solutions/robomimic",
            (),
            "registry does not match the manager-issued registry",
        ),
        (
            "private.invalid/robomimic",
            "private.invalid/robomimic",
            "public",
            "",
            "s3://manager-bucket/oss-solutions/robomimic",
            (),
            "registry visibility must be private",
        ),
        (
            "private.invalid/robomimic",
            "private.invalid/robomimic",
            "private",
            "",
            "s3://manager-bucket/oss-solutions/robomimic",
            ("--base-profile", "ubuntu"),
            "requires its immutable prebuilt profile",
        ),
    ),
)
def test_robomimic_runner_rejects_unsafe_manager_targets_before_build(
    tmp_path: Path,
    registry: str,
    manager_registry: str,
    visibility: str,
    image: str,
    output_root: str,
    extra_args: tuple[str, ...],
    expected_error: str,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    env = {
        **os.environ,
        "NPA_E2E_PROJECT": "manager-project",
        "NPA_BYOF_ROBOMIMIC_REGISTRY": manager_registry,
        "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY": visibility,
        "NPA_BYOF_KUBECONFIG": str(kubeconfig),
        "NPA_BYOF_K8S_CONTEXT": "manager-context",
        "NPA_BYOF_K8S_NAMESPACE": "robomimic-validation",
        "NPA_E2E_S3_BUCKET": "manager-bucket",
        "NPA_E2E_MK8S_RESERVED_CAPACITY": "1",
        "NPA_BYOF_LIVE_GPU": "1",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200": "1",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_PVC": "robomimic-runtime-exact",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256": "a" * 64,
    }
    accepted_image = f"{manager_registry.rstrip('/')}/npa-robomimic@sha256:{'a' * 64}"
    env["NPA_BYOF_ROBOMIMIC_IMAGE"] = accepted_image
    command = [
        sys.executable,
        str(ROOT / "npa" / "scripts" / "run_byof_repo.py"),
        "--repo-url",
        "https://github.com/ARISE-Initiative/robomimic.git",
        "--repo-ref",
        SOURCE_REVISION,
        "--solution-name",
        "robomimic",
        "--project",
        "manager-project",
        "--registry",
        registry,
        "--base-profile",
        "prebuilt",
        "--base-image",
        "tool://robomimic",
        "--output-root",
        output_root,
        "--skip-build",
        *extra_args,
    ]
    command.extend(["--image", image or accepted_image])
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 64
    assert payload["status"] == "refused"
    assert expected_error in payload["error"]
    assert payload["build_started"] is False
    assert payload["push_started"] is False
    assert payload["run_started"] is False


def test_robomimic_runner_accepts_only_the_exact_immutable_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    accepted_image = "private.invalid/robomimic/npa-robomimic@sha256:" + "a" * 64
    selectors = {
        "NPA_E2E_PROJECT": "manager-project",
        "NPA_BYOF_ROBOMIMIC_REGISTRY": "private.invalid/robomimic",
        "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY": "private",
        "NPA_BYOF_KUBECONFIG": str(tmp_path / "kubeconfig"),
        "NPA_BYOF_K8S_CONTEXT": "manager-context",
        "NPA_BYOF_K8S_NAMESPACE": "robomimic-validation",
        "NPA_E2E_S3_BUCKET": "manager-bucket",
        "NPA_E2E_MK8S_RESERVED_CAPACITY": "1",
        "NPA_BYOF_LIVE_GPU": "1",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200": "1",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_PVC": "robomimic-runtime-exact",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256": "a" * 64,
        "NPA_BYOF_ROBOMIMIC_IMAGE": accepted_image,
    }
    for variable, value in selectors.items():
        monkeypatch.setenv(variable, value)
    Path(selectors["NPA_BYOF_KUBECONFIG"]).write_text(
        "apiVersion: v1\n", encoding="utf-8"
    )
    run_id = "exact-contract"
    live_module = _live_e2e_module()
    profile = live_module._materialize_robomimic_attested_profile(
        tmp_path / "attested.yaml",
        namespace=selectors["NPA_BYOF_K8S_NAMESPACE"],
        service_account=live_module._robomimic_observer_name(run_id),
        runtime_pvc=selectors["NPA_BYOF_ROBOMIMIC_RUNTIME_PVC"],
        runtime_inventory_sha256=selectors[
            "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256"
        ],
    )
    config = _workflow_config()
    runner = _byof_runner_module()
    args = runner._parse_args(
        [
            "--repo-url",
            str(config["repo_url"]),
            "--repo-ref",
            str(config["repo_ref"]),
            "--repo-auth",
            "none",
            "--project",
            selectors["NPA_E2E_PROJECT"],
            "--registry",
            selectors["NPA_BYOF_ROBOMIMIC_REGISTRY"],
            "--base-profile",
            str(config["base_profile"]),
            "--base-image",
            str(config["base_image"]),
            "--build-command",
            str(config["build_command"]),
            "--workload",
            "solution-smoke",
            "--smoke-command",
            str(config["smoke_command"]),
            "--solution-name",
            "robomimic",
            "--capability-name",
            str(config["capability_name"]),
            "--smoke-artifact-name",
            "robomimic-smoke.json",
            "--yaml",
            str(profile),
            "--output-root",
            "s3://manager-bucket/oss-solutions/robomimic",
            "--wait-timeout",
            "-1",
            "--run-id",
            run_id,
            "--image",
            accepted_image,
            "--skip-build",
        ]
    )
    with pytest.raises(ValueError, match="runtime use remains deferred"):
        runner._require_robomimic_manager_context(
            args,
            registry=args.registry,
            image=accepted_image,
            base_profile=args.base_profile,
        )
    assert (
        runner.ROBOMIMIC_BUILD_COMMAND_SHA256
        == hashlib.sha256(str(config["build_command"]).encode()).hexdigest()
    )
    assert (
        runner.ROBOMIMIC_SMOKE_COMMAND_SHA256
        == hashlib.sha256(str(config["smoke_command"]).encode()).hexdigest()
    )
    wrong_image = "private.invalid/robomimic/npa-robomimic@sha256:" + "b" * 64
    with pytest.raises(ValueError, match="immutable inputs"):
        runner._require_robomimic_manager_context(
            args,
            registry=args.registry,
            image=wrong_image,
            base_profile=args.base_profile,
        )
    malformed_image = "private.invalid/robomimic/unreviewed@sha256:" + "c" * 64
    monkeypatch.setenv("NPA_BYOF_ROBOMIMIC_IMAGE", malformed_image)
    args.image = malformed_image
    with pytest.raises(ValueError, match="exact private npa-robomimic digest"):
        runner._require_robomimic_manager_context(
            args,
            registry=args.registry,
            image=malformed_image,
            base_profile=args.base_profile,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "skip-push",
        "skip-run",
        "no-cleanup",
        "build-enabled",
        "run-id",
        "repository",
        "source-revision",
        "base-image",
        "build-command",
        "smoke-command",
        "workload",
        "solution",
        "capability",
        "smoke-artifact",
        "image",
        "profile",
    ),
)
def test_robomimic_runner_rejects_every_immutable_contract_bypass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    module = _live_e2e_module()
    accepted_image = "private.invalid/robomimic/npa-robomimic@sha256:" + "a" * 64
    monkeypatch_env = {
        "NPA_E2E_PROJECT": "manager-project",
        "NPA_BYOF_ROBOMIMIC_REGISTRY": "private.invalid/robomimic",
        "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY": "private",
        "NPA_BYOF_KUBECONFIG": str(tmp_path / "kubeconfig"),
        "NPA_BYOF_K8S_CONTEXT": "manager-context",
        "NPA_BYOF_K8S_NAMESPACE": "robomimic-validation",
        "NPA_E2E_S3_BUCKET": "manager-bucket",
        "NPA_E2E_MK8S_RESERVED_CAPACITY": "1",
        "NPA_BYOF_LIVE_GPU": "1",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200": "1",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_PVC": "robomimic-runtime-exact",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256": "a" * 64,
        "NPA_BYOF_ROBOMIMIC_IMAGE": accepted_image,
    }
    Path(monkeypatch_env["NPA_BYOF_KUBECONFIG"]).write_text(
        "apiVersion: v1\n", encoding="utf-8"
    )
    for variable, value in monkeypatch_env.items():
        monkeypatch.setenv(variable, value)
    run_id = "immutable-contract"
    profile = module._materialize_robomimic_attested_profile(
        tmp_path / "attested.yaml",
        namespace="robomimic-validation",
        service_account=module._robomimic_observer_name(run_id),
        runtime_pvc=monkeypatch_env["NPA_BYOF_ROBOMIMIC_RUNTIME_PVC"],
        runtime_inventory_sha256=monkeypatch_env[
            "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256"
        ],
    )
    config = _workflow_config()
    options = {
        "repository": ("--repo-url", "https://github.com/example/robomimic.git"),
        "source-revision": ("--repo-ref", "main"),
        "base-image": ("--base-image", "ubuntu:24.04"),
        "build-command": ("--build-command", "true"),
        "smoke-command": ("--smoke-command", "true"),
        "workload": ("--workload", "container-verify"),
        "solution": ("--solution-name", "renamed"),
        "capability": ("--capability-name", "renamed"),
        "smoke-artifact": ("--smoke-artifact-name", "renamed.json"),
        "image": ("--image", "private.invalid/robomimic/other:tag"),
        "run-id": ("--run-id", "../../tmp"),
    }
    flag = (
        (f"--{mutation}",)
        if mutation
        in {
            "skip-push",
            "skip-run",
            "no-cleanup",
        }
        else options.get(mutation, ())
    )
    if mutation == "profile":
        profile.write_text("name: unrelated\n", encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "npa" / "scripts" / "run_byof_repo.py"),
        "--repo-url",
        str(config["repo_url"]),
        "--repo-ref",
        str(config["repo_ref"]),
        "--repo-auth",
        "none",
        "--project",
        "manager-project",
        "--registry",
        "private.invalid/robomimic",
        "--base-profile",
        "prebuilt",
        "--base-image",
        str(config["base_image"]),
        "--build-command",
        str(config["build_command"]),
        "--workload",
        "solution-smoke",
        "--smoke-command",
        str(config["smoke_command"]),
        "--solution-name",
        "robomimic",
        "--capability-name",
        str(config["capability_name"]),
        "--smoke-artifact-name",
        "robomimic-smoke.json",
        "--yaml",
        str(profile),
        "--output-root",
        "s3://manager-bucket/oss-solutions/robomimic",
        "--wait-timeout",
        "-1",
        "--run-id",
        run_id,
        "--image",
        accepted_image,
        *(("--skip-build",) if mutation != "build-enabled" else ()),
        *flag,
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **monkeypatch_env},
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 64, result.stdout + result.stderr
    assert payload["status"] == "refused"
    assert payload["build_started"] is False
    assert payload["push_started"] is False
    assert payload["run_started"] is False


def test_robomimic_workflow_direct_submit_refuses_before_preflight() -> None:
    for extra_args in (
        [],
        ["--var", "execution_policy=direct"],
        [
            "--var",
            "execution_policy=direct",
            "--var",
            "solution_name=renamed",
            "--var",
            "repo_url=https://github.com/example/unrelated.git",
        ],
    ):
        result = CliRunner().invoke(
            app,
            ["workbench", "workflow", "submit", str(WORKFLOW), *extra_args],
        )

        assert result.exit_code != 0
        assert "dedicated live gate" in result.output
        assert (
            "before any image/runtime pull, data fetch, or GPU submission"
            in result.output
        )


def test_robomimic_run_spec_execute_refuses_all_mutable_overrides() -> None:
    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "workflow",
            "run-spec",
            str(WORKFLOW),
            "--execute",
            "--var",
            "execution_policy=direct",
            "--var",
            "solution_name=renamed",
            "--var",
            "repo_url=https://github.com/example/unrelated.git",
        ],
    )

    assert result.exit_code != 0
    assert "dedicated live gate" in result.output
    assert (
        "before any image/runtime pull, data fetch, or GPU submission" in result.output
    )


def test_robomimic_workflow_validates_and_plans_real_byof_stage() -> None:
    spec = load_spec(WORKFLOW)
    plan = build_plan(spec, run_id="robomimic-plan-test")

    assert spec.name == "byof-robomimic"
    assert spec.config["execution_policy"] == "dedicated-live-gate-only"
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.tool_ref == "workbench.byof.repo"
    assert "byof-solution-smoke-robomimic-b200-gpu" in step.argv
    assert "robomimic-smoke.json" in step.argv
    assert "B200:1" in WORKFLOW.read_text(encoding="utf-8")


def test_robomimic_smoke_is_immutable_and_fails_closed() -> None:
    config = _workflow_config()
    build = str(config["build_command"])
    smoke = str(config["smoke_command"])

    assert config["repo_ref"] == SOURCE_REVISION
    assert config["base_profile"] == "prebuilt"
    assert str(config["controller_image"]).endswith(
        "/npa-robomimic:0.1.0-neutral-unbuilt"
    )
    assert config["base_image"] == "tool://robomimic"
    assert config["resource_profile_yaml"] == "byof-solution-smoke-robomimic-b200-gpu"
    assert config["capability_name"] == "lift_ph_lowdim_checkpoint_reload_action"
    assert config["smoke_artifact_name"] == "robomimic-smoke.json"
    assert config["wait_timeout"] == -1
    assert build == ""
    assert smoke == "robomimic-entrypoint smoke"
    assert hashlib.sha256(build.encode()).hexdigest() == BUILD_COMMAND_SHA256
    lock_bytes = BAKED_LOCK.read_bytes()
    assert (
        sum(
            bool(re.match(rb"^[A-Za-z0-9_.-]+==", line))
            for line in lock_bytes.splitlines()
        )
        == 40
    )
    assert hashlib.sha256(lock_bytes).hexdigest() == DEPENDENCY_LOCK_SHA256
    smoke = _smoke_python()

    assert 'allowed_hosts=("huggingface.co",)' in smoke
    for hf_redirect_host in (
        r"cdn-lfs(?:-[a-z0-9-]+)?\.hf\.co",
        "cas-bridge.xethub.hf.co",
        ".cdn.hf.co",
    ):
        assert hf_redirect_host in smoke
    for broad_redirect_domain in ("amazonaws.com", "cloudfront.net"):
        assert broad_redirect_domain not in smoke

    for required in (
        SOURCE_REVISION,
        DATASET_REVISION,
        DATASET_SHA256,
        '"split_train_val.py"',
        "np.random.seed(0)",
        "len(train_keys) + len(valid_keys) != len(demo_keys)",
        '"train.py"',
        "TRAIN_STEPS = 4",
        "VALIDATION_STEPS = 2",
        'Path("/workspace/byof-inputs") / output_dir.name',
        "partial.replace(dataset)",
        'config.train.hdf5_filter_key = "train"',
        'config.train.hdf5_validation_filter_key = "valid"',
        'source_metadata.get("observed_head") != SOURCE_REVISION',
        f'BAKED_LOCK_SHA256 = "{DEPENDENCY_LOCK_SHA256}"',
        "models/model_epoch_1.pth",
        "policy_from_checkpoint(",
        "optimizer_step_count != TRAIN_STEPS",
        "action.shape != (7,)",
        '"B200" not in gpu_name.upper()',
        'architecture != "sm_100"',
        'os.environ.get("NPA_ROBOMIMIC_STRICT_B200_ATTESTED") != "1"',
        'os.environ.get("NPA_ROBOMIMIC_EXPECTED_NAMESPACE", "").strip()',
        'os.environ["NPA_ROBOMIMIC_ACTIVE_RUNTIME_ROOT"]',
        "runtime_root == runtime_mount_root",
        "snapshot_parent.stat().st_mode & 0o077",
        'pod_status.get("spec", {}).get("serviceAccountName", "")',
        'status.get("name") == "ray-node"',
        'containers[0].get("image") != runtime_image',
        'runtime_mounts[0].get("readOnly") is True',
        '.get("readOnly")',
        "os.statvfs(runtime_root).f_flag & os.ST_RDONLY",
        '"architecture": architecture',
        '"strict_reserved_capacity_attested": True',
        '"observed_head": source_metadata["observed_head"]',
        '"trajectory_count": len(demo_keys)',
        '"sample_count": sum(sample_counts.values())',
        '"train_trajectory_count": len(train_keys)',
        '"validation_trajectory_count": len(valid_keys)',
        '"train_sample_count": sum(sample_counts[key] for key in train_keys)',
        '"validation_sample_count": sum(sample_counts[key] for key in valid_keys)',
        '"train_loss": train_loss',
        '"validation_loss": validation_loss',
        '"configured_validation_forward_steps": VALIDATION_STEPS',
        '"sha256": checkpoint_hash',
        '"finite": finite',
        '"within_allowed_range": within_range',
        '"accelerator_count": gpu_count',
        '"runtime_ref": runtime_image',
        '"observation_source": "Kubernetes Pod status.containerStatuses[].imageID"',
        '"pod_image"',
        '"workload_identity"',
        '"pretrained_weights": False',
        '"runtime_cache": "external-read-only-prepopulated"',
        '"manager_inventory_digest_matched": True',
        '"atomic_private_snapshot_published": True',
        '"snapshot_write_bits_absent": runtime_root.stat().st_mode & 0o222 == 0',
        '"exit_status": 0',
    ):
        assert required in smoke

    assert "image_policy_sweeps" in smoke
    assert "simulator_rollouts" in smoke
    assert "full_algorithm_matrix" in smoke
    assert "robosuite" not in smoke
    ast.parse(_smoke_python())


def test_robomimic_profile_is_exactly_one_compute_only_b200() -> None:
    documents = list(yaml.safe_load_all(PROFILE.read_text(encoding="utf-8")))
    task = documents[1]
    assert task["resources"]["accelerators"] == "B200:1"
    assert (
        task["config"]["kubernetes"]["pod_config"]["spec"]["serviceAccountName"]
        == "npa-robomimic-observer-placeholder"
    )
    assert task["envs"]["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility"
    assert task["envs"]["BYOF_IMAGE"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_STRICT_B200_ATTESTED"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_EXPECTED_NAMESPACE"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_EXPECTED_SERVICE_ACCOUNT"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256"] == ""
    assert 'export PATH="/opt/conda/bin:${PATH}"' not in task["run"]
    pod_spec = task["config"]["kubernetes"]["pod_config"]["spec"]
    mounts = pod_spec["containers"][0]["volumeMounts"]
    runtime_mount = next(item for item in mounts if item["name"] == "robomimic-runtime")
    assert runtime_mount == {
        "name": "robomimic-runtime",
        "mountPath": "/opt/npa-runtime/robomimic",
        "readOnly": True,
    }
    volumes = {item["name"]: item for item in pod_spec["volumes"]}
    assert volumes["robomimic-runtime"]["persistentVolumeClaim"] == {
        "claimName": "npa-robomimic-runtime-placeholder",
        "readOnly": True,
    }
    assert volumes["robomimic-inputs"]["emptyDir"] == {}
    assert volumes["robomimic-outputs"]["emptyDir"] == {}
    assert "${NPA_E2E_MK8S_RESERVED_CAPACITY}" not in PROFILE.read_text(
        encoding="utf-8"
    )
    assert "pip install --quiet boto3" not in PROFILE.read_text(encoding="utf-8")
    assert "runtime input must not be uploaded as output" in PROFILE.read_text(
        encoding="utf-8"
    )
    profile_text = PROFILE.read_text(encoding="utf-8")
    for endpoint_gate in (
        'endpoint_parts.scheme != "https"',
        "endpoint_parts.username is not None",
        're.fullmatch(r"storage\\.[a-z0-9-]+\\.nebius\\.cloud"',
        'RuntimeError("refusing S3 credentials for an unapproved endpoint")',
    ):
        assert endpoint_gate in profile_text
    assert "if smoke_exit_code == 0:" in PROFILE.read_text(encoding="utf-8")
    assert 'smoke_artifact["exit_status"] = smoke_exit_code' in PROFILE.read_text(
        encoding="utf-8"
    )
    text = PROFILE.read_text(encoding="utf-8").lower()
    for graphics_surface in ("vulkan", "egl"):
        assert graphics_surface not in text


def test_robomimic_readiness_binds_exact_workflow_bytes() -> None:
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    workflow_hash = hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()

    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert readiness["workflow_sha256"] == workflow_hash
    assert readiness["planning"]["validation"]["status"] in {
        "verified",
        "unverified",
    }
    assert readiness["planning"]["task_fidelity"]["status"] in {
        "verified",
        "unverified",
    }
    assert set(readiness["prerequisites"]) == {
        "output_storage",
        "worker_input",
        "credentials",
        "source_image",
        "target_runtime",
    }


def test_robomimic_delivery_boundaries_do_not_invent_runtime_consent() -> None:
    doc = DOC.read_text(encoding="utf-8")
    normalized_doc = " ".join(doc.split())
    workflow = WORKFLOW.read_text(encoding="utf-8")
    profile = PROFILE.read_text(encoding="utf-8")

    for boundary in (
        "| Source |",
        "| Baked runtime |",
        "| Weights |",
        "| Data and assets |",
        "| Runtime cache |",
        "| Outputs |",
    ):
        assert boundary in doc

    assert "changes delivery, not permission" in normalized_doc
    assert "No image has been built" in doc
    assert "pretrained weights" in doc
    assert "every layer and image history entry" in doc
    combined = "\n".join((doc, workflow, profile))
    for invented_proxy in (
        "NPA_ROBOMIMIC_ACCEPT",
        "NPA_ACCEPT_CUDA",
        "NPA_ACCEPT_CUDNN",
    ):
        assert invented_proxy not in combined
