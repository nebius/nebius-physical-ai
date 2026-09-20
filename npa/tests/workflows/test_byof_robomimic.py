"""Contract tests for the immutable, authorization-gated robomimic BYOF path."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
RUNTIME_LOCK = IMAGE_ROOT / "runtime-requirements.lock"
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
RUNTIME_LOCK_SHA256 = "60e13f9903e3d9966eb3ec1a8a88536e93fe17358c7adc40844927b6357ac283"


def _customer_binding(project: str) -> str:
    return hashlib.sha256(
        b"npa.robomimic.customer-binding.v1\0" + project.encode("utf-8")
    ).hexdigest()


def _entitlement_notice_sha256(lock: dict[str, object], lock_sha256: str) -> str:
    contract = lock["customer_entitlement"]
    assert isinstance(contract, dict)
    identity = {
        "schema": contract["schema"],
        "runtime_id": lock["runtime_id"],
        "runtime_lock_sha256": lock_sha256,
        "terms": contract["terms"],
        "customer_responsibilities": contract["responsibilities"],
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _customer_entitlement(
    tmp_path: Path,
    *,
    project: str,
    run_id: str,
    runtime_inventory_sha256: str = "a" * 64,
) -> Path:
    lock_path = IMAGE_ROOT / "runtime-requirements.lock"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    contract = lock["customer_entitlement"]
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    accepted = datetime.now(timezone.utc).replace(microsecond=0)
    record = {
        "schema": contract["schema"],
        "decision": "accepted",
        "customer_binding_sha256": _customer_binding(project),
        "run_id": run_id,
        "source_revision": lock["source_revision"],
        "runtime_id": lock["runtime_id"],
        "runtime_manifest_sha256": runtime_inventory_sha256,
        "runtime_lock_sha256": lock_sha256,
        "terms": contract["terms"],
        "customer_responsibilities": contract["responsibilities"],
        "notice_sha256": _entitlement_notice_sha256(lock, lock_sha256),
        "accepted_at": accepted.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": (accepted + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = tmp_path / f"{run_id}-entitlement.json"
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _workflow_config() -> dict[str, object]:
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    config = payload["config"]
    assert isinstance(config, dict)
    return config


def _smoke_python() -> str:
    return SMOKE.read_text(encoding="utf-8")


def _smoke_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    h5py = ModuleType("h5py")
    robomimic = ModuleType("robomimic")
    config = ModuleType("robomimic.config")
    utils = ModuleType("robomimic.utils")
    file_utils = ModuleType("robomimic.utils.file_utils")
    verifier = ModuleType("verify_image")
    setattr(config, "config_factory", lambda *_args: None)
    setattr(file_utils, "policy_from_checkpoint", lambda *_args, **_kwargs: None)
    setattr(verifier, "verified_source_identity", lambda *_args: None)
    setattr(verifier, "verify_customer_runtime_entitlement", lambda **_kwargs: None)
    for name, module in (
        ("h5py", h5py),
        ("robomimic", robomimic),
        ("robomimic.config", config),
        ("robomimic.utils", utils),
        ("robomimic.utils.file_utils", file_utils),
        ("verify_image", verifier),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    specification = importlib.util.spec_from_file_location("robomimic_smoke", SMOKE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _profile_upload_module(tmp_path: Path) -> ModuleType:
    """Load only the embedded uploader's declarations, never its live operations."""

    documents = list(yaml.safe_load_all(PROFILE.read_text(encoding="utf-8")))
    run = documents[1]["run"]
    script = run.split("python3 <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    parsed = ast.parse(script)
    declarations = []
    for node in parsed.body:
        if isinstance(
            node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)
        ):
            declarations.append(node)
        elif isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id.isupper()
            for target in node.targets
        ):
            declarations.append(node)
    helper_path = tmp_path / "robomimic_profile_upload.py"
    helper_path.write_text(
        ast.unparse(ast.Module(body=declarations, type_ignores=[])) + "\n",
        encoding="utf-8",
    )
    specification = importlib.util.spec_from_file_location(
        "robomimic_profile_upload", helper_path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


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


def _serialized_entitlement_refusal(error: BaseException) -> str:
    """Render every ordinary exception diagnostic surface for marker checks."""

    return json.dumps(
        {
            "type": type(error).__name__,
            "args": [str(argument) for argument in error.args],
            "cause": repr(error.__cause__),
            "context": repr(error.__context__),
            "notes": list(getattr(error, "__notes__", ())),
            "traceback": "".join(traceback.format_exception(error)),
        },
        sort_keys=True,
    )


class _SmokeProofBody:
    """Record bounded reads and cleanup for one mocked S3 response body."""

    def __init__(
        self,
        payload: bytes,
        *,
        read_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.read_error = read_error
        self.close_error = close_error
        self.read_amounts: list[int] = []
        self.closed = False

    def read(self, amount: int) -> bytes:
        self.read_amounts.append(amount)
        if self.read_error is not None:
            raise self.read_error
        return self.payload

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _mock_robomimic_artifact_client(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    response: dict[str, object] | Exception,
) -> list[dict[str, object]]:
    """Replace the live S3 client with one deterministic mocked response."""

    calls: list[dict[str, object]] = []

    class Client:
        def get_object(self, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(
        module, "s3_client_for_project", lambda *_args, **_kwargs: Client()
    )
    return calls


def _assert_value_free_artifact_refusal(
    module: ModuleType, marker: str
) -> RuntimeError:
    """Call the mocked artifact reader and require a constant safe refusal."""

    with pytest.raises(
        RuntimeError, match="^robomimic smoke proof validation failed$"
    ) as raised:
        module._robomimic_artifact(marker, marker, marker)
    diagnostics = _serialized_entitlement_refusal(raised.value)
    assert marker not in diagnostics
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not getattr(raised.value, "__notes__", ())
    return raised.value


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
        "NPA_BYOF_ROBOMIMIC_IMAGE",
        "NPA_BYOF_KUBECONFIG",
        "NPA_BYOF_K8S_CONTEXT",
        "NPA_BYOF_K8S_NAMESPACE",
        "NPA_E2E_S3_BUCKET",
        "NPA_E2E_MK8S_RESERVED_CAPACITY",
        "NPA_BYOF_LIVE_GPU",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_PVC",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_ENTITLEMENT_FILE",
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
        "NPA_BYOF_ROBOMIMIC_IMAGE": (
            "private.invalid/robomimic/npa-robomimic@sha256:" + "d" * 64
        ),
        "NPA_BYOF_KUBECONFIG": str(kubeconfig),
        "NPA_BYOF_K8S_CONTEXT": "manager-context",
        "NPA_BYOF_K8S_NAMESPACE": "robomimic-validation",
        "NPA_E2E_S3_BUCKET": "manager-bucket",
        "NPA_E2E_MK8S_RESERVED_CAPACITY": "1",
        "NPA_BYOF_LIVE_GPU": "1",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200": "1",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_PVC": "robomimic-runtime-exact",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256": "a" * 64,
        "NPA_BYOF_ROBOMIMIC_RUNTIME_ENTITLEMENT_FILE": str(
            _customer_entitlement(
                tmp_path, project="manager-project", run_id="selector-check"
            )
        ),
        "AWS_ENDPOINT_URL": "https://storage.test-region.nebius.cloud",
    }
    for variable, value in selectors.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.delenv(missing_variable, raising=False)

    with pytest.raises(RuntimeError, match="context-invalid"):
        _live_e2e_module()._robomimic_live_selectors("manager-project")


def test_robomimic_live_harness_refuses_entitlement_before_any_side_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _live_e2e_module()
    run_id = "preflight-refusal"
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    entitlement = _customer_entitlement(
        tmp_path, project="manager-project", run_id=run_id
    )
    record = json.loads(entitlement.read_text(encoding="utf-8"))
    record["decision"] = "declined"
    entitlement.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    entitlement.chmod(0o600)
    selectors = {
        "NPA_E2E_PROJECT": "manager-project",
        "NPA_BYOF_ROBOMIMIC_REGISTRY": "private.invalid/robomimic",
        "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY": "private",
        "NPA_BYOF_ROBOMIMIC_IMAGE": (
            "private.invalid/robomimic/npa-robomimic@sha256:" + "d" * 64
        ),
        "NPA_BYOF_KUBECONFIG": str(kubeconfig),
        "NPA_BYOF_K8S_CONTEXT": "manager-context",
        "NPA_BYOF_K8S_NAMESPACE": "robomimic-validation",
        "NPA_E2E_S3_BUCKET": "manager-bucket",
        "NPA_E2E_MK8S_RESERVED_CAPACITY": "1",
        "NPA_BYOF_LIVE_GPU": "1",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200": "1",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_PVC": "robomimic-runtime-exact",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256": "a" * 64,
        "NPA_BYOF_ROBOMIMIC_RUNTIME_ENTITLEMENT_FILE": str(entitlement),
        "NPA_BYOF_ROBOMIMIC_RUN_ID": run_id,
        "AWS_ENDPOINT_URL": "https://storage.test-region.nebius.cloud",
    }
    for variable, value in selectors.items():
        monkeypatch.setenv(variable, value)

    def unexpected_side_effect(*_args, **_kwargs):
        pytest.fail("invalid entitlement reached a live or local side effect")

    monkeypatch.setattr(module, "_activate_nebius_profile", unexpected_side_effect)
    monkeypatch.setattr(module, "load_spec", unexpected_side_effect)
    monkeypatch.setattr(module, "resolve_container_registry", unexpected_side_effect)
    monkeypatch.setattr(module, "live_bucket", unexpected_side_effect)
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", unexpected_side_effect)
    monkeypatch.setattr(module, "_robomimic_observer_rbac", unexpected_side_effect)
    monkeypatch.setattr(module.subprocess, "run", unexpected_side_effect)
    before = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    with pytest.raises(RuntimeError, match="binding-mismatch"):
        module._invoke_robomimic_gate("manager-project")

    after = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "image",
    (
        "",
        "private.invalid/robomimic/npa-robomimic:mutable",
        "other.invalid/robomimic/npa-robomimic@sha256:" + "d" * 64,
    ),
)
def test_robomimic_live_harness_freezes_exact_image_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image: str
) -> None:
    module = _live_e2e_module()
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    selectors = {
        "NPA_E2E_PROJECT": "manager-project",
        "NPA_BYOF_ROBOMIMIC_REGISTRY": "private.invalid/robomimic",
        "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY": "private",
        "NPA_BYOF_ROBOMIMIC_IMAGE": image,
        "NPA_BYOF_KUBECONFIG": str(kubeconfig),
        "NPA_BYOF_K8S_CONTEXT": "manager-context",
        "NPA_BYOF_K8S_NAMESPACE": "robomimic-validation",
        "NPA_E2E_S3_BUCKET": "manager-bucket",
        "NPA_E2E_MK8S_RESERVED_CAPACITY": "1",
        "NPA_BYOF_LIVE_GPU": "1",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200": "1",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_PVC": "robomimic-runtime-exact",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256": "a" * 64,
        "NPA_BYOF_ROBOMIMIC_RUNTIME_ENTITLEMENT_FILE": str(
            tmp_path / "must-not-be-read.json"
        ),
        "NPA_BYOF_ROBOMIMIC_RUN_ID": "image-refusal",
        "AWS_ENDPOINT_URL": "https://storage.test-region.nebius.cloud",
    }
    for variable, value in selectors.items():
        monkeypatch.setenv(variable, value)

    def unexpected_side_effect(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid image selector reached a side effect")

    monkeypatch.setattr(module, "_activate_nebius_profile", unexpected_side_effect)
    monkeypatch.setattr(
        module, "_preflight_robomimic_runtime_entitlement", unexpected_side_effect
    )
    monkeypatch.setattr(module, "_robomimic_observer_rbac", unexpected_side_effect)
    monkeypatch.setattr(module.subprocess, "run", unexpected_side_effect)

    with pytest.raises(module._RobomimicEntitlementRefusal, match="context-invalid"):
        module._invoke_robomimic_gate("manager-project")


@pytest.mark.parametrize(
    ("case", "expected_category"),
    (
        ("invalid-json", "record-invalid"),
        ("unsafe-private-record", "record-unsafe"),
        ("missing-record-path", "record-unsafe"),
        ("rejected-customer-identity", "binding-mismatch"),
        ("rejected-run-identity", "binding-mismatch"),
        ("malformed-timestamp", "time-invalid"),
        ("expired-timestamp", "time-invalid"),
        ("runtime-binding-mismatch", "binding-mismatch"),
    ),
)
def test_robomimic_e2e_entitlement_refusals_are_value_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    case: str,
    expected_category: str,
) -> None:
    module = _live_e2e_module()
    marker = f"private-{case}-marker"
    run_id = "diagnostic-redaction"
    entitlement = _customer_entitlement(
        tmp_path, project="manager-project", run_id=run_id
    )
    marker_path = tmp_path / f"{marker}.json"
    entitlement.replace(marker_path)
    entitlement = marker_path
    selectors = {
        "project": "manager-project",
        "runtime_inventory_sha256": "a" * 64,
        "runtime_entitlement_file": str(entitlement),
    }
    record = json.loads(entitlement.read_text(encoding="utf-8"))
    if case == "invalid-json":
        entitlement.write_text(f'{{"marker":"{marker}"', encoding="utf-8")
    elif case == "unsafe-private-record":
        entitlement.chmod(0o644)
    elif case == "missing-record-path":
        selectors["runtime_entitlement_file"] = str(tmp_path / marker / "missing")
    elif case == "rejected-customer-identity":
        record["customer_binding_sha256"] = marker
    elif case == "rejected-run-identity":
        record["run_id"] = marker
    elif case == "malformed-timestamp":
        record["accepted_at"] = marker
    elif case == "expired-timestamp":
        record["accepted_at"] = "2000-01-01T00:00:00Z"
        record["expires_at"] = "2000-01-01T01:00:00Z"
    elif case == "runtime-binding-mismatch":
        record["runtime_manifest_sha256"] = marker
    if case not in {"invalid-json", "unsafe-private-record", "missing-record-path"}:
        entitlement.write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )

    with pytest.raises(module._RobomimicEntitlementRefusal) as raised:
        module._preflight_robomimic_runtime_entitlement(
            selectors=selectors, run_id=run_id
        )

    error = raised.value
    expected_message = f"robomimic customer entitlement refused: {expected_category}"
    assert error.category == expected_category
    assert error.args == (expected_message,)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert not getattr(error, "__notes__", ())
    captured = capsys.readouterr()
    diagnostics = _serialized_entitlement_refusal(error) + captured.out + captured.err
    assert marker not in diagnostics


def test_robomimic_e2e_context_refusal_discards_sensitive_assertion(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _live_e2e_module()
    marker = "private-context-identity-marker"
    monkeypatch.setattr(
        module,
        "_robomimic_live_selectors",
        lambda *_args: (_ for _ in ()).throw(AssertionError(marker)),
    )
    monkeypatch.setattr(
        module,
        "_activate_nebius_profile",
        lambda: pytest.fail("context refusal reached a live or local side effect"),
    )

    with pytest.raises(module._RobomimicEntitlementRefusal) as raised:
        module._invoke_robomimic_gate(marker)

    error = raised.value
    assert error.category == "context-invalid"
    assert error.args == ("robomimic customer entitlement refused: context-invalid",)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert not getattr(error, "__notes__", ())
    captured = capsys.readouterr()
    diagnostics = _serialized_entitlement_refusal(error) + captured.out + captured.err
    assert marker not in diagnostics


def test_robomimic_live_preflight_matches_runner_entitlement_proof(
    tmp_path: Path,
) -> None:
    module = _live_e2e_module()
    runner = _byof_runner_module()
    run_id = "preflight-match"
    entitlement = _customer_entitlement(
        tmp_path, project="manager-project", run_id=run_id
    )
    selectors = {
        "project": "manager-project",
        "runtime_inventory_sha256": "a" * 64,
        "runtime_entitlement_file": str(entitlement),
    }

    harness_proof = module._preflight_robomimic_runtime_entitlement(
        selectors=selectors, run_id=run_id
    )
    runner_proof = runner._verify_robomimic_entitlement(
        path=entitlement,
        customer_identity="manager-project",
        run_id=run_id,
        runtime_inventory_sha256="a" * 64,
    )

    assert harness_proof == {
        "record_sha256": runner_proof["record_sha256"],
        "customer_binding_sha256": runner_proof["customer_binding_sha256"],
    }


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
    with pytest.raises(RuntimeError, match="context-invalid"):
        _live_e2e_module()._robomimic_storage_endpoint(endpoint)


def test_robomimic_storage_endpoint_returns_one_canonical_https_origin() -> None:
    endpoint = "https://storage.test-region.nebius.cloud"
    assert _live_e2e_module()._robomimic_storage_endpoint(endpoint) == endpoint


def test_robomimic_strict_attestation_is_resolved_only_in_run_local_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_E2E_MK8S_RESERVED_CAPACITY", "1")
    module = _live_e2e_module()
    entitlement = _customer_entitlement(
        tmp_path, project="manager-project", run_id="profile-run"
    )
    rendered = module._materialize_robomimic_attested_profile(
        tmp_path / "robomimic-attested.yaml",
        namespace="robomimic-validation",
        service_account="npa-robomimic-run-scoped",
        runtime_pvc="robomimic-runtime-exact",
        runtime_inventory_sha256="a" * 64,
        runtime_entitlement_file=str(entitlement),
        runtime_entitlement_sha256=hashlib.sha256(entitlement.read_bytes()).hexdigest(),
        customer_binding_sha256=_customer_binding("manager-project"),
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
    assert rendered_task["envs"]["NPA_ROBOMIMIC_RUNTIME_ENTITLEMENT_SHA256"] == (
        hashlib.sha256(entitlement.read_bytes()).hexdigest()
    )
    assert rendered_task["envs"]["NPA_ROBOMIMIC_CUSTOMER_BINDING_SHA256"] == (
        _customer_binding("manager-project")
    )
    assert rendered_task["file_mounts"][
        "/opt/npa-runtime-authorization/robomimic.json"
    ] == str(entitlement)
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


def _exception_diagnostics(error: BaseException) -> tuple[str, ...]:
    values = getattr(error, "__notes__", error.args)
    return tuple(str(value) for value in values)


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

    assert any(
        "cleanup failed" in note for note in _exception_diagnostics(raised.value)
    )
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
    assert any(
        "owned by another invocation" in note
        for note in _exception_diagnostics(raised.value)
    )


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

    assert any(
        "FileNotFoundError" in note for note in _exception_diagnostics(raised.value)
    )
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

    assert any(expected_note in note for note in _exception_diagnostics(raised.value))
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
    module = _live_e2e_module()
    project = "manager-project"
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    entitlement = _customer_entitlement(
        tmp_path, project=project, run_id="public-registry-gate"
    )
    values = {
        "NPA_E2E_PROJECT": project,
        "NPA_BYOF_ROBOMIMIC_REGISTRY": public_registry,
        "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY": "private",
        "NPA_BYOF_ROBOMIMIC_IMAGE": (
            f"{public_registry}/npa-robomimic@sha256:" + "a" * 64
        ),
        "NPA_BYOF_KUBECONFIG": str(kubeconfig),
        "NPA_BYOF_K8S_CONTEXT": "manager-context",
        "NPA_BYOF_K8S_NAMESPACE": "robomimic-validation",
        "NPA_E2E_S3_BUCKET": "manager-bucket",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_PVC": "robomimic-runtime-pvc",
        "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256": "a" * 64,
        "NPA_BYOF_ROBOMIMIC_RUNTIME_ENTITLEMENT_FILE": str(entitlement),
        "NPA_E2E_MK8S_RESERVED_CAPACITY": "1",
        "NPA_BYOF_LIVE_GPU": "1",
        "NPA_BYOF_ROBOMIMIC_LIVE_B200": "1",
        "AWS_ENDPOINT_URL": "https://storage.test-region.nebius.cloud",
    }
    for variable, value in values.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.delenv("NEBIUS_S3_ENDPOINT", raising=False)
    checked_registries: list[str] = []
    original_public_check = module.is_public_registry

    def record_public_check(registry: str) -> bool:
        checked_registries.append(registry)
        return original_public_check(registry)

    monkeypatch.setattr(module, "is_public_registry", record_public_check)

    with pytest.raises(RuntimeError, match="context-invalid"):
        module._robomimic_live_selectors(project)

    assert checked_registries == [public_registry]


def test_robomimic_allows_immutable_development_but_quarantines_releases() -> None:
    source_sha = "a" * 40
    assert images.development_image_for_tool("robomimic", git_sha=source_sha) == (
        f"ghcr.io/nebius/nebius-physical-ai/npa-robomimic:dev-{source_sha}"
    )
    for tag in (None, "0.1.0", "dev-abcd"):
        with pytest.raises(ValueError, match="publication-quarantined"):
            images.container_image_for_tool("robomimic", tag=tag)
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


def test_robomimic_customer_notice_has_no_external_or_file_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _byof_runner_module()
    record = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(
        runner,
        "_base_image_candidates",
        lambda **_kwargs: pytest.fail("notice reached image resolution"),
    )

    status = runner.main(
        [
            "--repo-url",
            "https://github.com/ARISE-Initiative/robomimic.git",
            "--repo-ref",
            SOURCE_REVISION,
            "--solution-name",
            "robomimic",
            "--robomimic-runtime-entitlement-action",
            "notice",
            "--robomimic-runtime-entitlement-file",
            str(record),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 64
    assert payload["status"] == "notice"
    assert payload["external_action_started"] is False
    assert payload["actions"] == ["accept", "decline", "resume"]
    assert len(payload["terms"]) == 3
    assert "field_of_use" not in payload
    assert not record.exists()


def test_robomimic_customer_accept_records_only_bound_value_free_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _byof_runner_module()
    customer = "private-customer-identity"
    record = tmp_path / "entitlement.json"
    monkeypatch.setenv("NPA_E2E_PROJECT", customer)
    monkeypatch.setenv("NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256", "a" * 64)
    monkeypatch.setattr(
        runner,
        "_base_image_candidates",
        lambda **_kwargs: pytest.fail("accept reached image resolution"),
    )
    expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    notice_sha256 = runner._robomimic_entitlement_notice()["notice_sha256"]

    status = runner.main(
        [
            "--repo-url",
            "https://github.com/ARISE-Initiative/robomimic.git",
            "--repo-ref",
            SOURCE_REVISION,
            "--solution-name",
            "robomimic",
            "--run-id",
            "customer-accept",
            "--robomimic-runtime-entitlement-action",
            "accept",
            "--robomimic-runtime-entitlement-file",
            str(record),
            "--robomimic-runtime-entitlement-expires-at",
            expiry,
            "--robomimic-runtime-entitlement-notice-sha256",
            notice_sha256,
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    stored = json.loads(record.read_text(encoding="utf-8"))
    assert status == 0
    assert payload["status"] == "entitlement-recorded"
    assert payload["external_action_started"] is False
    assert stat.S_IMODE(record.stat().st_mode) == 0o600
    assert stored["customer_binding_sha256"] == _customer_binding(customer)
    assert stored["run_id"] == "customer-accept"
    assert stored["runtime_manifest_sha256"] == "a" * 64
    assert stored["decision"] == "accepted"
    assert stored["notice_sha256"] == notice_sha256
    assert "field_of_use" not in stored
    assert "field_of_use" not in payload
    assert customer not in stdout
    assert customer not in record.read_text(encoding="utf-8")


def test_robomimic_customer_accept_refuses_stale_notice_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _byof_runner_module()
    record = tmp_path / "entitlement.json"
    monkeypatch.setenv("NPA_E2E_PROJECT", "customer-project")
    monkeypatch.setenv("NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256", "a" * 64)
    expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    status = runner.main(
        [
            "--repo-url",
            "https://github.com/ARISE-Initiative/robomimic.git",
            "--repo-ref",
            SOURCE_REVISION,
            "--solution-name",
            "robomimic",
            "--run-id",
            "stale-notice",
            "--robomimic-runtime-entitlement-action",
            "accept",
            "--robomimic-runtime-entitlement-file",
            str(record),
            "--robomimic-runtime-entitlement-expires-at",
            expiry,
            "--robomimic-runtime-entitlement-notice-sha256",
            "0" * 64,
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 64
    assert payload["status"] == "refused"
    assert "exact current notice digest" in payload["error"]
    assert payload["build_started"] is False
    assert payload["push_started"] is False
    assert payload["run_started"] is False
    assert not record.exists()


def test_robomimic_entitlement_refuses_symlinked_or_permissive_parent(
    tmp_path: Path,
) -> None:
    runner = _byof_runner_module()
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    symlinked_parent = tmp_path / "linked"
    symlinked_parent.symlink_to(private_parent, target_is_directory=True)
    permissive_parent = tmp_path / "permissive"
    permissive_parent.mkdir(mode=0o755)
    permissive_parent.chmod(0o755)

    for record in (
        symlinked_parent / "entitlement.json",
        permissive_parent / "entitlement.json",
    ):
        with pytest.raises(runner.RobomimicEntitlementError) as raised:
            runner._write_robomimic_entitlement(record, {"decision": "accepted"})
        assert raised.value.code == "unsafe-parent"
        assert not record.exists()

    source = private_parent / "entitlement.json"
    source.write_text('{"decision":"accepted"}\n', encoding="utf-8")
    source.chmod(0o600)
    with pytest.raises(runner.RobomimicEntitlementError) as raised:
        runner._read_robomimic_entitlement(symlinked_parent / "entitlement.json")
    assert raised.value.code == "unsafe-parent"


def test_robomimic_entitlement_publication_detects_parent_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _byof_runner_module()
    parent = tmp_path / "records"
    parent.mkdir(mode=0o700)
    displaced = tmp_path / "records-displaced"
    record = parent / "entitlement.json"
    real_link = runner.os.link

    def replace_parent_after_link(*args, **kwargs) -> None:
        real_link(*args, **kwargs)
        parent.rename(displaced)
        parent.mkdir(mode=0o700)

    monkeypatch.setattr(runner.os, "link", replace_parent_after_link)

    with pytest.raises(runner.RobomimicEntitlementError) as raised:
        runner._write_robomimic_entitlement(record, {"decision": "accepted"})

    assert raised.value.code == "parent-changed"
    assert not record.exists()
    assert list(parent.iterdir()) == []
    assert list(displaced.iterdir()) == []


def test_robomimic_entitlement_read_detects_parent_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _byof_runner_module()
    parent = tmp_path / "records"
    parent.mkdir(mode=0o700)
    record = parent / "entitlement.json"
    record.write_text('{"decision":"accepted"}\n', encoding="utf-8")
    record.chmod(0o600)
    displaced = tmp_path / "records-displaced"
    real_fdopen = runner.os.fdopen
    replaced = False

    def replace_parent_before_read(*args, **kwargs):
        nonlocal replaced
        if not replaced:
            parent.rename(displaced)
            parent.mkdir(mode=0o700)
            replaced = True
        return real_fdopen(*args, **kwargs)

    monkeypatch.setattr(runner.os, "fdopen", replace_parent_before_read)

    with pytest.raises(runner.RobomimicEntitlementError) as raised:
        runner._read_robomimic_entitlement(record)

    assert raised.value.code == "parent-changed"
    assert not record.exists()
    assert (displaced / "entitlement.json").is_file()


def test_robomimic_entitlement_fifo_refuses_without_blocking(tmp_path: Path) -> None:
    runner = _byof_runner_module()
    fifo = tmp_path / "entitlement.json"
    os.mkfifo(fifo, mode=0o600)

    started = time.monotonic()
    with pytest.raises(runner.RobomimicEntitlementError) as raised:
        runner._read_robomimic_entitlement(fifo)

    assert time.monotonic() - started < 1.0
    assert raised.value.code == "unsafe-file"


def test_robomimic_entitlement_errors_never_disclose_exception_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _byof_runner_module()
    marker = "private-customer-path-marker"
    monkeypatch.setenv("NPA_E2E_PROJECT", "customer-project")
    monkeypatch.setenv("NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256", "a" * 64)
    monkeypatch.setattr(
        runner,
        "_write_robomimic_entitlement",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(marker)),
    )
    expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    status = runner.main(
        [
            "--repo-url",
            "https://github.com/ARISE-Initiative/robomimic.git",
            "--repo-ref",
            SOURCE_REVISION,
            "--solution-name",
            "robomimic",
            "--run-id",
            "redacted-refusal",
            "--robomimic-runtime-entitlement-action",
            "accept",
            "--robomimic-runtime-entitlement-file",
            str(tmp_path / marker / "entitlement.json"),
            "--robomimic-runtime-entitlement-expires-at",
            expiry,
            "--robomimic-runtime-entitlement-notice-sha256",
            runner._robomimic_entitlement_notice()["notice_sha256"],
        ]
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert status == 64
    assert payload["error_code"] == "entitlement-refused"
    assert payload["error"] == "robomimic customer runtime entitlement refused"
    assert marker not in output


def test_robomimic_customer_decline_does_not_create_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _byof_runner_module()
    record = tmp_path / "declined.json"
    monkeypatch.setattr(
        runner,
        "_base_image_candidates",
        lambda **_kwargs: pytest.fail("decline reached image resolution"),
    )

    status = runner.main(
        [
            "--repo-url",
            "https://github.com/ARISE-Initiative/robomimic.git",
            "--repo-ref",
            SOURCE_REVISION,
            "--solution-name",
            "robomimic",
            "--robomimic-runtime-entitlement-action",
            "decline",
            "--robomimic-runtime-entitlement-file",
            str(record),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 64
    assert payload["status"] == "declined"
    assert payload["external_action_started"] is False
    assert not record.exists()


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
        "NPA_BYOF_ROBOMIMIC_RUNTIME_ENTITLEMENT_FILE",
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


def test_robomimic_runner_recognizes_accepted_digest_under_renamed_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _byof_runner_module()
    digest = "sha256:" + "a" * 64
    monkeypatch.setenv(
        "NPA_BYOF_ROBOMIMIC_IMAGE",
        f"private.invalid/robomimic/npa-robomimic@{digest}",
    )
    args = runner._parse_args(
        [
            "--repo-url",
            "https://github.com/example/unrelated.git",
            "--repo-ref",
            "b" * 40,
            "--solution-name",
            "renamed",
            "--capability-name",
            "renamed",
            "--smoke-artifact-name",
            "renamed.json",
            "--smoke-command",
            "python train.py",
            "--image",
            f"docker:private.invalid/opaque/renamed@{digest}",
        ]
    )
    assert runner._is_robomimic_request(args) is True


@pytest.mark.parametrize(
    "registry",
    (
        "docker.io:443/example",
        "quay.io:5000/example",
        "public.ecr.aws.:80/example",
    ),
)
def test_robomimic_registry_guard_normalizes_public_hosts(registry: str) -> None:
    runner = _byof_runner_module()
    assert runner._is_public_robomimic_registry(registry) is True


def test_robomimic_registry_guard_normalizes_configured_public_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _byof_runner_module()
    monkeypatch.setattr(
        runner,
        "public_container_registry",
        lambda: "ghcr.io.:443/example/workbench/",
    )
    assert runner._is_public_robomimic_registry("ghcr.io/example/workbench") is True
    assert runner._is_public_robomimic_registry("ghcr.io/operator/private") is False


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


def test_workflow_refusal_recognizes_accepted_digest_under_renamed_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "a" * 64
    monkeypatch.setenv(
        "NPA_BYOF_ROBOMIMIC_IMAGE",
        f"private.invalid/robomimic/npa-robomimic@{digest}",
    )
    spec = SimpleNamespace(
        name="renamed",
        config={
            "repo_url": "https://github.com/example/unrelated.git",
            "repo_ref": "b" * 40,
            "solution_name": "renamed",
            "execution_policy": "direct",
            "controller_image": f"docker:private.invalid/opaque/renamed@{digest}",
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
            "image does not target the operator-selected private registry",
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
            "operator-selected bucket and solution prefix",
        ),
        (
            "private.invalid/actual",
            "private.invalid/manager",
            "private",
            "",
            "s3://manager-bucket/oss-solutions/robomimic",
            (),
            "registry does not match the operator-selected registry",
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
    run_id = "unsafe-target"
    entitlement = _customer_entitlement(
        tmp_path, project="manager-project", run_id=run_id
    )
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
        "--run-id",
        run_id,
        "--robomimic-runtime-entitlement-file",
        str(entitlement),
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
    entitlement = _customer_entitlement(
        tmp_path, project=selectors["NPA_E2E_PROJECT"], run_id=run_id
    )
    live_module = _live_e2e_module()
    profile = live_module._materialize_robomimic_attested_profile(
        tmp_path / "attested.yaml",
        namespace=selectors["NPA_BYOF_K8S_NAMESPACE"],
        service_account=live_module._robomimic_observer_name(run_id),
        runtime_pvc=selectors["NPA_BYOF_ROBOMIMIC_RUNTIME_PVC"],
        runtime_inventory_sha256=selectors[
            "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256"
        ],
        runtime_entitlement_file=str(entitlement),
        runtime_entitlement_sha256=hashlib.sha256(entitlement.read_bytes()).hexdigest(),
        customer_binding_sha256=_customer_binding(selectors["NPA_E2E_PROJECT"]),
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
            "--robomimic-runtime-entitlement-file",
            str(entitlement),
        ]
    )
    assert runner._handle_robomimic_entitlement_action(args) is None
    assert (
        runner._require_robomimic_execution_context(
            args,
            registry=args.registry,
            image=accepted_image,
            base_profile=args.base_profile,
        )
        is None
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
        runner._require_robomimic_execution_context(
            args,
            registry=args.registry,
            image=wrong_image,
            base_profile=args.base_profile,
        )
    malformed_image = "private.invalid/robomimic/unreviewed@sha256:" + "c" * 64
    monkeypatch.setenv("NPA_BYOF_ROBOMIMIC_IMAGE", malformed_image)
    args.image = malformed_image
    with pytest.raises(ValueError, match="exact private npa-robomimic digest"):
        runner._require_robomimic_execution_context(
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
    entitlement = _customer_entitlement(
        tmp_path, project="manager-project", run_id=run_id
    )
    monkeypatch_env["NPA_BYOF_ROBOMIMIC_RUNTIME_ENTITLEMENT_FILE"] = str(entitlement)
    profile = module._materialize_robomimic_attested_profile(
        tmp_path / "attested.yaml",
        namespace="robomimic-validation",
        service_account=module._robomimic_observer_name(run_id),
        runtime_pvc=monkeypatch_env["NPA_BYOF_ROBOMIMIC_RUNTIME_PVC"],
        runtime_inventory_sha256=monkeypatch_env[
            "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256"
        ],
        runtime_entitlement_file=str(entitlement),
        runtime_entitlement_sha256=hashlib.sha256(entitlement.read_bytes()).hexdigest(),
        customer_binding_sha256=_customer_binding("manager-project"),
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
        "--robomimic-runtime-entitlement-file",
        str(entitlement),
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
    runtime_lock_bytes = RUNTIME_LOCK.read_bytes()
    runtime_lock = json.loads(runtime_lock_bytes)
    assert len(runtime_lock["packages"]) == 62
    assert hashlib.sha256(runtime_lock_bytes).hexdigest() == RUNTIME_LOCK_SHA256
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
        'source_identity["revision"] != SOURCE_REVISION',
        f'RUNTIME_LOCK_SHA256 = "{RUNTIME_LOCK_SHA256}"',
        "RUNTIME_DISTRIBUTION_COUNT = 62",
        "models/model_epoch_1.pth",
        "policy_from_checkpoint(",
        "optimizer_steps != TRAIN_STEPS",
        "action.shape != (7,)",
        '"B200" not in gpu_name.upper()',
        'architecture != "sm_100"',
        "_require_strict_capacity_observation()",
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
        '"observed_head": source_identity["observed_head"]',
        '"git_tree_sha1": source_identity["git_tree_sha1"]',
        '"trajectory_count": len(demo_keys)',
        '"sample_count": sum(sample_counts.values())',
        '"train_trajectory_count": len(train_keys)',
        '"validation_trajectory_count": len(valid_keys)',
        '"train_sample_count": sum(sample_counts[key] for key in train_keys)',
        '"validation_sample_count": sum(sample_counts[key] for key in valid_keys)',
        '"train_loss": train_loss',
        '"validation_loss": validation_loss',
        '"configured_validation_forward_steps": VALIDATION_STEPS',
        '"sha256": training["checkpoint_hash"]',
        '"finite": finite',
        '"within_allowed_range": training["within_range"]',
        '"accelerator_count": gpu_count',
        '"runtime_ref": runtime_image',
        '"observation_source": "Kubernetes Pod status.containerStatuses[].imageID"',
        '"pod_image"',
        '"workload_identity"',
        '"pretrained_weights": False',
        '"runtime_cache": "external-read-only-prepopulated"',
        '"runtime_manifest_digest_matched": True',
        "verify_customer_runtime_entitlement(",
        '"customer_runtime_entitlement": context["entitlement"]',
        '"atomic_private_snapshot_published": True',
        '"snapshot_write_bits_absent": context["runtime_root"].stat().st_mode & 0o222',
        '"exit_status": 0',
    ):
        assert required in smoke

    assert "image_policy_sweeps" in smoke
    assert "simulator_rollouts" in smoke
    assert "full_algorithm_matrix" in smoke
    assert "robosuite" not in smoke
    assert '"strict_reserved_capacity_attested": True' not in smoke
    ast.parse(_smoke_python())


def test_robomimic_dataset_stream_stops_and_removes_oversized_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedResponse:
        def __init__(self) -> None:
            self.chunks = iter((b"abc", b"d"))

        def __enter__(self) -> OversizedResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return next(self.chunks)

        def close(self) -> None:
            return None

    module = _smoke_module(monkeypatch)
    monkeypatch.setattr(module, "verify_customer_runtime_entitlement", lambda **_kw: {})
    response = OversizedResponse()
    connection_closed: list[bool] = []
    connection = SimpleNamespace(close=lambda: connection_closed.append(True))
    monkeypatch.setattr(module, "DATASET_BYTES", 3)
    monkeypatch.setattr(
        module,
        "_open_allowed_https",
        lambda *_args, **_kwargs: (connection, response),
    )

    with pytest.raises(RuntimeError, match="exceeds its locked byte count"):
        module._download_dataset(tmp_path / "inputs", entitlement_arguments={})

    assert connection_closed == [True]
    assert list((tmp_path / "inputs").iterdir()) == []


def test_robomimic_dataset_retry_cleans_owned_stale_partial_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _smoke_module(monkeypatch)
    monkeypatch.setattr(module, "verify_customer_runtime_entitlement", lambda **_kw: {})
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    partial = input_dir / "lift_ph_lowdim_v15.download"
    partial.write_bytes(b"interrupted")
    opened: list[bool] = []

    def fail_after_cleanup(*_args: object, **_kwargs: object) -> object:
        assert not partial.exists()
        opened.append(True)
        raise RuntimeError("network sentinel")

    monkeypatch.setattr(module, "_open_allowed_https", fail_after_cleanup)

    with pytest.raises(RuntimeError, match="network sentinel"):
        module._download_dataset(input_dir, entitlement_arguments={})

    assert opened == [True]
    assert list(input_dir.iterdir()) == []


def test_robomimic_dataset_retry_rejects_stale_partial_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _smoke_module(monkeypatch)
    monkeypatch.setattr(module, "verify_customer_runtime_entitlement", lambda **_kw: {})
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"preserve")
    partial = input_dir / "lift_ph_lowdim_v15.download"
    partial.symlink_to(outside)

    with pytest.raises(RuntimeError, match="unsafe stale dataset partial"):
        module._download_dataset(input_dir, entitlement_arguments={})

    assert partial.is_symlink()
    assert outside.read_bytes() == b"preserve"


@pytest.mark.parametrize("assertion", ("", "1", '{"policy":"STRICT"}'))
def test_robomimic_strict_claim_refuses_caller_assertions_without_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, assertion: str
) -> None:
    """Inert interface refusal, never a recorded or live allocation proof."""
    module = _smoke_module(monkeypatch)
    output = tmp_path / "output"
    monkeypatch.setenv("NPA_SMOKE_OUTPUT_DIR", str(output))
    monkeypatch.setenv("NPA_ROBOMIMIC_STRICT_B200_ATTESTED", assertion)
    monkeypatch.setattr(module, "_pod_identity", lambda *_a: pytest.fail("Pod query"))
    monkeypatch.setattr(module, "_hardware", lambda: pytest.fail("GPU query"))
    monkeypatch.setattr(module, "_download_dataset", lambda *_a: pytest.fail("fetch"))
    with pytest.raises(RuntimeError, match="STRICT capacity qualification deferred"):
        module.main()
    assert not output.exists()


def test_robomimic_training_mode_keeps_runtime_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _smoke_module(monkeypatch)
    monkeypatch.setattr(module.sys, "argv", ["smoke.py", "--train-smoke"])
    monkeypatch.setenv("NPA_SMOKE_OUTPUT_DIR", str(tmp_path / "output"))

    def refuse_context(_output: Path) -> None:
        raise RuntimeError("runtime context refused")

    monkeypatch.setattr(module, "_load_smoke_context", refuse_context)
    monkeypatch.setattr(module, "_download_dataset", lambda *_a: pytest.fail("fetch"))
    with pytest.raises(RuntimeError, match="runtime context refused"):
        module.main()


def test_robomimic_inert_local_b200_shape_does_not_claim_strict_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked hardware shape only: no GPU, scheduler or provider qualification."""
    module = _smoke_module(monkeypatch)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(module.torch.cuda, "get_device_name", lambda _i: "NVIDIA B200")
    monkeypatch.setattr(module.torch.cuda, "get_device_capability", lambda _i: (10, 0))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_a, **_kw: SimpleNamespace(stdout="0, NVIDIA B200"),
    )
    observed = module._hardware()
    assert observed["accelerator_count"] == 1
    assert observed["architecture"] == "sm_100"
    assert not any("strict" in key or "reserved" in key for key in observed)


def _inert_smoke_entitlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    expire_at_check: int | None = None,
) -> SimpleNamespace:
    """Synthetic customer/run record checked by the real pure verifier, no acceptance."""
    path = _customer_entitlement(tmp_path, project="inert-project", run_id="inert-run")
    record = json.loads(path.read_text())
    spec = importlib.util.spec_from_file_location(
        "inert_verifier", IMAGE_ROOT / "verify_image.py"
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    arguments = {
        "entitlement_path": path,
        "runtime_lock_path": IMAGE_ROOT / "runtime-requirements.lock",
        "expected_entitlement_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected_customer_binding_sha256": _customer_binding("inert-project"),
        "expected_run_id": "inert-run",
        "expected_inventory_sha256": "a" * 64,
    }
    calls: list[dict[str, object]] = []

    def verify(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        key = "expires_at" if expire_at_check == len(calls) else "accepted_at"
        now = datetime.strptime(record[key], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        return verifier.verify_customer_runtime_entitlement(**kwargs, now=now)

    monkeypatch.setattr(module, "verify_customer_runtime_entitlement", verify)
    return SimpleNamespace(arguments=arguments, calls=calls, path=path)


class _InertDatasetBody(_SmokeProofBody):
    def __enter__(self) -> _InertDatasetBody:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@pytest.mark.parametrize("expire_at_check", (1, 2, 3, 4, 5))
def test_robomimic_dataset_expiry_at_each_side_effect_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expire_at_check: int,
) -> None:
    module = _smoke_module(monkeypatch)
    binding = _inert_smoke_entitlement(
        tmp_path, monkeypatch, module, expire_at_check=expire_at_check
    )
    events: list[str] = []
    response = _InertDatasetBody(b"abc")
    connection = SimpleNamespace(close=lambda: events.append("close"))

    def open_inert(*_args: object, **kwargs: object) -> tuple[object, object]:
        kwargs["before_request"]()
        events.append("request")
        return connection, response

    monkeypatch.setattr(module, "_open_allowed_https", open_inert)
    inputs = tmp_path / "inputs"
    with pytest.raises(
        RuntimeError, match="customer runtime entitlement refused"
    ) as error:
        module._download_dataset(inputs, entitlement_arguments=binding.arguments)
    assert len(binding.calls) == expire_at_check
    assert all(call == binding.arguments for call in binding.calls)
    assert str(binding.path) not in _serialized_entitlement_refusal(error.value)
    assert not inputs.exists() if expire_at_check == 1 else list(inputs.iterdir()) == []
    assert ("request" in events) == (expire_at_check > 3)
    if "request" in events:
        assert response.closed and events[-1] == "close"


@pytest.mark.parametrize(
    "field",
    (
        "expected_entitlement_sha256",
        "expected_customer_binding_sha256",
        "expected_run_id",
        "expected_inventory_sha256",
    ),
)
def test_robomimic_dataset_binding_mismatch_precedes_cache_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    module = _smoke_module(monkeypatch)
    binding = _inert_smoke_entitlement(tmp_path, monkeypatch, module)
    module._value_free_customer_entitlement(**binding.arguments)
    binding.arguments[field] = (
        "different-run" if field == "expected_run_id" else "b" * 64
    )
    monkeypatch.setattr(
        module, "_open_allowed_https", lambda *_a, **_k: pytest.fail("network")
    )
    inputs = tmp_path / "inputs"
    with pytest.raises(RuntimeError, match="customer runtime entitlement refused"):
        module._download_dataset(inputs, entitlement_arguments=binding.arguments)
    assert not inputs.exists()


def test_robomimic_expiry_preserves_preexisting_partial_until_authorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _smoke_module(monkeypatch)
    binding = _inert_smoke_entitlement(tmp_path, monkeypatch, module, expire_at_check=2)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    partial = inputs / "lift_ph_lowdim_v15.download"
    partial.write_bytes(b"inert-owned-partial")
    monkeypatch.setattr(
        module, "_open_allowed_https", lambda *_a, **_k: pytest.fail("network")
    )
    with pytest.raises(RuntimeError, match="customer runtime entitlement refused"):
        module._download_dataset(inputs, entitlement_arguments=binding.arguments)
    assert partial.read_bytes() == b"inert-owned-partial"


def test_robomimic_https_reauthorizes_before_redirect_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _smoke_module(monkeypatch)
    events: list[str] = []
    response = SimpleNamespace(
        status=302,
        getheader=lambda _n: "https://huggingface.co/next",
        close=lambda: events.append("response-close"),
    )
    connection = SimpleNamespace(
        request=lambda *_a, **_k: events.append("request"),
        getresponse=lambda: response,
        close=lambda: events.append("connection-close"),
    )
    monkeypatch.setattr(
        module.http.client, "HTTPSConnection", lambda *_a, **_k: connection
    )

    def authorize() -> None:
        if "request" in events:
            raise RuntimeError("expired inert record")
        events.append("authorized")

    with pytest.raises(RuntimeError, match="expired inert record"):
        module._open_allowed_https(
            "https://huggingface.co/start",
            headers={},
            allowed_hosts=("huggingface.co",),
            before_request=authorize,
        )
    assert events == ["authorized", "request", "response-close", "connection-close"]


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
    assert "path.read_bytes()" not in profile_text
    assert "MAX_OUTPUT_FILE_BYTES" in profile_text
    assert "MAX_OUTPUT_TOTAL_BYTES" in profile_text
    assert "MAX_OUTPUT_ENTRY_COUNT" in profile_text
    assert "MAX_OUTPUT_PENDING_DIRECTORY_COUNT" in profile_text
    assert "MAX_OUTPUT_OBJECT_COUNT" in profile_text
    assert "with os.scandir(current) as entries:" in profile_text
    assert 'root.rglob("*")' not in profile_text
    assert "RESERVED_JSON_MAX_BYTES" in profile_text
    assert "smoke_artifact_path.read_text" not in profile_text
    assert '(root / "npa_byof_summary.json").write_text' not in profile_text
    assert "ContentLength=size" in profile_text
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


def test_robomimic_profile_streams_outputs_in_bounded_chunks(tmp_path: Path) -> None:
    module = _profile_upload_module(tmp_path)
    root = tmp_path / "outputs"
    root.mkdir()
    payload = root / "checkpoint.pth"
    payload.write_bytes(b"x" * (2 * 1024 * 1024 + 7))

    class RecordingS3:
        def __init__(self) -> None:
            self.chunks: list[int] = []
            self.digest = hashlib.sha256()

        def put_object(self, **kwargs: object) -> None:
            assert kwargs["ContentLength"] == payload.stat().st_size
            assert kwargs["IfNoneMatch"] == "*"
            body = kwargs["Body"]
            assert body.read(0) == b""
            while chunk := body.read():
                self.chunks.append(len(chunk))
                self.digest.update(chunk)

    s3 = RecordingS3()
    assert module.upload_outputs(s3, "bucket", "prefix/", root) == 1
    assert s3.chunks
    assert max(s3.chunks) <= module.OUTPUT_READ_CHUNK_BYTES
    assert s3.digest.hexdigest() == hashlib.sha256(payload.read_bytes()).hexdigest()


def test_robomimic_profile_refuses_directory_entry_fanout_before_upload(
    tmp_path: Path,
) -> None:
    module = _profile_upload_module(tmp_path)
    root = tmp_path / "outputs"
    root.mkdir()
    for index in range(module.MAX_OUTPUT_ENTRY_COUNT + 1):
        (root / f"directory-{index:04d}").mkdir()

    class NoUploadS3:
        def put_object(self, **_kwargs: object) -> None:
            pytest.fail("entry fanout must be rejected before upload")

    with pytest.raises(RuntimeError, match="filesystem entry count limit"):
        module.upload_outputs(NoUploadS3(), "bucket", "prefix/", root)


def test_robomimic_profile_bounds_pending_directories_as_entries_arrive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _profile_upload_module(tmp_path)
    root = tmp_path / "outputs"
    root.mkdir()
    (root / "first").mkdir()
    (root / "second").mkdir()
    monkeypatch.setattr(module, "MAX_OUTPUT_PENDING_DIRECTORY_COUNT", 1)

    with pytest.raises(RuntimeError, match="pending directory limit"):
        module.checked_output_files(root)


def test_robomimic_profile_refuses_zero_byte_object_fanout_before_upload(
    tmp_path: Path,
) -> None:
    module = _profile_upload_module(tmp_path)
    root = tmp_path / "outputs"
    root.mkdir()
    for index in range(module.MAX_OUTPUT_OBJECT_COUNT + 1):
        (root / f"empty-{index:04d}.json").touch()

    class NoUploadS3:
        def put_object(self, **_kwargs: object) -> None:
            pytest.fail("object fanout must be rejected before upload")

    with pytest.raises(RuntimeError, match="regular object count limit"):
        module.upload_outputs(NoUploadS3(), "bucket", "prefix/", root)


def test_robomimic_profile_consumes_recursive_entries_lazily(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _profile_upload_module(tmp_path)
    root = tmp_path / "outputs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("preserve", encoding="utf-8")
    unsafe = root / "first-entry"
    unsafe.symlink_to(outside)

    class HostileScandir:
        def __init__(self) -> None:
            self.consumed = False
            self.closed = False

        def __enter__(self) -> HostileScandir:
            return self

        def __exit__(self, *_args: object) -> None:
            self.closed = True

        def __iter__(self) -> HostileScandir:
            return self

        def __next__(self) -> object:
            if self.consumed:
                pytest.fail("recursive entries were materialized before validation")
            self.consumed = True
            return SimpleNamespace(
                path=str(unsafe),
                stat=lambda *, follow_symlinks: unsafe.lstat(),
            )

    iterator = HostileScandir()
    monkeypatch.setattr(module.os, "scandir", lambda _root: iterator)

    with pytest.raises(RuntimeError, match="output symlink is forbidden"):
        module.checked_output_files(root)
    assert iterator.closed is True


def test_robomimic_profile_uploads_legal_bounded_output_set(tmp_path: Path) -> None:
    module = _profile_upload_module(tmp_path)
    root = tmp_path / "outputs"
    nested = root / "metrics"
    nested.mkdir(parents=True)
    (root / "checkpoint.pth").write_bytes(b"checkpoint")
    (nested / "validation.json").write_text('{"loss": 0.25}\n', encoding="utf-8")

    class RecordingS3:
        def __init__(self) -> None:
            self.keys: list[str] = []

        def put_object(self, **kwargs: object) -> None:
            self.keys.append(str(kwargs["Key"]))
            body = kwargs["Body"]
            while body.read():
                pass

    s3 = RecordingS3()
    assert module.upload_outputs(s3, "bucket", "prefix/", root) == 2
    assert set(s3.keys) == {
        "prefix/checkpoint.pth",
        "prefix/metrics/validation.json",
    }


@pytest.mark.parametrize("limit_kind", ("per-file", "aggregate"))
def test_robomimic_profile_refuses_oversized_output_sets(
    tmp_path: Path, limit_kind: str
) -> None:
    module = _profile_upload_module(tmp_path)
    root = tmp_path / "outputs"
    root.mkdir()
    if limit_kind == "per-file":
        (root / "oversized").touch()
        os.truncate(root / "oversized", module.MAX_OUTPUT_FILE_BYTES + 1)
        expected = "per-file byte limit"
    else:
        for index in range(3):
            path = root / f"part-{index}"
            path.touch()
            os.truncate(path, module.MAX_OUTPUT_FILE_BYTES)
        expected = "aggregate byte limit"

    class NoUploadS3:
        def put_object(self, **_kwargs: object) -> None:
            pytest.fail("oversized outputs must be rejected before upload")

    with pytest.raises(RuntimeError, match=expected):
        module.upload_outputs(NoUploadS3(), "bucket", "prefix/", root)


def test_robomimic_profile_detects_output_growth_during_upload(
    tmp_path: Path,
) -> None:
    module = _profile_upload_module(tmp_path)
    root = tmp_path / "outputs"
    root.mkdir()
    payload = root / "checkpoint.pth"
    payload.write_bytes(b"checkpoint")

    class GrowingS3:
        def put_object(self, **kwargs: object) -> None:
            body = kwargs["Body"]
            while body.read():
                pass
            with payload.open("ab") as handle:
                handle.write(b"unexpected-growth")

    with pytest.raises(RuntimeError, match="size changed during upload"):
        module.upload_outputs(GrowingS3(), "bucket", "prefix/", root)


@pytest.mark.parametrize("source_kind", ("reserved-json", "output"))
def test_robomimic_profile_refuses_fifo_sources_without_blocking(
    tmp_path: Path, source_kind: str
) -> None:
    module = _profile_upload_module(tmp_path)
    root = tmp_path / "outputs"
    root.mkdir()
    fifo = root / ("reserved.json" if source_kind == "reserved-json" else "output.bin")
    os.mkfifo(fifo)
    started = time.monotonic()

    if source_kind == "reserved-json":
        with pytest.raises(RuntimeError, match="one regular file"):
            module.read_reserved_json_object(fifo)
    else:
        with pytest.raises(RuntimeError, match="one regular unlinked file"):
            module.upload_outputs(SimpleNamespace(), "bucket", "prefix/", root)

    assert time.monotonic() - started < 1.0


@pytest.mark.parametrize("source_kind", ("reserved-json", "output"))
def test_robomimic_profile_refuses_regular_file_to_fifo_open_race(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source_kind: str
) -> None:
    module = _profile_upload_module(tmp_path)
    root = tmp_path / "outputs"
    root.mkdir()
    source = root / (
        "reserved.json" if source_kind == "reserved-json" else "output.bin"
    )
    source.write_text("{}\n", encoding="utf-8")
    original_open = module.os.open
    replaced = False

    def replace_before_open(
        path: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        nonlocal replaced
        if Path(path) == source and not replaced:
            replaced = True
            source.unlink()
            os.mkfifo(source)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", replace_before_open)
    started = time.monotonic()
    if source_kind == "reserved-json":
        with pytest.raises(RuntimeError, match="one regular file"):
            module.read_reserved_json_object(source)
    else:
        with pytest.raises(RuntimeError, match="identity changed before upload"):
            module.upload_outputs(SimpleNamespace(), "bucket", "prefix/", root)

    assert replaced is True
    assert time.monotonic() - started < 1.0


@pytest.mark.parametrize("transport_kind", ("upload", "client", "verification"))
def test_robomimic_profile_transport_refusals_are_value_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transport_kind: str,
) -> None:
    module = _profile_upload_module(tmp_path)
    marker = f"private-{transport_kind}-transport-marker"

    class RefusingS3:
        def put_object(self, **_kwargs: object) -> None:
            raise RuntimeError(marker)

        def head_object(self, **_kwargs: object) -> None:
            raise RuntimeError(marker)

    if transport_kind == "upload":
        root = tmp_path / "outputs"
        root.mkdir()
        (root / "result.json").write_text("{}\n", encoding="utf-8")

        def operation() -> object:
            return module.upload_outputs(
                RefusingS3(), "private-bucket", "private-prefix/", root
            )

    elif transport_kind == "client":
        monkeypatch.setattr(
            module.boto3,
            "client",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(marker)),
        )

        def operation() -> object:
            return module.create_output_client({"endpoint_url": marker})

    else:

        def operation() -> object:
            return module.verify_uploaded_objects(
                RefusingS3(), "private-bucket", ["private-key"]
            )

    with pytest.raises(RuntimeError) as raised:
        operation()

    error = raised.value
    serialized = json.dumps(
        {
            "args": error.args,
            "cause": repr(error.__cause__),
            "context": repr(error.__context__),
            "traceback": "".join(traceback.format_exception(error)),
        }
    )
    assert marker not in serialized
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("malformation", ("port", "bracketed-host"))
def test_robomimic_profile_endpoint_refusal_is_value_free(
    tmp_path: Path, malformation: str
) -> None:
    module = _profile_upload_module(tmp_path)
    marker = "private-uploader-port-marker"
    endpoint = (
        f"https://storage.test-region.nebius.cloud:{marker}"
        if malformation == "port"
        else f"https://[{marker}"
    )

    with pytest.raises(RuntimeError, match="endpoint is malformed") as raised:
        module.checked_storage_endpoint(endpoint)

    error = raised.value
    serialized = json.dumps(
        {
            "args": error.args,
            "cause": repr(error.__cause__),
            "context": repr(error.__context__),
            "traceback": "".join(traceback.format_exception(error)),
        }
    )
    assert marker not in serialized
    assert error.__cause__ is None
    assert error.__context__ is None


def test_robomimic_profile_refuses_symlinked_smoke_artifact(
    tmp_path: Path,
) -> None:
    module = _profile_upload_module(tmp_path)
    outside = tmp_path / "outside-smoke.json"
    outside.write_text('{"preserve": true}\n', encoding="utf-8")
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    smoke = output_root / "robomimic-smoke.json"
    smoke.symlink_to(outside)

    with pytest.raises(RuntimeError, match="not a safe regular file"):
        module.read_reserved_json_object(smoke)

    assert smoke.is_symlink()
    assert outside.read_text(encoding="utf-8") == '{"preserve": true}\n'


def test_robomimic_profile_atomically_replaces_symlinked_summary(
    tmp_path: Path,
) -> None:
    module = _profile_upload_module(tmp_path)
    outside = tmp_path / "outside-summary.json"
    outside.write_text('{"preserve": true}\n', encoding="utf-8")
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    summary = output_root / "npa_byof_summary.json"
    summary.symlink_to(outside)

    module.atomic_write_json_object(summary, {"status": "failed"})

    assert not summary.is_symlink()
    assert json.loads(summary.read_text(encoding="utf-8")) == {"status": "failed"}
    assert outside.read_text(encoding="utf-8") == '{"preserve": true}\n'


@pytest.mark.parametrize("malformation", ("port", "bracketed-host"))
def test_robomimic_download_refuses_a_malformed_allowed_host(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    module = _smoke_module(monkeypatch)
    marker = "private-invalid-port-marker"
    url = (
        f"https://huggingface.co:{marker}/file"
        if malformation == "port"
        else f"https://[{marker}"
    )
    with pytest.raises(RuntimeError, match="malformed approved HTTPS URL") as raised:
        module._open_allowed_https(
            url,
            headers={},
            allowed_hosts=("huggingface.co",),
        )

    error = raised.value
    serialized = json.dumps(
        {
            "args": error.args,
            "cause": repr(error.__cause__),
            "context": repr(error.__context__),
            "traceback": "".join(traceback.format_exception(error)),
        }
    )
    assert marker not in serialized
    assert error.__cause__ is None
    assert error.__context__ is None
    assert not getattr(error, "__notes__", ())


def test_robomimic_download_refuses_a_malformed_redirect_value_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _smoke_module(monkeypatch)
    marker = "private-redirect-authority-marker"

    class RedirectResponse:
        status = 302

        def getheader(self, _name: str) -> str:
            return f"https://[{marker}"

        def close(self) -> None:
            return None

    class RedirectConnection:
        def request(self, *_args: object, **_kwargs: object) -> None:
            return None

        def getresponse(self) -> RedirectResponse:
            return RedirectResponse()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        module.http.client,
        "HTTPSConnection",
        lambda *_args, **_kwargs: RedirectConnection(),
    )
    with pytest.raises(
        RuntimeError, match="malformed approved HTTPS redirect"
    ) as raised:
        module._open_allowed_https(
            "https://huggingface.co/approved",
            headers={},
            allowed_hosts=("huggingface.co",),
        )

    diagnostics = _serialized_entitlement_refusal(raised.value)
    assert marker not in diagnostics
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not getattr(raised.value, "__notes__", ())


@pytest.mark.parametrize("target_kind", ("path", "query"))
def test_robomimic_download_refuses_non_ascii_target_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    module = _smoke_module(monkeypatch)
    marker = "private-non-ascii-request-marker"
    url = (
        f"https://huggingface.co/{marker}-\N{SNOWMAN}"
        if target_kind == "path"
        else f"https://huggingface.co/approved?token={marker}-\N{SNOWMAN}"
    )
    connection_attempts = 0

    def fail_connection(*_args: object, **_kwargs: object) -> None:
        nonlocal connection_attempts
        connection_attempts += 1
        pytest.fail("connection creation reached after request-target refusal")

    monkeypatch.setattr(module.http.client, "HTTPSConnection", fail_connection)
    with pytest.raises(RuntimeError, match="approved HTTPS transport failed") as raised:
        module._open_allowed_https(
            url,
            headers={},
            allowed_hosts=("huggingface.co",),
        )
    assert connection_attempts == 0
    assert marker not in _serialized_entitlement_refusal(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert not getattr(raised.value, "__notes__", ())


def test_robomimic_download_transport_refusal_is_value_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _smoke_module(monkeypatch)
    marker = "private-" + "download-transport-marker"

    class RefusingConnection:
        def request(self, *_args: object, **_kwargs: object) -> None:
            raise OSError(marker)

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        module.http.client,
        "HTTPSConnection",
        lambda *_args, **_kwargs: RefusingConnection(),
    )

    with pytest.raises(RuntimeError, match="approved HTTPS transport failed") as raised:
        module._open_allowed_https(
            "https://huggingface.co/approved",
            headers={},
            allowed_hosts=("huggingface.co",),
        )

    error = raised.value
    serialized = json.dumps(
        {
            "args": error.args,
            "cause": repr(error.__cause__),
            "context": repr(error.__context__),
            "traceback": "".join(traceback.format_exception(error)),
        }
    )
    assert marker not in serialized
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("declared_size", (None, "12", -1, 64 * 1024 + 1))
def test_robomimic_artifact_refuses_invalid_size_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    declared_size: object,
) -> None:
    module = _live_e2e_module()
    marker = "private-artifact-size-marker"
    body = _SmokeProofBody(b'{"solution":"robomimic"}')
    response = {"Body": body}
    if declared_size is not None:
        response["ContentLength"] = declared_size
    calls = _mock_robomimic_artifact_client(monkeypatch, module, response)
    _assert_value_free_artifact_refusal(module, marker)
    assert len(calls) == 1
    assert body.read_amounts == []
    assert body.closed is True


@pytest.mark.parametrize("stream_kind", ("lying-length", "oversized"))
def test_robomimic_artifact_refuses_invalid_stream_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    stream_kind: str,
) -> None:
    module = _live_e2e_module()
    marker = "private-artifact-stream-marker"
    payload = (
        b"{}"
        if stream_kind == "lying-length"
        else b"x" * (module.ROBOMIMIC_SMOKE_PROOF_MAX_BYTES + 1)
    )
    body = _SmokeProofBody(payload)
    response = {"Body": body, "ContentLength": 1}
    _mock_robomimic_artifact_client(monkeypatch, module, response)
    _assert_value_free_artifact_refusal(module, marker)
    assert body.read_amounts == [module.ROBOMIMIC_SMOKE_PROOF_MAX_BYTES + 1]
    assert body.closed is True


@pytest.mark.parametrize("failure_kind", ("read", "decode", "json"))
def test_robomimic_artifact_sanitizes_content_failures_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    module = _live_e2e_module()
    marker = "private-artifact-content-marker"
    payload = b"\xff" if failure_kind == "decode" else marker.encode()
    read_error = OSError(marker) if failure_kind == "read" else None
    body = _SmokeProofBody(payload, read_error=read_error)
    response = {"Body": body, "ContentLength": len(payload)}
    _mock_robomimic_artifact_client(monkeypatch, module, response)
    _assert_value_free_artifact_refusal(module, marker)
    assert body.read_amounts == [module.ROBOMIMIC_SMOKE_PROOF_MAX_BYTES + 1]
    assert body.closed is True


def test_robomimic_artifact_sanitizes_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _live_e2e_module()
    marker = "private-artifact-transport-marker"
    calls = _mock_robomimic_artifact_client(monkeypatch, module, OSError(marker))
    _assert_value_free_artifact_refusal(module, marker)
    assert len(calls) == 1


def test_robomimic_artifact_sanitizes_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _live_e2e_module()
    marker = "private-artifact-close-marker"
    payload = b'{"solution":"robomimic"}'
    body = _SmokeProofBody(payload, close_error=OSError(marker))
    response = {"Body": body, "ContentLength": len(payload)}
    _mock_robomimic_artifact_client(monkeypatch, module, response)
    _assert_value_free_artifact_refusal(module, marker)
    assert body.read_amounts == [module.ROBOMIMIC_SMOKE_PROOF_MAX_BYTES + 1]
    assert body.closed is True


def test_robomimic_artifact_reads_once_bounded_and_closes_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _live_e2e_module()
    payload = b'{"solution":"robomimic"}'
    body = _SmokeProofBody(payload)
    response = {"Body": body, "ContentLength": len(payload)}
    calls = _mock_robomimic_artifact_client(monkeypatch, module, response)
    artifact = module._robomimic_artifact("project", "bucket", "run")
    assert artifact == {"solution": "robomimic"}
    assert len(calls) == 1
    assert body.read_amounts == [module.ROBOMIMIC_SMOKE_PROOF_MAX_BYTES + 1]
    assert body.closed is True


def test_robomimic_smoke_entitlement_refusal_is_value_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _smoke_module(monkeypatch)
    marker = "private-runtime-record-marker"
    monkeypatch.setattr(
        module,
        "verify_customer_runtime_entitlement",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(marker)),
    )

    with pytest.raises(
        RuntimeError, match="customer runtime entitlement refused"
    ) as raised:
        module._value_free_customer_entitlement(record=marker)

    error = raised.value
    serialized = json.dumps(
        {
            "args": error.args,
            "cause": repr(error.__cause__),
            "context": repr(error.__context__),
            "traceback": "".join(traceback.format_exception(error)),
        }
    )
    assert marker not in serialized
    assert error.__cause__ is None
    assert error.__context__ is None


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
    assert "These records do not assert a completed image build" in doc
    assert "pretrained weights" in doc
    assert "every layer and image history entry" in doc
    combined = "\n".join((doc, workflow, profile))
    for invented_proxy in (
        "NPA_ROBOMIMIC_ACCEPT",
        "NPA_ACCEPT_CUDA",
        "NPA_ACCEPT_CUDNN",
    ):
        assert invented_proxy not in combined
