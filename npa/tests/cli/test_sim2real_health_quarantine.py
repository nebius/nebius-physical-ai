"""Keep image-independent Sim2Real health checks available during quarantine."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench import health
from npa.clients import credentials
from npa.deploy.images import container_image_for_tool
from npa.workflows.sim2real import config as execution_config
from npa.workflows.sim2real import diagnostic_config, models
from npa.workflows.sim2real_health import (
    IMAGE_FIELDS,
    IMAGE_DEPENDENT_CHECKS,
    DoctorProbes,
    KubeResult,
    run_preflight,
)

runner = CliRunner()
_IMAGE_FIELDS = (
    "augment_image",
    "envgen_image",
    "policy_image",
    "trainer_image",
    "vlm_image",
    "vlm_reason2_image",
    "vlm_cosmos3_image",
    "eval_image",
    "isaac_image",
)


def _cluster_probe(args: list[str]) -> KubeResult:
    if args == ["config", "current-context"]:
        return KubeResult(0, "diagnostic-context")
    if args[:2] == ["auth", "can-i"]:
        return KubeResult(0, "yes")
    if args[:2] == ["get", "pvc"]:
        return KubeResult(
            0,
            json.dumps(
                {
                    "spec": {
                        "volumeName": "cache-volume",
                        "accessModes": ["ReadWriteMany"],
                    },
                    "status": {"phase": "Bound"},
                }
            ),
        )
    if args[:2] == ["get", "nodes"]:
        return KubeResult(
            0,
            json.dumps(
                {
                    "items": [
                        {
                            "metadata": {
                                "labels": {
                                    "nvidia.com/gpu.product": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
                                }
                            },
                            "status": {"allocatable": {"nvidia.com/gpu": "1"}},
                        }
                    ]
                }
            ),
        )
    raise AssertionError(f"unexpected Kubernetes probe: {args}")


@pytest.fixture
def diagnostic_spies(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    for field in _IMAGE_FIELDS:
        monkeypatch.delenv(field.upper(), raising=False)
    monkeypatch.delenv("NPA_SIM2REAL_REGISTRY", raising=False)
    forbidden = Mock(side_effect=AssertionError("execution image resolution attempted"))
    monkeypatch.setattr(models, "container_image_for_tool", forbidden)
    execution = Mock(side_effect=AssertionError("execution config constructed"))
    monkeypatch.setattr(health, "build_config_from_env", execution)
    image_inspector = Mock(side_effect=AssertionError("image inspection attempted"))
    monkeypatch.setattr(health, "_image_inspector", image_inspector)
    loaded = Mock(
        return_value=SimpleNamespace(
            hf_token="fixture-token",
            ngc_api_key="fixture-key",
            s3_access_key_id="fixture-access",
            s3_secret_access_key="fixture-secret",
        )
    )
    monkeypatch.setattr(health, "load_credentials", loaded)
    storage = SimpleNamespace(list_checkpoints=Mock(return_value=[]))
    storage_factory = Mock(return_value=storage)
    monkeypatch.setattr(health.StorageClient, "from_environment", storage_factory)
    kube = Mock(side_effect=_cluster_probe)
    kube_factory = Mock(return_value=kube)
    monkeypatch.setattr(health, "_kube_runner_factory", kube_factory)
    writes = Mock(side_effect=AssertionError("credentials were changed"))
    monkeypatch.setattr(credentials, "write_credentials_file", writes)
    monkeypatch.setattr(credentials, "persist_supported_env_credentials", writes)
    return SimpleNamespace(
        defaults=forbidden,
        execution=execution,
        image_inspector=image_inspector,
        loaded=loaded,
        storage=storage,
        storage_factory=storage_factory,
        kube=kube,
        kube_factory=kube_factory,
        writes=writes,
    )


@pytest.mark.parametrize(
    ("selected", "name"),
    [
        ("coherence", "compositional-workflow-coherence"),
        ("tokens", "tokens"),
        ("s3", "s3"),
        ("cluster", "cluster"),
    ],
)
def test_image_independent_checks_do_not_construct_execution_images(
    diagnostic_spies: SimpleNamespace,
    selected: str,
    name: str,
    tmp_path: Path,
) -> None:
    kubeconfig = str(tmp_path / "kubeconfig.yaml")
    result = runner.invoke(
        app,
        [
            "workbench",
            "health",
            "sim2real",
            "--checks",
            selected,
            "--s3-bucket",
            "example-bucket",
            "--s3-endpoint",
            "https://s3.example.invalid",
            "--k8s-context",
            "diagnostic-context",
            "--k8s-kubeconfig",
            kubeconfig,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [(row["name"], row["status"]) for row in payload["checks"]] == [
        (name, "PASS")
    ]
    assert payload["selected_checks"] == [selected]
    assert payload["image_policy_evaluated"] is False
    diagnostic_spies.defaults.assert_not_called()
    diagnostic_spies.execution.assert_not_called()
    diagnostic_spies.image_inspector.assert_not_called()
    diagnostic_spies.writes.assert_not_called()
    if selected != "s3":
        diagnostic_spies.storage_factory.assert_not_called()
    if selected != "cluster":
        diagnostic_spies.kube.assert_not_called()
    diagnostic_spies.kube_factory.assert_called_once_with(
        "diagnostic-context", kubeconfig
    )


def test_tokens_and_s3_use_explicit_storage_without_image_resolution(
    diagnostic_spies: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_BUCKET", "environment-bucket")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://environment.example.invalid")
    result = runner.invoke(
        app,
        [
            "workbench",
            "health",
            "sim2real",
            "--checks",
            "tokens,s3",
            "--run-id",
            "diagnostic-run",
            "--s3-bucket",
            "explicit-bucket",
            "--s3-endpoint",
            "https://explicit.example.invalid",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_id"] == "diagnostic-run"
    assert [(row["name"], row["status"]) for row in payload["checks"]] == [
        ("s3", "PASS"),
        ("tokens", "PASS"),
    ]
    diagnostic_spies.storage_factory.assert_called_once_with(
        endpoint_url="https://explicit.example.invalid"
    )
    diagnostic_spies.storage.list_checkpoints.assert_called_once_with(
        "s3://explicit-bucket/"
    )
    diagnostic_spies.defaults.assert_not_called()
    diagnostic_spies.image_inspector.assert_not_called()
    diagnostic_spies.writes.assert_not_called()


@pytest.mark.parametrize(
    "selected",
    [*IMAGE_DEPENDENT_CHECKS, "all", f"{IMAGE_DEPENDENT_CHECKS[0]},coherence"],
)
def test_execution_image_checks_keep_quarantine_as_a_cli_error(
    diagnostic_spies: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    selected: str,
) -> None:
    monkeypatch.setattr(
        health, "build_config_from_env", execution_config.build_config_from_env
    )
    monkeypatch.setattr(models, "container_image_for_tool", container_image_for_tool)
    result = runner.invoke(
        app,
        [
            "workbench",
            "health",
            "sim2real",
            "--checks",
            selected,
            "--s3-bucket",
            "example-bucket",
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert "no consumable public release" in result.output
    assert "operator-controlled registry/image" in result.output
    assert "ValueError" not in result.output
    diagnostic_spies.loaded.assert_not_called()
    diagnostic_spies.storage_factory.assert_not_called()
    diagnostic_spies.image_inspector.assert_not_called()
    diagnostic_spies.writes.assert_not_called()


def test_explicit_images_remain_selected_for_registry_check(
    diagnostic_spies: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = Mock(wraps=execution_config.build_config_from_env)
    monkeypatch.setattr(health, "build_config_from_env", builder)
    for field in _IMAGE_FIELDS:
        monkeypatch.setenv(field.upper(), f"registry.example.invalid/{field}:accepted")
    diagnostic_spies.image_inspector.side_effect = None
    diagnostic_spies.image_inspector.return_value = True
    result = runner.invoke(
        app, ["workbench", "health", "sim2real", "--checks", "registry", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["selected_checks"] == ["registry"]
    assert payload["image_policy_evaluated"] is True
    builder.assert_called_once()
    assert {
        call.args[0] for call in diagnostic_spies.image_inspector.call_args_list
    } == {f"registry.example.invalid/{field}:accepted" for field in IMAGE_FIELDS}
    diagnostic_spies.defaults.assert_not_called()


def test_preflight_config_uses_the_canonical_image_dependent_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = object()
    diagnostic = object()
    monkeypatch.setattr(health, "IMAGE_DEPENDENT_CHECKS", ("tokens",))
    monkeypatch.setattr(health, "build_config_from_env", lambda **_kwargs: execution)
    monkeypatch.setattr(
        health, "build_diagnostic_config_from_env", lambda **_kwargs: diagnostic
    )
    overrides = {
        "run_id": "diagnostic-run",
        "s3_bucket": "example-bucket",
        "s3_endpoint": "https://storage.example.invalid",
        "k8s_namespace": "",
        "k8s_context": "",
        "k8s_kubeconfig": "",
    }
    assert health._sim2real_preflight_config(["tokens"], overrides) is execution


def test_unknown_checks_fail_before_any_config_or_credentials(
    diagnostic_spies: SimpleNamespace,
) -> None:
    result = runner.invoke(
        app, ["workbench", "health", "sim2real", "--checks", "unknown"]
    )
    assert result.exit_code == 2
    assert "unknown check" in result.output
    diagnostic_spies.execution.assert_not_called()
    diagnostic_spies.defaults.assert_not_called()
    diagnostic_spies.loaded.assert_not_called()


def test_empty_checks_fail_before_any_config_or_credentials(
    diagnostic_spies: SimpleNamespace,
) -> None:
    result = runner.invoke(app, ["workbench", "health", "sim2real", "--checks", ""])
    assert result.exit_code == 2
    assert "at least one check is required" in result.output
    diagnostic_spies.execution.assert_not_called()
    diagnostic_spies.defaults.assert_not_called()
    diagnostic_spies.loaded.assert_not_called()


@pytest.mark.parametrize("warn_only", [False, True])
def test_s3_failure_preserves_report_and_exit_contract(
    diagnostic_spies: SimpleNamespace,
    warn_only: bool,
) -> None:
    diagnostic_spies.storage.list_checkpoints.side_effect = RuntimeError(
        "fixture read denied"
    )
    args = [
        "workbench",
        "health",
        "sim2real",
        "--checks",
        "s3",
        "--s3-bucket",
        "example-bucket",
        "--s3-endpoint",
        "https://s3.example.invalid",
        "--json",
    ]
    if warn_only:
        args.append("--warn-only")
    result = runner.invoke(app, args)
    assert result.exit_code == (0 if warn_only else 1)
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["checks"][0]["status"] == "FAIL"
    diagnostic_spies.execution.assert_not_called()
    diagnostic_spies.writes.assert_not_called()


def _diagnostic_environment(monkeypatch: pytest.MonkeyPatch, level: int) -> None:
    groups = (
        ("NPA_SIM2REAL_BUCKET", "NPA_S3_BUCKET", "S3_BUCKET"),
        ("AWS_ENDPOINT_URL", "S3_ENDPOINT_URL"),
        ("KUBECONFIG", "NPA_SIM2REAL_KUBECONFIG"),
        ("NPA_SIM2REAL_K8S_NAMESPACE",),
    )
    for group in groups:
        for index, name in enumerate(group):
            if index < level:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, f"configured-{name.lower()}")
    monkeypatch.setenv("NPA_SIM2REAL_K8S_CONTEXT", "configured-context")
    monkeypatch.setenv("NPA_SIM2REAL_ISAAC_CACHE_PVC", "configured-cache")
    monkeypatch.setenv("NPA_SIM2REAL_K8S_GPU_RESOURCE", "example.invalid/gpu")
    monkeypatch.setenv("NPA_SIM2REAL_K8S_GPU_PRODUCT", "configured-product")
    monkeypatch.setenv(
        "NPA_SIM2REAL_K8S_GPU_CANDIDATES", "fallback-a;fallback-b, fallback-c"
    )
    monkeypatch.setattr(
        diagnostic_config, "_serviceaccount_namespace", lambda: "pod-namespace"
    )
    monkeypatch.setattr(
        execution_config, "_serviceaccount_namespace", lambda: "pod-namespace"
    )


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("explicit", [False, True])
def test_diagnostic_settings_preserve_execution_environment_precedence(
    diagnostic_spies: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    level: int,
    explicit: bool,
    tmp_path: Path,
) -> None:
    _diagnostic_environment(monkeypatch, level)
    settings = {"run_id": "diagnostic-run"}
    if explicit:
        settings.update(
            s3_bucket="explicit-bucket",
            s3_endpoint="https://explicit.example.invalid",
            k8s_namespace="explicit-namespace",
            k8s_context="explicit-context",
            k8s_kubeconfig=str(tmp_path / "explicit.yaml"),
        )
    context = diagnostic_config.build_diagnostic_config_from_env(**settings)
    config = execution_config.build_config_from_env(
        **settings,
        **{
            field: f"registry.example.invalid/{field}:accepted"
            for field in _IMAGE_FIELDS
        },
    )
    for field in dataclasses.fields(context):
        assert getattr(context, field.name) == getattr(config, field.name)
        assert "image" not in field.name
    assert context.k8s_gpu_candidates == ("fallback-a", "fallback-b", "fallback-c")
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.s3_bucket = "changed-bucket"
    diagnostic_spies.defaults.assert_not_called()


def test_diagnostic_default_kubernetes_settings_match_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "NPA_SIM2REAL_ISAAC_CACHE_PVC",
        "NPA_SIM2REAL_K8S_GPU_RESOURCE",
        "NPA_SIM2REAL_K8S_GPU_PRODUCT",
        "NPA_SIM2REAL_K8S_GPU_CANDIDATES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NPA_SIM2REAL_REGISTRY", "registry.example.invalid/operator")
    monkeypatch.setattr(
        diagnostic_config, "_serviceaccount_namespace", lambda: "pod-namespace"
    )
    monkeypatch.setattr(
        execution_config, "_serviceaccount_namespace", lambda: "pod-namespace"
    )
    settings = {
        "run_id": "default-settings-run",
        "s3_bucket": "example-bucket",
        "s3_endpoint": "https://storage.example.invalid",
    }
    context = diagnostic_config.build_diagnostic_config_from_env(**settings)
    config = execution_config.build_config_from_env(
        **settings,
        **{
            field: f"registry.example.invalid/{field}:accepted"
            for field in _IMAGE_FIELDS
        },
    )
    for field in (
        "k8s_isaac_cache_pvc",
        "k8s_gpu_resource",
        "k8s_gpu_product",
        "k8s_gpu_candidates",
    ):
        assert getattr(context, field) == getattr(config, field)


def test_empty_run_id_uses_existing_environment_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_RUN_ID", "environment-run")
    assert (
        diagnostic_config.build_diagnostic_config_from_env(run_id="").run_id
        == "environment-run"
    )


@pytest.mark.parametrize(
    "checks", [*((name,) for name in IMAGE_DEPENDENT_CHECKS), None]
)
def test_image_free_context_cannot_satisfy_execution_image_checks(checks) -> None:
    context = diagnostic_config.build_diagnostic_config_from_env(
        run_id="diagnostic-run"
    )
    with pytest.raises(ValueError, match="require resolved execution images"):
        run_preflight(
            context, repo_root=Path.cwd(), probes=DoctorProbes(), checks=checks
        )
