"""Tests for canonical Sim2Real runbook direct-Kubernetes submission."""

from __future__ import annotations

import hashlib
import stat
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import pytest
import yaml


def _operator():
    from npa.workflows.sim2real.monitor import OperatorConfig

    return OperatorConfig(
        bucket="unit-bucket",
        endpoint_url="https://storage.example.invalid",
        registry="registry.unit.test/team",
        k8s_context="unit-context",
    )


def _patch_operator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from npa.workflows.sim2real import k8s_submit

    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    monkeypatch.setattr(k8s_submit, "load_operator_config", _operator)
    monkeypatch.setattr(k8s_submit, "resolve_kubeconfig", lambda _context: kubeconfig)
    real_defaults = k8s_submit._default_image_env

    def immutable_defaults(registry: str, *, orchestrator_image: str = ""):
        values, _ = real_defaults(registry)
        digest = "a" * 64
        for key in k8s_submit._required_real_image_envs(values):
            values[key] = f"{registry}/{key.lower().replace('_', '-')}@sha256:{digest}"
        orchestrator = orchestrator_image or f"{registry}/orchestrator@sha256:{digest}"
        return values, orchestrator

    monkeypatch.setattr(k8s_submit, "_default_image_env", immutable_defaults)
    archived = tmp_path / "retired-controller.yaml"
    controller = {
        "apiVersion": "npa.sim2real/v1alpha1",
        "kind": "Sim2RealController",
        "metadata": {"name": "retired-test-controller"},
        "spec": {
            "stageCount": 14,
            "execution": "durable-kubernetes-controller",
            "stateStore": "content-addressed-s3-journal",
            "kubernetesClient": "official-python-client",
            "scheduler": "kueue",
        },
    }
    task = {
        "name": "retired-sim2real-test-task",
        "resources": {
            "cloud": "kubernetes",
            "cpus": "16+",
            "memory": "64+",
            "image_id": "docker:example.invalid/retired:latest",
        },
        "envs": {
            "INNER_ITERATIONS": "3",
            "OUTER_ITERATIONS": "3",
            "LOOP_OF_LOOPS_ITERATIONS": "1",
            "ROLLOUT_COUNT": "1",
            "STEPS_PER_ROLLOUT": "32",
            "VALIDATION_ENV_COUNT": "1",
            "HELDOUT_ENV_COUNT": "1",
            "NPA_ENV_COUNT": "12",
            "NPA_SIM2REAL_HELDOUT_EVAL_LIMIT": "0",
            "NPA_SIM2REAL_K8S_JOB_TIMEOUT_S": "0",
            "NPA_BYO_ISAAC_JOB_TIMEOUT_S": "0",
            "NPA_BYO_ISAAC_VALIDATION_INTERVAL": "1",
            "NPA_SIM2REAL_ROLLOUT_HORIZON_STEPS": "32",
            "NPA_COSMOS_REASON_MAX_FRAMES": "32",
            "NPA_COSMOS_REASON_MAX_NEW_TOKENS": "2048",
            "SUCCESS_THRESHOLD": "0.50",
            "NPA_BYO_ISAAC_SUCCESS_DIST_M": "0.05",
            "NPA_SIM2REAL_EARLY_EXIT": "0",
            "NPA_SIM2REAL_RERUN": "1",
            "NPA_SIM2REAL_RERUN_SERVE": "1",
            "BYO_TRAINER_COMMAND": "python3 -m npa.workflows.sim2real.byo_isaac_trainer",
            "BYO_POLICY_COMMAND": "python3 -m npa.workflows.sim2real.byo_isaac_policy_rollout",
            "BYO_EVAL_COMMAND": "python3 -m npa.workflows.sim2real.byo_isaac_eval",
            "NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS": "1",
            "NPA_SIM2REAL_ISAAC_CACHE_PVC": "isaac-cache",
            "NPA_SIM2REAL_K8S_REGISTRY_CONFIG_SECRET": "npa-nebius-registry",
            "NPA_SIM2REAL_K8S_IMAGE_PULL_SECRETS": "npa-nebius-registry",
            "NPA_SIM2REAL_K8S_SERVICE_ACCOUNT": "agent-sa",
            "NPA_SIM2REAL_CONTROLLER_RETRIES": "3",
        },
        "setup": "python3 -m npa.workflows.sim2real.runtime_attestation",
        "run": "python3 -m npa.workflows.sim2real run --upload-artifacts",
    }
    archived.write_text(yaml.safe_dump_all([controller, task], sort_keys=False))
    monkeypatch.setattr(k8s_submit, "default_runbook_path", lambda: archived)


def _capture_ephemeral_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, str]:
    from npa.workflows.sim2real import k8s_submit

    captured: dict[str, str] = {}
    real = k8s_submit._secure_temporary_manifest

    @contextmanager
    def capture(**kwargs):
        captured.update({key: str(value) for key, value in kwargs.items()})
        with real(**kwargs) as path:
            captured["path"] = str(path)
            yield path

    monkeypatch.setattr(k8s_submit, "_secure_temporary_manifest", capture)
    return captured


def test_operator_config_prefers_canonical_registry_over_legacy_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workflows.sim2real.monitor import load_operator_config

    config_dir = tmp_path / ".npa"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        """container_registry: registry.canonical.test/team
storage:
  bucket: unit-bucket
  endpoint_url: https://storage.unit.test
  registry: registry.legacy.test/team
  k8s_context: unit-context
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NPA_REGISTRY", raising=False)
    monkeypatch.delenv("NPA_REGISTRY_ID", raising=False)

    assert load_operator_config().registry == "registry.canonical.test/team"


def test_operator_config_explicit_path_isolated_from_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workflows.sim2real.monitor import load_operator_config

    home = tmp_path / "home"
    (home / ".npa").mkdir(parents=True)
    (home / ".npa" / "config.yaml").write_text(
        """container_registry: registry.wrong.test/team
storage:
  bucket: wrong-bucket
  endpoint_url: https://storage.wrong.test
  k8s_context: wrong-context
""",
        encoding="utf-8",
    )
    isolated = tmp_path / "launcher" / "npa" / "config.yaml"
    isolated.parent.mkdir(parents=True)
    isolated.write_text(
        """container_registry: registry.isolated.test/team
storage:
  bucket: isolated-bucket
  endpoint_url: https://storage.isolated.test
  k8s_context: isolated-context
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NPA_SIM2REAL_OPERATOR_CONFIG", str(isolated))
    monkeypatch.delenv("NPA_REGISTRY", raising=False)
    monkeypatch.delenv("NPA_REGISTRY_ID", raising=False)

    config = load_operator_config()

    assert config.bucket == "isolated-bucket"
    assert config.endpoint_url == "https://storage.isolated.test"
    assert config.registry == "registry.isolated.test/team"
    assert config.k8s_context == "isolated-context"


def test_operator_config_rejects_relative_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workflows.sim2real.monitor import load_operator_config

    monkeypatch.setenv("NPA_SIM2REAL_OPERATOR_CONFIG", "relative/config.yaml")

    with pytest.raises(ValueError, match="must be an absolute path"):
        load_operator_config()


def test_canonical_sim2real_uses_standard_workflow_detection() -> None:
    from npa.orchestration.npa_workflow.detect import detect_submit_format

    root = Path(__file__).resolve().parents[2]
    canonical = root / "workflows" / "workbench" / "npa-workflows" / "sim2real.yaml"
    assert detect_submit_format(canonical) == "npa.workflow"
    assert not (root / "workflows" / "sim2real.yaml").exists()


def test_plan_only_materializes_qualified_real_images_without_external_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workflows.sim2real import k8s_submit

    _patch_operator(monkeypatch, tmp_path)
    captured = _capture_ephemeral_manifest(monkeypatch)

    result = k8s_submit.submit_sim2real_staged_job(
        run_id="unit-plan",
        trigger_dataset_uri="s3://unit-bucket/trigger/",
        env_overrides={
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ISAACSIM_ACCEPT_EULA": "YES",
        },
        plan_only=True,
    )

    assert result.status == "planned"
    manifest = yaml.safe_load(captured["manifest_yaml"])
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert manifest["metadata"]["name"] == "sim2real-unit-plan"
    assert container["image"].startswith("registry.unit.test/team/")
    assert env["NPA_SIM2REAL_SOURCE_SHA"]
    assert env["NPA_SIM2REAL_RUNTIME_IMAGE"].endswith("a" * 64)
    assert "NPA_SIM2REAL_SOURCE_TARBALL_URI" not in env
    for key in k8s_submit._required_real_image_envs(env):
        assert k8s_submit._registry_qualified(env[key]), (key, env[key])
    assert "manifest_path" not in {field.name for field in fields(result)}
    assert (
        result.manifest_sha256
        == hashlib.sha256(captured["manifest_yaml"].encode("utf-8")).hexdigest()
    )
    assert not Path(captured["path"]).exists()


def test_submit_refreshes_all_images_and_creates_typed_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workflows.sim2real import k8s_submit, registry_auth

    _patch_operator(monkeypatch, tmp_path)
    refreshed: list[tuple[str, ...]] = []
    applied: list[dict[str, object]] = []
    monkeypatch.setattr(
        registry_auth,
        "ensure_registry_pull_secret_for_images",
        lambda *images, **_kwargs: refreshed.append(tuple(images)),
    )

    from npa.workflows.sim2real import k8s_client

    class FakeClient:
        @classmethod
        def from_environment(cls, **_kwargs):
            return cls()

        def create_or_adopt(self, manifest, **_identity):
            applied.append(manifest)
            return "job-uid", False

    monkeypatch.setattr(k8s_client, "KubernetesJobClient", FakeClient)

    result = k8s_submit.submit_sim2real_staged_job(
        run_id="unit-submit",
        trigger_dataset_uri="s3://unit-bucket/trigger/",
        env_overrides={
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ISAACSIM_ACCEPT_EULA": "YES",
            "BYO_TRAINER_COMMAND": "python3 -m npa.workflows.sim2real.byo_isaac_trainer",
            "BYO_POLICY_COMMAND": "python3 -m npa.workflows.sim2real.byo_isaac_policy_rollout",
            "BYO_EVAL_COMMAND": "python3 -m npa.workflows.sim2real.byo_isaac_eval",
        },
    )

    assert result.status == "submitted"
    assert len(refreshed) == 1
    expected_images, _ = k8s_submit._default_image_env("registry.unit.test/team")
    required_names = k8s_submit._required_real_image_envs(expected_images)
    assert set(refreshed[0][1:]) == {expected_images[name] for name in required_names}
    assert len(applied) == 1
    assert applied[0]["metadata"]["name"] == result.job_name
    assert len(result.manifest_sha256) == 64


def test_unqualified_real_component_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workflows.sim2real import k8s_submit

    _patch_operator(monkeypatch, tmp_path)
    with pytest.raises(
        ValueError, match="AUGMENT_IMAGE must be a registry-qualified image"
    ):
        k8s_submit.submit_sim2real_staged_job(
            run_id="unit-bad-image",
            env_overrides={
                "AUGMENT_IMAGE": "npa-cosmos2-transfer:latest",
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "ISAACSIM_ACCEPT_EULA": "YES",
            },
            plan_only=True,
        )


def test_real_submit_requires_mounted_registry_config_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workflows.sim2real import k8s_submit

    _patch_operator(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="mounted registry config Secret"):
        k8s_submit.submit_sim2real_staged_job(
            run_id="unit-missing-registry-config",
            env_overrides={
                "NPA_SIM2REAL_K8S_REGISTRY_CONFIG_SECRET": "",
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "ISAACSIM_ACCEPT_EULA": "YES",
            },
            plan_only=True,
        )


@pytest.mark.parametrize(
    "environment_variable",
    ["VLM_REASON2_IMAGE", "VLM_REASON3_IMAGE", "NPA_RERUN_VIEWER_IMAGE"],
)
def test_new_real_path_image_guards_reject_unqualified_or_placeholder_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    environment_variable: str,
) -> None:
    from npa.workflows.sim2real import k8s_submit

    _patch_operator(monkeypatch, tmp_path)
    bad_image = (
        "example.invalid/team/viewer:latest"
        if environment_variable == "NPA_RERUN_VIEWER_IMAGE"
        else "unqualified-image:latest"
    )
    with pytest.raises(ValueError, match=f"{environment_variable} must be"):
        k8s_submit.submit_sim2real_staged_job(
            run_id=f"unit-bad-{environment_variable.lower()}",
            env_overrides={
                environment_variable: bad_image,
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "ISAACSIM_ACCEPT_EULA": "YES",
            },
            plan_only=True,
        )


def test_image_aliases_resolve_before_guarding_and_materialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.workflows.sim2real import k8s_submit

    _patch_operator(monkeypatch, tmp_path)
    captured = _capture_ephemeral_manifest(monkeypatch)
    vlm = f"registry.valid.test/team/custom-reason@sha256:{'b' * 64}"
    viewer = f"registry.valid.test/team/custom-viewer@sha256:{'c' * 64}"

    k8s_submit.submit_sim2real_staged_job(
        run_id="unit-image-aliases",
        env_overrides={
            "VLM_IMAGE": vlm,
            "NPA_SIM2REAL_RERUN_IMAGE": viewer,
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ISAACSIM_ACCEPT_EULA": "YES",
        },
        plan_only=True,
    )

    manifest = yaml.safe_load(captured["manifest_yaml"])
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["VLM_IMAGE"] == vlm
    assert env["VLM_REASON2_IMAGE"] == vlm
    assert env["VLM_REASON3_IMAGE"] == vlm
    assert env["NPA_RERUN_VIEWER_IMAGE"] == viewer


def test_disabled_visualization_does_not_require_viewer_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.workflows.sim2real import k8s_submit

    _patch_operator(monkeypatch, tmp_path)
    result = k8s_submit.submit_sim2real_staged_job(
        run_id="unit-no-viewer",
        env_overrides={
            "NPA_SIM2REAL_RERUN": "0",
            "NPA_RERUN_VIEWER_IMAGE": "not-qualified:latest",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ISAACSIM_ACCEPT_EULA": "YES",
        },
        plan_only=True,
    )
    assert result.status == "planned"


@pytest.mark.parametrize("name", ["NPA_SIM2REAL_RERUN", "NPA_SIM2REAL_RERUN_SERVE"])
def test_visualization_toggle_must_be_boolean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    from npa.workflows.sim2real import k8s_submit

    _patch_operator(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=rf"{name} must be boolean"):
        k8s_submit.submit_sim2real_staged_job(
            run_id="unit-bad-viewer-toggle",
            env_overrides={
                name: "maybe",
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "ISAACSIM_ACCEPT_EULA": "YES",
            },
            plan_only=True,
        )


def test_secure_manifest_is_unique_restrictive_and_cleaned(tmp_path: Path) -> None:
    del tmp_path
    from npa.workflows.sim2real.k8s_submit import _secure_temporary_manifest

    paths: list[Path] = []
    parents: list[Path] = []
    for _ in range(2):
        with _secure_temporary_manifest(
            run_id="unit-secure",
            job_name="sim2real-unit-secure",
            manifest_yaml="apiVersion: batch/v1\n",
        ) as path:
            paths.append(path)
            parents.append(path.parent)
            assert path.is_file()
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert not path.exists()
        assert not path.parent.exists()
    assert paths[0] != paths[1]
    assert parents[0] != parents[1]


def test_secure_manifest_cleanup_preserves_body_error() -> None:
    from npa.workflows.sim2real.k8s_submit import _secure_temporary_manifest

    manifest_path: Path | None = None
    with pytest.raises(RuntimeError, match="body failed"):
        with _secure_temporary_manifest(
            run_id="unit-error",
            job_name="sim2real-unit-error",
            manifest_yaml="kind: Job\n",
        ) as path:
            manifest_path = path
            raise RuntimeError("body failed")
    assert manifest_path is not None
    assert not manifest_path.exists()
    assert not manifest_path.parent.exists()


def test_runtime_is_immutable_and_has_no_source_staging_surface() -> None:
    from npa.workflows.sim2real import k8s_submit

    assert not hasattr(k8s_submit, "_stage_orchestrator_source")
    source = Path(k8s_submit.__file__).read_text(encoding="utf-8")
    assert "SOURCE_TARBALL" not in source
    assert "tarfile" not in source


def test_workflow_var_aliases_route_to_runbook_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.workflows.sim2real import k8s_submit

    captured: dict[str, object] = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)
        return k8s_submit.Sim2RealSubmitResult(
            run_id="unit-vars",
            job_name="sim2real-unit-vars",
            k8s_context="unit-context",
            run_prefix_uri="s3://unit-bucket/custom/unit-vars/",
        )

    monkeypatch.setattr(k8s_submit, "submit_sim2real_staged_job", fake_submit)
    k8s_submit.submit_sim2real_from_workflow_vars(
        run_id="unit-vars",
        substitutions={
            "bucket": "unit-bucket",
            "prefix": "custom",
            "trigger_uri": "s3://unit-bucket/trigger/",
            "INNER_ITERATIONS": "3",
            "OUTER_ITERATIONS": "2",
        },
        plan_only=True,
    )

    assert captured["s3_bucket"] == "unit-bucket"
    assert captured["s3_prefix"] == "custom"
    assert captured["trigger_dataset_uri"] == "s3://unit-bucket/trigger/"
    assert captured["inner_iterations"] == 3
    assert captured["outer_iterations"] == 2
    assert captured["plan_only"] is True
