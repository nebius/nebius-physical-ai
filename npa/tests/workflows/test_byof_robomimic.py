from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow import build_plan, load_spec


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows" / "testing" / "byof-robomimic.yaml"
READINESS = ROOT / "workflows" / "testing" / "byof-robomimic.readiness.json"
DOC = ROOT / "docs" / "workbench" / "byof-robomimic.md"
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
BASE_DIGEST = "sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2"
BUILD_COMMAND_SHA256 = "40934ad75e79127e2b494adf73c59e0cd2164dda71dac2a142891cdda7a43bf6"
DEPENDENCY_LOCK_SHA256 = "910b762eb9fa6bb31bfb05d339845d8c68c81f0b3e85ab95ec3eefbca29cad71"


def _workflow_config() -> dict[str, object]:
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    config = payload["config"]
    assert isinstance(config, dict)
    return config


def _smoke_python() -> str:
    smoke = str(_workflow_config()["smoke_command"])
    return smoke.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def _live_e2e_module():
    """Import the live module so its pure contracts run in the default suite."""

    tests_root = str(ROOT / "npa")
    sys.path.insert(0, tests_root)
    try:
        return importlib.import_module(
            "tests.e2e.test_byof_onboarding_live_e2e"
        )
    finally:
        sys.path.remove(tests_root)


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
    }
    for variable, value in selectors.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.delenv(missing_variable, raising=False)

    with pytest.raises(AssertionError):
        _live_e2e_module()._robomimic_live_selectors("manager-project")


def test_robomimic_strict_attestation_is_resolved_only_in_run_local_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NPA_E2E_MK8S_RESERVED_CAPACITY", "1")
    module = _live_e2e_module()
    rendered = module._materialize_robomimic_attested_profile(
        tmp_path / "robomimic-attested.yaml",
        service_account="npa-robomimic-run-scoped",
    )

    source_task = list(yaml.safe_load_all(PROFILE.read_text(encoding="utf-8")))[1]
    rendered_task = list(yaml.safe_load_all(rendered.read_text(encoding="utf-8")))[1]
    assert source_task["envs"]["NPA_ROBOMIMIC_STRICT_B200_ATTESTED"] == ""
    assert rendered_task["envs"]["NPA_ROBOMIMIC_STRICT_B200_ATTESTED"] == "1"
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
    )

    assert len(manifests) == 3
    assert {manifest["kind"] for manifest in manifests} == {
        "ServiceAccount",
        "Role",
        "RoleBinding",
    }
    assert all(manifest["metadata"]["name"] == name for manifest in manifests)
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


def test_robomimic_observer_cleanup_runs_after_gate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _live_e2e_module()
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "auth" in command:
            verb = command[command.index("can-i") + 1]
            resource = command[command.index("can-i") + 2]
            allowed = (verb, resource) == ("get", "pods")
            return subprocess.CompletedProcess(
                command, 0 if allowed else 1, "yes\n" if allowed else "no\n", ""
            )
        if "get" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

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
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "auth" in command:
            verb = command[command.index("can-i") + 1]
            resource = command[command.index("can-i") + 2]
            allowed = (verb, resource) == ("get", "pods")
            return subprocess.CompletedProcess(
                command, 0 if allowed else 1, "yes\n" if allowed else "no\n", ""
            )
        if "delete" in command and "rolebinding/" in " ".join(command):
            return subprocess.CompletedProcess(command, 1, "", "simulated delete error")
        if "get" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

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
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "get" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "create" in command:
            return subprocess.CompletedProcess(command, 1, "", "response lost")
        return subprocess.CompletedProcess(command, 0, "", "")

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
    assert "serviceaccount/" in " ".join(deleted[0])


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


@pytest.mark.parametrize(
    ("registry", "manager_registry", "visibility", "image", "output_root", "expected_error"),
    (
        (
            "private.invalid/robomimic",
            "private.invalid/robomimic",
            "private",
            "quay.io/example/robomimic:latest",
            "s3://manager-bucket/oss-solutions/robomimic",
            "image does not target the manager-issued private registry",
        ),
        (
            "quay.io:443/example",
            "quay.io:443/example",
            "private",
            "",
            "s3://manager-bucket/oss-solutions/robomimic",
            "requires an operator-private registry",
        ),
        (
            "private.invalid/robomimic",
            "private.invalid/robomimic",
            "private",
            "",
            "s3://other-bucket/oss-solutions/robomimic",
            "manager-issued bucket and solution prefix",
        ),
        (
            "private.invalid/actual",
            "private.invalid/manager",
            "private",
            "",
            "s3://manager-bucket/oss-solutions/robomimic",
            "registry does not match the manager-issued registry",
        ),
        (
            "private.invalid/robomimic",
            "private.invalid/robomimic",
            "public",
            "",
            "s3://manager-bucket/oss-solutions/robomimic",
            "registry visibility must be private",
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
    }
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
        "--output-root",
        output_root,
        "--skip-build",
        "--skip-push",
        "--skip-run",
    ]
    if image:
        command.extend(["--image", image])
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
        assert "before any build, runtime pull, or GPU submission" in result.output


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
    assert config["base_image"].endswith(f"@{BASE_DIGEST}")
    assert config["resource_profile_yaml"] == "byof-solution-smoke-robomimic-b200-gpu"
    assert config["capability_name"] == "lift_ph_lowdim_checkpoint_reload_action"
    assert config["smoke_artifact_name"] == "robomimic-smoke.json"
    assert config["wait_timeout"] == -1
    assert "low_dim_v15.hdf5" not in build
    assert DATASET_REVISION not in build
    assert hashlib.sha256(build.encode()).hexdigest() == BUILD_COMMAND_SHA256
    assert "--only-binary=:all: --no-deps --require-hashes" in build
    assert "--no-deps --no-build-isolation -e ." in build
    assert "boto3==1.38.33 --hash=sha256:" in build
    assert "torch.__version__.split('+')[0] == '2.7.1'" in build
    lock_lines = [
        line[3:-3]
        for line in build.splitlines()
        if line.startswith("  '") and line.endswith("' \\")
    ]
    assert len(lock_lines) == 48
    lock_bytes = ("\n".join(lock_lines) + "\n").encode()
    assert hashlib.sha256(lock_bytes).hexdigest() == DEPENDENCY_LOCK_SHA256

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
        "raw_dataset.replace(dataset)",
        'config.train.hdf5_filter_key = "train"',
        'config.train.hdf5_validation_filter_key = "valid"',
        '["git", "-C", str(repo_root), "rev-parse", "HEAD"]',
        "source_head != SOURCE_REVISION",
        f'EXPECTED_BUILD_COMMAND_SHA256 = "{BUILD_COMMAND_SHA256}"',
        "build_metadata.get(\"build_command_sha256\") != EXPECTED_BUILD_COMMAND_SHA256",
        f'DEPENDENCY_LOCK_SHA256 = "{DEPENDENCY_LOCK_SHA256}"',
        "dependency_lock_hash != DEPENDENCY_LOCK_SHA256",
        "dependency_artifact_count != DEPENDENCY_ARTIFACT_COUNT",
        "policy_from_checkpoint(",
        "optimizer_step_count != TRAIN_STEPS",
        "action.shape != (7,)",
        '"B200" not in gpu_name.upper()',
        'architecture != "sm_100"',
        'os.environ.get("NPA_ROBOMIMIC_STRICT_B200_ATTESTED") != "1"',
        '"architecture": architecture',
        '"strict_reserved_capacity_attested": True',
        '"observed_head": source_head',
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
        '"image_id": pod_image_id',
        '"observation_source": "Kubernetes Pod status.containerStatuses[].imageID"',
        '"pod_image"',
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
    assert "${NPA_E2E_MK8S_RESERVED_CAPACITY}" not in PROFILE.read_text(
        encoding="utf-8"
    )
    assert "pip install --quiet boto3" not in PROFILE.read_text(encoding="utf-8")
    assert "runtime input must not be uploaded as output" in PROFILE.read_text(
        encoding="utf-8"
    )
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
