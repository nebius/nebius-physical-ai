"""Unit tests for npa.workflow → SkyPilot rendering and submit detection."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow.detect import (
    detect_submit_format,
    is_npa_workflow_spec,
)
from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    NpaWorkflowRenderError,
    SkypilotRenderOptions,
    assert_no_unresolved_placeholders,
    normalize_resources,
    render_skypilot_yaml,
    resolve_task_image,
    tool_image_key,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit
from npa.orchestration.skypilot.workflow import WorkflowResult

REPO_ROOT = Path(__file__).resolve().parents[4]
NPA_SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
SKYPILOT_SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "skypilot"
RUNNER = CliRunner()


def test_is_npa_workflow_spec_true_for_golden() -> None:
    path = NPA_SPECS / "vlm-eval-single.yaml"
    assert is_npa_workflow_spec(path)
    assert detect_submit_format(path) == "npa.workflow"


def test_is_npa_workflow_spec_false_for_skypilot() -> None:
    path = SKYPILOT_SPECS / "vlm-eval.yaml"
    assert not is_npa_workflow_spec(path)
    assert detect_submit_format(path) == "skypilot"


def test_normalize_resources_strips_gi_suffix() -> None:
    assert normalize_resources({"memory": "80Gi", "cpus": 16, "cloud": "k8s"}) == {
        "cloud": "k8s",
        "cpus": "16+",
        "memory": "80+",
    }


def test_normalize_resources_leaves_exact_nebius_shapes() -> None:
    assert normalize_resources({"memory": "16Gi", "cpus": 4, "cloud": "nebius"}) == {
        "cloud": "nebius",
        "cpus": 4,
        "memory": "16",
    }


def test_nebius_cloud_render_injects_docker_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "test-token")
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "iam")
    # Use a GPU twin that resolves a Nebius registry image (Token Factory no
    # longer pins npa-cosmos — it is API-only and uses SkyPilot's default image).
    path = NPA_SPECS / "vlm-eval-single.yaml"
    spec = load_spec(path)
    for profile in spec.resources.values():
        if isinstance(profile, dict):
            profile["cloud"] = "nebius"
    plan = build_plan(spec, run_id="demo")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="demo",
        options=SkypilotRenderOptions(registry="cr.eu-north1.nebius.cloud/reg"),
    )
    docs = [doc for doc in yaml.safe_load_all(rendered) if doc is not None]
    task = docs[1]
    assert task["resources"]["cloud"] == "nebius"
    assert "image_id" in task["resources"]
    assert task["secrets"]["SKYPILOT_DOCKER_SERVER"] == "cr.eu-north1.nebius.cloud"
    assert task["secrets"]["SKYPILOT_DOCKER_USERNAME"] == "iam"
    assert task["secrets"]["SKYPILOT_DOCKER_PASSWORD"] == "test-token"


def test_tool_image_key_prefix_match() -> None:
    assert tool_image_key("workbench.vlm_eval.run") == "cosmos"
    assert tool_image_key("workbench.token_factory.caption") is None
    assert tool_image_key("workbench.lancedb.import_bdd100k") == "lancedb"
    assert tool_image_key("workbench.sonic.train") == "sonic"
    assert tool_image_key("unknown.tool") is None


def test_render_token_factory_uses_env_aws_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.us-central1.nebius.cloud")
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    prepared = prepare_npa_workflow_for_submit(
        NPA_SPECS / "token-factory-caption.yaml",
        run_id="caption-demo",
        render_options=SkypilotRenderOptions(registry="cr.example.invalid/reg"),
    )
    try:
        docs = [
            doc
            for doc in yaml.safe_load_all(
                prepared.skypilot_yaml_path.read_text(encoding="utf-8")
            )
            if doc is not None
        ]
        assert "image_id" not in docs[1]["resources"]
        assert docs[1]["envs"]["AWS_ENDPOINT_URL"] == (
            "https://storage.us-central1.nebius.cloud"
        )
        assert docs[1]["envs"]["NPA_SRC_S3_URI"] == "s3://example-bucket/npa-src/npa"
    finally:
        prepared.temp_dir.cleanup()


def test_render_vlm_eval_single_produces_serial_pipeline() -> None:
    spec = load_spec(NPA_SPECS / "vlm-eval-single.yaml")
    plan = build_plan(spec, run_id="demo")
    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="demo",
        options=SkypilotRenderOptions(registry="cr.example.invalid/reg"),
    )
    assert_no_unresolved_placeholders(text)
    docs = [doc for doc in yaml.safe_load_all(text) if doc is not None]
    assert docs[0]["name"] == "vlm-eval-single"
    assert docs[0]["execution"] == "serial"
    assert len(docs) == 2
    task = docs[1]
    assert task["name"] == "score-rollouts"
    assert task["resources"]["accelerators"] == "H100:1"
    assert task["resources"]["cpus"] == "16+"
    assert task["resources"]["memory"] == "80+"
    assert task["resources"]["image_id"].startswith("docker:cr.example.invalid/reg/")
    assert "npa workbench vlm-eval run" in task["run"]
    assert "set -euo pipefail" in task["run"]


def test_render_self_hosted_vlm_includes_vllm_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "test-token")
    spec = load_spec(NPA_SPECS / "vlm-eval-single.yaml")
    plan = build_plan(spec, run_id="demo")
    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="demo",
        options=SkypilotRenderOptions(registry="cr.example.invalid/reg"),
    )
    docs = [doc for doc in yaml.safe_load_all(text) if doc is not None]
    assert "vllm" in docs[1]["setup"]


def test_render_token_factory_caption_cpu_and_secret_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    prepared = prepare_npa_workflow_for_submit(
        NPA_SPECS / "token-factory-caption.yaml",
        run_id="caption-demo",
        render_options=SkypilotRenderOptions(registry="cr.example.invalid/reg"),
    )
    try:
        assert "NEBIUS_TOKEN_FACTORY_KEY" in prepared.secret_env_hints
        docs = [
            doc
            for doc in yaml.safe_load_all(
                prepared.skypilot_yaml_path.read_text(encoding="utf-8")
            )
            if doc is not None
        ]
        assert docs[0]["execution"] == "serial"
        assert "accelerators" not in docs[1]["resources"]
        # Token Factory uses the default SkyPilot image (no cosmos pin).
        assert "image_id" not in docs[1]["resources"]
        assert docs[1]["envs"]["NPA_SRC_S3_URI"] == "s3://example-bucket/npa-src/npa"
        assert "token-factory caption" in docs[1]["run"]
        assert "NEBIUS_TOKEN_FACTORY_KEY" in docs[1]["setup"]
    finally:
        prepared.temp_dir.cleanup()


def test_render_token_factory_requires_npa_src_s3_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_E2E_NPA_SRC_S3_URI", raising=False)
    with pytest.raises(NpaWorkflowRenderError, match="NPA_SRC_S3_URI is unset"):
        prepare_npa_workflow_for_submit(
            NPA_SPECS / "token-factory-caption.yaml",
            run_id="caption-demo",
            render_options=SkypilotRenderOptions(registry="cr.example.invalid/reg"),
        )


def test_render_token_factory_sets_npa_src_s3_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    prepared = prepare_npa_workflow_for_submit(
        NPA_SPECS / "token-factory-caption.yaml",
        run_id="caption-demo",
        render_options=SkypilotRenderOptions(registry="cr.example.invalid/reg"),
    )
    try:
        docs = [
            doc
            for doc in yaml.safe_load_all(
                prepared.skypilot_yaml_path.read_text(encoding="utf-8")
            )
            if doc is not None
        ]
        assert "image_id" not in docs[1]["resources"]
        assert docs[1]["envs"]["NPA_SRC_S3_URI"] == "s3://example-bucket/npa-src/npa"
        assert "file_mounts" not in docs[1]
    finally:
        prepared.temp_dir.cleanup()


def test_plan_only_registry_secrets_use_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--plan-only must not mint or embed live SKYPILOT_DOCKER_PASSWORD values."""

    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "live-should-not-appear")
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "iam")
    path = NPA_SPECS / "vlm-eval-single.yaml"
    spec = load_spec(path)
    for profile in spec.resources.values():
        if isinstance(profile, dict):
            profile["cloud"] = "nebius"
    plan = build_plan(spec, run_id="demo")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="demo",
        options=SkypilotRenderOptions(
            registry="cr.eu-north1.nebius.cloud/reg",
            materialize_registry_secrets=False,
        ),
    )
    docs = [doc for doc in yaml.safe_load_all(rendered) if doc is not None]
    task = docs[1]
    assert task["secrets"]["SKYPILOT_DOCKER_PASSWORD"] == "<SKYPILOT_DOCKER_PASSWORD>"
    assert "live-should-not-appear" not in rendered


def test_render_bdd100k_task_count() -> None:
    spec = load_spec(NPA_SPECS / "bdd100k-pipeline.yaml")
    plan = build_plan(spec, run_id="bdd-demo")
    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="bdd-demo",
        options=SkypilotRenderOptions(registry="cr.example.invalid/reg"),
    )
    docs = [doc for doc in yaml.safe_load_all(text) if doc is not None]
    assert docs[0]["execution"] == "serial"
    assert len(docs) - 1 == len(plan.steps)
    assert len(plan.steps) >= 10


def test_render_rejects_parallel_execution() -> None:
    spec = load_spec(NPA_SPECS / "vlm-eval-single.yaml")
    plan = build_plan(spec, run_id="demo")
    with pytest.raises(NpaWorkflowRenderError, match="execution=serial"):
        render_skypilot_yaml(
            spec,
            plan,
            run_id="demo",
            options=SkypilotRenderOptions(execution="parallel"),
        )


def test_resolve_task_image_uses_override() -> None:
    image = resolve_task_image(
        "workbench.vlm_eval.run",
        {},
        options=SkypilotRenderOptions(image_overrides={"*": "cr.example/custom:1"}),
    )
    assert image == "cr.example/custom:1"


def test_prepare_requires_assume_decision_for_dynamic_specs() -> None:
    with pytest.raises(Exception, match="assume-decision"):
        prepare_npa_workflow_for_submit(
            NPA_SPECS / "sim2real-vlm-rl.yaml",
            run_id="dyn-demo",
        )


def test_workbench_workflow_submit_npa_workflow_renders_and_submits(mocker) -> None:
    captured: dict[str, object] = {}

    def fake_submit(path, run_id, **kwargs):
        captured["content"] = Path(path).read_text(encoding="utf-8")
        captured["run_id"] = run_id
        captured["path"] = str(path)
        return WorkflowResult(status="SUBMITTED", job_id="42", returncode=0)

    mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        side_effect=fake_submit,
    )

    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(NPA_SPECS / "vlm-eval-single.yaml"),
            "--run-id",
            "npa-submit-1",
            "--registry",
            "cr.example.invalid/reg",
            "--submit-timeout",
            "30",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "SUBMITTED" in result.output
    assert captured["run_id"] == "npa-submit-1"
    assert "vlm-eval-single.yaml" not in str(captured["path"])
    content = str(captured["content"])
    assert "execution: serial" in content
    assert "score-rollouts" in content
    assert "${" not in content


def test_workbench_workflow_submit_npa_plan_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "live-plan-only-token")
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(NPA_SPECS / "token-factory-caption.yaml"),
            "--run-id",
            "plan-only-1",
            "--plan-only",
            "--registry",
            "cr.example.invalid/reg",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "status: PLANNED" in result.output
    assert "token-factory caption" in result.output
    assert "NEBIUS_TOKEN_FACTORY_KEY" in result.output
    assert "live-plan-only-token" not in result.output


def test_workbench_workflow_submit_plan_only_redacts_registry_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "live-plan-only-token")
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(NPA_SPECS / "vlm-eval-single.yaml"),
            "--run-id",
            "plan-only-redact",
            "--plan-only",
            "--registry",
            "cr.eu-north1.nebius.cloud/reg",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "status: PLANNED" in result.output
    assert "<SKYPILOT_DOCKER_PASSWORD>" in result.output
    assert "live-plan-only-token" not in result.output


def test_workbench_workflow_submit_npa_var_merges_config(
    mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    captured: dict[str, object] = {}

    def fake_submit(path, run_id, **kwargs):
        captured["content"] = Path(path).read_text(encoding="utf-8")
        return WorkflowResult(status="SUBMITTED", job_id="7", returncode=0)

    mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        side_effect=fake_submit,
    )
    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(NPA_SPECS / "token-factory-caption.yaml"),
            "--run-id",
            "var-demo",
            "--var",
            "bucket=my-live-bucket",
            "--registry",
            "cr.example.invalid/reg",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "my-live-bucket" in str(captured["content"])
