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
    plan_image_pull_secrets,
    render_skypilot_yaml,
    resolve_task_image,
    tool_image_key,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit
from npa.orchestration.npa_workflow.submission_state import load_submission_state
from npa.orchestration.skypilot.workflow import WorkflowResult

REPO_ROOT = Path(__file__).resolve().parents[4]
NPA_SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
SKYPILOT_FIXTURES = REPO_ROOT / "npa" / "tests" / "fixtures" / "skypilot"
RUNNER = CliRunner()


def test_is_npa_workflow_spec_true_for_golden() -> None:
    path = NPA_SPECS / "vlm-eval-single.yaml"
    assert is_npa_workflow_spec(path)
    assert detect_submit_format(path) == "npa.workflow"


def test_is_npa_workflow_spec_false_for_skypilot() -> None:
    path = SKYPILOT_FIXTURES / "sonic-train-standalone.yaml"
    assert not is_npa_workflow_spec(path)
    assert detect_submit_format(path) == "skypilot"


def test_sonic_stage_setup_installs_torch_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SONIC train/export/eval need torch + ONNX. On a run with no baked image
    # (the daily rotation clears image pins) the stage would otherwise reach the
    # GPU and fail with "requires torch".
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://bucket/prefix/npa")
    spec = load_spec(NPA_SPECS / "sonic-export-eval.yaml")
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="demo"),
        run_id="demo",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
    )
    docs = [d for d in yaml.safe_load_all(rendered) if d]
    assert docs, rendered
    for doc in docs[1:]:
        setup = doc.get("setup", "")
        assert "onnxruntime>=1.18" in setup, doc["name"]
        assert "torch>=2.12.1" in setup, doc["name"]


def test_sonic_specs_train_with_the_in_job_runtime() -> None:
    # `serverless` (and vm/container) delegate to more infrastructure, which a
    # stage that already holds a GPU cannot provision.
    for name in (
        "sonic-train.yaml",
        "sonic-export.yaml",
        "sonic-export-eval.yaml",
        "sonic-locomotion-finetuning.yaml",
    ):
        spec = load_spec(NPA_SPECS / name)
        assert spec.config["sonic_runtime"] == "local", name


def test_self_hosted_vlm_eval_run_starts_vllm_server() -> None:
    # The self-hosted vlm-eval twin must launch a background vLLM server in its
    # run script (the eval client waits for /v1/models readiness). Without this
    # the server is never up and the eval fails with connection-refused.
    spec = load_spec(NPA_SPECS / "vlm-eval-single.yaml")
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="demo"),
        run_id="demo",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
    )
    docs = [d for d in yaml.safe_load_all(rendered) if d]
    run = next(d["run"] for d in docs if "vlm-eval run" in d.get("run", ""))
    assert 'vllm serve "$npa_vlm_model"' in run
    assert "--served-model-name" in run
    assert "npa_vlm_pid=$!" in run  # backgrounded + trap-killed on exit
    # This branch's preamble also WAITS for readiness before the command runs, rather than
    # relying on the client to retry a connection-refused (EVIDENCE.md §R21).
    assert "npa_vlm_log" in run
    # The served model is exported so the eval client asks for it instead of the
    # library default, and the twin picks a model whose cold start is bounded.
    assert "export NPA_VLM_SELF_HOSTED_MODEL=Qwen/Qwen2-VL-2B-Instruct" in run
    # A server that dies during startup must fail the stage immediately with its
    # own log, not stall until the client's readiness window expires.
    assert "vLLM server exited before becoming ready" in run
    # ... and prints the server's own log rather than leaving the operator to find it.
    assert "npa_vlm_log" in run
    # No CUDA toolkit in the task image, so nothing may JIT-compile a kernel.
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in run
    # Console scripts that vLLM's dependencies install (ninja, for the JIT paths)
    # live next to the stage interpreter, not on the stage shell's PATH.
    assert 'export PATH="$PATH:$npa_scripts"' in run
    setup = next(d["setup"] for d in docs if "vlm-eval run" in d.get("run", ""))
    # Weights are pulled in setup so the run phase only loads local files.
    assert "snapshot_download(MODEL)" in setup


def test_vlm_eval_benchmark_starts_a_server_because_its_twin_scores_for_real() -> None:
    """#236's benchmark twin was `sample` + backend=stub, so it needed no server.

    This branch's twin seeds a real labeled benchmark in S3 and scores it on the self-hosted
    backend (EVIDENCE.md §R22), so it does need one — and the decision is made by the backend
    the spec asks for, not by the toolRef's name.
    """

    spec = load_spec(NPA_SPECS / "vlm-eval-benchmark.yaml")
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="demo"),
        run_id="demo",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
    )
    backend = str(spec.config.get("vlm_backend") or "").replace("_", "-")
    if backend == "self-hosted":
        assert "vllm serve" in rendered
    else:
        assert "vllm serve" not in rendered


def _unused_test_stub_vlm_eval_benchmark_does_not_start_vllm_server() -> None:
    # The benchmark twin runs backend=stub; it must NOT launch a vLLM server.
    spec = load_spec(NPA_SPECS / "vlm-eval-benchmark.yaml")
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="demo"),
        run_id="demo",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
    )
    assert "vllm serve" not in rendered


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


def test_nebius_cloud_render_injects_docker_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def _nebius_gpu_spec():
    path = NPA_SPECS / "vlm-eval-single.yaml"
    spec = load_spec(path)
    for profile in spec.resources.values():
        if isinstance(profile, dict):
            profile["cloud"] = "nebius"
    return spec, build_plan(spec, run_id="demo")


def test_render_errors_on_registry_credentials_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Image pinned to us-central1 but Docker creds authenticate to eu-north1 →
    # a 403 ErrImagePull for EVERY stage image. Must fail fast at render, not stall.
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "test-token")
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "iam")
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "cr.eu-north1.nebius.cloud")
    spec, plan = _nebius_gpu_spec()
    with pytest.raises(NpaWorkflowRenderError, match="registry mismatch"):
        render_skypilot_yaml(
            spec,
            plan,
            run_id="demo",
            options=SkypilotRenderOptions(registry="cr.us-central1.nebius.cloud/reg"),
        )


def test_render_ok_when_registry_matches_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "test-token")
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "iam")
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "cr.eu-north1.nebius.cloud")
    spec, plan = _nebius_gpu_spec()
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="demo",
        options=SkypilotRenderOptions(registry="cr.eu-north1.nebius.cloud/reg"),
    )
    task = [doc for doc in yaml.safe_load_all(rendered) if doc is not None][1]
    assert task["secrets"]["SKYPILOT_DOCKER_SERVER"] == "cr.eu-north1.nebius.cloud"


def test_kubernetes_private_image_references_the_refreshed_pull_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "test-token")
    spec = load_spec(NPA_SPECS / "vlm-eval-single.yaml")
    plan = build_plan(spec, run_id="demo")

    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="demo",
        options=SkypilotRenderOptions(registry="cr.eu-north1.nebius.cloud/reg"),
    )
    task = [doc for doc in yaml.safe_load_all(rendered) if doc is not None][1]

    assert task["config"]["kubernetes"]["pod_config"]["spec"]["imagePullSecrets"] == [
        {"name": "npa-nebius-registry"}
    ]


def test_nurec_plan_exposes_its_ngc_pull_authority_to_preflight() -> None:
    spec = load_spec(NPA_SPECS / "nurec-reconstruct.yaml")
    plan = build_plan(spec, run_id="demo")

    authorities = plan_image_pull_secrets(
        spec,
        plan.steps,
        run_id="demo",
        options=SkypilotRenderOptions(),
    )

    assert authorities["nvcr.io/nvidia/nre/nre-ga:26.04"] == (
        "ngc-nvcr-imagepullsecret",
    )


def test_tool_image_key_prefix_match() -> None:
    assert tool_image_key("workbench.vlm_eval.run") == "cosmos"
    assert tool_image_key("workbench.token_factory.caption") is None
    assert tool_image_key("workbench.lancedb.import_bdd100k") == "lancedb"
    assert tool_image_key("workbench.sonic.train") == "sonic"
    assert tool_image_key("unknown.tool") is None


def test_cosmos3_generate_and_reason_resolve_to_different_images() -> None:
    """Generation runs in the framework image, reasoning in the Reason VLM image.

    An exact-match entry must beat the ``workbench.cosmos3`` prefix: rendering a
    generate step into npa-cosmos3-reason would schedule a container that has no
    cosmos-framework in it.
    """

    assert tool_image_key("workbench.cosmos3.generate") == "cosmos3"
    assert tool_image_key("workbench.cosmos3.reason") == "cosmos3-reason"


def test_render_token_factory_uses_env_aws_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_render_transfer_forwards_explicit_runtime_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    monkeypatch.setenv("NPA_COSMOS_VARIANT_PARALLELISM", "4")
    monkeypatch.setenv("NPA_COSMOS_DISABLE_CONTENT_GUARDRAILS", "1")
    spec = load_spec(REPO_ROOT / "npa" / "workflows" / "physical-ai-data-factory.yaml")
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="demo", assume_decision="promote_checkpoint"),
        run_id="demo",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
    )
    docs = [doc for doc in yaml.safe_load_all(rendered) if doc]
    transfer = next(doc for doc in docs if "cosmos2 transfer" in doc.get("run", ""))
    assert transfer["envs"]["NPA_COSMOS_VARIANT_PARALLELISM"] == "4"
    assert transfer["envs"]["NPA_COSMOS_DISABLE_CONTENT_GUARDRAILS"] == "1"


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


def test_render_token_factory_sets_npa_src_s3_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_first_party_image_rejects_uid_zero_pod_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "NPA_REGISTRY", "cr.us-central1.nebius.cloud/project"
    )
    spec = load_spec(NPA_SPECS / "vlm-eval-single.yaml")
    for profile in spec.resources.values():
        if isinstance(profile, dict):
            profile["kubernetes"] = {
                "pod_config": {
                    "spec": {
                        "containers": [
                            {
                                "name": "worker",
                                "securityContext": {"runAsUser": 0},
                            }
                        ]
                    }
                }
            }
    with pytest.raises(NpaWorkflowRenderError, match="runAsUser: 0"):
        render_skypilot_yaml(
            spec,
            build_plan(spec, run_id="no-root"),
            run_id="no-root",
            options=SkypilotRenderOptions(
                image_overrides={
                    "*": "cr.us-central1.nebius.cloud/project/npa-fiftyone:validation"
                },
                materialize_registry_secrets=False,
            ),
        )


def test_resolve_task_image_uses_longest_tool_family_override() -> None:
    image = resolve_task_image(
        "workbench.fiftyone.curate_augmented",
        {},
        options=SkypilotRenderOptions(
            image_overrides={
                "*": "cr.example/default:1",
                "workbench.fiftyone": "cr.example/fiftyone:1",
                "workbench.fiftyone.curate_augmented": "cr.example/curate:1",
            }
        ),
    )

    assert image == "cr.example/curate:1"


def test_tool_image_prefix_does_not_cross_tool_boundary() -> None:
    image = resolve_task_image(
        "workbench.fiftyone_extra.curate",
        {},
        options=SkypilotRenderOptions(
            image_overrides={"workbench.fiftyone": "cr.example/fiftyone:1"}
        ),
    )
    assert image == ""


def test_resolve_task_image_can_clear_tool_family_image() -> None:
    image = resolve_task_image(
        "workbench.cosmos_evaluator.evaluate",
        {},
        options=SkypilotRenderOptions(
            image_overrides={
                "*": "cr.example/default:1",
                "workbench.cosmos_evaluator": "",
            }
        ),
    )

    assert image == ""


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
            # Rendering test: the real SkyPilot CLI / npa-source prerequisites
            # are mocked out, so skip the submit preflight.
            "--skip-preflight",
            "--no-preflight-images",
            "--no-resolve-accelerators",
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
    receipt = load_submission_state("default", "npa-submit-1")
    assert receipt["launch"]["sky_job_id"] == "42"
    assert receipt["workflow"]["name"] == "vlm-eval-single"
    assert receipt["workflow"]["run_prefix_uri"] == (
        "s3://example-bucket/runs/npa-submit-1/vlm-eval"
    )
    assert receipt["workflow"]["manifest_uri"].endswith("/npa-workflow/manifest.json")
    assert receipt["workflow"]["steps"][0]["state"] == "score-rollouts"


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
            "--details",
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
            "--details",
            "--registry",
            "cr.eu-north1.nebius.cloud/reg",
            "--skip-preflight",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "status: PLANNED" in result.output
    assert "<SKYPILOT_DOCKER_PASSWORD>" in result.output
    assert "live-plan-only-token" not in result.output


def test_e2e_clear_workbench_images_env_is_not_global_cli_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_E2E_CLEAR_WORKBENCH_IMAGES", "1")
    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_E2E_NPA_SRC_S3_URI", raising=False)

    result = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(NPA_SPECS / "vlm-eval-single.yaml"),
            "--run-id",
            "env-clear-is-test-only",
            "--plan-only",
            "--details",
            "--registry",
            "cr.example.invalid/reg",
            # This spec's steps keep their workbench images (that is the assertion
            # below), but the prerequisite check cannot know that before planning
            # and asks for a source it will not need.
            "--skip-preflight",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "status: PLANNED" in result.output
    assert "image_id: docker:cr.example.invalid/reg/npa-cosmos:" in result.output


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
            "--skip-preflight",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "my-live-bucket" in str(captured["content"])


def test_default_npa_setup_has_optin_source_overlay() -> None:
    from npa.orchestration.npa_workflow.skypilot_render import default_npa_setup

    setup = default_npa_setup()
    # Opt-in overlay: gated on NPA_SRC_OVERLAY, reinstalls branch npa on top of a
    # baked image so branch code runs on GPU without an image rebuild. Default off.
    assert 'if [ "$NPA_SRC_OVERLAY" = "1" ]' in setup
    assert "/tmp/npa-src-overlay" in setup
    # Installs route through the PEP 668-tolerant helper (see npa_pip_install).
    assert "npa_pip_install -e /tmp/npa-src-overlay --no-deps" in setup
