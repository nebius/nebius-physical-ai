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
from npa.orchestration.npa_workflow.submit import (
    merge_config_overrides,
    prepare_npa_workflow_for_submit,
)
from npa.orchestration.npa_workflow.submission_state import load_submission_state
from npa.orchestration.skypilot.workflow import WorkflowResult

REPO_ROOT = Path(__file__).resolve().parents[4]
NPA_SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
PAIDF = NPA_SPECS / "physical-ai-data-factory.yaml"
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


@pytest.mark.parametrize(
    ("name", "expected_image"),
    [
        (
            "byof-openpi.yaml",
            "docker:nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04@sha256:24c8e3581ea6330038b0d374920721983312627f8adbfcf390bdb4b399d280ed",
        ),
        ("byof-wan2.2.yaml", "docker:registry.example/npa-wan2-2:"),
    ],
)
def test_non_isaac_byof_specs_render_their_declared_runtime_image(
    name: str, expected_image: str
) -> None:
    spec = load_spec(NPA_SPECS / name)
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="byof-image"),
        run_id="byof-image",
        options=SkypilotRenderOptions(
            registry="registry.example", materialize_registry_secrets=False
        ),
    )
    task = [doc for doc in yaml.safe_load_all(rendered) if doc][-1]
    assert task["resources"]["image_id"].startswith(expected_image)
    assert "ACCEPT_EULA" not in task["envs"]


def test_kubernetes_profile_disk_size_renders_as_ephemeral_storage() -> None:
    spec = load_spec(NPA_SPECS / "byof-wan2.2.yaml")
    plan = build_plan(spec, run_id="disk-contract")
    assert plan.steps[0].resources_profile["disk_size"] == 200
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="disk-contract",
        options=SkypilotRenderOptions(
            registry="registry.example", materialize_registry_secrets=False
        ),
    )
    task = [doc for doc in yaml.safe_load_all(rendered) if doc][-1]
    assert task["resources"]["ephemeral_storage"] == 200
    assert "disk_size" not in task["resources"]


def test_every_byof_spec_declares_its_outer_runtime_image() -> None:
    paths = sorted(NPA_SPECS.glob("byof*.yaml"))

    # Pinned so a new BYOF spec cannot skip the per-profile image assertion
    # below by simply not being globbed. Bump it when you add one.
    assert len(paths) == 10
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        base_image = raw["config"].get("base_image")
        assert isinstance(base_image, str) and base_image, path.name
        for profile in raw["resources"].values():
            assert profile["image"] == "{{config.base_image}}", path.name


def test_isaac_byof_config_routes_image_and_preserves_cli_opt_out() -> None:
    base = load_spec(NPA_SPECS / "byof.yaml")
    spec = merge_config_overrides(
        base,
        {"base_profile": "isaac-lab", "base_image": "tool://isaac-lab"},
    )
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="byof-isaac"),
        run_id="byof-isaac",
        options=SkypilotRenderOptions(
            registry="registry.example",
            materialize_registry_secrets=False,
            accept_eula=False,
        ),
    )
    task = [doc for doc in yaml.safe_load_all(rendered) if doc][-1]
    assert task["resources"]["image_id"].startswith(
        "docker:registry.example/npa-isaac-lab:"
    )
    assert task["envs"]["ACCEPT_EULA"] == ""


def test_resolved_isaac_image_routes_all_five_raw_shell_sweep_states() -> None:
    spec = load_spec(NPA_SPECS / "isaac-lab-rl-sweep.yaml")
    plan = build_plan(spec, run_id="resolved-isaac-sweep")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="resolved-isaac-sweep",
        options=SkypilotRenderOptions(
            image_overrides={"*": "registry.example/npa-isaac-lab:runtime"},
            materialize_registry_secrets=False,
            accept_eula=False,
        ),
    )
    tasks = [
        doc for doc in yaml.safe_load_all(rendered) if doc and doc.get("resources")
    ]

    assert {task["name"] for task in tasks} == {
        "variant-lr-1e-3",
        "variant-lr-3e-4",
        "variant-entropy-0",
        "variant-entropy-0-01",
        "select-best",
    }
    assert all(task["envs"]["ACCEPT_EULA"] == "" for task in tasks)


def test_resolved_non_isaac_image_does_not_create_false_gate() -> None:
    spec = load_spec(NPA_SPECS / "isaac-lab-rl-sweep.yaml")
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="resolved-ubuntu-control"),
        run_id="resolved-ubuntu-control",
        options=SkypilotRenderOptions(
            image_overrides={"*": "ubuntu:22.04"},
            materialize_registry_secrets=False,
        ),
    )
    tasks = [
        doc for doc in yaml.safe_load_all(rendered) if doc and doc.get("resources")
    ]

    assert len(tasks) == 5
    assert all("ACCEPT_EULA" not in task["envs"] for task in tasks)


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


def test_setup_prefers_the_dependency_complete_baked_npa_interpreter() -> None:
    spec = load_spec(NPA_SPECS / "vlm-eval-single.yaml")
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="demo"),
        run_id="demo",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
    )
    setup = [doc for doc in yaml.safe_load_all(rendered) if doc][1]["setup"]

    candidate_loop = setup.split("for candidate in ", 1)[1].split("; do", 1)[0]
    assert candidate_loop.index('"${NPA_BAKED_PYTHON:-}"') < candidate_loop.index(
        "sys.executable"
    )


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


def test_normalize_resources_translates_kubernetes_disk_capacity() -> None:
    assert normalize_resources(
        {"cloud": "kubernetes", "disk_size": 200, "cpus": 4}
    ) == {
        "cloud": "kubernetes",
        "cpus": "4+",
        "ephemeral_storage": 200,
    }


def test_normalize_resources_preserves_non_kubernetes_renderer_behavior() -> None:
    assert normalize_resources({"cloud": "nebius", "disk_size": 200}) == {
        "cloud": "nebius",
    }


def test_accelerator_name_override_preserves_each_profile_gpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "NPA_WORKFLOW_GPU_ACCELERATOR",
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION",
    )
    assert normalize_resources({"accelerators": "RTXPRO6000:1"})["accelerators"] == (
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    )
    assert normalize_resources({"accelerators": "RTXPRO6000:8"})["accelerators"] == (
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION:8"
    )


def test_gpu_memory_override_targets_only_accelerator_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_WORKFLOW_GPU_MEMORY", "384Gi")
    assert normalize_resources(
        {
            "cloud": "kubernetes",
            "accelerators": "RTXPRO6000:4",
            "memory": "128Gi",
        }
    )["memory"] == "384+"
    assert normalize_resources(
        {"cloud": "kubernetes", "cpus": 4, "memory": "16Gi"}
    )["memory"] == "16+"


def test_submit_time_accelerator_override_preserves_profile_gpu_count() -> None:
    assert (
        normalize_resources(
            {"accelerators": "RTXPRO6000:8"},
            accelerator_overrides={"RTXPRO6000:8": "resolved-product"},
        )["accelerators"]
        == "resolved-product:8"
    )


def test_nebius_cloud_render_injects_exact_host_docker_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "test-token")
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "iam")
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "registry.example")
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
        options=SkypilotRenderOptions(registry="registry.example/reg"),
    )
    docs = [doc for doc in yaml.safe_load_all(rendered) if doc is not None]
    task = docs[1]
    assert task["resources"]["cloud"] == "nebius"
    assert "image_id" in task["resources"]
    assert task["secrets"]["SKYPILOT_DOCKER_SERVER"] == "registry.example"
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
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "test-token")
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "iam")
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "registry.example")
    spec, plan = _nebius_gpu_spec()
    with pytest.raises(NpaWorkflowRenderError) as exc_info:
        render_skypilot_yaml(
            spec,
            plan,
            run_id="demo",
            options=SkypilotRenderOptions(registry="registry-us.example/reg"),
        )

    message = str(exc_info.value)
    assert "registry mismatch" in message
    assert "registry-us.example" in message
    assert "registry.example" in message
    assert "test-token" not in message


def test_render_public_image_ignores_unrelated_private_registry_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "test-token")
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "operator")
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "registry.example")
    spec, plan = _nebius_gpu_spec()

    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="demo",
        options=SkypilotRenderOptions(
            registry="ghcr.io/nebius/nebius-physical-ai"
        ),
    )

    task = [doc for doc in yaml.safe_load_all(rendered) if doc is not None][1]
    assert "secrets" not in task or "SKYPILOT_DOCKER_PASSWORD" not in task["secrets"]


def test_render_ok_when_registry_matches_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "test-token")
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "iam")
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "registry.example")
    spec, plan = _nebius_gpu_spec()
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="demo",
        options=SkypilotRenderOptions(registry="registry.example/reg"),
    )
    task = [doc for doc in yaml.safe_load_all(rendered) if doc is not None][1]
    assert task["secrets"]["SKYPILOT_DOCKER_SERVER"] == "registry.example"


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
        options=SkypilotRenderOptions(registry="registry.example/reg"),
    )
    task = [doc for doc in yaml.safe_load_all(rendered) if doc is not None][1]

    assert "config" not in task or "imagePullSecrets" not in (
        ((task.get("config") or {}).get("kubernetes") or {}).get("pod_config") or {}
    ).get("spec", {})

    authorities = plan_image_pull_secrets(
        spec,
        plan.steps,
        run_id="demo",
        options=SkypilotRenderOptions(registry="registry.example/reg"),
    )
    assert set(authorities.values()) == {()}


def test_public_plan_has_no_implicit_kubernetes_pull_authority() -> None:
    spec = load_spec(NPA_SPECS / "vlm-eval-single.yaml")
    plan = build_plan(spec, run_id="demo")

    authorities = plan_image_pull_secrets(
        spec,
        plan.steps,
        run_id="demo",
        options=SkypilotRenderOptions(registry="registry.example/customer"),
    )

    assert set(authorities.values()) == {()}


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


def test_alpamayo2_super_resolves_configured_image() -> None:
    tool_ref = "workbench.alpamayo2_super.infer"

    assert tool_image_key(tool_ref) == "alpamayo2-super"
    assert resolve_task_image(
        tool_ref,
        {},
        options=SkypilotRenderOptions(registry="cr.example.invalid/reg"),
    ) == "cr.example.invalid/reg/npa-alpamayo2-super:0.1.0-cu128"


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
    monkeypatch.setenv("NPA_COSMOS_VALIDATION_SCOPE", "demo")
    monkeypatch.setenv("NPA_COSMOS_VALIDATION_DELAY_RANK", "1")
    monkeypatch.setenv("NPA_COSMOS_DISABLE_CONTENT_GUARDRAILS", "1")
    spec = load_spec(
        REPO_ROOT
        / "npa"
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "physical-ai-data-factory.yaml"
    )
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="demo", assume_decision="promote_checkpoint"),
        run_id="demo",
        options=SkypilotRenderOptions(materialize_registry_secrets=False),
    )
    docs = [doc for doc in yaml.safe_load_all(rendered) if doc]
    transfer = next(doc for doc in docs if "cosmos2 transfer" in doc.get("run", ""))
    assert transfer["envs"]["NPA_COSMOS_VARIANT_PARALLELISM"] == "4"
    assert transfer["envs"]["NPA_COSMOS_VALIDATION_SCOPE"] == "demo"
    assert transfer["envs"]["NPA_COSMOS_VALIDATION_DELAY_RANK"] == "1"
    assert transfer["envs"]["NPA_COSMOS_DISABLE_CONTENT_GUARDRAILS"] == "1"
    ray_node = next(
        container
        for container in transfer["config"]["kubernetes"]["pod_config"]["spec"][
            "containers"
        ]
        if container["name"] == "ray-node"
    )
    assert ray_node["imagePullPolicy"] == "IfNotPresent"


def test_transfer_execute_old_spec_uses_backward_compatible_optional_defaults(
    tmp_path: Path,
) -> None:
    """A pre-refinement/SAM2 external spec must still validate and render."""

    old_spec = tmp_path / "old-transfer-execute.yaml"
    old_spec.write_text(
        """
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: old-transfer-execute
config:
  trigger_uri: s3://example/input/
  augment_uri: s3://example/output/
  configs_uri: s3://example/configs/
resources:
  gpu:
    cloud: kubernetes
    accelerators: RTXPRO6000:1
initial: transfer
states:
  transfer:
    toolRef: workbench.cosmos2.transfer_execute
    resources: gpu
    terminal: true
""".lstrip()
    )
    spec = load_spec(old_spec)
    plan = build_plan(spec, run_id="old-compatible")
    argv = plan.steps[0].argv
    assert argv[argv.index("--control-weight") + 1] == "1.0"
    assert argv[argv.index("--guidance") + 1] == "3.0"
    assert argv[argv.index("--refinement-uri") + 1] == ""
    assert argv[argv.index("--protected-chroma-mode") + 1] == "off"
    assert argv[argv.index("--segmentation-mode") + 1] == "off"

    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="old-compatible",
        options=SkypilotRenderOptions(
            registry="cr.example.invalid/reg",
            materialize_registry_secrets=False,
        ),
    )
    assert_no_unresolved_placeholders(rendered)
    assert "--condition-on-input" in rendered
    assert "--execute" in rendered


def test_paidf_planner_uses_compatible_first_pass_and_committed_retry_pointer() -> None:
    spec = load_spec(PAIDF)
    plan = build_plan(spec, run_id="planner-settings")
    prepare = next(step for step in plan.steps if step.state == "prepare-refinement")
    augment = next(step for step in plan.steps if step.state == "augment")

    assert augment.argv[augment.argv.index("--control-weight") + 1] == "1.0"
    assert augment.argv[augment.argv.index("--guidance") + 1] == "3.0"
    assert (
        augment.argv[augment.argv.index("--refinement-uri") + 1]
        == "s3://example-bucket/physical-ai-data-factory/planner-settings/configs/refinement.json"
    )
    # The prepare argv carries the exact baseline, adaptive bounds, authoritative
    # decision artifact, and matching quality threshold used to create a retry.
    assert prepare.argv[-11:] == [
        "true",
        "1.0",
        "3.0",
        "0.25",
        "1.0",
        "1.0",
        "1.0",
        "s3://example-bucket/physical-ai-data-factory/planner-settings/grade/decision.json",
        "0.75",
        "1",
        "",
    ]


def test_paidf_refinement_iterations_use_append_only_artifact_prefixes() -> None:
    spec = load_spec(PAIDF)
    plan = build_plan(
        spec,
        run_id="append-only-refinement",
        assume_decision="loop_back",
    )

    augments = [step for step in plan.steps if step.state == "augment"]
    evaluates = [step for step in plan.steps if step.state == "evaluate"]
    gates = [step for step in plan.steps if step.state == "quality-gate"]
    prepares = [step for step in plan.steps if step.state == "prepare-refinement"]
    assert [step.outputs[0]["uri"] for step in augments] == [
        "s3://example-bucket/physical-ai-data-factory/append-only-refinement/"
        "cosmos_augmented/iteration-1/manifest.json",
        "s3://example-bucket/physical-ai-data-factory/append-only-refinement/"
        "cosmos_augmented/iteration-2/manifest.json",
    ]
    assert [
        step.argv[step.argv.index("--output-uri") + 1] for step in evaluates
    ] == [
        "s3://example-bucket/physical-ai-data-factory/append-only-refinement/"
        "grade/iteration-1/ranking/",
        "s3://example-bucket/physical-ai-data-factory/append-only-refinement/"
        "grade/iteration-2/ranking/",
    ]
    assert [step.outputs[0]["uri"] for step in gates] == [
        "s3://example-bucket/physical-ai-data-factory/append-only-refinement/"
        "grade/iteration-1/decision.json",
        "s3://example-bucket/physical-ai-data-factory/append-only-refinement/"
        "grade/iteration-2/decision.json",
    ]
    assert [step.argv[-2] for step in prepares] == ["1", "2"]

    reject = next(step for step in plan.steps if step.state == "reject-quality")
    assert (
        reject.argv[-3]
        == "s3://example-bucket/physical-ai-data-factory/append-only-refinement/"
        "grade/iteration-2/"
    )


def test_paidf_bare_static_plan_previews_promoted_path_with_fail_closed_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preview is useful, but its accepted stages stay behind a real guard."""

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    spec = load_spec(NPA_SPECS / "physical-ai-data-factory.yaml")
    plan = build_plan(spec, run_id="paidf-static-promote")
    states = [step.state for step in plan.steps]

    assert plan.assume_decision == "promote_checkpoint"
    assert states.index("quality-disposition") < states.index(
        "require-accepted-quality"
    ) < states.index("annotate-augmented")
    assert states[-2:] == ["visualize", "finalize"]
    assert "visualize-rejected" not in states
    assert "reject-quality" not in states

    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="paidf-static-promote",
        options=SkypilotRenderOptions(
            registry="cr.example.invalid/reg",
            materialize_registry_secrets=False,
        ),
    )
    docs = [doc for doc in yaml.safe_load_all(rendered) if doc]
    task_names = [doc.get("name") for doc in docs[1:]]
    guard_index = task_names.index("require-accepted-quality") + 1
    guard_run = docs[guard_index]["run"]
    assert "enforce_quality_disposition" in guard_run
    assert "set -euo pipefail" in guard_run
    assert task_names.index("require-accepted-quality") < task_names.index(
        "annotate-augmented"
    )


def test_paidf_static_rejection_plan_cannot_render_accepted_or_final_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/npa")
    spec = load_spec(NPA_SPECS / "physical-ai-data-factory.yaml")
    plan = build_plan(
        spec,
        run_id="paidf-static-reject",
        assume_decision="loop_back",
    )
    states = [step.state for step in plan.steps]

    assert states[-2:] == ["visualize-rejected", "reject-quality"]
    for forbidden in (
        "require-accepted-quality",
        "annotate-augmented",
        "cosmos-curate",
        "curate",
        "visualize",
        "finalize",
    ):
        assert forbidden not in states

    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="paidf-static-reject",
        options=SkypilotRenderOptions(
            registry="cr.example.invalid/reg",
            materialize_registry_secrets=False,
        ),
    )
    docs = [doc for doc in yaml.safe_load_all(rendered) if doc]
    task_names = [doc.get("name") for doc in docs[1:]]
    assert task_names[-2:] == ["visualize-rejected", "reject-quality"]
    assert "enforce_quality_disposition" in docs[-1]["run"]
    assert "set -euo pipefail" in docs[-1]["run"]
    assert not set(task_names) & {
        "require-accepted-quality",
        "annotate-augmented",
        "cosmos-curate",
        "curate",
        "visualize",
        "finalize",
    }


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


@pytest.mark.parametrize("nested_shell_key", ["run", "setup"])
def test_placeholder_guard_distinguishes_shell_from_declarative_fields(
    nested_shell_key: str,
) -> None:
    rendered = """\
name: placeholder-contract
execution: serial
---
name: task
# ${COMMENT_ONLY} is removed by YAML parsing and is not executable.
setup: |
  export CACHE_DIR="${CACHE_DIR:-/workspace/cache}"
run: |
  printf '%s\\n' "${CACHE_DIR}"
envs:
  MATERIALIZED: ready
resources:
  image_id: docker:registry.example/npa-tool:tag
"""
    assert_no_unresolved_placeholders(rendered)

    unresolved_env = rendered.replace(
        "MATERIALIZED: ready", 'MATERIALIZED: "${MISSING}"'
    )
    with pytest.raises(NpaWorkflowRenderError, match=r"\$\{MISSING\}"):
        assert_no_unresolved_placeholders(unresolved_env)

    nested_shell_name = rendered.replace(
        "MATERIALIZED: ready",
        f'{nested_shell_key}: "${{NESTED_SHELL_VALUE}}"',
    )
    with pytest.raises(NpaWorkflowRenderError, match=r"\$\{NESTED_SHELL_VALUE\}"):
        assert_no_unresolved_placeholders(nested_shell_name)

    unresolved_image = rendered.replace(
        "docker:registry.example/npa-tool:tag",
        'docker:registry.example/npa-tool:"${IMAGE_TAG}"',
    )
    with pytest.raises(NpaWorkflowRenderError, match=r"\$\{IMAGE_TAG\}"):
        assert_no_unresolved_placeholders(unresolved_image)


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
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "registry.example")
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
            registry="registry.example/reg",
            materialize_registry_secrets=False,
        ),
    )
    docs = [doc for doc in yaml.safe_load_all(rendered) if doc is not None]
    task = docs[1]
    assert task["secrets"]["SKYPILOT_DOCKER_PASSWORD"] == "<SKYPILOT_DOCKER_PASSWORD>"
    assert "live-should-not-appear" not in rendered


def test_plan_only_anonymous_public_registry_omits_empty_username(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "ghcr.io")
    monkeypatch.delenv("SKYPILOT_DOCKER_USERNAME", raising=False)
    monkeypatch.delenv("SKYPILOT_DOCKER_PASSWORD", raising=False)
    monkeypatch.delenv("NPA_REGISTRY_USERNAME", raising=False)
    monkeypatch.delenv("NPA_REGISTRY_PASSWORD", raising=False)
    spec, plan = _nebius_gpu_spec()

    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="demo",
        options=SkypilotRenderOptions(
            registry="ghcr.io/nebius/nebius-physical-ai",
            materialize_registry_secrets=False,
        ),
    )
    task = [doc for doc in yaml.safe_load_all(rendered) if doc is not None][1]
    assert "secrets" not in task
    assert "SKYPILOT_DOCKER_USERNAME: ''" not in rendered


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
    monkeypatch.setenv("NPA_REGISTRY", "registry-us.example/project")
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
                    "*": "registry-us.example/project/npa-fiftyone:validation"
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
            NPA_SPECS / "tokenfactory-cosmos-gate.yaml",
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
    assert_no_unresolved_placeholders(content)
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
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "operator")
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "registry.example")
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
            "registry.example/reg",
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
