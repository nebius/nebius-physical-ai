from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_WORKBENCH_IMAGE = "cr.eu-north1.nebius.cloud/<your-registry-id>/npa-genesis:0.4.6"
EXPECTED_RETARGETING_IMAGE = "cr.eu-north1.nebius.cloud/<your-registry-id>/npa-retargeting:0.1.1"
# Frozen raw-task fixtures, not shipped templates: the three materializer tests below
# exercise the submit WRAPPER, which still accepts a customer's own SkyPilot YAML.
# See npa/tests/fixtures/skypilot/README.md.
PIPELINE_YAML = ROOT / "npa/tests/fixtures/skypilot/sonic-locomotion-finetuning.yaml"
# The raw sonic-export / sonic-eval / sonic-export-eval templates are retired; their
# npa.workflow specs are the surface now (each live-verified — see EVIDENCE §R4/§R5).
NPA_WORKFLOWS = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
SONIC_TRAIN_STANDALONE_YAML = ROOT / "npa/tests/fixtures/skypilot/sonic-train-standalone.yaml"
EXPECTED_SONIC_IMAGE = (
    "registry.example/workbench/npa-sonic:cuda13-b300-0.1.2-k8s-runtime-"
    "sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
)


def _docs(path: Path) -> list[dict]:
    return [
        doc
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if doc is not None
    ]




def test_sonic_workflow_materializer_resolves_images_and_s3_literals() -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    plan = materialize_sonic_workflow(
        PIPELINE_YAML,
        run_id="sonic-run",
        registry="registry.example/workbench",
        npa_image="registry.example/workbench/npa:tools",
        gpu_target="gpu-rtx6000",
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
        s3_prefix="sonic-proof/sonic-run",
        accelerators="RTXPRO-6000-BLACKWELL-SERVER-EDITION:1",
    )
    docs = [doc for doc in yaml.safe_load_all(plan.yaml_text) if doc is not None]
    retarget, train, eval_task = docs[1:]

    assert retarget["resources"]["image_id"] == "docker:registry.example/workbench/npa-retargeting:0.1.1"
    assert train["resources"]["image_id"] == f"docker:{EXPECTED_SONIC_IMAGE}"
    assert retarget["envs"]["AWS_PROFILE"] == "nebius"
    assert retarget["envs"]["AWS_ENDPOINT_URL"] == "https://storage.example"
    assert train["resources"]["cloud"] == "kubernetes"
    assert train["resources"]["accelerators"] == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    assert eval_task["resources"]["image_id"] == (
        f"docker:{EXPECTED_SONIC_IMAGE}"
    )
    assert eval_task["resources"]["cloud"] == "kubernetes"
    assert eval_task["resources"]["accelerators"] == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    assert train["envs"]["SONIC_GPU_TYPE"] == "gpu-rtx6000"
    assert train["envs"]["SONIC_IMAGE_VARIANT"] == "sonic-k8s-host-mounted"
    assert train["envs"]["AWS_PROFILE"] == "nebius"
    assert train["envs"]["POLICY_IMAGE"] == (
        EXPECTED_SONIC_IMAGE
    )
    assert eval_task["envs"]["POLICY_IMAGE"] == (
        EXPECTED_SONIC_IMAGE
    )
    assert eval_task["envs"]["AWS_PROFILE"] == "nebius"
    assert train["envs"]["SONIC_TRAIN_OUTPUT_URI"] == "s3://proof-bucket/sonic-proof/sonic-run/training/"
    assert train["envs"]["RETARGETED_MOTION_URI"] == "s3://proof-bucket/sonic-proof/sonic-run/retargeted/"
    assert eval_task["envs"]["SONIC_FINE_TUNED_CHECKPOINT_URI"] == (
        "s3://proof-bucket/sonic-proof/sonic-run/training/checkpoints/last.pt"
    )
    assert eval_task["envs"]["SONIC_MUJOCO_OUTPUT_URI"] == (
        "s3://proof-bucket/sonic-proof/sonic-run/mujoco-eval/"
    )
    assert train["envs"]["AWS_ENDPOINT_URL"] == "https://storage.example"
    assert eval_task["envs"]["AWS_ENDPOINT_URL"] == "https://storage.example"
    for task in (retarget, train, eval_task):
        assert "${" not in task["resources"]["image_id"]
        assert "${" not in "\n".join(str(value) for value in task["envs"].values())
    assert "<your-" not in plan.yaml_text


def test_sonic_sdk_submit_passes_secret_envs(mocker) -> None:
    from npa.orchestration.skypilot.workflow import WorkflowResult
    from npa.workbench.sonic import workflow as sonic_workflow

    captured: dict[str, object] = {}

    def fake_submit_workflow(path, run_id, **kwargs):
        captured["content"] = path.read_text(encoding="utf-8")
        captured["run_id"] = run_id
        captured["kwargs"] = kwargs
        return WorkflowResult(status="SUBMITTED", job_id="42", returncode=0)

    mocker.patch.object(
        sonic_workflow,
        "_submit_skypilot_workflow",
        side_effect=fake_submit_workflow,
    )

    result = sonic_workflow.submit_sonic_workflow(
        SONIC_TRAIN_STANDALONE_YAML,
        run_id="sonic-run",
        registry="registry.example/workbench",
        gpu_target="gpu-rtx6000",
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
        s3_prefix="sonic-proof/sonic-run",
        secret_envs=["AWS_ACCESS_KEY_ID"],
        accept_eula=True,
    )

    assert result.job_id == "42"
    assert captured["run_id"] == "sonic-run"
    assert captured["kwargs"]["secret_envs"] == ["AWS_ACCESS_KEY_ID"]
    assert EXPECTED_SONIC_IMAGE in str(captured["content"])


def test_sonic_workflow_materializer_supports_docker_payload_mode() -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    plan = materialize_sonic_workflow(
        SONIC_TRAIN_STANDALONE_YAML,
        run_id="sonic-run",
        registry="registry.example/workbench",
        gpu_target="gpu-rtx6000",
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
        env_overrides={"SONIC_PAYLOAD_MODE": "docker"},
    )
    docs = [doc for doc in yaml.safe_load_all(plan.yaml_text) if doc is not None]
    task = docs[1]

    assert "image_id" not in task["resources"]
    assert task["envs"]["POLICY_IMAGE"] == EXPECTED_SONIC_IMAGE
    assert task["envs"]["SONIC_PAYLOAD_MODE"] == "docker"
    assert task["envs"]["SONIC_DOCKER_GPU_REQUEST"] == "all"
    assert '--gpus "${SONIC_DOCKER_GPU_REQUEST}"' in task["run"]
    assert "nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml" in task["run"]
    assert "NVIDIA_VISIBLE_DEVICES=${SONIC_DOCKER_GPU_REQUEST}" in task["run"]
    assert 'docker run --rm "${docker_gpu_args[@]}"' in task["run"]


def test_sonic_locomotion_spec_runs_the_three_stages_in_order() -> None:
    """Replaces the retired template's serial/task-name assertions.

    The template said `execution: serial` over three named tasks. The spec's equivalent is a
    plan whose steps carry the three toolRefs in the same order, which is a stronger statement:
    it is what the engine will actually run, not what a document claims.
    """

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(NPA_WORKFLOWS / "sonic-locomotion-finetuning.yaml")
    steps = build_plan(spec, run_id="probe").steps
    assert [step.tool_ref for step in steps] == [
        "workbench.retargeting.run",
        "workbench.sonic.train",
        "workbench.mjlab.eval",
    ]
    # Serial, in the engine's terms: no step belongs to a `parallel:` fan-out group, so the
    # scheduler launches them one wave at a time in this order.
    assert [step.group for step in steps] == ["", "", ""]

def test_retargeting_spec_invokes_the_real_cli_surface() -> None:
    """Replaces the retired retargeting template's raw-YAML assertions.

    The template pinned an image and a source frame rate through `envs`; the spec
    declares CPU-only resources (retargeting needs no GPU) and reaches the same CLI
    through its toolRef. The image is chosen by the engine from the resource profile,
    which is why `--image` is a pinned `spec_gap` (see test_three_tier_contract.py).
    """

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(NPA_WORKFLOWS / "retargeting.yaml")
    step = build_plan(spec, run_id="probe").steps[0]
    argv = " ".join(step.argv)

    assert step.tool_ref == "workbench.retargeting.run"
    assert "npa workbench sonic retargeting run" in argv
    assert "accelerators" not in spec.resources[step.resources]
    for flag in ("--input-path", "--output-path", "--embodiment", "--source-format"):
        assert flag in argv



def test_mjlab_eval_spec_invokes_the_real_cli_surface() -> None:
    """Replaces the retired mjlab-eval template's raw-YAML assertions."""

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(NPA_WORKFLOWS / "mjlab-eval.yaml")
    step = build_plan(spec, run_id="probe").steps[0]
    argv = " ".join(step.argv)

    assert step.tool_ref == "workbench.mjlab.eval"
    assert "npa workbench mjlab eval" in argv
    assert spec.resources[step.resources]["accelerators"] == "H100:1"
    for flag in ("--input-path", "--checkpoint", "--output-path", "--suite",
                 "--embodiment", "--episodes"):
        assert flag in argv


def test_sonic_export_and_eval_specs_invoke_the_real_cli_surfaces() -> None:
    """Replaces the raw-YAML `envs` assertions for the three retired templates.

    The equivalent contract on the npa.workflow side is: the spec declares the right
    ``toolRef``, wires every config key the toolRef's argv references (``load_spec``
    resolves them), and the *result path* is the declared artifact rather than a format
    word — the bug that made both eval stages succeed while writing nothing (EVIDENCE
    §R5).
    """

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    export = load_spec(NPA_WORKFLOWS / "sonic-export.yaml")
    assert export.name == "sonic-export"
    assert export.states["export-onnx"].tool_ref == "workbench.sonic.export"
    # #238 made the export twin self-contained by prepending a train stage, so find the export
    # step by its toolRef rather than assuming it is first.
    export_argv = " ".join(
        next(
            step.argv
            for step in build_plan(export, run_id="probe").steps
            if step.tool_ref == "workbench.sonic.export"
        )
    )
    assert "npa workbench sonic export" in export_argv
    assert "--checkpoint s3://" in export_argv and "--output s3://" in export_argv

    evaluate = load_spec(NPA_WORKFLOWS / "sonic-eval.yaml")
    assert evaluate.states["eval-onnx"].tool_ref == "workbench.sonic.eval"
    eval_argv = build_plan(evaluate, run_id="probe").steps[0].argv
    assert "npa workbench sonic eval" in " ".join(eval_argv)
    # `--output` is the RESULT PATH; `--output-format` is the format.
    assert eval_argv[eval_argv.index("--output") + 1].endswith("/eval.json")
    assert eval_argv[eval_argv.index("--output-format") + 1] == "json"
    # The env name comes from the spec's config, not a literal the test decides.
    assert eval_argv[eval_argv.index("--env") + 1] == evaluate.config["env"]

    chained = load_spec(NPA_WORKFLOWS / "sonic-export-eval.yaml")
    steps = build_plan(chained, run_id="probe").steps
    # #238 made the chain self-contained by prepending a train stage, so the export twin can
    # run as a standalone submit instead of needing a checkpoint from somewhere else.
    assert [step.tool_ref for step in steps] == [
        "workbench.sonic.train",
        "workbench.sonic.export",
        "workbench.sonic.eval",
    ]
    # The eval stage consumes exactly what the export stage produced: both argv lists
    # carry the SAME resolved ONNX URI, so the chain cannot silently drift apart.
    export_step = next(s for s in steps if s.tool_ref == "workbench.sonic.export")
    eval_step = next(s for s in steps if s.tool_ref == "workbench.sonic.eval")
    produced = export_step.argv[export_step.argv.index("--output") + 1]
    consumed = eval_step.argv[eval_step.argv.index("--onnx") + 1]
    assert produced.startswith("s3://") and produced.endswith("/sonic_policy.onnx")
    assert consumed == produced
    # And the eval result goes to its own declared artifact, not to a format word.
    assert eval_step.argv[eval_step.argv.index("--output") + 1].endswith("/eval.json")
    assert eval_step.argv[eval_step.argv.index("--output-format") + 1] == "json"


def test_sonic_locomotion_assets_do_not_add_python_runner() -> None:
    scripts = {path.name for path in (ROOT / "npa" / "scripts").glob("run_*sonic*")}

    assert scripts == set()
