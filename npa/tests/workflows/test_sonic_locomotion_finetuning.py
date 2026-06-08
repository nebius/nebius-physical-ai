from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_WORKBENCH_IMAGE = "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-genesis:0.4.6"
EXPECTED_RETARGETING_IMAGE = "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-retargeting:0.1.0"
PIPELINE_YAML = (
    ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "skypilot"
    / "sonic-locomotion-finetuning.yaml"
)
RETARGETING_YAML = (
    ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "retargeting.yaml"
)
MJLAB_YAML = ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "mjlab-eval.yaml"
SONIC_EXPORT_YAML = (
    ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "sonic-export.yaml"
)
SONIC_EVAL_YAML = (
    ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "sonic-eval.yaml"
)
SONIC_EXPORT_EVAL_YAML = (
    ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "skypilot"
    / "sonic-export-eval.yaml"
)
SONIC_TRAIN_STANDALONE_YAML = (
    ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "skypilot"
    / "sonic-train-standalone.yaml"
)


def _docs(path: Path) -> list[dict]:
    return [
        doc
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if doc is not None
    ]


def test_sonic_locomotion_pipeline_yaml_is_serial_and_uses_expected_tools() -> None:
    docs = _docs(PIPELINE_YAML)

    assert docs[0] == {"name": "sonic-locomotion-finetuning", "execution": "serial"}
    tasks = docs[1:]
    assert [task["name"] for task in tasks] == [
        "sonic-retarget-motion",
        "sonic-g1-finetune",
        "sonic-mujoco-eval",
    ]
    assert "npa workbench retargeting run" in tasks[0]["run"]
    assert "/entrypoint.sh finetune" in tasks[1]["run"]
    assert "mujoco-eval" in tasks[2]["run"]


def test_sonic_locomotion_pipeline_uses_h100_mujoco_mvp_image() -> None:
    docs = _docs(PIPELINE_YAML)
    retarget, train, eval_task = docs[1:]

    assert retarget["resources"] == {
        "cloud": "kubernetes",
        "cpus": 4,
        "memory": 16,
        "image_id": "docker:${NPA_RETARGETING_IMAGE}",
    }
    assert retarget["envs"]["NPA_RETARGETING_IMAGE"] == EXPECTED_RETARGETING_IMAGE
    assert retarget["envs"]["SOURCE_FORMAT"] == "auto"
    assert retarget["envs"]["RETARGET_FRAME_RATE"] == "30"
    assert retarget["envs"]["RETARGET_SOURCE_FRAME_RATE"] == "120"
    assert retarget["envs"]["AWS_PROFILE"] == "nebius"
    assert retarget["envs"]["AWS_ENDPOINT_URL"] == "https://storage.eu-north1.nebius.cloud"
    assert train["resources"]["cloud"] == "nebius"
    assert train["resources"]["region"] == "eu-north1"
    assert train["resources"]["accelerators"] == "H100:1"
    assert train["resources"]["use_spot"] is True
    assert train["resources"]["image_id"] == (
        "docker:example.invalid/npa-sonic-mujoco:0.1.3-mvp"
    )
    assert eval_task["resources"]["cloud"] == "nebius"
    assert eval_task["resources"]["region"] == "eu-north1"
    assert eval_task["resources"]["accelerators"] == "H100:1"
    assert eval_task["resources"]["use_spot"] is True
    assert eval_task["resources"]["image_id"] == (
        "docker:example.invalid/npa-sonic-mujoco:0.1.3-mvp"
    )
    assert train["envs"]["POLICY_IMAGE"] == "example.invalid/npa-sonic-mujoco:0.1.3-mvp"
    assert eval_task["envs"]["POLICY_IMAGE"] == "example.invalid/npa-sonic-mujoco:0.1.3-mvp"
    assert train["envs"]["SONIC_GPU_TYPE"] == "h100"
    assert train["envs"]["SONIC_IMAGE_VARIANT"] == "sonic-mujoco-h100-mvp"
    assert train["envs"]["AWS_PROFILE"] == "nebius"
    assert train["envs"]["RETARGETED_MOTION_URI"].endswith("/retargeted/")
    assert train["envs"]["SONIC_TRAIN_MODE"] == "finetune"
    assert train["envs"]["SONIC_RUN_REAL_TRAIN"] == "1"
    assert eval_task["envs"]["SONIC_FINE_TUNED_CHECKPOINT_URI"].endswith(
        "/training/checkpoints/last.pt"
    )
    assert eval_task["envs"]["AWS_PROFILE"] == "nebius"
    assert eval_task["envs"]["SONIC_MUJOCO_STEPS"] == "64"


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

    assert retarget["resources"]["image_id"] == "docker:registry.example/workbench/npa-retargeting:0.1.0"
    assert train["resources"]["image_id"] == "docker:registry.example/workbench/npa-sonic:0.1.2-k8s-runtime"
    assert retarget["envs"]["AWS_PROFILE"] == "nebius"
    assert retarget["envs"]["AWS_ENDPOINT_URL"] == "https://storage.example"
    assert train["resources"]["cloud"] == "kubernetes"
    assert train["resources"]["accelerators"] == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    assert eval_task["resources"]["image_id"] == (
        "docker:registry.example/workbench/npa-sonic:0.1.2-k8s-runtime"
    )
    assert eval_task["resources"]["cloud"] == "kubernetes"
    assert eval_task["resources"]["accelerators"] == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    assert train["envs"]["SONIC_GPU_TYPE"] == "gpu-rtx6000"
    assert train["envs"]["SONIC_IMAGE_VARIANT"] == "sonic-k8s-host-mounted"
    assert train["envs"]["AWS_PROFILE"] == "nebius"
    assert train["envs"]["POLICY_IMAGE"] == (
        "registry.example/workbench/npa-sonic:0.1.2-k8s-runtime"
    )
    assert eval_task["envs"]["POLICY_IMAGE"] == (
        "registry.example/workbench/npa-sonic:0.1.2-k8s-runtime"
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
        gpu_target="l40s",
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
        s3_prefix="sonic-proof/sonic-run",
        secret_envs=["AWS_ACCESS_KEY_ID"],
    )

    assert result.job_id == "42"
    assert captured["run_id"] == "sonic-run"
    assert captured["kwargs"]["secret_envs"] == ["AWS_ACCESS_KEY_ID"]
    assert "registry.example/workbench/npa-sonic:0.1.2" in str(captured["content"])


def test_sonic_workflow_materializer_supports_docker_payload_mode() -> None:
    from npa.workbench.sonic.workflow import materialize_sonic_workflow

    plan = materialize_sonic_workflow(
        SONIC_TRAIN_STANDALONE_YAML,
        run_id="sonic-run",
        registry="registry.example/workbench",
        gpu_target="l40s",
        s3_endpoint="https://storage.example",
        s3_bucket="proof-bucket",
        env_overrides={"SONIC_PAYLOAD_MODE": "docker"},
    )
    docs = [doc for doc in yaml.safe_load_all(plan.yaml_text) if doc is not None]
    task = docs[1]

    assert "image_id" not in task["resources"]
    assert task["envs"]["POLICY_IMAGE"] == "registry.example/workbench/npa-sonic:0.1.2"
    assert task["envs"]["SONIC_PAYLOAD_MODE"] == "docker"
    assert "docker run --rm --gpus all" in task["run"]


def test_tool_yamls_match_registered_cli_surfaces() -> None:
    retarget_docs = _docs(RETARGETING_YAML)
    mjlab_docs = _docs(MJLAB_YAML)
    sonic_export_docs = _docs(SONIC_EXPORT_YAML)
    sonic_eval_docs = _docs(SONIC_EVAL_YAML)

    assert retarget_docs[0] == {"name": "retargeting", "execution": "serial"}
    assert retarget_docs[1]["name"] == "retarget-motion"
    assert "npa workbench retargeting run" in retarget_docs[1]["run"]
    assert "accelerators" not in retarget_docs[1]["resources"]
    assert retarget_docs[1]["resources"]["image_id"] == "docker:${NPA_RETARGETING_IMAGE}"
    assert retarget_docs[1]["envs"]["NPA_RETARGETING_IMAGE"] == EXPECTED_RETARGETING_IMAGE
    assert retarget_docs[1]["envs"]["RETARGET_SOURCE_FRAME_RATE"] == "120"

    assert mjlab_docs[0] == {"name": "mjlab-eval", "execution": "serial"}
    assert mjlab_docs[1]["name"] == "mjlab-locomotion-eval"
    assert "npa workbench mjlab eval" in mjlab_docs[1]["run"]
    assert mjlab_docs[1]["resources"]["accelerators"] == "H100:1"
    assert mjlab_docs[1]["resources"]["image_id"] == "docker:${NPA_WORKBENCH_IMAGE}"
    assert mjlab_docs[1]["envs"]["NPA_WORKBENCH_IMAGE"] == EXPECTED_WORKBENCH_IMAGE

    assert sonic_export_docs[0] == {"name": "sonic-export", "execution": "serial"}
    assert sonic_export_docs[1]["name"] == "sonic-export-onnx"
    assert "npa workbench sonic export" in sonic_export_docs[1]["run"]
    assert sonic_export_docs[1]["resources"]["accelerators"] == "L40S:1"
    assert sonic_export_docs[1]["envs"]["SONIC_GPU_TARGET"] == "L40S"
    assert sonic_export_docs[1]["envs"]["SONIC_IMAGE_VARIANT"] == "sonic-l40s-baked"
    assert sonic_export_docs[1]["envs"]["SONIC_OPSET"] == "17"
    assert sonic_export_docs[1]["envs"]["SONIC_AXES"] == "dynamic"
    assert sonic_export_docs[1]["envs"]["SONIC_NORMALIZE"] == "baked"
    assert sonic_export_docs[1]["envs"]["SONIC_METADATA"] == "sidecar"

    assert sonic_eval_docs[0] == {"name": "sonic-eval", "execution": "serial"}
    assert sonic_eval_docs[1]["name"] == "sonic-eval-onnx"
    assert "npa workbench sonic eval" in sonic_eval_docs[1]["run"]
    assert sonic_eval_docs[1]["resources"]["cloud"] == "nebius"
    assert sonic_eval_docs[1]["resources"]["accelerators"] == "L40S:1"
    assert sonic_eval_docs[1]["resources"]["image_id"] == "docker:${NPA_WORKBENCH_IMAGE}"
    assert sonic_eval_docs[1]["envs"]["NPA_WORKBENCH_IMAGE"] == EXPECTED_WORKBENCH_IMAGE
    assert sonic_eval_docs[1]["envs"]["SONIC_EVAL_BACKEND"] == "reference"
    assert sonic_eval_docs[1]["envs"]["SONIC_EVAL_ENV"] == "smoke"
    assert sonic_eval_docs[1]["envs"]["SONIC_EVAL_CONTAINER_GPUS"] == "all"
    assert sonic_eval_docs[1]["envs"]["SONIC_EVAL_CONTAINER_GPU_TARGET"] == "L40S"
    assert sonic_eval_docs[1]["envs"]["SONIC_EVAL_CONTAINER_IMAGE_VARIANT"] == "sonic-l40s-baked"
    assert sonic_eval_docs[1]["envs"]["SONIC_EVAL_CONTAINER_ARGS"] == "eval"
    assert sonic_eval_docs[1]["envs"]["SONIC_EVAL_CONTAINER_OUTPUT_PATH"].endswith(
        "sonic_eval_results.json"
    )


def test_sonic_export_eval_blueprint_chains_real_cli_commands() -> None:
    docs = _docs(SONIC_EXPORT_EVAL_YAML)

    assert docs[0] == {"name": "sonic-export-eval", "execution": "serial"}
    assert len(docs) == 2

    task = docs[1]
    assert task["name"] == "sonic-export-eval"
    assert task["resources"]["cloud"] == "nebius"
    assert task["resources"]["accelerators"] == "L40S:1"

    envs = task["envs"]
    assert envs["POLICY_CKPT"].startswith("s3://")
    assert envs["OUTPUT_DIR"].startswith("s3://")
    assert envs["EVAL_BACKEND"] == "reference"
    assert envs["EVAL_ENV"] == "sonic-locomotion-smoke"
    assert envs["EPISODES"] == "8"
    assert envs["CONTAINER_IMAGE"] == ""
    assert envs["CONTAINER_GPU_TARGET"] == "L40S"
    assert envs["CONTAINER_IMAGE_VARIANT"] == "sonic-l40s-baked"
    assert envs["CONTAINER_GPUS"] == "all"
    assert envs["CONTAINER_ARGS"] == "eval"
    assert envs["GPU"] == "L40S:1"

    run = task["run"]
    assert "npa workbench sonic export" in run
    assert "npa workbench sonic eval" in run
    assert "NPA_SONIC_E2E_METRICS_JSON_BEGIN" in run
    assert "--container-image" in run
    assert "--container-driver-capabilities" in run


def test_sonic_locomotion_assets_do_not_add_python_runner() -> None:
    scripts = {path.name for path in (ROOT / "npa" / "scripts").glob("run_*sonic*")}

    assert scripts == set()
