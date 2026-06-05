from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_WORKBENCH_IMAGE = "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-genesis:0.4.6"
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
        "sonic-finetune",
        "sonic-mjlab-eval",
    ]
    assert "npa workbench retargeting run" in tasks[0]["run"]
    assert "/entrypoint.sh train" in tasks[1]["run"]
    assert "npa workbench mjlab eval" in tasks[2]["run"]


def test_sonic_locomotion_pipeline_routes_first_party_sonic_to_l40s_manifest_image() -> None:
    docs = _docs(PIPELINE_YAML)
    retarget, train, eval_task = docs[1:]

    assert retarget["resources"] == {
        "cloud": "kubernetes",
        "cpus": 4,
        "memory": 16,
        "image_id": "docker:${NPA_WORKBENCH_IMAGE}",
    }
    assert retarget["envs"]["NPA_WORKBENCH_IMAGE"] == EXPECTED_WORKBENCH_IMAGE
    assert train["resources"]["cloud"] == "kubernetes"
    assert train["resources"]["accelerators"] == "L40S:1"
    assert eval_task["resources"]["cloud"] == "kubernetes"
    assert eval_task["resources"]["accelerators"] == "H100:1"
    assert eval_task["resources"]["image_id"] == "docker:${NPA_WORKBENCH_IMAGE}"
    assert eval_task["envs"]["NPA_WORKBENCH_IMAGE"] == EXPECTED_WORKBENCH_IMAGE
    assert train["resources"]["image_id"].endswith("/npa-sonic:<sonic-image-tag>")
    assert train["envs"]["SONIC_GPU_TYPE"] == "l40s"
    assert train["envs"]["SONIC_IMAGE_VARIANT"] == "sonic-l40s-baked"


def test_tool_yamls_match_registered_cli_surfaces() -> None:
    retarget_docs = _docs(RETARGETING_YAML)
    mjlab_docs = _docs(MJLAB_YAML)
    sonic_export_docs = _docs(SONIC_EXPORT_YAML)
    sonic_eval_docs = _docs(SONIC_EVAL_YAML)

    assert retarget_docs[0] == {"name": "retargeting", "execution": "serial"}
    assert retarget_docs[1]["name"] == "retarget-motion"
    assert "npa workbench retargeting run" in retarget_docs[1]["run"]
    assert "accelerators" not in retarget_docs[1]["resources"]
    assert retarget_docs[1]["resources"]["image_id"] == "docker:${NPA_WORKBENCH_IMAGE}"
    assert retarget_docs[1]["envs"]["NPA_WORKBENCH_IMAGE"] == EXPECTED_WORKBENCH_IMAGE

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
