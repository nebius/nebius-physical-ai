from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from npa.guardrails.skypilot import load_yaml_documents
from npa.guardrails.three_tier import (
    CapabilityContract,
    ParameterContract,
    registered_workbench_tools,
    validate_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


CONTRACTS: tuple[CapabilityContract, ...] = (
    CapabilityContract(
        name="sonic/train",
        cli_module="npa.cli.workbench.sonic.train",
        cli_callback="train_cmd",
        sdk_module="npa.sdk.workbench.sonic",
        sdk_attr="train",
        yaml_path=Path("npa/workflows/workbench/skypilot/sonic-train-standalone.yaml"),
        params=(
            ParameterContract("checkpoint", "checkpoint", "SONIC_CHECKPOINT", "--checkpoint"),
            ParameterContract("data_path", "data_path", "SONIC_DATA_PATH", "--data-path"),
            ParameterContract("sample_data", "sample_data", "SONIC_SAMPLE_DATA", "--sample-data"),
            ParameterContract("embodiment", "embodiment", "SONIC_EMBODIMENT", "--embodiment"),
            ParameterContract("num_envs", "num_envs", "SONIC_NUM_ENVS", "--num-envs"),
            ParameterContract("headless", "headless", "SONIC_HEADLESS", "--headless"),
            ParameterContract("max_iterations", "max_iterations", "SONIC_MAX_ITERATIONS", "--max-iterations"),
            ParameterContract("output_path", "output_path", "SONIC_OUTPUT_PREFIX", "--output-path"),
            ParameterContract("image", "image", "POLICY_IMAGE", "--image"),
            ParameterContract("gpu_type", "gpu_type", "SONIC_GPU_TYPE", "--gpu-type"),
            ParameterContract("image_variant", "image_variant", "SONIC_IMAGE_VARIANT", "--image-variant"),
        ),
    ),
    CapabilityContract(
        name="sonic/export",
        cli_module="npa.cli.workbench.sonic.export",
        cli_callback="export_cmd",
        sdk_module="npa.sdk.workbench.sonic",
        sdk_attr="export_onnx",
        yaml_path=Path("npa/workflows/workbench/skypilot/sonic-export.yaml"),
        params=(
            ParameterContract("checkpoint", "checkpoint", "SONIC_CHECKPOINT", "--checkpoint"),
            ParameterContract("output_path", "output", "SONIC_OUTPUT", "--output"),
            ParameterContract("opset", "opset", "SONIC_OPSET", "--opset"),
            ParameterContract("axes", "axes", "SONIC_AXES", "--axes"),
            ParameterContract("normalize", "normalize", "SONIC_NORMALIZE", "--normalize"),
            ParameterContract("metadata", "metadata", "SONIC_METADATA", "--metadata"),
            ParameterContract("obs_spec", "obs_spec", "SONIC_OBS_SPEC", "--obs-spec"),
            ParameterContract("action_spec", "action_spec", "SONIC_ACTION_SPEC", "--action-spec"),
            ParameterContract("config", "config", "SONIC_CONFIG", "--config"),
            ParameterContract("verify", "verify", "SONIC_VERIFY", "--verify"),
            ParameterContract("parity_atol", "parity_atol", "SONIC_PARITY_ATOL", "--parity-atol"),
        ),
    ),
    CapabilityContract(
        name="vlm-eval/run",
        cli_module="npa.cli.workbench.vlm_eval",
        cli_callback="run_cmd",
        sdk_module="npa.sdk.workbench.vlm_eval",
        sdk_attr="run",
        yaml_path=Path("npa/workflows/workbench/skypilot/vlm-eval.yaml"),
        params=(
            ParameterContract("input_path", "input_path", "EVAL_INPUT_URI", "--input-path"),
            ParameterContract("output_path", "output_path", "VLM_EVAL_OUTPUT_URI", "--output-path"),
            ParameterContract("task", "task", "VLM_EVAL_TASK", "--task"),
            ParameterContract("backend", "backend", "VLM_BACKEND", "--backend"),
            ParameterContract("model", "model", "VLM_MODEL", "--model"),
            ParameterContract("endpoint_url", "endpoint_url", "VLM_ENDPOINT_URL", "--endpoint-url"),
            ParameterContract("frame_selection", "frame_selection", "VLM_FRAME_SELECTION", "--frame-selection"),
            ParameterContract("max_frames", "max_frames", "VLM_MAX_FRAMES", "--max-frames"),
            ParameterContract("success_threshold", "success_threshold", "VLM_SUCCESS_THRESHOLD", "--success-threshold"),
        ),
    ),
    CapabilityContract(
        name="detection-training/train",
        cli_module="npa.cli.workbench.detection_training",
        cli_callback="train_cmd",
        sdk_module="npa.sdk.workbench.detection_training",
        sdk_attr="train",
        yaml_path=Path("npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml"),
        params=(
            ParameterContract("view", "view", "VIEW_NAME", "--view"),
            ParameterContract("output_uri", "output_uri", "TRAIN_OUTPUT_URI", "--output-uri"),
            ParameterContract("lance_uri", "lance_uri", "LANCE_URI", "--lance-uri"),
            ParameterContract("epochs", "epochs", "TRAIN_EPOCHS", "--epochs"),
            ParameterContract("batch_size", "batch_size", "TRAIN_BATCH_SIZE", "--batch-size"),
            ParameterContract("learning_rate", "learning_rate", "TRAIN_LEARNING_RATE", "--learning-rate"),
        ),
    ),
    CapabilityContract(
        name="detection-training/eval",
        cli_module="npa.cli.workbench.detection_training",
        cli_callback="eval_cmd",
        sdk_module="npa.sdk.workbench.detection_training",
        sdk_attr="eval",
        yaml_path=Path("npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml"),
        params=(
            ParameterContract("eval_view", "eval_view", "VIEW_NAME", "--eval-view"),
            ParameterContract("output_uri", "output_uri", "EVAL_OUTPUT_URI", "--output-uri"),
            ParameterContract("lance_uri", "lance_uri", "LANCE_URI", "--lance-uri"),
        ),
    ),
)


def test_current_three_tier_contracts_are_coherent() -> None:
    failures: list[str] = []
    for contract in CONTRACTS:
        failures.extend(validate_contract(contract, repo_root=REPO_ROOT))
    assert not failures, "\n".join(failures)


def test_new_workbench_tools_require_contract_or_explicit_seam() -> None:
    contracted = {contract.name.split("/", 1)[0] for contract in CONTRACTS}
    seam = {
        "cosmos",
        "data",
        "fiftyone",
        "genesis",
        "groot",
        "isaac-lab",
        "lancedb",
        "lerobot",
        "mjlab",
        "retargeting",
        "workflow",
    }
    discovered = registered_workbench_tools()
    assert discovered == contracted | seam


def test_contract_catches_deliberately_broken_yaml_fixture(tmp_path: Path) -> None:
    source = REPO_ROOT / CONTRACTS[0].yaml_path
    docs = load_yaml_documents(source)
    task_doc = docs[1]
    envs = dict(task_doc["envs"])
    envs.pop("SONIC_CHECKPOINT")
    task_doc["envs"] = envs
    broken = tmp_path / "broken.yaml"
    import yaml

    broken.write_text(yaml.safe_dump_all(docs, sort_keys=False), encoding="utf-8")
    broken_contract = replace(CONTRACTS[0], yaml_path=broken)

    failures = validate_contract(broken_contract, repo_root=REPO_ROOT)

    assert any("YAML env missing: SONIC_CHECKPOINT" in failure for failure in failures)


def test_standalone_policy_yaml_is_parameterized_and_endpoint_safe() -> None:
    path = REPO_ROOT / "npa/workflows/workbench/skypilot/sonic-train-standalone.yaml"
    text = path.read_text(encoding="utf-8")
    docs = load_yaml_documents(path)
    task = docs[1]
    envs = task["envs"]

    assert task["resources"]["image_id"] == "docker:${POLICY_IMAGE}"
    assert {
        "POLICY_IMAGE",
        "SONIC_GPU_TYPE",
        "SONIC_IMAGE_VARIANT",
        "S3_ENDPOINT_URL",
        "S3_BUCKET",
    } <= set(envs)
    assert envs["POLICY_IMAGE"].startswith("example.invalid/")
    assert envs["SONIC_GPU_TYPE"] == "l40s"
    assert envs["SONIC_IMAGE_VARIANT"] == "sonic-l40s-baked"
    assert envs["S3_ENDPOINT_URL"] == ""
    assert envs["S3_BUCKET"] == "example-bucket"
    assert "nebius.cloud" not in text
