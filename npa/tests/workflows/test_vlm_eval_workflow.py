from __future__ import annotations

from pathlib import Path

import yaml

from npa.workbench.vlm_eval import DEFAULT_MODEL


ROOT = Path(__file__).resolve().parents[3]
VLM_EVAL_YAML = ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "vlm-eval.yaml"
VLM_EVAL_BENCHMARK_YAML = (
    ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "vlm-eval-benchmark.yaml"
)


def _docs(path: Path) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc is not None]


def test_vlm_eval_workflow_serves_open_vlm_and_runs_cli() -> None:
    docs = _docs(VLM_EVAL_YAML)

    assert docs[0] == {"name": "vlm-eval", "execution": "serial"}
    task = docs[1]
    assert task["name"] == "vlm-eval-self-hosted"
    assert task["resources"]["cloud"] == "kubernetes"
    assert task["resources"]["accelerators"] == "H100:1"
    assert task["envs"]["VLM_MODEL"] == DEFAULT_MODEL
    assert task["envs"]["VLM_BACKEND"] == "self-hosted"
    assert task["envs"]["VLM_FRAME_SELECTION"] == "keyframes"
    assert "python3 -m vllm.entrypoints.openai.api_server" in task["run"]
    assert "npa workbench vlm-eval run" in task["run"]
    for flag in ("--backend", "--model", "--endpoint-url", "--frame-selection", "--success-threshold"):
        assert flag in task["run"]


def test_vlm_eval_benchmark_workflow_serves_open_vlm_and_runs_sweep_cli() -> None:
    docs = _docs(VLM_EVAL_BENCHMARK_YAML)

    assert docs[0] == {"name": "vlm-eval-benchmark", "execution": "serial"}
    task = docs[1]
    assert task["name"] == "vlm-eval-benchmark-self-hosted"
    assert task["resources"]["cloud"] == "kubernetes"
    assert task["resources"]["accelerators"] == "H100:1"
    assert task["envs"]["VLM_MODEL"] == DEFAULT_MODEL
    assert task["envs"]["VLM_MODELS"] == DEFAULT_MODEL
    assert task["envs"]["VLM_BACKEND"] == "self-hosted"
    assert task["envs"]["VLM_RUBRICS"] == "default,strict"
    assert task["envs"]["VLM_THRESHOLDS"] == "0.5,0.8,0.9"
    assert "python3 -m vllm.entrypoints.openai.api_server" in task["run"]
    assert "npa workbench vlm-eval benchmark" in task["run"]
    for flag in ("--dataset", "--output", "--models", "--rubrics", "--thresholds", "--format json"):
        assert flag in task["run"]
