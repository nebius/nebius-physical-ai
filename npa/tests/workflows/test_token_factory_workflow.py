from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
SKYPILOT = ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot"
CAPTION_YAML = SKYPILOT / "token-factory-caption.yaml"
GENERATE_YAML = SKYPILOT / "token-factory-generate.yaml"
REASON_YAML = SKYPILOT / "token-factory-cosmos-reason.yaml"
VLM_EVAL_YAML = SKYPILOT / "vlm-eval-token-factory.yaml"


def _docs(path: Path) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc is not None]


def test_caption_workflow_is_cpu_only_and_runs_cli() -> None:
    docs = _docs(CAPTION_YAML)
    assert docs[0] == {"name": "token-factory-caption", "execution": "serial"}
    task = docs[1]
    assert task["resources"]["cloud"] == "kubernetes"
    assert "accelerators" not in task["resources"]
    assert "npa workbench token-factory caption" in task["run"]
    assert "NEBIUS_TOKEN_FACTORY_KEY" in task["run"]
    for flag in ("--input-path", "--output-path", "--model", "--instruction", "--max-images"):
        assert flag in task["run"]


def test_generate_workflow_is_cpu_only_and_runs_cli() -> None:
    docs = _docs(GENERATE_YAML)
    assert docs[0] == {"name": "token-factory-generate", "execution": "serial"}
    task = docs[1]
    assert task["resources"]["cloud"] == "kubernetes"
    assert "accelerators" not in task["resources"]
    assert "npa workbench token-factory generate" in task["run"]
    for flag in ("--input-path", "--output-path", "--model", "--system-prompt", "--max-prompts"):
        assert flag in task["run"]


def test_cosmos_reason_workflow_is_cpu_only_and_runs_cli() -> None:
    docs = _docs(REASON_YAML)
    assert docs[0] == {"name": "token-factory-cosmos-reason", "execution": "serial"}
    task = docs[1]
    assert task["resources"]["cloud"] == "kubernetes"
    assert "accelerators" not in task["resources"]
    assert task["envs"]["MODEL"] == "nvidia/Cosmos3-Super-Reasoner"
    assert "python3 -m vllm" not in task["run"]
    assert "npa workbench token-factory reason" in task["run"]
    assert "NEBIUS_TOKEN_FACTORY_KEY" in task["run"]
    for flag in ("--input-path", "--output-path", "--model", "--task", "--max-images"):
        assert flag in task["run"]


def test_vlm_eval_token_factory_workflow_uses_api_backend_no_gpu() -> None:
    docs = _docs(VLM_EVAL_YAML)
    assert docs[0] == {"name": "vlm-eval-token-factory", "execution": "serial"}
    task = docs[1]
    assert task["resources"]["cloud"] == "kubernetes"
    assert "accelerators" not in task["resources"]
    assert task["envs"]["VLM_BACKEND"] == "api"
    assert "python3 -m vllm" not in task["run"]
    assert "npa workbench vlm-eval run" in task["run"]
    assert "--api-key-env NEBIUS_TOKEN_FACTORY_KEY" in task["run"]
