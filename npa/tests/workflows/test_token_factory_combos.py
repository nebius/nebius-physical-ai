from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from npa.workflows.token_factory_combos import (
    DEFAULT_TRIAGE_SYSTEM_PROMPT,
    build_triage_prompt,
    default_triage_run_id,
    join_uri,
    render_triage_prompts_jsonl,
    summarize_run_artifacts,
    triage_job_name,
    triage_prompt_record,
    triage_report_uri,
    utc_stamp,
)

ROOT = Path(__file__).resolve().parents[3]
ROLLOUT_JUDGE_YAML = ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "tokenfactory-rollout-judge.yaml"
TRIAGE_RUNNER = ROOT / "npa" / "scripts" / "run_tokenfactory_train_triage.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tokenfactory_train_triage", TRIAGE_RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _docs(path: Path) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc is not None]


# --- pure helpers ---------------------------------------------------------


def test_utc_stamp_and_run_id_are_deterministic_and_safe() -> None:
    moment = datetime(2026, 6, 11, 18, 30, 5, tzinfo=timezone.utc)
    assert utc_stamp(moment) == "20260611T183005Z"
    assert default_triage_run_id(moment) == "tf-train-triage-20260611T183005Z"


def test_triage_job_name_sanitizes_to_nebius_safe() -> None:
    name = triage_job_name("TF Train/Triage_2026!!")
    assert name == "tf-train-triage-2026"
    assert name == name.lower()
    assert all(ch.isalnum() or ch == "-" for ch in name)
    assert not name.startswith("-") and not name.endswith("-")
    assert len(triage_job_name("x" * 200)) <= 48


def test_join_and_report_uris() -> None:
    assert join_uri("s3://b/run/", "triage") == "s3://b/run/triage"
    assert join_uri("s3://b/run", "a", "b") == "s3://b/run/a/b"
    assert join_uri("s3://b/run/", "") == "s3://b/run/"
    assert triage_report_uri("s3://b/run/triage") == "s3://b/run/triage/generations.jsonl"


def test_summarize_run_artifacts_reads_text_skips_binary_and_truncates(tmp_path: Path) -> None:
    (tmp_path / "train_config.json").write_text(json.dumps({"steps": 50, "policy": "act"}), encoding="utf-8")
    (tmp_path / "train.log").write_text("step 0 loss 1.2\nstep 50 loss 0.3\n", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"\x00\x01\x02binaryweights")
    nested = tmp_path / "checkpoints" / "last"
    nested.mkdir(parents=True)
    (nested / "config.yaml").write_text("device: cuda\n", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.json").write_text("{}", encoding="utf-8")

    digest = summarize_run_artifacts(tmp_path)
    assert "train_config.json" in digest
    assert "train.log" in digest
    assert "checkpoints/last/config.yaml" in digest
    assert "safetensors" not in digest  # binary weights excluded
    assert "secret.json" not in digest  # dotfiles excluded


def test_summarize_truncates_large_files(tmp_path: Path) -> None:
    (tmp_path / "big.log").write_text("x" * 50_000, encoding="utf-8")
    digest = summarize_run_artifacts(tmp_path, max_file_bytes=1000, max_total_bytes=5000)
    assert "[truncated]" in digest
    assert len(digest.encode("utf-8")) < 8000


def test_summarize_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        summarize_run_artifacts(tmp_path / "nope")


def test_summarize_empty_dir_is_explicit(tmp_path: Path) -> None:
    assert "no textual artifacts" in summarize_run_artifacts(tmp_path)


def test_build_triage_prompt_includes_context_and_digest() -> None:
    prompt = build_triage_prompt(
        job_name="job-1",
        output_uri="s3://b/run/",
        artifact_digest="### train_config.json\n{...}",
        extra_context="ran on H200",
    )
    assert "job-1" in prompt
    assert "s3://b/run/" in prompt
    assert "ran on H200" in prompt
    assert "### train_config.json" in prompt


def test_triage_prompt_record_and_jsonl_roundtrip() -> None:
    record = triage_prompt_record(job_name="My Job", output_uri="s3://b/run/", artifact_digest="d")
    assert record["id"] == "triage-my-job"
    jsonl = render_triage_prompts_jsonl([record])
    parsed = json.loads(jsonl.strip())
    assert parsed["id"] == "triage-my-job"
    assert "prompt" in parsed
    assert jsonl.endswith("\n")


def test_default_triage_system_prompt_is_grounded() -> None:
    assert "only use facts" in DEFAULT_TRIAGE_SYSTEM_PROMPT.lower()


# --- runner (render-only, no infrastructure) ------------------------------


def test_runner_render_only_plan_serverless() -> None:
    module = _load_runner()
    args = module._parse_args(["--run-id", "demo", "--gpu-type", "h200"])
    plan = module.build_plan(args)
    assert plan["compute"] == "nebius-serverless-gpu"
    assert plan["hosted_inference"] == "nebius-token-factory"
    assert plan["job_name"] == "demo"
    assert plan["skip_train"] is False
    cmd = plan["train_command"]
    assert cmd[:3] == ["workbench", "lerobot", "train"]
    assert "--runtime" in cmd and "serverless" in cmd
    assert "--smoke" in cmd
    assert "--gpu-type" in cmd
    assert "--output" in cmd and "json" in cmd


def test_runner_from_output_path_skips_train() -> None:
    module = _load_runner()
    args = module._parse_args(["--from-output-path", "s3://b/run/"])
    plan = module.build_plan(args)
    assert plan["skip_train"] is True
    assert plan["artifacts_uri"] == "s3://b/run/"
    assert plan["triage_root"] == "s3://b/run/triage"
    assert "train_command" not in plan


def test_runner_no_smoke_flag() -> None:
    module = _load_runner()
    args = module._parse_args(["--no-smoke", "--steps", "200"])
    plan = module.build_plan(args)
    assert "--smoke" not in plan["train_command"]
    assert "--steps" in plan["train_command"]


# --- rollout-judge SkyPilot YAML (k8s GPU + Token Factory) ----------------


def test_rollout_judge_is_two_stage_gpu_then_hosted_judge() -> None:
    docs = _docs(ROLLOUT_JUDGE_YAML)
    assert docs[0] == {"name": "tokenfactory-rollout-judge", "execution": "serial"}
    gpu_stage, judge_stage = docs[1], docs[2]

    # Stage 1 genuinely uses a Nebius k8s GPU.
    assert gpu_stage["name"] == "rollout-gpu"
    assert gpu_stage["resources"]["cloud"] == "kubernetes"
    assert "accelerators" in gpu_stage["resources"]
    assert "lerobot-eval" in gpu_stage["run"]
    assert gpu_stage["envs"]["ROLLOUTS_URI"].startswith("s3://")

    # Stage 2 is zero-GPU hosted Token Factory judging.
    assert judge_stage["name"] == "tokenfactory-judge"
    assert judge_stage["resources"]["cloud"] == "kubernetes"
    assert "accelerators" not in judge_stage["resources"]
    assert "python3 -m vllm" not in judge_stage["run"]
    assert "npa workbench vlm-eval run" in judge_stage["run"]
    assert "--backend api" in judge_stage["run"]
    assert "--api-key-env NEBIUS_API_KEY" in judge_stage["run"]
    # The judge reads exactly what the GPU stage wrote.
    assert judge_stage["envs"]["ROLLOUTS_URI"] == gpu_stage["envs"]["ROLLOUTS_URI"]


def test_rollout_judge_has_no_hardcoded_infra_ids() -> None:
    text = ROLLOUT_JUDGE_YAML.read_text(encoding="utf-8")
    assert "<your-registry-id>" in text
    assert "<your-bucket-name>" in text
