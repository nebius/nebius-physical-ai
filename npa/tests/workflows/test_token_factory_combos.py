from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from npa.workflows.token_factory_combos import (
    DEFAULT_SWEEP_DESIGN_SYSTEM_PROMPT,
    DEFAULT_SWEEP_RANKING_SYSTEM_PROMPT,
    DEFAULT_SWEEP_STEPS,
    DEFAULT_TRIAGE_SYSTEM_PROMPT,
    build_ranking_prompt,
    build_sweep_design_prompt,
    build_triage_prompt,
    default_sweep_run_id,
    default_triage_run_id,
    join_uri,
    render_triage_prompts_jsonl,
    summarize_run_artifacts,
    sweep_variant_output_uri,
    sweep_variants,
    triage_job_name,
    triage_prompt_record,
    triage_report_uri,
    utc_stamp,
)

ROOT = Path(__file__).resolve().parents[3]
SKYPILOT = ROOT / "npa" / "workflows" / "workbench" / "skypilot"
ROLLOUT_JUDGE_YAML = SKYPILOT / "tokenfactory-rollout-judge.yaml"
SCENE_JUDGE_YAML = SKYPILOT / "tokenfactory-scene-to-rollout-judge.yaml"
TRAIN_TRIAGE_YAML = SKYPILOT / "tokenfactory-train-triage.yaml"
TRIAGE_RUNNER = ROOT / "npa" / "scripts" / "run_tokenfactory_train_triage.py"
SWEEP_RUNNER = ROOT / "npa" / "scripts" / "run_tokenfactory_sim_sweep.py"

# Every combo workflow that has a submittable SkyPilot YAML form.
COMBO_YAMLS = [ROLLOUT_JUDGE_YAML, SCENE_JUDGE_YAML, TRAIN_TRIAGE_YAML]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runner():
    return _load_module("run_tokenfactory_train_triage", TRIAGE_RUNNER)


def _load_sweep_runner():
    return _load_module("run_tokenfactory_sim_sweep", SWEEP_RUNNER)


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


# --- sim-sweep pure helpers ----------------------------------------------


def test_sweep_run_id_and_variants_are_deterministic() -> None:
    moment = datetime(2026, 6, 11, 18, 30, 5, tzinfo=timezone.utc)
    assert default_sweep_run_id(moment) == "tf-sim-sweep-20260611T183005Z"

    variants = sweep_variants(2)
    assert [v["id"] for v in variants] == ["v0-steps50", "v1-steps100"]
    assert [v["steps"] for v in variants] == [50, 100]


def test_sweep_variants_clamps_to_grid_bounds() -> None:
    assert len(sweep_variants(0)) == 1  # clamps up to at least one
    assert len(sweep_variants(99)) == len(DEFAULT_SWEEP_STEPS)  # clamps down to grid
    with pytest.raises(ValueError):
        sweep_variants(2, steps_grid=[])


def test_sweep_variant_output_uri_nests_under_variants() -> None:
    uri = sweep_variant_output_uri("s3://b/sweep", "v0-steps50")
    assert uri == "s3://b/sweep/variants/v0-steps50"


def test_build_sweep_design_prompt_lists_only_grid_variants() -> None:
    variants = sweep_variants(2)
    prompt = build_sweep_design_prompt(
        objective="maximize success",
        dataset="lerobot/pusht",
        policy_type="act",
        variants=variants,
    )
    assert "maximize success" in prompt
    assert "v0-steps50" in prompt and "steps=50" in prompt
    assert "v1-steps100" in prompt
    assert "v2-" not in prompt  # only the two requested variants


def test_build_ranking_prompt_includes_each_run_digest() -> None:
    prompt = build_ranking_prompt(
        objective="best policy",
        runs=[
            {"id": "v0", "uri": "s3://b/v0/", "digest": "### train.log\nloss 0.3"},
            {"id": "v1", "uri": "s3://b/v1/", "digest": "### train.log\nloss 0.9"},
        ],
    )
    assert "best policy" in prompt
    assert "Variant v0" in prompt and "Variant v1" in prompt
    assert "loss 0.3" in prompt and "loss 0.9" in prompt


def test_build_ranking_prompt_handles_no_runs() -> None:
    assert "no completed variants" in build_ranking_prompt(objective="x", runs=[])


def test_sweep_system_prompts_are_grounded() -> None:
    assert "only use facts" in DEFAULT_SWEEP_RANKING_SYSTEM_PROMPT.lower()
    assert "do not invent" in DEFAULT_SWEEP_DESIGN_SYSTEM_PROMPT.lower()


# --- sim-sweep runner (render-only, no infrastructure) --------------------


def test_sweep_runner_render_only_full_sweep() -> None:
    module = _load_sweep_runner()
    args = module._parse_args(
        ["--run-id", "demo", "--num-variants", "2", "--bucket", "s3://b/tf-sim-sweep"]
    )
    plan = module.build_plan(args)
    assert plan["mode"] == "full-sweep"
    assert plan["compute"] == "nebius-serverless-gpu"
    assert plan["hosted_inference"] == "nebius-token-factory"
    assert len(plan["variants"]) == 2
    cmd = plan["variants"][0]["train_command"]
    assert cmd[:3] == ["workbench", "lerobot", "train"]
    assert "--runtime" in cmd and "serverless" in cmd
    assert "--steps" in cmd
    assert "--seed" not in cmd  # lerobot train has no --seed flag
    assert plan["variants"][0]["output_uri"].endswith("variants/v0-steps50")
    assert "Design the rationale" in plan["design_prompt"]


def test_sweep_runner_rank_existing_skips_design_and_gpu() -> None:
    module = _load_sweep_runner()
    args = module._parse_args(["--rank-existing", "s3://b/runA/, s3://b/runB/"])
    plan = module.build_plan(args)
    assert plan["mode"] == "rank-existing"
    assert plan["variant_uris"] == ["s3://b/runA/", "s3://b/runB/"]
    assert "variants" not in plan
    assert "design_prompt" not in plan


def test_sweep_runner_disambiguates_colliding_run_labels() -> None:
    module = _load_sweep_runner()
    # Distinct last segments keep their names...
    runs = module._label_existing_runs(["s3://b/runA/", "s3://b/runB/"])
    assert [r["id"] for r in runs] == ["runA", "runB"]
    # ...colliding last segments are suffixed by position.
    collide = module._label_existing_runs(
        ["s3://b/r1/checkpoints/pretrained_model/", "s3://b/r2/checkpoints/pretrained_model/"]
    )
    assert [r["id"] for r in collide] == ["pretrained_model-0", "pretrained_model-1"]
    assert len({r["id"] for r in collide}) == 2


# --- scene-to-rollout-judge SkyPilot YAML (reason -> k8s GPU -> VLM judge) -


def test_scene_judge_is_three_stage_reason_gpu_judge() -> None:
    docs = _docs(SCENE_JUDGE_YAML)
    assert docs[0] == {"name": "tokenfactory-scene-to-rollout-judge", "execution": "serial"}
    reason_stage, gpu_stage, judge_stage = docs[1], docs[2], docs[3]

    # Stage 1: hosted reasoner, zero-GPU.
    assert reason_stage["name"] == "scene-reason"
    assert "accelerators" not in reason_stage["resources"]
    assert "npa workbench token-factory reason" in reason_stage["run"]
    assert reason_stage["envs"]["PLAN_URI"].startswith("s3://")

    # Stage 2: genuine Nebius k8s GPU rollout.
    assert gpu_stage["name"] == "rollout-gpu"
    assert gpu_stage["resources"]["cloud"] == "kubernetes"
    assert "accelerators" in gpu_stage["resources"]
    assert "lerobot-eval" in gpu_stage["run"]

    # Stage 3: hosted VLM judge that consumes the plan and the rollout.
    assert judge_stage["name"] == "scene-judge"
    assert "accelerators" not in judge_stage["resources"]
    assert "npa workbench vlm-eval run" in judge_stage["run"]
    assert "--backend api" in judge_stage["run"]
    assert judge_stage["envs"]["PLAN_URI"] == reason_stage["envs"]["PLAN_URI"]
    assert judge_stage["envs"]["ROLLOUTS_URI"] == gpu_stage["envs"]["ROLLOUTS_URI"]


def test_scene_judge_has_no_hardcoded_infra_ids() -> None:
    text = SCENE_JUDGE_YAML.read_text(encoding="utf-8")
    assert "<your-registry-id>" in text
    assert "<your-bucket-name>" in text


# --- train-triage SkyPilot YAML (k8s GPU train -> hosted Token Factory triage) -


def test_train_triage_yaml_is_two_stage_gpu_then_hosted_triage() -> None:
    docs = _docs(TRAIN_TRIAGE_YAML)
    assert docs[0] == {"name": "tokenfactory-train-triage", "execution": "serial"}
    gpu_stage, triage_stage = docs[1], docs[2]

    # Stage 1: a genuine Nebius k8s GPU training run.
    assert gpu_stage["name"] == "train-gpu"
    assert gpu_stage["resources"]["cloud"] == "kubernetes"
    assert "accelerators" in gpu_stage["resources"]
    assert "lerobot-train" in gpu_stage["run"]
    assert gpu_stage["envs"]["ARTIFACTS_URI"].startswith("s3://")

    # Stage 2: zero-GPU hosted Token Factory triage over what the GPU stage wrote.
    assert triage_stage["name"] == "tokenfactory-triage"
    assert "accelerators" not in triage_stage["resources"]
    assert "npa workbench token-factory generate" in triage_stage["run"]
    assert "summarize_run_artifacts" in triage_stage["run"]
    assert triage_stage["envs"]["ARTIFACTS_URI"] == gpu_stage["envs"]["ARTIFACTS_URI"]


def test_train_triage_yaml_has_no_hardcoded_infra_ids() -> None:
    text = TRAIN_TRIAGE_YAML.read_text(encoding="utf-8")
    assert "<your-registry-id>" in text
    assert "<your-bucket-name>" in text


# --- CLI / SDK / YAML support matrix for the combos -----------------------


def test_all_combo_yamls_are_well_formed_serial_pipelines() -> None:
    """Every combo YAML is a serial multi-doc with named stages and a GPU stage."""
    for path in COMBO_YAMLS:
        docs = _docs(path)
        assert docs[0]["execution"] == "serial", f"{path.name} is not serial"
        stages = docs[1:]
        assert len(stages) >= 2, f"{path.name} should have >=2 stages"
        assert all(stage.get("name") for stage in stages), f"{path.name} has an unnamed stage"
        # At least one GPU stage (Nebius compute) and at least one fail-fast on the key.
        assert any("accelerators" in stage.get("resources", {}) for stage in stages), (
            f"{path.name} has no GPU compute stage"
        )
        full_text = path.read_text(encoding="utf-8")
        assert "NEBIUS_API_KEY" in full_text, f"{path.name} never references the Token Factory key"


def test_sdk_exposes_workflow_submit_for_combo_yamls() -> None:
    """The SDK can submit the combo YAMLs via npa.workflow.submit."""
    from npa import workflow

    assert "submit" in workflow.__all__
    assert callable(workflow.submit)


def test_sdk_workflow_submit_delegates_to_orchestrator(mocker) -> None:
    """npa.workflow.submit forwards a combo YAML to the SkyPilot orchestrator."""
    import types

    from npa import workflow

    fake = types.SimpleNamespace(status="SUBMITTED", job_id="job-1")
    submit_mock = mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow", return_value=fake
    )

    workflow.submit(
        ROLLOUT_JUDGE_YAML,
        run_id="rj-test",
        secret_env=["NEBIUS_API_KEY", "AWS_ACCESS_KEY_ID"],
    )

    submit_mock.assert_called_once()
    assert submit_mock.call_args.args[1] == "rj-test"
    assert "NEBIUS_API_KEY" in submit_mock.call_args.kwargs["secret_envs"]


def test_sdk_exposes_token_factory_and_vlm_eval_building_blocks() -> None:
    """The hosted Token Factory stages are SDK-callable building blocks."""
    from npa.sdk.workbench import token_factory, vlm_eval

    for name in ("generate", "reason", "caption", "verify"):
        assert callable(getattr(token_factory, name))
    assert hasattr(vlm_eval, "run") or hasattr(vlm_eval, "benchmark")
