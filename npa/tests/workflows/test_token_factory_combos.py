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
SKYPILOT = ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot"
ROLLOUT_JUDGE_YAML = SKYPILOT / "tokenfactory-rollout-judge.yaml"
SCENE_JUDGE_YAML = SKYPILOT / "tokenfactory-scene-to-rollout-judge.yaml"
TRAIN_TRIAGE_YAML = SKYPILOT / "tokenfactory-train-triage.yaml"
TRIAGE_RUNNER = ROOT / "npa" / "scripts" / "run_tokenfactory_train_triage.py"
SWEEP_RUNNER = ROOT / "npa" / "scripts" / "run_tokenfactory_sim_sweep.py"

# Combo workflows that still have a raw SkyPilot YAML form. train-triage is retired: its twin
# `npa-workflows/tokenfactory-train-triage.yaml` was verified live (job 256, EVIDENCE.md §R32–R33),
# and the shape assertions moved onto the spec below.
COMBO_YAMLS: list = []


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


def test_rollout_judge_combo_spec_is_gpu_producer_then_hosted_judge() -> None:
    """The retired template's contract, on the spec that replaced it.

    Live proof: job 261 (`npa-wf-multi-tokenfactory-rollout-judge-combo-d4798e41`) — the GPU stage
    rendered two real MP4 episodes, and the hosted judge scored exactly that prefix from CPU.
    See EVIDENCE.md §R34.
    """

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(
        ROOT
        / "npa"
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "tokenfactory-rollout-judge-combo.yaml"
    )
    plan = build_plan(spec, run_id="rollout-judge-test")
    steps = {step.state: step for step in plan.steps if step.argv}

    assert sorted(steps) == ["judge", "rollout-gpu"]
    # Stage 1 rolls out on a GPU, in the stage's own pod.
    assert steps["rollout-gpu"].tool_ref == "workbench.lerobot.policy_rollout"
    assert "accelerators" in spec.resources[steps["rollout-gpu"].resources]
    # Stage 2 is the hosted backend, holding no GPU.
    assert steps["judge"].tool_ref == "workbench.vlm_eval.run"
    assert "accelerators" not in spec.resources[steps["judge"].resources]
    assert spec.config["vlm_backend"] == "api"
    # And the judge reads EXACTLY what the rollout wrote — the point of the combo.
    rollout_argv, judge_argv = steps["rollout-gpu"].argv, steps["judge"].argv
    rollouts = rollout_argv[rollout_argv.index("--rollouts-s3-uri") + 1]
    assert judge_argv[judge_argv.index("--input-path") + 1] == rollouts


def test_the_retired_rollout_judge_template_is_gone() -> None:
    assert not ROLLOUT_JUDGE_YAML.exists(), "tokenfactory-rollout-judge.yaml came back"


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


def test_sweep_runner_full_mode_resolves_rank_root_without_keyerror(monkeypatch) -> None:
    """Regression: full-sweep mode must derive rank_root from sweep_root, not variant_uris."""
    module = _load_sweep_runner()
    captured: dict[str, str] = {}

    monkeypatch.setattr(module, "_hydrate_credentials", lambda: None)
    monkeypatch.setattr(module, "_design_sweep", lambda *a, **k: {"status": "completed"})
    monkeypatch.setattr(
        module,
        "_launch_variants",
        lambda variants: [{"id": v["id"], "uri": v["output_uri"]} for v in variants],
    )

    def _fake_rank(*, objective, runs, rank_root, model, max_tokens):
        captured["rank_root"] = rank_root
        return {"status": "completed", "ranked_variants": [r["id"] for r in runs]}

    monkeypatch.setattr(module, "_rank_runs", _fake_rank)

    args = module._parse_args(["--run-id", "demo", "--num-variants", "2", "--bucket", "s3://b/sweeps"])
    plan = module.build_plan(args)
    assert module._run(args, plan) == 0
    assert captured["rank_root"] == "s3://b/sweeps/demo/ranking"


def test_sweep_runner_job_names_track_resolved_run_id() -> None:
    """Without --run-id, per-variant Job names must use the resolved timestamped
    run_id (matching sweep_root), not a fixed 'tf-sweep' literal that collides
    across sweeps."""
    module = _load_sweep_runner()
    args = module._parse_args(["--num-variants", "2", "--bucket", "s3://b/tf"])
    plan = module.build_plan(args)

    run_id = plan["run_id"]
    assert run_id.startswith("tf-sim-sweep-")
    assert plan["sweep_root"].endswith(run_id)

    job_names = []
    for variant in plan["variants"]:
        cmd = variant["train_command"]
        job_names.append(cmd[cmd.index("--job-name") + 1])

    # Job names embed the (sanitized) resolved run_id, so they are unique per
    # sweep and never the bare fallback literal.
    sanitized_run_id = run_id.lower()
    for name in job_names:
        assert name.startswith(sanitized_run_id)
        assert not name.startswith("tf-sweep-v")
    assert len(set(job_names)) == len(job_names)


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


def test_scene_to_rollout_judge_spec_chains_reason_to_judge() -> None:
    """The three-stage chain, on the spec that replaced the template.

    Live proof: job 262 (`npa-wf-multi-tokenfactory-scene-to-rollout-judge-c9b64b65`), all three
    stages SUCCEEDED and the judge's task literally contained the reasoner's analysis.
    See EVIDENCE.md §R35.
    """

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(
        ROOT
        / "npa"
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "tokenfactory-scene-to-rollout-judge.yaml"
    )
    plan = build_plan(spec, run_id="scene-judge-test")
    steps = {step.state: step for step in plan.steps if step.argv}

    assert sorted(steps) == ["rollout-gpu", "scene-judge", "scene-reason"]
    # Only the middle stage holds a GPU.
    assert "accelerators" not in spec.resources[steps["scene-reason"].resources]
    assert "accelerators" in spec.resources[steps["rollout-gpu"].resources]
    assert "accelerators" not in spec.resources[steps["scene-judge"].resources]

    reason_argv = steps["scene-reason"].argv
    rollout_argv = steps["rollout-gpu"].argv
    judge_argv = steps["scene-judge"].argv
    # The judge scores what the rollout rendered ...
    rollouts = rollout_argv[rollout_argv.index("--rollouts-s3-uri") + 1]
    assert judge_argv[judge_argv.index("--input-path") + 1] == rollouts
    # ... against the plan the reasoner wrote. Without this link the third stage is decorative.
    plan_uri = reason_argv[reason_argv.index("--output-path") + 1]
    assert judge_argv[judge_argv.index("--task-from") + 1] == f"{plan_uri}scene_reasoning.json"


def test_the_retired_scene_judge_template_is_gone() -> None:
    assert not SCENE_JUDGE_YAML.exists(), "tokenfactory-scene-to-rollout-judge.yaml came back"


def test_combo_specs_have_no_hardcoded_infra_ids() -> None:
    """Every combo is a spec now, and a spec parameterises infra instead of placeholdering it."""

    specs = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
    for name in (
        "tokenfactory-rollout-judge-combo.yaml",
        "tokenfactory-scene-to-rollout-judge.yaml",
        "tokenfactory-train-triage.yaml",
    ):
        text = (specs / name).read_text(encoding="utf-8")
        assert "bucket: example-bucket" in text, name
        # No registry id anywhere: images come from --registry / the image resolver.
        assert "nebius.cloud/" not in text, name


# --- train-triage SkyPilot YAML (k8s GPU train -> hosted Token Factory triage) -


def test_train_triage_spec_is_two_stage_gpu_then_hosted_triage() -> None:
    """The retired template's contract, asserted on the spec that replaced it.

    Live proof: job 256 (`npa-wf-multi-tokenfactory-train-triage-6732d78a`) — train-gpu
    SUCCEEDED on the LeRobot image and produced a real 206 MB checkpoint, then the CPU triage
    stage wrote a 2,529-character report from that run's artifacts. See EVIDENCE.md §R32–R33.
    """

    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(
        ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "tokenfactory-train-triage.yaml"
    )
    plan = build_plan(spec, run_id="train-triage-test")
    steps = {step.state: step for step in plan.steps if step.argv}

    assert sorted(steps) == ["train-gpu", "triage"]
    # Stage 1 is a genuine GPU training run, in the stage's own pod.
    assert steps["train-gpu"].tool_ref == "workbench.lerobot.policy_train"
    assert "accelerators" in spec.resources[steps["train-gpu"].resources]
    # Stage 2 is zero-GPU hosted triage over exactly what stage 1 wrote.
    assert steps["triage"].tool_ref == "workbench.token_factory.triage"
    assert "accelerators" not in spec.resources[steps["triage"].resources]
    train_argv, triage_argv = steps["train-gpu"].argv, steps["triage"].argv
    artifacts = train_argv[train_argv.index("--artifacts-s3-uri") + 1]
    assert artifacts.startswith("s3://")
    assert triage_argv[triage_argv.index("--artifacts-uri") + 1] == artifacts


def test_the_retired_train_triage_template_is_gone() -> None:
    assert not TRAIN_TRIAGE_YAML.exists(), "tokenfactory-train-triage.yaml came back"


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
        assert "NEBIUS_TOKEN_FACTORY_KEY" in full_text, f"{path.name} never references the Token Factory key"


def test_sdk_exposes_workflow_submit_for_combo_yamls() -> None:
    """The SDK can submit the combo YAMLs via npa.workflow.submit."""
    from npa import workflow

    assert "submit" in workflow.__all__
    assert callable(workflow.submit)


def test_sdk_workflow_submit_delegates_to_orchestrator(mocker, monkeypatch) -> None:
    """npa.workflow.submit forwards a combo YAML to the SkyPilot orchestrator.

    Every combo is a spec now, so this submits one — `workflow.submit` accepts both formats and
    routes on the apiVersion (DESIGN.md §D5).
    """
    import types

    from npa import workflow

    # Rendering a spec needs somewhere for a stage to get npa from; the submit itself is mocked.
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    fake = types.SimpleNamespace(status="SUBMITTED", job_id="job-1")
    submit_mock = mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow", return_value=fake
    )

    workflow.submit(
        ROOT
        / "npa"
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "tokenfactory-scene-to-rollout-judge.yaml",
        run_id="rj-test",
        secret_env=["NEBIUS_TOKEN_FACTORY_KEY", "AWS_ACCESS_KEY_ID"],
        # Clear the workbench image pins: resolving them would mint a registry token.
        image="none",
    )

    submit_mock.assert_called_once()
    assert submit_mock.call_args.args[1] == "rj-test"
    assert "NEBIUS_TOKEN_FACTORY_KEY" in submit_mock.call_args.kwargs["secret_envs"]


def test_sdk_exposes_token_factory_and_vlm_eval_building_blocks() -> None:
    """The hosted Token Factory stages are SDK-callable building blocks."""
    from npa.sdk.workbench import token_factory, vlm_eval

    for name in ("generate", "reason", "caption", "verify"):
        assert callable(getattr(token_factory, name))
    assert hasattr(vlm_eval, "run") or hasattr(vlm_eval, "benchmark")
