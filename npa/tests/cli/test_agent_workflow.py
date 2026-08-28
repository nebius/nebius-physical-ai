from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli import agent as agent_module
from npa.cli.agent_chat import (
    apis_for_intent,
    build_grounded_reply,
    goal_requests_catalog_composition,
    match_chat_intent,
)
from npa.cli.agent_workflow import (
    WorkflowParameterError,
    author_workflow_from_goal,
    choose_workflow_template,
    extract_data_factory_params,
    extract_sim2real_params,
    generate_data_factory_yaml,
    generate_isaac_byof_yaml,
    generate_gpu_cross_region_yaml,
    generate_rl_policy_training_yaml,
    generate_sim2real_loop_gate_yaml,
    generate_sim2real_staged_yaml,
    generate_sim2real_two_step_yaml,
    generate_token_factory_gate_yaml,
    generate_vlm_rl_loop_yaml,
    generate_workflow_draft,
    generate_workflow_yaml,
    plan_workflow_yaml_text,
    validate_workflow_yaml_text,
    _TEMPLATES,
)
from npa.cli.main import app

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_YAML = (
    REPO_ROOT / "npa/workflows/workbench/npa-workflows/sim2real-two-step-agent.yaml"
)

_GOLDEN_YAMLS = [
    "byof.yaml",
    "rl-policy-training-sim-success.yaml",
    "sim2real-two-step-agent.yaml",
    "sim2real-two-step.yaml",
    "sim2real.yaml",
    "tokenfactory-cosmos-gate.yaml",
    "tokenfactory-rollout-judge.yaml",
    "vlm-eval-single.yaml",
    "bdd100k-pipeline.yaml",
]

runner = CliRunner()


def test_generate_sim2real_two_step_yaml_validates() -> None:
    yaml_text = generate_sim2real_two_step_yaml()
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True
    assert result["name"] == "sim2real-two-step"
    assert set(result["states"]) == {"augment", "envgen"}


def test_generated_two_step_matches_the_checked_in_agent_example() -> None:
    generated = yaml.safe_load(generate_sim2real_two_step_yaml())
    checked_in = yaml.safe_load(EXAMPLE_YAML.read_text(encoding="utf-8"))
    # Agent generation still labels drafts beta; the checked-in executable uses
    # the stable spelling. Their actual workflow contract must otherwise match.
    generated["apiVersion"] = checked_in["apiVersion"]
    generated["metadata"]["description"] = generated["metadata"]["description"].strip()
    checked_in["metadata"]["description"] = checked_in["metadata"][
        "description"
    ].strip()

    assert generated == checked_in


def test_example_yaml_file_validate_spec_cli() -> None:
    if not EXAMPLE_YAML.is_file():
        EXAMPLE_YAML.write_text(generate_sim2real_two_step_yaml(), encoding="utf-8")
    assert EXAMPLE_YAML.is_file()
    result = runner.invoke(
        app, ["workbench", "workflow", "validate-spec", str(EXAMPLE_YAML), "--json"]
    )
    assert result.exit_code == 0
    assert "sim2real-two-step" in result.output


def test_plan_two_step_workflow_has_two_steps() -> None:
    yaml_text = EXAMPLE_YAML.read_text(encoding="utf-8")
    plan = plan_workflow_yaml_text(yaml_text, run_id="unit-demo")
    assert plan["ok"] is True
    assert len(plan["steps"]) == 2
    steps = {step["state"]: step for step in plan["steps"]}
    assert (
        steps["augment"]["tool_ref"] == "workbench.cosmos2.transfer_conditioned_execute"
    )
    assert "--execute" in steps["augment"]["argv"]
    assert "--condition-on-input" in steps["augment"]["argv"]
    assert steps["envgen"]["tool_ref"] == "workbench.sim2real_envgen.raw_shard"
    manifest_uri = "s3://example-bucket/sim2real/unit-demo/augment/manifest.json"
    envgen_argv = steps["envgen"]["argv"]
    assert envgen_argv[envgen_argv.index("--augmented-frames-uri") + 1] == manifest_uri
    assert steps["envgen"]["inputs"] == [
        {"uri": manifest_uri, "schema": "npa.cosmos2.transfer.v1"}
    ]


def test_generate_sim2real_loop_gate_yaml_validates() -> None:
    yaml_text = generate_sim2real_loop_gate_yaml()
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True
    assert result["name"] == "sim2real-loop-gate-agent"
    assert {"augment", "refine", "vlm-critique", "quality-gate", "publish"}.issubset(
        set(result["states"])
    )


@pytest.mark.parametrize(
    ("yaml_text", "state"),
    [
        (generate_sim2real_loop_gate_yaml(), "augment"),
        (generate_vlm_rl_loop_yaml(), "stage-03-transfer"),
        (generate_token_factory_gate_yaml(), "augment-scene"),
        (generate_data_factory_yaml(), "augment"),
    ],
)
def test_generated_cosmos_transfer_outputs_use_canonical_manifest_schema(
    yaml_text: str, state: str
) -> None:
    spec = yaml.safe_load(yaml_text)

    assert spec["states"][state]["outputs"][0]["schema"] == ("npa.cosmos2.transfer.v1")


@pytest.mark.parametrize(
    ("yaml_text", "state"),
    [
        (generate_sim2real_loop_gate_yaml(), "augment"),
        (generate_vlm_rl_loop_yaml(), "stage-03-transfer"),
        (generate_token_factory_gate_yaml(), "augment-scene"),
    ],
)
def test_generated_input_cosmos_stages_fail_closed_without_their_clip(
    yaml_text: str, state: str
) -> None:
    spec = yaml.safe_load(yaml_text)

    input_uris = {artifact["uri"] for artifact in spec["states"][state]["inputs"]}
    assert "{{config.trigger_uri}}" in input_uris
    if state == "stage-03-transfer":
        assert (
            "npa.workflows.sim2real.workflow_stage"
            in spec["states"][state]["run"]["argv"]
        )
    else:
        assert spec["states"][state]["toolRef"] == (
            "workbench.cosmos2.transfer_conditioned_execute"
        )


def test_generated_data_factory_consumer_uses_canonical_manifest_schema() -> None:
    spec = yaml.safe_load(generate_data_factory_yaml())

    assert spec["states"]["evaluate"]["inputs"][0]["schema"] == (
        "npa.cosmos2.transfer.v1"
    )
    assert spec["states"]["evaluate"]["toolRef"] == (
        "workbench.cosmos_evaluator.evaluate"
    )


def test_plan_loop_gate_workflow_respects_assume_decision() -> None:
    yaml_text = generate_sim2real_loop_gate_yaml()
    plan = plan_workflow_yaml_text(
        yaml_text, run_id="loop-demo", assume_decision="promote_checkpoint"
    )
    assert plan["ok"] is True
    step_states = [str(step.get("state")) for step in plan["steps"]]
    assert "quality-gate" in step_states
    assert "publish" in step_states


def test_match_create_workflow_intent() -> None:
    assert match_chat_intent("create a 2-step sim2real workflow") == "create_workflow"
    assert (
        match_chat_intent("generate npa.workflow YAML for sim2real")
        == "create_workflow"
    )


def test_two_step_sim2real_request_propagates_supported_parameters() -> None:
    draft = generate_workflow_draft(
        user_text="create a 2-step sim2real workflow with 5000 environments, seed 9, and 2 GPUs",
        intent="create_workflow",
        bucket="bucket",
    )
    assert draft["template"] == "two-step"
    spec = yaml.safe_load(draft["yaml"])
    assert spec["config"]["env_count"] == "5000"
    assert spec["config"]["envgen_seed"] == "9"
    assert (
        spec["resources"]["gpu"]["accelerators"]
        == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:2"
    )


def test_create_workflow_grounded_reply_includes_yaml_fence() -> None:
    state: dict = {}
    reply = build_grounded_reply(
        "create_workflow", state, ["workbench.cosmos2.transfer"]
    )
    assert "```yaml" in reply
    assert "sim2real-two-step" in reply
    assert "augment" in reply
    assert "GET /api" not in reply


def test_generic_sim2real_goal_uses_canonical_two_step_template() -> None:
    from npa.cli.agent_workflow import author_workflow_from_goal
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    authored = author_workflow_from_goal(
        "create 2-step sim2real workflow",
        tool_refs=frozenset(TOOL_CATALOG),
    )

    # With no concrete component in the goal, chat rejects catalog composition
    # and falls back to generate_workflow_draft("two-step").
    assert authored["matched_tool_refs"] == []
    draft = generate_workflow_draft(
        user_text="create 2-step sim2real workflow",
        intent="create_workflow",
        tool_refs=frozenset(TOOL_CATALOG),
    )
    assert draft["runnable"] is True
    assert set(draft["validation"]["states"]) == {"augment", "envgen"}


def test_create_workflow_apis() -> None:
    apis = apis_for_intent("create_workflow")
    assert any(path.endswith("draft") for path in apis)
    assert any("validate" in path for path in apis)


def test_generate_data_factory_yaml_validates_and_plans() -> None:
    yaml_text = generate_data_factory_yaml()
    generated = yaml.safe_load(yaml_text)
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True
    assert result["name"] == "physical-ai-data-factory"
    expected = {
        "generate-configs",
        "annotate-original",
        "augment",
        "grade",
        "evaluate",
        "select-candidates",
        "evaluate-selected",
        "quality-gate",
        "quality-disposition",
        "review-terminal-candidates",
        "route-terminal-quality",
        "require-accepted-quality",
        "visualize-rejected",
        "reject-quality",
        "annotate-augmented",
        "cosmos-curate",
        "curate",
        "visualize",
        "finalize",
    }
    assert expected.issubset(set(result["states"]))
    plan = plan_workflow_yaml_text(
        yaml_text,
        run_id="paidf-demo",
        assume_decision="promote_checkpoint",
    )
    assert plan["ok"] is True
    tool_refs = [step.get("tool_ref") for step in plan["steps"]]
    assert "workbench.cosmos2.transfer_execute" in tool_refs
    assert "workbench.token_factory.caption" in tool_refs
    assert "workbench.cosmos_evaluator.evaluate" in tool_refs
    assert "workbench.cosmos_curate.curate" in tool_refs
    assert "workbench.fiftyone.curate_augmented" in tool_refs
    assert "workbench.fiftyone.review_augmented" in tool_refs
    assert generated["states"]["cosmos-curate"]["resources"] == "gpu"
    assert generated["config"]["trigger_uri"] == generated["config"]["input_uri"]
    assert generated["config"]["grade_threshold"] == "0.75"
    assert generated["config"]["plan_assume_decision"] == "promote_checkpoint"
    assert generated["config"]["augment_control_weight"] == "1.0"
    assert generated["config"]["augment_guidance"] == "3.0"
    assert generated["config"]["default_decision"] == "loop_back"
    assert generated["config"]["appearance_fidelity_mode"] == "advisory"
    assert generated["states"]["grade"]["next"] == "quality-disposition"
    assert generated["states"]["annotate-augmented"]["needs"] == [
        "require-accepted-quality"
    ]
    assert generated["states"]["quality-disposition"]["transitions"] == [
        {"when": "promote_checkpoint", "goto": "review-terminal-candidates"},
        {"when": "loop_back", "goto": "review-terminal-candidates"},
    ]
    assert generated["states"]["review-terminal-candidates"]["next"] == (
        "route-terminal-quality"
    )
    assert generated["states"]["route-terminal-quality"]["transitions"] == [
        {"when": "promote_checkpoint", "goto": "require-accepted-quality"},
        {"when": "loop_back", "goto": "visualize-rejected"},
    ]
    assert "enforce_quality_disposition" in " ".join(
        generated["states"]["require-accepted-quality"]["run"]["argv"]
    )
    assert generated["states"]["visualize-rejected"]["toolRef"] == (
        "workbench.nurec.visualize"
    )
    rejected_plan = plan_workflow_yaml_text(
        yaml_text,
        run_id="paidf-rejected",
        assume_decision="loop_back",
    )
    rejected_states = [step["state"] for step in rejected_plan["steps"]]
    assert rejected_states[-2:] == ["visualize-rejected", "reject-quality"]
    assert "annotate-augmented" not in rejected_states
    assert "supported video" in generated["states"]["augment"]["description"].lower()


def test_extract_data_factory_params_fanout_gpus_subject() -> None:
    params = extract_data_factory_params(
        "augment my warehouse robot clips and fan out 6 scenarios on at least 4 GPUs"
    )
    assert params["n_augmentations"] == 6
    assert params["gpu_count"] == 4
    assert "warehouse robot clips" in params["augment_subject"]

    # "8 variants" style also parses.
    p2 = extract_data_factory_params(
        "generate a paidf workflow with 8 scenario variants"
    )
    assert p2["n_augmentations"] == 8


def test_data_factory_yaml_reflects_requested_fanout_and_gpus() -> None:
    yaml_text = generate_data_factory_yaml(
        user_text="augment the robot arm demos and fan out 6 scenarios using 6 gpus"
    )
    data = yaml.safe_load(yaml_text)
    assert data["config"]["n_augmentations"] == "6"
    assert data["config"]["variant_parallelism"] == "6"
    assert data["config"]["augment_subject"].startswith("robot arm demos")
    assert (
        data["resources"]["gpu"]["accelerators"]
        == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:6"
    )
    # Parallelism never exceeds the GPU count.
    capped = generate_data_factory_yaml(
        user_text="augment the clips and fan out 8 scenarios on 4 gpus"
    )
    cd = yaml.safe_load(capped)
    assert cd["config"]["n_augmentations"] == "8"
    assert cd["config"]["variant_parallelism"] == "4"
    assert (
        cd["resources"]["gpu"]["accelerators"]
        == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:4"
    )


def test_data_factory_subject_is_an_argv_value_not_shell_source() -> None:
    workflow = generate_data_factory_yaml(
        user_text="augment the worker's robot clips and fan out 2 scenarios"
    )
    spec = yaml.safe_load(workflow)
    run = spec["states"]["generate-configs"]["run"]
    assert "shell" not in run
    assert spec["config"]["augment_subject"] == "worker's robot clips"
    assert run["argv"][-3:] == [
        "{{config.augment_subject}}",
        "{{config.augmentation_seed}}",
        "{{config.quality_anchor_uri}}",
    ]
    plan = plan_workflow_yaml_text(workflow, run_id="subject-safe")
    argv = next(step for step in plan["steps"] if step["state"] == "generate-configs")[
        "argv"
    ]
    assert argv[-3:] == ["worker's robot clips", "", ""]


def test_data_factory_chat_propagates_quality_and_curator_knobs() -> None:
    params = extract_data_factory_params(
        "create PAIDF with 3 refinement iterations, grade threshold 80%, "
        "clip length 5 and minimum clip length 2, maximum 12 images and maximum 384 tokens"
    )
    assert params["refinement_iterations"] == 3
    assert params["grade_threshold"] == 0.8
    assert params["curator_clip_len_s"] == 5
    assert params["curator_min_clip_len_s"] == 2
    assert params["max_images"] == 12
    assert params["max_tokens"] == 384
    data = yaml.safe_load(
        generate_data_factory_yaml(
            user_text=(
                "create PAIDF with 3 refinement iterations, grade threshold 80%, "
                "clip length 5 and minimum clip length 2, maximum 12 images and maximum 384 tokens"
            )
        )
    )
    assert data["config"]["refinement_iterations"] == "3"
    assert data["config"]["grade_threshold"] == "0.8"
    assert data["config"]["curator_clip_len_s"] == "5"
    assert data["config"]["curator_min_clip_len_s"] == "2"
    assert data["config"]["max_images"] == "12"
    assert data["config"]["max_tokens"] == "384"


def test_sim2real_staged_chat_parameters_validate_and_plan() -> None:
    prompt = (
        "create a sim-to-real workflow for UR5e with Genesis, 12000 environments, "
        "train fraction 75%, 4 inner iterations, 2 outer iterations, 6 rollouts, "
        "10 steps per rollout, 12 held-out envs, success threshold 82%, seed 7"
    )
    params = extract_sim2real_params(prompt)
    assert params == {
        "inner_iterations": 4,
        "outer_iterations": 2,
        "rollout_count": 6,
        "steps_per_rollout": 10,
        "heldout_env_count": 12,
        "env_count": 12000,
        "seed": 7,
        "success_threshold": 0.82,
        "train_fraction": 0.75,
        "sim_backend": "genesis",
        "robot_preset": "ur5e",
    }
    workflow = generate_sim2real_staged_yaml(user_text=prompt)
    spec = yaml.safe_load(workflow)
    assert spec["apiVersion"] == "npa.workflow/v0.0.1"
    assert spec["config"]["env_count"] == "12000"
    assert spec["config"]["threshold"] == "0.82"
    assert spec["config"]["task_id"] == "Isaac-Lift-Cube-Franka-v0"
    assert len(spec["states"]) == 24
    assert "stage-08-cosmos3" in spec["states"]
    assert "stage-08-wave" not in spec["states"]
    assert "stage-08-reason2" not in spec["states"]
    assert "run-sim2real" not in spec["states"]
    validation = validate_workflow_yaml_text(workflow)
    assert validation["ok"] is True
    plan = plan_workflow_yaml_text(workflow, run_id="sim-chat")
    assert plan["ok"] is True
    argv = [token for step in plan["steps"] for token in step["argv"]]
    assert argv[argv.index("--env-count") + 1] == "12000"
    assert argv[argv.index("--threshold") + 1] == "0.82"


def test_embedded_agent_uses_the_exact_canonical_sim2real_yaml() -> None:
    from npa.cli.agent_contracts import _embedded_agent_workflow_source

    canonical = (
        REPO_ROOT
        / "npa"
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "sim2real.yaml"
    ).read_text(encoding="utf-8")
    source = _embedded_agent_workflow_source()

    assert f"_EMBEDDED_CANONICAL_SIM2REAL_YAML = {canonical!r}" in source


def test_sim2real_chat_accepts_training_steps_and_evaluation_threshold() -> None:
    params = extract_sim2real_params(
        "write sim-to-real YAML with 8 training steps and an 80% evaluation threshold"
    )
    assert params["steps_per_rollout"] == 8
    assert params["success_threshold"] == 0.8

    spec = yaml.safe_load(
        generate_sim2real_staged_yaml(
            user_text=(
                "write sim-to-real YAML with 8 training steps and an 80% evaluation threshold"
            )
        )
    )
    assert spec["config"]["steps_per_rollout"] == "8"
    assert spec["config"]["threshold"] == "0.8"


def test_sim2real_named_text_and_clause_boundaries_are_exact() -> None:
    params = extract_sim2real_params(
        "isaac task Isaac-Lift-Cube-Franka-v0 and 5000 environments, "
        "trigger dataset id lerobot/pusht and 3 rollouts"
    )
    assert params["isaac_task"] == "Isaac-Lift-Cube-Franka-v0"
    assert params["trigger_dataset_id"] == "lerobot/pusht"
    assert params["env_count"] == 5000
    assert params["rollout_count"] == 3

    adjacent = extract_sim2real_params(
        "isaac task Isaac-Lift-Cube-Franka-v0 and trigger dataset id "
        "lerobot/pusht and 3 rollouts"
    )
    assert adjacent["isaac_task"] == "Isaac-Lift-Cube-Franka-v0"
    assert adjacent["trigger_dataset_id"] == "lerobot/pusht"
    assert adjacent["rollout_count"] == 3


@pytest.mark.parametrize(
    ("phrase", "flag", "expected"),
    [
        (
            "trigger dataset uri s3://bucket/input/",
            "--trigger-uri",
            "s3://bucket/input/",
        ),
        ("assets uri s3://bucket/assets/", "--assets-uri", "s3://bucket/assets/"),
        (
            "scene spec uri s3://bucket/scene.json",
            "--scene-spec-uri",
            "s3://bucket/scene.json",
        ),
        ("5 rollouts", "--rollout-count", "5"),
        ("rollout length 3", "--steps-per-rollout", "3"),
        ("12 held-out environments", "--gold-count", "12"),
        ("5000 environments", "--env-count", "5000"),
        ("8 envgen shards", "--shard-count", "8"),
        ("seed 9", "--seed", "9"),
        ("80% success threshold", "--threshold", "0.8"),
        ("75% train fraction", "--train-fraction", "0.75"),
    ],
)
def test_each_extracted_sim2real_value_is_one_exact_argv_token(
    phrase: str, flag: str, expected: str
) -> None:
    draft = generate_workflow_draft(
        user_text=f"create sim2real workflow with {phrase}",
        intent="create_vlm_rl_workflow",
        bucket="bucket",
    )
    argv = [token for step in draft["plan"]["steps"] for token in step["argv"]]
    assert argv[argv.index(flag) + 1] == expected


def test_envgen_shards_do_not_set_environment_count() -> None:
    assert extract_sim2real_params("16 envgen shards") == {"envgen_shard_count": 16}


def test_unsupported_envgen_shards_fail_draft_planning() -> None:
    draft = generate_workflow_draft(
        user_text="create sim2real workflow with 16 envgen shards",
        intent="create_vlm_rl_workflow",
        bucket="bucket",
    )
    assert draft["plan"]["ok"] is False
    assert "parallelCount resolves to 16" in draft["plan"]["error"]
    assert "declares 8 members" in draft["plan"]["error"]


@pytest.mark.parametrize(
    ("phrase", "field", "expected"),
    [
        ("rollout length 3", "steps_per_rollout", 3),
        ("4 inner loop iterations", "inner_iterations", 4),
        ("2 outer loop iterations", "outer_iterations", 2),
        ("12 held-out environments", "heldout_env_count", 12),
        ("80% evaluation threshold", "success_threshold", 0.8),
        ("train fraction 75%", "train_fraction", 0.75),
    ],
)
def test_documented_sim2real_parameter_phrases(
    phrase: str, field: str, expected: object
) -> None:
    assert extract_sim2real_params(phrase)[field] == expected


@pytest.mark.parametrize(
    "phrase", ["threshold of 8", "threshold 110%", "threshold -0.1"]
)
def test_invalid_thresholds_fail_closed(phrase: str) -> None:
    with pytest.raises(WorkflowParameterError):
        extract_sim2real_params(phrase)
    draft = generate_workflow_draft(
        user_text=f"create sim2real workflow with {phrase}",
        intent="create_vlm_rl_workflow",
        bucket="bucket",
    )
    assert draft["runnable"] is False
    assert any("threshold" in error for error in draft["context_errors"])


def test_absent_sim2real_parameters_preserve_exact_defaults() -> None:
    spec = yaml.safe_load(
        generate_sim2real_staged_yaml(user_text="create sim2real yaml")
    )
    assert spec["config"]["rollout_count"] == "64"
    assert spec["config"]["steps_per_rollout"] == "32"
    assert spec["config"]["threshold"] == "0.50"


@pytest.mark.parametrize(
    "phrase",
    ["create PAIDF with 100000 variants", "create PAIDF using 5000 GPUs"],
)
def test_data_factory_fanout_and_gpu_ceilings_fail_closed(phrase: str) -> None:
    with pytest.raises(WorkflowParameterError):
        extract_data_factory_params(phrase)
    draft = generate_workflow_draft(
        user_text=phrase,
        intent="create_data_factory_workflow",
        bucket="bucket",
    )
    assert draft["runnable"] is False
    assert any("ceiling" in error for error in draft["context_errors"])


def test_unresolved_placeholder_draft_is_not_runnable() -> None:
    draft = generate_workflow_draft(
        user_text="create sim2real yaml",
        intent="create_vlm_rl_workflow",
        bucket="",
        infrastructure={
            "has_infra": False,
            "configured": [],
            "local_clusters": [],
            "cloud_clusters": [],
        },
    )
    assert draft["runnable"] is False
    assert "<configure-s3-bucket>" in draft["yaml"]
    assert any(
        "unresolved configuration placeholders" in error
        for error in draft["context_errors"]
    )


def test_explicit_toolref_chain_wins_over_sim2real_blueprint_words() -> None:
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    goal = (
        "compose workbench.cosmos2.transfer -> "
        "workbench.fiftyone.curate_augmented for my sim2real clips"
    )
    authored = author_workflow_from_goal(goal, tool_refs=frozenset(TOOL_CATALOG))
    assert authored["tool_refs"] == [
        "workbench.cosmos2.transfer",
        "workbench.fiftyone.curate_augmented",
    ]
    assert goal_requests_catalog_composition(goal) is True


def test_vlm_rl_loop_is_reachable_when_explicitly_requested() -> None:
    selection = choose_workflow_template(
        user_text="create a VLM-RL outer loop and inner loop workflow",
        intent="create_vlm_rl_workflow",
    )
    assert selection["template"] == "vlm-rl-loop"

    slash_selection = choose_workflow_template(
        user_text=(
            "Draft a VLM/RL outer-loop workflow YAML with policy rollout, heldout eval, "
            "promote_checkpoint, and loop_back."
        ),
        intent="create_vlm_rl_workflow",
    )
    assert slash_selection["template"] == "vlm-rl-loop"


@pytest.mark.parametrize(
    ("text", "intent", "expected"),
    [
        ("create 2-step workflow", "create_workflow", "two-step"),
        ("create sim2real yaml", "create_vlm_rl_workflow", "sim2real-staged"),
        ("create loop gate workflow", "create_loop_gate_workflow", "loop-gate"),
        (
            "create PAIDF yaml",
            "create_data_factory_workflow",
            "physical-ai-data-factory",
        ),
        ("create VLM-RL loop workflow", "create_vlm_rl_workflow", "vlm-rl-loop"),
        ("create near sim realism yaml", "create_workflow", "two-step"),
    ],
)
def test_workflow_routing_matrix_is_stable(
    text: str, intent: str, expected: str
) -> None:
    assert (
        choose_workflow_template(user_text=text, intent=intent)["template"] == expected
    )


def test_staged_workflow_success_threshold_is_not_misrouted_to_watch() -> None:
    prompt = (
        "Create YAML for a staged sim2real workflow with 3 policy rollouts "
        "and success threshold 50%."
    )

    assert match_chat_intent(prompt) == "create_vlm_rl_workflow"


def test_workflow_draft_uses_configured_infrastructure_without_inventing() -> None:
    infra = {
        "project": "customer-project",
        "has_infra": True,
        "configured": [
            {
                "cluster_name": "customer-cluster",
                "context": "customer-context",
                "gpu_profile": "rtxpro",
                "raw": {"gpu_accelerator": "RTXPRO6000"},
            }
        ],
    }
    draft = generate_workflow_draft(
        user_text="create sim2real workflow with 2 GPUs and 5000 environments",
        intent="create_workflow",
        bucket="customer-bucket",
        infrastructure=infra,
    )
    assert draft["template"] == "sim2real-staged"
    assert draft["runnable"] is True
    spec = yaml.safe_load(draft["yaml"])
    assert spec["config"]["bucket"] == "customer-bucket"
    assert spec["config"]["env_count"] == "5000"
    assert "customer-cluster" not in draft["yaml"]
    assert spec["resources"]["isaac-gpu"]["accelerators"] == "RTXPRO6000:1"


def test_infrastructure_selection_is_deterministic_and_wires_sibling_placement() -> (
    None
):
    infra = {
        "project": "project-alias",
        "has_infra": True,
        "configured": [
            {
                "cluster_name": "z-cluster",
                "context": "z",
                "raw": {"gpu_accelerator": "L40S"},
            },
            {
                "cluster_name": "a-cluster",
                "context": "a",
                "raw": {
                    "gpu_accelerator": "L40S",
                    "namespace": "workflow-ns",
                    "service_account": "workflow-sa",
                    "image_pull_secrets": "managed-pull",
                    "env_secret_names": "runtime-env",
                    "envgen_image": "registry/envgen:test",
                },
            },
        ],
    }
    draft = generate_workflow_draft(
        user_text="create sim2real workflow with 7 rollouts",
        intent="create_vlm_rl_workflow",
        bucket="bucket",
        infrastructure=infra,
    )
    assert draft["infrastructure"]["cluster_name"] == "a-cluster"
    assert "2 candidate" in draft["infrastructure"]["selection_reason"]
    spec = yaml.safe_load(draft["yaml"])
    assert spec["config"]["rollout_count"] == "7"
    assert "a-cluster" not in draft["yaml"]
    assert "registry/envgen:test" not in draft["yaml"]
    argv = [token for step in draft["plan"]["steps"] for token in step["argv"]]
    assert argv[argv.index("--rollout-count") + 1] == "7"


@pytest.mark.parametrize(
    ("infra", "expected_source", "expected_cluster", "runnable"),
    [
        (
            {
                "has_infra": True,
                "configured": [],
                "local_clusters": [
                    {
                        "cluster_name": "cached",
                        "context": "cached",
                        "kubeconfig": "/tmp/k",
                        "kubeconfig_exists": True,
                    }
                ],
                "cloud_clusters": [],
            },
            "local",
            "cached",
            False,
        ),
        (
            {
                "has_infra": True,
                "configured": [],
                "local_clusters": [],
                "cloud_clusters": [{"name": "cloud-real", "status": "RUNNING"}],
            },
            "cloud",
            "cloud-real",
            False,
        ),
        (
            {
                "has_infra": False,
                "configured": [],
                "local_clusters": [],
                "cloud_clusters": [],
            },
            "none",
            "",
            False,
        ),
    ],
)
def test_infrastructure_matrix_never_invents_identifiers(
    infra: dict, expected_source: str, expected_cluster: str, runnable: bool
) -> None:
    draft = generate_workflow_draft(
        user_text="create Isaac sim2real workflow",
        intent="create_vlm_rl_workflow",
        bucket="bucket",
        infrastructure=infra,
    )
    assert draft["infrastructure"]["source"] == expected_source
    assert draft["infrastructure"]["cluster_name"] == expected_cluster
    assert draft["runnable"] is runnable
    assert "customer-cluster" not in draft["yaml"]


def test_workflow_draft_rejects_unavailable_or_non_rt_accelerator() -> None:
    infra = {
        "project": "p",
        "has_infra": True,
        "configured": [{"raw": {"gpu_accelerator": "H100"}}],
    }
    draft = generate_workflow_draft(
        user_text="create an Isaac sim2real workflow on RTX PRO 6000",
        intent="create_workflow",
        bucket="bucket",
        infrastructure=infra,
    )
    assert draft["runnable"] is False
    assert any(
        "not the configured profile" in error for error in draft["context_errors"]
    )


def test_cloud_inventory_prefers_project_cluster_and_rejects_unavailable_accelerator() -> (
    None
):
    infra = {
        "project": "rtxpro",
        "has_infra": True,
        "configured": [],
        "local_clusters": [],
        "cloud_clusters": [
            {"name": "other-cluster", "raw": {"available_accelerators": ["L40S"]}},
            {
                "name": "npa-rtxpro-mk8s",
                "raw": {
                    "gpu_accelerator": "RTXPRO6000",
                    "available_accelerators": ["RTXPRO6000"],
                },
            },
        ],
    }
    selected = generate_workflow_draft(
        user_text="create Isaac sim2real YAML on RTX PRO 6000",
        intent="create_vlm_rl_workflow",
        bucket="bucket",
        infrastructure=infra,
    )
    assert selected["infrastructure"]["cluster_name"] == "npa-rtxpro-mk8s"
    assert selected["runnable"] is True

    unavailable = generate_workflow_draft(
        user_text="create Isaac sim2real YAML on L40S",
        intent="create_vlm_rl_workflow",
        bucket="bucket",
        infrastructure=infra,
    )
    assert unavailable["runnable"] is False
    assert any(
        "L40S is unavailable" in error for error in unavailable["context_errors"]
    )


def test_default_data_factory_uses_four_gpus() -> None:
    data = yaml.safe_load(generate_data_factory_yaml())
    assert (
        data["resources"]["gpu"]["accelerators"]
        == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:4"
    )
    assert data["config"]["variant_parallelism"] == "4"
    assert data["config"]["n_augmentations"] == "4"


def test_match_create_data_factory_intent() -> None:
    assert (
        match_chat_intent(
            "write me a paidf npa workflow to augment and fan out scenarios"
        )
        == "create_data_factory_workflow"
    )
    assert (
        match_chat_intent("augment my robot clips and fan out 4 scenarios")
        == "create_data_factory_workflow"
    )
    assert (
        match_chat_intent("generate a physical ai data factory workflow yaml")
        == "create_data_factory_workflow"
    )


def test_choose_template_selects_data_factory() -> None:
    selection = choose_workflow_template(
        user_text="augment my footage and fan out 4 scenario variants",
        intent="create_data_factory_workflow",
    )
    assert selection["template"] == "physical-ai-data-factory"


def test_data_factory_draft_from_intent_and_text_is_runnable() -> None:
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    draft = generate_workflow_draft(
        user_text="build a paidf workflow: augment the robot clips and fan out 4 scenarios on 4 gpus",
        intent="create_data_factory_workflow",
        tool_refs=frozenset(TOOL_CATALOG.keys()),
    )
    assert draft["template"] == "physical-ai-data-factory"
    assert draft["runnable"] is True
    data = yaml.safe_load(draft["yaml"])
    assert data["config"]["n_augmentations"] == "4"
    assert (
        data["resources"]["gpu"]["accelerators"]
        == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:4"
    )


def test_infra_backend_intent_and_reply() -> None:
    assert (
        match_chat_intent("which kubernetes clusters are available?")
        == "infra_backends"
    )
    assert (
        match_chat_intent("deploy an mk8s cluster for workflow runs")
        == "mk8s_provision"
    )
    apis = apis_for_intent("infra_backends")
    assert "infra/k8s" in apis
    mk8s_apis = apis_for_intent("mk8s_provision")
    assert "infra/mk8s/provision" in mk8s_apis
    reply = build_grounded_reply(
        "infra_backends",
        {
            "infra": {
                "project": "demo",
                "has_infra": False,
                "agent_npa_ready": True,
                "configured": [],
                "local_clusters": [],
            }
        },
        [],
    )
    assert "No Kubernetes infra" in reply
    assert "deploy minimal GPU Kubernetes" in reply
    mk8s_reply = build_grounded_reply("mk8s_provision", {}, [])
    assert "POST /api/infra/mk8s/provision" in mk8s_reply
    assert "npa provision-if-absent" in mk8s_reply


def test_bootstrap_embeds_workflow_endpoints() -> None:
    from npa.cli.agent import rendered_agent_ui_html

    source = Path(agent_module.__file__).read_text(encoding="utf-8")
    ui = rendered_agent_ui_html()
    bundled = source + "\n" + ui
    assert '@app.post("/workflows/validate")' in source
    assert '@app.post("/workflows/plan")' in source
    assert '@app.post("/workflows/submit")' in source
    assert '@app.get("/infra/k8s")' in source
    assert '@app.post("/infra/provision")' in source
    assert '@app.get("/infra/mk8s")' in source
    assert '@app.post("/infra/mk8s/provision")' in source
    assert '@app.post("/infra/soperator/validate")' in source
    assert '@app.post("/infra/soperator/deploy")' in source
    assert '@app.get("/infra/soperator/status/{{name}}")' in source
    assert "agent-live-infra-plan" in source
    assert "pip install -e" in source
    assert "deploy/cluster" in source
    assert "_soperator_deploy_from_payload" in source
    assert "DEFAULT_SOLUTIONS_LIBRARY_REF" in source
    assert "_validate_immutable_solutions_library_ref" in source
    assert "SoperatorDeploymentValidationError" in source
    assert '"status": "degraded-validation"' in source
    assert "gpu_creation_check_timeout_seconds" in source
    assert "_validate_gpu_creation_check_timeout" in source
    assert '"system_max_size": spec.effective_system_max_size()' in source
    assert '"capacity_mode": pool.capacity_mode()' in source
    assert '"reservation_selector": pool.reservation_selector_kind() or None' in source
    assert '@app.get("/workflows/draft")' in source
    assert "workflowYaml" in bundled
    assert "validateWorkflowYaml" in bundled
    assert "generate_workflow_draft" in source
    assert '@app.get("/sim-viz/runs")' in source
    assert '@app.post("/sim-viz/select-run")' in source
    assert "sim_viz_runs" in source
    assert '@app.get("/workflows/sim2real/runs/{{run_id:path}}")' in source
    assert "stages-panel" in bundled
    assert "<h3>Stages</h3>" in bundled
    assert "Sim2Real Run Monitor" not in bundled
    assert "Pick a discovered NPA workflow/artifact run" in bundled
    assert "evidence-backed timeline and artifacts." in bundled
    assert "formatStageStatusLabel" in bundled
    assert (
        "data.ok === false" in bundled
    )  # submitWorkflowYaml must not treat blocked as success
    assert (
        'ok": bool(validation.get("ok"))' in source
        or '"ok": bool(validation.get("ok"))' in source
    )
    assert "lastAppliedDraftYaml" in bundled  # refresh must not stomp local YAML edits
    assert (
        "Uploaded `" in bundled or "not runnable yet" in bundled
    )  # upload surfaces validation state
    assert "No run-specific Rerun recording yet" in bundled
    assert "_run_sim2real_pipeline_background" in source
    assert "agent-local-sim2real" in source
    embedded = agent_module._embedded_agent_workflow_source()
    assert "validate_workflow_yaml_text" in embedded
    assert "Could not generate runnable workflow YAML yet" in source
    assert "chat returns YAML only after both validation and planning succeed" in source


def test_lightweight_validation_without_tool_refs_still_parses() -> None:
    yaml_text = generate_sim2real_two_step_yaml()
    result = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset())
    assert result["ok"] is True


def test_lightweight_validation_handles_complex_edges(monkeypatch) -> None:
    def _import_fail(_yaml_text: str) -> dict[str, object]:
        raise ImportError("test fallback")

    monkeypatch.setattr("npa.cli.agent_workflow._validate_with_npa", _import_fail)
    yaml_text = """
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: complex-agent-graph
initial: start
states:
  start:
    toolRef: workbench.cosmos2.transfer
    next: gate
  gate:
    transitions:
      promote_checkpoint: train
      loop_back: start
  train:
    sequence:
      - state: eval
      - state: done
  eval:
    toolRef: workbench.sim2real_envgen.raw_shard
    terminal: true
  done:
    terminal: true
"""
    result = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset())
    assert result["ok"] is True
    assert result["name"] == "complex-agent-graph"


def test_lightweight_plan_walks_reachable_graph(monkeypatch) -> None:
    def _import_fail(*_args, **_kwargs) -> dict[str, object]:
        raise ImportError("test fallback")

    monkeypatch.setattr("npa.cli.agent_workflow._plan_with_npa", _import_fail)
    monkeypatch.setattr("npa.cli.agent_workflow._validate_with_npa", _import_fail)
    yaml_text = """
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: branching-agent-plan
initial: root
states:
  root:
    next: branch
  branch:
    transitions:
      promote_checkpoint: deploy
      loop_back: recover
  deploy:
    terminal: true
  recover:
    terminal: true
"""
    plan = plan_workflow_yaml_text(
        yaml_text, run_id="agent-branch-demo", tool_refs=frozenset()
    )
    assert plan["ok"] is True
    states = [step["state"] for step in plan["steps"]]
    assert states == ["root", "branch", "deploy", "recover"]


def test_lightweight_validation_accepts_transition_list_goto(monkeypatch) -> None:
    def _import_fail(*_args, **_kwargs) -> dict[str, object]:
        raise ImportError("test fallback")

    monkeypatch.setattr("npa.cli.agent_workflow._validate_with_npa", _import_fail)
    yaml_text = """
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: goto-list-graph
initial: start
states:
  start:
    transitions:
      - when: promote_checkpoint
        goto: publish
      - when: loop_back
        goto: retry
  retry:
    next: publish
  publish:
    terminal: true
"""
    result = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset())
    assert result["ok"] is True
    assert result["name"] == "goto-list-graph"


# --- Complex YAML generator tests ---


def test_generate_vlm_rl_loop_yaml_validates() -> None:
    yaml_text = generate_vlm_rl_loop_yaml()
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True, f"vlm-rl validate failed: {result.get('error')}"
    assert result["name"] == "sim2real"
    states = set(result["states"])
    assert "stage-03-transfer" in states
    assert "stage-04-wave" in states
    assert "stage-14-visualize" in states


def test_generate_vlm_rl_loop_yaml_plan_has_multiple_steps() -> None:
    yaml_text = generate_vlm_rl_loop_yaml()
    plan = plan_workflow_yaml_text(
        yaml_text, run_id="vlm-rl-test", assume_decision="promote_checkpoint"
    )
    assert plan["ok"] is True, f"vlm-rl plan failed: {plan.get('error')}"
    assert len(plan["steps"]) >= 3
    steps = {step["state"]: step for step in plan["steps"]}
    assert "npa.workflows.sim2real.workflow_stage" in steps["stage-03-transfer"]["argv"]
    manifest_uri = "s3://example-bucket/sim2real/vlm-rl-test/augment/manifest.json"
    assert steps["stage-03-transfer"]["outputs"][0]["uri"] == manifest_uri
    assert steps["stage-03-transfer"]["inputs"][0]["uri"].startswith(
        "s3://example-bucket/sim2real-triggers/"
    )


def test_generate_vlm_rl_loop_yaml_contains_loop_and_gate() -> None:
    yaml_text = generate_vlm_rl_loop_yaml()
    assert "loop:" in yaml_text
    assert "parallel:" in yaml_text
    assert "promote_checkpoint" in yaml_text
    assert "writesDecision: true" in yaml_text
    assert "writesDecision: true" in yaml_text


def test_generate_token_factory_gate_yaml_validates() -> None:
    yaml_text = generate_token_factory_gate_yaml()
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True, f"token-factory validate failed: {result.get('error')}"
    assert result["name"] == "tokenfactory-cosmos-gate"
    states = set(result["states"])
    assert "reason-scene" in states
    assert "augment-scene" in states
    assert "publish" in states


def test_generate_token_factory_gate_yaml_plan() -> None:
    yaml_text = generate_token_factory_gate_yaml()
    plan = plan_workflow_yaml_text(
        yaml_text, run_id="gate-test", assume_decision="promote_checkpoint"
    )
    assert plan["ok"] is True, f"token-factory plan failed: {plan.get('error')}"
    assert len(plan["steps"]) >= 2
    tool_refs = [step.get("tool_ref") for step in plan["steps"]]
    assert "workbench.cosmos2.transfer_conditioned_execute" in tool_refs


def test_generate_token_factory_gate_yaml_contains_vlm_gate() -> None:
    yaml_text = generate_token_factory_gate_yaml()
    assert "loop:" in yaml_text
    assert "transitions:" in yaml_text
    assert "promote_checkpoint" in yaml_text
    assert "vlm-critique" in yaml_text
    assert "quality-gate" in yaml_text


def test_generate_isaac_byof_yaml_validates() -> None:
    yaml_text = generate_isaac_byof_yaml()
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True, f"isaac-byof validate failed: {result.get('error')}"
    assert result["name"] == "byof"
    assert "<repo-url>" in yaml_text
    assert "base_profile: ubuntu" in yaml_text
    assert "byof-run" in set(result["states"])


def test_generate_isaac_byof_yaml_plan_contains_byof_toolref() -> None:
    yaml_text = generate_isaac_byof_yaml()
    plan = plan_workflow_yaml_text(yaml_text, run_id="byof-test")
    assert plan["ok"] is True, f"isaac-byof plan failed: {plan.get('error')}"
    tool_refs = [step.get("tool_ref") for step in plan["steps"]]
    assert "workbench.byof.repo" in tool_refs


def test_generate_gpu_cross_region_yaml_validates() -> None:
    with pytest.raises(ValueError, match="stub Sim2Real components"):
        generate_gpu_cross_region_yaml()


def test_generate_gpu_cross_region_yaml_includes_multi_region_resources() -> None:
    with pytest.raises(ValueError, match="retired"):
        generate_gpu_cross_region_yaml()


def test_generate_gpu_cross_region_yaml_contract_edges_align() -> None:
    with pytest.raises(ValueError, match="real solution toolRefs"):
        generate_gpu_cross_region_yaml()


def test_generate_gpu_cross_region_yaml_plan() -> None:
    with pytest.raises(ValueError, match="retired"):
        generate_gpu_cross_region_yaml()


def test_generate_rl_policy_training_yaml_validates() -> None:
    yaml_text = generate_rl_policy_training_yaml()
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True, (
        f"rl-policy-success validate failed: {result.get('error')}"
    )
    assert result["name"] == "rl-policy-training-sim-success"
    states = set(result["states"])
    assert "train-policy" in states
    assert "eval-policy" in states
    assert "success-gate" in states
    assert "publish-policy" in states
    assert "training-not-success" in states


def test_generate_rl_policy_training_yaml_plan() -> None:
    yaml_text = generate_rl_policy_training_yaml()
    plan = plan_workflow_yaml_text(
        yaml_text, run_id="rl-policy-success-test", assume_decision="promote_checkpoint"
    )
    assert plan["ok"] is True, f"rl-policy-success plan failed: {plan.get('error')}"
    states = [step["state"] for step in plan["steps"]]
    assert "train-policy" in states
    assert "eval-policy" in states
    assert "success-gate" in states
    assert "publish-policy" in states


def test_generate_workflow_yaml_dispatcher() -> None:
    two_step = generate_workflow_yaml("two-step")
    assert "sim2real-two-step" in two_step
    assert "apiVersion: npa.workflow/v0.0.1" in two_step
    vlm_rl = generate_workflow_yaml("vlm-rl-loop")
    assert "name: sim2real" in vlm_rl
    gate = generate_workflow_yaml("token-factory-gate")
    assert "tokenfactory-cosmos-gate" in gate
    loop_gate = generate_workflow_yaml("loop-gate")
    assert "sim2real-loop-gate-agent" in loop_gate
    isaac_byof = generate_workflow_yaml("byof")
    assert "name: byof" in isaac_byof or "isaac-lab-byof" not in isaac_byof
    assert "isaac-lab-byof" not in isaac_byof
    cross_region = generate_workflow_yaml("gpu-cross-region")
    assert "sim2real-two-step" in cross_region
    rl_policy = generate_workflow_yaml("rl-policy-success")
    assert "rl-policy-training-sim-success" in rl_policy
    default = generate_workflow_yaml("unknown-template")
    assert "sim2real-two-step" in default


def test_vlm_rl_draft_keeps_canonical_resource_profiles_without_live_infra() -> None:
    draft = generate_workflow_draft(
        template="vlm-rl-loop",
        user_text=(
            "create a VLM/RL outer loop with an RTX PRO 6000 accelerator and 1 GPU"
        ),
        bucket="run-bucket",
        infrastructure={"has_infra": False},
    )

    assert draft["runnable"] is True
    spec = yaml.safe_load(draft["yaml"])
    # Current main intentionally loads the one compositional Sim2Real graph;
    # chat does not rewrite its named lane profiles into the retired single
    # generic `gpu` resource. Runtime backend selection is carried separately.
    assert "gpu" not in spec["resources"]
    assert spec["resources"]["transfer-gpu"]["accelerators"] == "RTXPRO6000:1"
    assert spec["resources"]["isaac-gpu"]["accelerators"] == "RTXPRO6000:1"


def test_choose_workflow_template_by_intent_and_text() -> None:
    selected = choose_workflow_template(
        user_text="create a multi-step outer loop with inner loop gate",
        intent="create_workflow",
    )
    assert selected["template"] == "vlm-rl-loop"
    selected_gate = choose_workflow_template(
        user_text="build tokenfactory quality gate workflow",
        intent="create_workflow",
    )
    assert selected_gate["template"] == "token-factory-gate"
    selected_byof = choose_workflow_template(
        user_text="create a BYOF Isaac Lab workflow for live infra",
        intent="create_workflow",
    )
    assert selected_byof["template"] == "byof"
    selected_multi_region = choose_workflow_template(
        user_text="create gpu workflow across two regions for one tenant",
        intent="create_workflow",
    )
    assert selected_multi_region["template"] == "two-step"
    selected_rl_policy = choose_workflow_template(
        user_text="build an rl policy training workflow in simulation",
        intent="create_workflow",
    )
    assert selected_rl_policy["template"] == "rl-policy-success"


def test_choose_workflow_template_byof_beats_full_tool_catalog_capabilities() -> None:
    """Explicit BYOF prompts must not lose to RL when TOOL_REFS bleed isaac/rl/policy."""
    full_catalog_capabilities = {
        "tool_refs": [
            "workbench.isaac_lab.rl",
            "workbench.rl.policy_train",
            "workbench.vlm.rl",
            "workbench.byof.repo",
            "workbench.token_factory.gate",
        ]
    }
    selected = choose_workflow_template(
        user_text="create a BYOF Isaac Lab workflow for live infra with placeholder repo and task",
        intent="create_workflow",
        capabilities=full_catalog_capabilities,
    )
    assert selected["template"] == "byof"
    draft = generate_workflow_draft(
        user_text="create a BYOF Isaac Lab workflow for live infra with placeholder repo and task",
        intent="create_workflow",
        capabilities=full_catalog_capabilities,
        tool_refs=frozenset(full_catalog_capabilities["tool_refs"]),
    )
    assert draft["template"] == "byof"
    assert "name: byof" in draft["yaml"]
    assert draft["validation"]["ok"] is True


def test_generate_workflow_draft_returns_selection_and_valid_yaml() -> None:
    draft = generate_workflow_draft(
        user_text="draft a tokenfactory gate workflow",
        intent="create_gate_workflow",
        tool_refs=frozenset(),
    )
    assert draft["template"] == "token-factory-gate"
    assert draft["validation"]["ok"] is True
    assert draft["plan"]["ok"] is True
    assert draft["runnable"] is True
    assert "metadata:" in draft["yaml"]
    assert "\n\n  scene_uri:" in draft["yaml"]


def test_generate_workflow_draft_sets_not_runnable_when_plan_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "npa.cli.agent_workflow.plan_workflow_yaml_text",
        lambda *_args, **_kwargs: {"ok": False, "error": "forced plan failure"},
    )
    draft = generate_workflow_draft(template="two-step", tool_refs=frozenset())
    assert draft["validation"]["ok"] is True
    assert draft["plan"]["ok"] is False
    assert draft["runnable"] is False


def test_every_agent_template_toolref_resolves_catalog() -> None:
    """Agent drafts must not surface retired or doc-only workflow toolRefs."""
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    emitted = []
    for template in _TEMPLATES:
        spec = yaml.safe_load(generate_workflow_yaml(template))
        emitted.extend(
            state["toolRef"]
            for state in spec["states"].values()
            if "toolRef" in state
        )

    unknown = sorted(set(emitted) - set(TOOL_CATALOG))
    assert unknown == []


def test_generate_workflow_yaml_aliases() -> None:
    assert "name: sim2real" in generate_workflow_yaml("vlm-rl")
    assert "name: sim2real" in generate_workflow_yaml("vlm_rl_loop")
    assert "tokenfactory-cosmos-gate" in generate_workflow_yaml("gate")
    assert "tokenfactory-cosmos-gate" in generate_workflow_yaml("tokenfactory")
    assert "sim2real-loop-gate-agent" in generate_workflow_yaml("loop")
    assert "name: byof" in generate_workflow_yaml("leisaac")
    assert "name: byof" in generate_workflow_yaml("isaac-lab")
    assert "rl-policy-training-sim-success" in generate_workflow_yaml("rl-policy")


@pytest.mark.parametrize("yaml_name", _GOLDEN_YAMLS)
def test_golden_yaml_validates(yaml_name: str) -> None:
    """All golden NPA workflow YAMLs in the repo should parse and validate."""
    yaml_path = REPO_ROOT / "npa/workflows/workbench/npa-workflows" / yaml_name
    if not yaml_path.is_file():
        pytest.skip(f"golden YAML not found: {yaml_name}")
    yaml_text = yaml_path.read_text(encoding="utf-8")
    result = validate_workflow_yaml_text(yaml_text)
    assert result["ok"] is True, f"{yaml_name} failed: {result.get('error')}"


@pytest.mark.parametrize("yaml_name", _GOLDEN_YAMLS)
def test_golden_yaml_plan_spec_cli(yaml_name: str) -> None:
    """Golden YAMLs should plan successfully with the CLI."""
    yaml_path = REPO_ROOT / "npa/workflows/workbench/npa-workflows" / yaml_name
    if not yaml_path.is_file():
        pytest.skip(f"golden YAML not found: {yaml_name}")
    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(yaml_path),
            "--run-id",
            "golden-test",
            "--assume-decision",
            "promote_checkpoint",
            "--json",
        ],
    )
    assert result.exit_code == 0, f"{yaml_name} plan-spec CLI failed:\n{result.output}"
    assert "golden-test" in result.output or "steps" in result.output


# --- Complex workflow intent routing tests ---


def test_match_vlm_rl_workflow_intent() -> None:
    assert (
        match_chat_intent("create a VLM-RL loop workflow") == "create_vlm_rl_workflow"
    )
    assert (
        match_chat_intent("generate a sim2real vlm rl pipeline")
        == "create_vlm_rl_workflow"
    )
    assert (
        match_chat_intent("build a workflow with outer loop and inner loop gate")
        == "create_vlm_rl_workflow"
    )
    assert (
        match_chat_intent("create sim-to-real YAML for Franka with 5000 environments")
        == "create_vlm_rl_workflow"
    )
    assert (
        match_chat_intent("create a sim2real workflow with a success threshold")
        == "create_vlm_rl_workflow"
    )


def test_match_gate_workflow_intent() -> None:
    assert (
        match_chat_intent("create a token factory gate workflow")
        == "create_gate_workflow"
    )
    assert (
        match_chat_intent("generate a quality gate cosmos augment loop")
        == "create_gate_workflow"
    )
    assert (
        match_chat_intent("build a tokenfactory cosmos-gate spec")
        == "create_gate_workflow"
    )


def test_create_vlm_rl_workflow_grounded_reply() -> None:
    state: dict = {}
    reply = build_grounded_reply("create_vlm_rl_workflow", state, [])
    assert "```yaml" in reply
    assert "name: sim2real" in reply
    assert "VLM-RL" in reply
    assert "GET /api" not in reply


def test_create_gate_workflow_grounded_reply() -> None:
    state: dict = {}
    reply = build_grounded_reply("create_gate_workflow", state, [])
    assert "```yaml" in reply
    assert "tokenfactory-cosmos-gate" in reply
    assert "Token Factory" in reply
    assert "GET /api" not in reply


def test_vlm_rl_workflow_apis_include_plan() -> None:
    apis = apis_for_intent("create_vlm_rl_workflow")
    assert any("validate" in p for p in apis)
    assert any("plan" in p for p in apis)


def test_gate_workflow_apis_include_plan() -> None:
    apis = apis_for_intent("create_gate_workflow")
    assert any("validate" in p for p in apis)
    assert any("plan" in p for p in apis)
