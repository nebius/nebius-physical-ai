from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "workflows" / "main" / "paidf-cosmos3.yaml"
runner = CliRunner()


@pytest.mark.parametrize("flags", [["--assume-decision", "promote_checkpoint"],
                                    ["--runtime", "--assume-decision", "promote_checkpoint"]])
def test_submit_requires_actual_runtime_decisions_before_side_effects(monkeypatch, flags):
    def unexpected(**kwargs):
        raise AssertionError("credential resolution must not be reached")

    monkeypatch.setattr("npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials", unexpected)
    result = runner.invoke(app, ["workbench", "workflow", "submit", str(SPEC),
                                "--run-id", "runtime-contract", *flags])
    assert result.exit_code == 1
    assert "reject --assume-decision for execution" in result.output


def test_source_overlay_is_selected_by_the_spec(monkeypatch):
    from npa.deploy.images import container_image_for_tool
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions, render_skypilot_yaml
    from npa.orchestration.npa_workflow.spec import load_spec

    monkeypatch.delenv("NPA_SRC_OVERLAY", raising=False)
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/staged-source.tar.gz")
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="overlay-contract", assume_decision="promote_checkpoint")
    rendered = render_skypilot_yaml(spec, plan, run_id="overlay-contract",
                                   options=SkypilotRenderOptions(registry="ghcr.io/nebius/nebius-physical-ai", materialize_registry_secrets=False))
    tasks = [task for task in yaml.safe_load_all(rendered) if task and "resources" in task]
    assert tasks
    assert all(task["envs"]["NPA_SRC_S3_URI"] == "s3://example-bucket/staged-source.tar.gz" for task in tasks)
    baked = [task for task in tasks if task["resources"].get("image_id")]
    assert baked
    assert all(task["envs"]["NPA_SRC_OVERLAY"] == "1" for task in baked)
    annotation = next(task for task in tasks if "paidf_cosmos3_annotation" in task["run"])
    assert annotation["resources"]["image_id"] == "docker:" + container_image_for_tool(
        "rerun-viewer", registry="ghcr.io/nebius/nebius-physical-ai")
    assert not annotation["resources"].get("accelerators")


def test_legacy_evaluator_argv_omits_new_options():
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(ROOT / "workflows/testing/physical-ai-data-factory.yaml")
    plan = build_plan(spec, run_id="legacy-contract", assume_decision="promote_checkpoint")
    for step in plan.steps:
        if step.tool_ref == "workbench.cosmos_evaluator.evaluate":
            assert "--alignment-mode" not in step.argv
            assert "--attribute-threshold" not in step.argv


def _doc() -> dict:
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))


def test_custom_appearance_profiles_reach_the_real_sampler(tmp_path):
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.workflows.data_factory_stages import generate_configs

    profile = {"lighting": "soft warm room lighting", "background": "gray work surface",
               "color_grade": "neutral balanced color palette",
               "surface_finish": "matte low-gloss work surface"}
    document = _doc()
    document["config"]["appearance_profiles_json"] = json.dumps([profile])
    document["config"]["configs_uri"] = str(tmp_path / "configs")
    document["config"]["images_uri"] = str(tmp_path / "input")
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(document))
    plan = build_plan(load_spec(path), run_id="custom-profiles",
                      assume_decision="promote_checkpoint")
    stage = next(step for step in plan.steps if step.state == "generate-configs")
    manifest = generate_configs(*stage.argv[3:])
    assert manifest["appearance_profiles"] == [profile]
    assert all(item["lighting"] == profile["lighting"] for item in manifest["augmentations"])


def test_invalid_appearance_profile_override_fails_during_planning():
    result = runner.invoke(app, ["workbench", "workflow", "plan-spec", str(SPEC),
                                "--run-id", "invalid-profiles", "--var",
                                'appearance_profiles_json=[{"lighting":"warm"}]', "--json"])
    assert result.exit_code != 0
    assert "appearance profile" in result.output


def test_edge_preset_reaches_generation_cli_and_settings(tmp_path, monkeypatch):
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.workflows import paidf_cosmos3

    document = _doc()
    document["config"]["transfer_edge_threshold"] = "very_low"
    document["config"]["transfer_rgb_weight"] = 0.5
    document["config"]["transfer_first_chunk_conditional_frames"] = "0"
    document["config"]["transfer_cfg_normalization"] = "enabled"
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(document))
    plan = build_plan(load_spec(path), run_id="edge-detail", assume_decision="promote_checkpoint")
    stage = next(step for step in plan.steps if step.state == "generate-variants")
    calls = []
    monkeypatch.setattr(paidf_cosmos3, "generate_variants", lambda *args, **kwargs: calls.append(kwargs) or {})
    result = runner.invoke(app, list(stage.argv[1:]))
    assert result.exit_code == 0, result.output
    assert calls[0]["transfer_edge_threshold"] == "very_low"
    assert calls[0]["transfer_rgb_weight"] == 0.5
    assert calls[0]["transfer_first_chunk_conditional_frames"] == 0
    assert calls[0]["transfer_cfg_normalization"] == "enabled"


def test_invalid_edge_preset_fails_during_planning():
    result = runner.invoke(app, ["workbench", "workflow", "plan-spec", str(SPEC),
                                "--run-id", "invalid-edge", "--var", "transfer_edge_threshold=auto", "--json"])
    assert result.exit_code != 0
    assert "transfer_edge_threshold" in result.output


def test_legacy_generation_config_uses_default_rgb_weight(tmp_path, monkeypatch):
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.workflows import paidf_cosmos3

    document = _doc()
    document["config"].pop("transfer_rgb_weight")
    document["config"].pop("transfer_first_chunk_conditional_frames")
    document["config"].pop("transfer_cfg_normalization")
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(document))
    plan = build_plan(load_spec(path), run_id="legacy-rgb", assume_decision="promote_checkpoint")
    stage = next(step for step in plan.steps if step.state == "generate-variants")
    calls = []
    monkeypatch.setattr(paidf_cosmos3, "generate_variants", lambda *args, **kwargs: calls.append(kwargs) or {})
    result = runner.invoke(app, list(stage.argv[1:]))
    assert result.exit_code == 0, result.output
    assert "--transfer-rgb-weight" not in stage.argv
    assert calls[0]["transfer_rgb_weight"] == 0.0
    assert "--transfer-first-chunk-conditional-frames" not in stage.argv
    assert calls[0]["transfer_first_chunk_conditional_frames"] == 1
    assert "--transfer-cfg-normalization" not in stage.argv
    assert calls[0]["transfer_cfg_normalization"] == "disabled"


@pytest.mark.parametrize("value", ["true", "false", "auto", "1", ""])
def test_invalid_cfg_normalization_fails_during_planning(value):
    result = runner.invoke(app, ["workbench", "workflow", "plan-spec", str(SPEC),
                                "--run-id", "invalid-normalization", "--var",
                                f"transfer_cfg_normalization={value}", "--json"])
    assert result.exit_code != 0
    assert "transfer_cfg_normalization" in result.output


def test_cfg_normalization_requires_native_transfer():
    result = runner.invoke(app, ["workbench", "workflow", "plan-spec", str(SPEC),
                                "--run-id", "normalization-without-edge", "--var",
                                "transfer_cfg_normalization=enabled", "--var",
                                "structural_control=none", "--json"])
    assert result.exit_code != 0
    assert "transfer_cfg_normalization requires structural_control=edge" in result.output


@pytest.mark.parametrize("value", ["-1", "2", "0.5", "true", "invalid"])
def test_invalid_first_chunk_conditioning_fails_during_planning(value):
    result = runner.invoke(app, ["workbench", "workflow", "plan-spec", str(SPEC),
                                "--run-id", "invalid-first-frame", "--var",
                                f"transfer_first_chunk_conditional_frames={value}", "--json"])
    assert result.exit_code != 0
    assert "transfer_first_chunk_conditional_frames" in result.output


def test_first_chunk_conditioning_requires_edge_transfer():
    result = runner.invoke(app, ["workbench", "workflow", "plan-spec", str(SPEC),
                                "--run-id", "first-frame-without-edge", "--var",
                                "transfer_first_chunk_conditional_frames=0",
                                "--var", "structural_control=none", "--json"])
    assert result.exit_code != 0
    assert "transfer_first_chunk_conditional_frames requires structural_control=edge" in result.output


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "true", "invalid"])
def test_invalid_rgb_weight_fails_during_planning(value):
    result = runner.invoke(app, ["workbench", "workflow", "plan-spec", str(SPEC),
                                "--run-id", "invalid-rgb", "--var", f"transfer_rgb_weight={value}", "--json"])
    assert result.exit_code != 0
    assert "transfer_rgb_weight" in result.output


def test_rgb_conditioning_requires_edge_transfer_during_planning():
    result = runner.invoke(app, ["workbench", "workflow", "plan-spec", str(SPEC),
                                "--run-id", "rgb-without-edge", "--var", "transfer_rgb_weight=0.5",
                                "--var", "structural_control=none", "--json"])
    assert result.exit_code != 0
    assert "transfer_rgb_weight requires structural_control=edge" in result.output


def test_source_caption_instruction_uses_task_context(tmp_path):
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    document = _doc()
    document["config"]["augment_subject"] = "two grippers inserting a battery"
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(document))
    plan = build_plan(load_spec(path), run_id="caption-context",
                      assume_decision="promote_checkpoint")
    stage = next(step for step in plan.steps if step.state == "annotate-original")
    instruction = stage.argv[stage.argv.index("--instruction") + 1]
    assert "two grippers inserting a battery" in instruction
    assert "Do not infer motion or task completion" in instruction
    assert "{{" not in instruction


def test_existing_caption_workflow_retains_default_instruction():
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    spec = load_spec(ROOT / "workflows/testing/token-factory-caption.yaml")
    plan = build_plan(spec, run_id="legacy-caption")
    stage = next(step for step in plan.steps
                 if step.tool_ref == "workbench.token_factory.caption")
    assert stage.argv[stage.argv.index("--instruction") + 1] == ""


def test_custom_caption_instruction_is_forwarded_literally(tmp_path):
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.spec import load_spec

    document = _doc()
    instruction = 'Describe the "left wrist" view; do not infer hidden contacts.'
    document["config"]["caption_instruction"] = instruction
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(document))
    plan = build_plan(load_spec(path), run_id="custom-caption",
                      assume_decision="promote_checkpoint")
    stage = next(step for step in plan.steps if step.state == "annotate-original")
    assert stage.argv[stage.argv.index("--instruction") + 1] == instruction


def test_paidf_cosmos3_schema_and_real_component_contract() -> None:
    doc = _doc()
    assert doc["apiVersion"] == "npa.workflow/v0.0.1"
    assert doc["metadata"]["executionMode"] == "runtime"
    states = doc["states"]
    assert states["prepare-input"]["toolRef"] == "workbench.cosmos3.prepare_video_input"
    assert (
        states["generate-variants"]["toolRef"] == "workbench.cosmos3.generate_variants"
    )
    assert states["evaluate"]["toolRef"] == "workbench.cosmos_evaluator.evaluate"
    assert states["cosmos-curate"]["toolRef"] == "workbench.cosmos_curate.curate"
    assert states["curate"]["toolRef"] == "workbench.fiftyone.curate_augmented"
    assert states["visualize"]["toolRef"] == "workbench.nurec.visualize"
    assert TOOL_CATALOG[states["generate-variants"]["toolRef"]].stub is False
    argv = TOOL_CATALOG[states["generate-variants"]["toolRef"]].argv_template
    assert argv[:4] == ["npa", "workbench", "cosmos3", "generate-variants"]
    assert "--input-path" in argv and "--guardrails" in argv
    assert "echo" not in argv


def test_paidf_cosmos3_cannot_be_prepared_as_a_one_shot_submit() -> None:
    with pytest.raises(NpaWorkflowError, match="requires runtime execution"):
        prepare_npa_workflow_for_submit(
            SPEC,
            run_id="paidf-one-shot",
            assume_decision="promote_checkpoint",
            config_overrides={"bucket": "example-bucket"},
        )


def test_general_cosmos3_toolref_forwards_conditioning_and_sampling() -> None:
    argv = TOOL_CATALOG["workbench.cosmos3.generate"].argv_template
    for flag in ("--input-path", "--seed", "--guidance", "--num-steps"):
        assert flag in argv


def test_dynamic_paths_stop_rejected_runs_before_downstream_components() -> None:
    states = _doc()["states"]
    assert states["refinement"]["sequence"] == [
        "generate-variants",
        "evaluate",
        "quality-gate",
    ]
    assert states["quality-disposition"]["next"] == "visualize-quality-evidence"
    assert states["visualize-quality-evidence"]["next"] == "quality-route"
    transitions = states["quality-route"]["transitions"]
    assert transitions == [
        {"when": "promote_checkpoint", "goto": "require-accepted-quality"},
        {"when": "loop_back", "goto": "reject-quality"},
    ]
    assert states["require-accepted-quality"]["next"] == "annotate-augmented"
    assert states["reject-quality"]["terminal"] is True
    rejected = {"visualize-quality-evidence", "quality-route", "reject-quality"}
    assert rejected.isdisjoint(
        {"annotate-augmented", "cosmos-curate", "curate", "finalize"}
    )


def test_configuration_surface_and_privacy_defaults() -> None:
    doc = _doc()
    config = doc["config"]
    for key in (
        "cosmos3_checkpoint",
        "cosmos3_mode",
        "seed",
        "guidance",
        "steps",
        "variant_count",
        "variant_parallelism",
        "augmentation_seed",
        "input_kind",
        "input_episode",
        "input_camera",
        "grade_threshold",
        "refinement_iterations",
        "retry_seed_stride",
        "retry_guidance_delta",
        "retry_steps_delta",
    ):
        assert key in config
    assert config["cosmos3_mode"] == "video2video"
    assert float(config["source_motion_weight"]) == 0.0
    assert float(config["grade_threshold"]) == 0.2
    assert float(config["attribute_threshold"]) == 0.25
    assert config["augmentation_seed"] == "30"
    assert (
        doc["states"]["generate-configs"]["run"]["argv"][-3]
        == "{{config.augmentation_seed}}"
    )
    assert config["bucket"] == "example-bucket"
    text = SPEC.read_text(encoding="utf-8")
    for forbidden in ("tenant", "project-id", "cluster-name", "hf_", "AKIA"):
        assert forbidden not in text
    assert "SAM" not in text and "SAM2" not in text


def test_validate_and_plan_both_decision_paths() -> None:
    validate = runner.invoke(
        app, ["workbench", "workflow", "validate-spec", str(SPEC), "--json"]
    )
    assert validate.exit_code == 0, validate.output
    assert json.loads(validate.output)["status"] == "valid"
    for decision in ("promote_checkpoint", "loop_back"):
        result = runner.invoke(
            app,
            [
                "workbench",
                "workflow",
                "plan-spec",
                str(SPEC),
                "--run-id",
                "test-run",
                "--assume-decision",
                decision,
                "--var",
                "bucket=example-bucket",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        names = [step["state"] for step in payload["steps"]]
        if decision == "promote_checkpoint":
            assert "finalize" in names and "cosmos-curate" in names
            visualizers = {step["state"]: step for step in payload["steps"]
                           if step["state"] in {"visualize-quality-evidence", "visualize"}}
            quality_argv = visualizers["visualize-quality-evidence"]["argv"]
            final_argv = visualizers["visualize"]["argv"]
            assert any(value.endswith("/reports/quality-evidence.rrd") for value in quality_argv)
            assert any(value.endswith("/reports/sim2real.rrd") for value in final_argv)
        else:
            assert names[-1] == "reject-quality"
            assert "cosmos-curate" not in names
            assert "visualize-quality-evidence" in names
        if decision == "promote_checkpoint":
            assert names.index("require-accepted-quality") < names.index(
                "annotate-augmented"
            )
            assert names.index("visualize-quality-evidence") < names.index(
                "require-accepted-quality"
            )


def test_cosmos3_cli_exposes_conditioned_variant_commands() -> None:
    result = runner.invoke(app, ["workbench", "cosmos3", "generate-variants", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--input-path",
        "--variant-count",
        "--retry-seed-stride",
        "--guardrails",
    ):
        assert flag in result.output
    prepare = runner.invoke(
        app, ["workbench", "cosmos3", "prepare-video-input", "--help"]
    )
    assert prepare.exit_code == 0
    assert "--lerobot-dataset-uri" in prepare.output
    assert "--episode" in prepare.output and "--camera" in prepare.output
