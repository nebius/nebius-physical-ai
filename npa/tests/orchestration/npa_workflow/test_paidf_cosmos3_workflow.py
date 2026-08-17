from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "paidf-cosmos3.yaml"
runner = CliRunner()


def _doc() -> dict:
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))


def test_paidf_cosmos3_schema_and_real_component_contract() -> None:
    doc = _doc()
    assert doc["apiVersion"] == "npa.workflow/v0.0.1"
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
