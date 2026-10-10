"""Keep the fresh ALOHA review faithful to its source, sampling and rejection contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from npa.deploy.images import public_release_manifest
from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.interpreter import _make_context
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    plan_images,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX
from npa.workflows import data_factory_stages


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "workflows/testing/paidf-aloha-cups-fanout.yaml"
EXPECTED_STAGES = [
    "prepare-input",
    "generate-configs",
    "annotate-original",
    "generate-variants",
    "evaluate",
    "quality-disposition",
    "visualize-quality-evidence",
]


def _live_helpers():
    path = ROOT / "npa/tests/e2e/npa_workflow_live_helpers.py"
    module_spec = importlib.util.spec_from_file_location("aloha_live_helpers", path)
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_fresh_source_review_has_no_refinement_or_promotion():
    spec = load_spec(SPEC)
    validate_spec(spec)
    plan = build_plan(spec, run_id="fresh-aloha-review")
    assert [step.state for step in plan.steps] == EXPECTED_STAGES
    assert spec.name != "paidf-cosmos3"
    assert spec.metadata["executionMode"] == "runtime"
    assert all(
        not state.loop and not state.transitions for state in spec.states.values()
    )
    assert spec.states[EXPECTED_STAGES[-1]].terminal
    config = _make_context(spec, run_id="fresh-aloha-review").config
    assert config["input_kind"] == "lerobot"
    assert config["input_episode"] == "0"
    assert config["input_camera"] == "observation.images.cam_high"
    assert config["input_video_uri"] == ""
    assert all(
        "fresh-aloha-review" in config[key]
        for key in (
            "lerobot_dataset_uri",
            "input_uri",
            "conditioning_video_uri",
            "input_provenance_uri",
            "captions_uri",
            "configs_uri",
            "augment_uri",
            "scores_uri",
            "quality_rrd_uri",
        )
    )


def test_sampling_controls_and_native_guardrails_reach_generation():
    plan = build_plan(load_spec(SPEC), run_id="sampling-review")
    argv = next(step.argv for step in plan.steps if step.state == "generate-variants")
    expected = {
        "--checkpoint": "Cosmos3-Nano",
        "--variant-count": "12",
        "--variant-parallelism": "1",
        "--seed": "17",
        "--steps": "35",
        "--guidance": "5.0",
        "--control-guidance": "1.0",
        "--conditioning-fps": "24",
        "--transfer-chunk-frames": "93",
        "--transfer-edge-threshold": "low",
        "--transfer-rgb-weight": "0.25",
        "--transfer-first-chunk-conditional-frames": "0",
        "--transfer-cfg-normalization": "disabled",
        "--source-motion-weight": "0.0",
    }
    assert {flag: argv[argv.index(flag) + 1] for flag in expected} == expected
    assert "--guardrails" in argv
    evaluator = next(step.argv for step in plan.steps if step.state == "evaluate")
    for flag, value in {
        "--attribute-threshold": "1.0",
        "--alignment-mode": "required",
        "--temporal-mode": "required",
        "--temporal-threshold": "0.8",
        "--appearance-mode": "advisory",
    }.items():
        assert evaluator[evaluator.index(flag) + 1] == value


def test_profiles_reproduce_the_recorded_twelve_scenario_order(tmp_path):
    document = yaml.safe_load(SPEC.read_text())
    example = json.loads(
        (ROOT / "docs/workbench/examples/paidf-cups-fanout-profiles.json").read_text()
    )
    assert json.loads(document["config"]["appearance_profiles_json"]) == example
    document["config"]["configs_uri"] = str(tmp_path / "configs")
    document["config"]["images_uri"] = str(tmp_path / "input")
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(document))
    plan = build_plan(load_spec(path), run_id="fresh-sampler-review")
    step = next(step for step in plan.steps if step.state == "generate-configs")
    manifest = data_factory_stages.generate_configs(*step.argv[3:])
    observed = manifest["augmentations"]
    previous = json.loads(
        (
            ROOT / "docs/workbench/evidence/paidf-lerobot/cups-fanout-results.json"
        ).read_text()
    )["variants"]
    assert len(observed) == 12
    assert [{key: item[key] for key in example[0]} for item in observed] == [
        item["profile"] for item in previous
    ]


def test_public_defaults_and_submitted_source_overlay_are_used(monkeypatch):
    monkeypatch.delenv("NPA_SRC_OVERLAY", raising=False)
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/fresh-source.tar.gz")
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="fresh-public-review")
    options = SkypilotRenderOptions(materialize_registry_secrets=False)
    images = plan_images(
        spec, plan.steps, run_id="fresh-public-review", options=options
    )
    candidates = public_release_manifest()["workflow_validation_candidates"]
    for tool in ("cosmos3", "cosmos-evaluator"):
        assert any(
            image.endswith("@" + candidates[tool]["published_digest"])
            for image in images
        )
    rendered = render_skypilot_yaml(
        spec, plan, run_id="fresh-public-review", options=options
    )
    tasks = [
        task for task in yaml.safe_load_all(rendered) if task and "resources" in task
    ]
    assert len(tasks) == 7
    assert all(
        task["envs"]["NPA_SRC_S3_URI"].endswith("fresh-source.tar.gz") for task in tasks
    )
    baked = [task for task in tasks if task["resources"].get("image_id")]
    assert baked
    assert all(task["envs"]["NPA_SRC_OVERLAY"] == "1" for task in baked)


def test_terminal_review_persists_a_strict_rejection(monkeypatch):
    published = []
    monkeypatch.setattr(
        data_factory_stages,
        "_download_json",
        lambda uri: {"status": "completed", "score": 0.116148, "passed": False},
    )
    monkeypatch.setattr(
        data_factory_stages,
        "_upload_json",
        lambda payload, uri: published.append((payload.copy(), uri)) or uri,
    )
    decisions = []
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: decisions.append(decision),
    )
    config = load_spec(SPEC).config
    result = data_factory_stages.write_quality_disposition(
        "scores/", "quality.json", "decision.json", config["grade_threshold"]
    )
    assert result["quality_status"] == "rejected"
    assert result["threshold"] == 0.75
    assert result["hard_checks_passed"] is False
    assert decisions == ["loop_back"]
    assert published[0][0]["quality_status"] == "rejected"


def test_live_registration_keeps_full_review_and_public_defaults(monkeypatch, tmp_path):
    case = next(case for case in SUBMIT_LIVE_MATRIX if case.spec == SPEC.name)
    assert case.runtime and case.tier == "gpu" and not case.plan_only
    assert case.requires_token_factory
    assert not case.config_vars and not case.image_overrides
    monkeypatch.setenv(
        "NPA_E2E_PAIDF_ALOHA_DATASET_URI", "s3://example-bucket/fresh-pinned-aloha/"
    )
    helpers = _live_helpers()
    path = helpers.materialize_live_spec(
        tmp_path, SPEC.name, bucket="example-bucket", run_id="fresh-e2e-review"
    )
    materialized = load_spec(path)
    plan = build_plan(materialized, run_id="fresh-e2e-review")
    config = _make_context(materialized, run_id="fresh-e2e-review").config
    assert config["lerobot_dataset_uri"].endswith("/fresh-pinned-aloha/")
    assert config["prefix"].endswith("/paidf-aloha-cups-fanout")
    assert len(plan.steps) == 7
    assert int(config["variant_count"]) == 12


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "https://example.invalid/data",
        "s3://example-bucket",
        "s3://example-bucket/data?query=value",
    ],
)
def test_live_materialization_requires_an_explicit_dataset_directory(
    monkeypatch, tmp_path, uri
):
    monkeypatch.setenv("NPA_E2E_PAIDF_ALOHA_DATASET_URI", uri)
    with pytest.raises(pytest.fail.Exception, match="NPA_E2E_PAIDF_ALOHA_DATASET_URI"):
        _live_helpers().materialize_live_spec(
            tmp_path, SPEC.name, bucket="example-bucket", run_id="fresh-review"
        )
