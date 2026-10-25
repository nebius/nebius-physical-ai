"""Exercise the single inspection workflow through its native preset interface."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.marble_credentials import marble_secret_names
from npa.orchestration.npa_workflow.presets import available_presets, preset_overrides
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    assert_no_unresolved_placeholders,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.submit import merge_config_overrides
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX
from npa.orchestration.npa_workflow.waves import build_wave_plan

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "workflows/testing/marble-go1-inspection.yaml"
SITES = {
    "warehouse": (500, 1.2),
    "shipyard": (750, 0.7),
    "home": (1000, 0.35),
    "factory": (1000, 0.5),
    "utility": (1000, 0.5),
}
runner = CliRunner()


def _spec(site):
    spec = load_spec(SPEC)
    return merge_config_overrides(
        spec, preset_overrides(workflow_name=spec.name, preset=site)
    )


def _flag(step, flag):
    return step.argv[step.argv.index(flag) + 1]


@pytest.mark.parametrize("site", SITES)
def test_preset_selects_exactly_one_world_one_gpu_and_one_continuous_report(
    site, monkeypatch
):
    spec = _spec(site)
    plan = build_plan(spec, run_id="preset-test")
    assert marble_secret_names(spec) == ("WLT_API_KEY",)
    waves = build_wave_plan(spec, run_id="preset-test").waves
    assert len(waves) == 3 and all(len(wave.steps) == 1 for wave in waves)
    acquire, collect, report = plan.steps
    assert _flag(acquire, "--source") == "generate"
    assert _flag(acquire, "--prompt") == spec.config["world_prompt"]
    assert _flag(acquire, "--output-path") == _flag(collect, "--input-path")
    assert _flag(collect, "--output-path") == _flag(report, "--input-path")
    assert f"/marble-go1-{site}/" in _flag(report, "--output-path")
    frames, speed = SITES[site]
    assert int(_flag(collect, "--frames")) == frames
    assert float(_flag(collect, "--speed-mps")) == speed
    assert _flag(collect, "--motion-profile") == "turnaround"
    assert _flag(collect, "--sensor-hz") == "25"
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="preset-test",
        options=SkypilotRenderOptions(
            registry="registry.example", materialize_registry_secrets=False
        ),
    )
    assert_no_unresolved_placeholders(rendered)
    jobs = [doc for doc in yaml.safe_load_all(rendered) if doc and "run" in doc]
    assert len(jobs) == 3
    assert "accelerators" not in jobs[0]["resources"]
    assert jobs[1]["resources"]["accelerators"] == "RTXPRO6000:1"
    assert "quadruped-collect" in jobs[1]["run"]
    assert "--motion-profile turnaround" in jobs[1]["run"]
    assert "onnxruntime==1.23.2" in jobs[1]["setup"]
    assert "pycollada==0.9.3" in jobs[1]["setup"]
    assert "accelerators" not in jobs[2]["resources"]
    assert "imageio-ffmpeg==0.6.0" in jobs[2]["setup"]


@pytest.mark.parametrize("site", SITES)
def test_cli_validation_and_planning_accept_the_same_selected_preset(site):
    base = ["workbench", "workflow"]
    validated = runner.invoke(
        app, [*base, "validate-spec", str(SPEC), "--preset", site, "--json"]
    )
    assert validated.exit_code == 0, validated.output
    planned = runner.invoke(
        app,
        [
            *base,
            "plan-spec",
            str(SPEC),
            "--preset",
            site,
            "--run-id",
            "cli-test",
            "--json",
        ],
    )
    assert planned.exit_code == 0, planned.output
    steps = json.loads(planned.stdout)["steps"]
    assert len(steps) == 3
    assert f"/marble-go1-{site}/" in steps[2]["argv"][-1] or any(
        f"/marble-go1-{site}/" in value for value in steps[2]["argv"]
    )


def test_default_warehouse_and_custom_world():
    assert build_plan(load_spec(SPEC), run_id="default") == build_plan(
        _spec("warehouse"), run_id="default"
    )
    assert set(available_presets("marble-go1-inspection")) == set(SITES)
    assert len({str(_spec(site).config["world_prompt"]) for site in SITES}) == 5
    overrides = preset_overrides(
        workflow_name="marble-go1-inspection",
        preset="",
        explicit={
            "world_prompt": "A hangar with a clear flat service aisle.",
            "workflow_preset": "hangar",
        },
    )
    custom = merge_config_overrides(load_spec(SPEC), overrides)
    acquire = build_plan(custom, run_id="custom").steps[0]
    assert _flag(acquire, "--prompt") == overrides["world_prompt"]
    assert "/marble-go1-hangar/" in _flag(acquire, "--output-path")


@pytest.mark.parametrize(
    "args, message",
    [
        (["--preset", "all"], "has no preset"),
        (["--preset", "home", "--var", "world_prompt=other"], "owns config keys"),
    ],
)
def test_invalid_selection_fails_before_submission(args, message, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Invalid preset reached credential resolution")

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.submit_credentials.resolve_submit_credentials",
        forbidden,
    )
    result = runner.invoke(
        app, ["workbench", "workflow", "submit", str(SPEC), *args, "--plan-only"]
    )
    assert result.exit_code == 1
    assert message.lower() in result.output.lower(), result.output


def test_one_live_matrix_entry_uses_a_selected_preset():
    cases = [case for case in SUBMIT_LIVE_MATRIX if case.spec.startswith("marble-go1-")]
    assert len(cases) == 1
    case = cases[0]
    assert case.spec == SPEC.name and case.preset == "warehouse"
    assert case.runtime and not case.plan_only and not case.rotation_skip
    assert "WLT_API_KEY" in case.secret_envs
    assert not case.config_vars and not case.expected_parallel_tasks
