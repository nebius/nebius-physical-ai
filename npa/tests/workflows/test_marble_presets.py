"""Verify generated-site presets through native planning and actual worker rendering."""

from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.marble_credentials import marble_secret_names
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    assert_no_unresolved_placeholders,
    render_skypilot_job_group_yaml,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.submit import merge_config_overrides
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX
from npa.orchestration.npa_workflow.waves import build_wave_plan
from npa.workbench.marble.schemas import QuadrupedRequest

ROOT = Path(__file__).resolve().parents[3]
SITES = ("warehouse", "shipyard", "home", "factory", "utility")


def _spec(site):
    return load_spec(ROOT / f"workflows/testing/marble-go1-{site}.yaml")


def _options(monkeypatch):
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source")
    return SkypilotRenderOptions(
        registry="registry.example", materialize_registry_secrets=False
    )


def _flag(step, flag):
    return step.argv[step.argv.index(flag) + 1]


@pytest.mark.parametrize("site", SITES)
def test_single_site_generates_collects_and_reports_on_the_right_devices(
    site, monkeypatch
):
    spec = _spec(site)
    plan = build_plan(spec, run_id="preset-test")
    assert marble_secret_names(spec) == ("WLT_API_KEY",)
    acquire, collect, report = plan.steps
    assert _flag(acquire, "--source") == "generate"
    assert _flag(acquire, "--prompt") == spec.config["world_prompt"]
    assert _flag(acquire, "--output-path") == _flag(collect, "--input-path")
    assert _flag(collect, "--output-path") == _flag(report, "--input-path")
    settings = {
        name: spec.config[name]
        for name in QuadrupedRequest.model_fields
        if name in spec.config
    }
    request = QuadrupedRequest(
        **settings,
        input_path=_flag(collect, "--input-path"),
        output_path=_flag(collect, "--output-path"),
        run_id="preset-test",
    )
    assert request.motion_profile == "turnaround"
    rendered = render_skypilot_yaml(
        spec, plan, run_id="preset-test", options=_options(monkeypatch)
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


def test_five_sites_render_three_native_groups_with_separate_artifacts(monkeypatch):
    spec = _spec("five-sites")
    waves = build_wave_plan(spec, run_id="fanout-test").waves
    assert [(wave.kind, len(wave.steps)) for wave in waves] == [("parallel", 5)] * 3
    assert [wave.max_concurrency for wave in waves] == [5, 5, 5]
    assert marble_secret_names(spec) == ("WLT_API_KEY",)
    for index, wave in enumerate(waves):
        rendered = render_skypilot_job_group_yaml(
            spec, wave.steps, run_id="fanout-test", options=_options(monkeypatch)
        )
        assert_no_unresolved_placeholders(rendered)
        documents = list(yaml.safe_load_all(rendered))
        assert documents[0]["execution"] == "parallel"
        assert "primary_tasks" not in documents[0]
        assert len(documents[1:]) == 5
        outputs = [_flag(step, "--output-path") for step in wave.steps]
        assert len(set(outputs)) == 5
        for site, step, job in zip(SITES, wave.steps, documents[1:]):
            assert f"/marble-go1-five-sites/{site}/" in _flag(step, "--output-path")
            if index == 1:
                assert job["resources"]["accelerators"] == "RTXPRO6000:1"
                assert "npa-envgen" in job["resources"]["image_id"]
            else:
                assert "accelerators" not in job["resources"]
    for acquire, collect, report in zip(*(wave.steps for wave in waves)):
        assert _flag(acquire, "--output-path") == _flag(collect, "--input-path")
        assert _flag(collect, "--output-path") == _flag(report, "--input-path")


def test_fanout_matches_each_individual_preset():
    groups = build_wave_plan(_spec("five-sites"), run_id="same-inputs").waves
    prompts = []
    for index, site in enumerate(SITES):
        acquire, collect, _ = build_plan(_spec(site), run_id="same-inputs").steps
        prompt = _flag(acquire, "--prompt")
        prompts.append(prompt)
        assert _flag(groups[0].steps[index], "--prompt") == prompt
        for flag in ("--frames", "--speed-mps", "--motion-profile", "--sensor-hz"):
            assert _flag(groups[1].steps[index], flag) == _flag(collect, flag)
    assert len(set(prompts)) == 5


def test_incorrect_preset_count_is_rejected_before_submission():
    with pytest.raises(NpaWorkflowError, match="parallelCount resolves to 4"):
        merge_config_overrides(_spec("five-sites"), {"preset_count": "4"})


@pytest.mark.parametrize("site", (*SITES, "five-sites"))
def test_presets_have_live_coverage_without_a_sample_world_override(site):
    case = next(
        case for case in SUBMIT_LIVE_MATRIX if case.spec == f"marble-go1-{site}.yaml"
    )
    assert case.runtime and not case.plan_only
    assert not case.rotation_skip
    assert "WLT_API_KEY" in case.secret_envs
    assert not case.config_vars
    if site == "five-sites":
        assert case.expected_parallel_tasks == 5
