"""VLM-eval workflow coverage, at the spec level.

This file used to assert the raw shape of three SkyPilot templates — `vlm-eval.yaml`,
`vlm-eval-benchmark.yaml` and `sim-to-real-loop.yaml`. All three are retired (their twins
reached SUCCEEDED live: jobs 219, 220, 218 — EVIDENCE.md §R18–R20), so the assertions move
onto the specs that replaced them: same properties, checked where they now live.

What each template guaranteed and what checks it here:

* it served an open VLM itself → the *renderer* starts and health-checks vLLM for any
  `self-hosted` stage (`test_self_hosted_vlm_preamble.py`), and the specs below ask for
  that backend;
* it passed a specific set of CLI flags → the resolved `toolRef` argv;
* the loop aggregated a rollout set → `tests/workbench/test_vlm_eval_loop.py`, plus the
  spec's declared artifact here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec
from npa.workbench.vlm_eval import (
    BENCHMARK_RESULT_FILENAME,
    DEFAULT_MODEL,
    LOOP_REPORT_FILENAME,
    RESULT_FILENAME,
)

ROOT = Path(__file__).resolve().parents[3]
SPECS = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"


def _only_step(spec_name: str):
    spec = load_spec(SPECS / spec_name)
    plan = build_plan(spec, run_id="vlm-eval-workflow-test")
    steps = [step for step in plan.steps if step.argv]
    assert len(steps) == 1, f"{spec_name} should have exactly one executing stage"
    return spec, steps[0]


def _flag(argv: list[str], flag: str) -> str:
    assert flag in argv, f"argv is missing {flag}: {argv}"
    return argv[argv.index(flag) + 1]


@pytest.mark.parametrize(
    "spec_name", ["vlm-eval-single.yaml", "vlm-eval-benchmark.yaml", "vlm-eval-loop.yaml"]
)
def test_specs_ask_for_the_self_hosted_backend_the_templates_served(spec_name: str) -> None:
    """All three templates ran their own vLLM; the specs must still request that backend.

    `vlm-eval-benchmark.yaml` in particular used to say `stub`, which meant the twin never
    touched a VLM and could not stand in for the template (EVIDENCE.md §R20).
    """

    spec, step = _only_step(spec_name)

    assert spec.config["vlm_backend"] == "self-hosted"
    assert _flag(step.argv, "--backend") == "self-hosted"


def test_single_rollout_spec_scores_one_prefix() -> None:
    _spec, step = _only_step("vlm-eval-single.yaml")

    assert step.tool_ref == "workbench.vlm_eval.run"
    assert step.argv[:4] == ["npa", "workbench", "vlm-eval", "run"]
    # argv carries the RESOLVED uris; config still holds the templated form.
    assert _flag(step.argv, "--input-path").endswith("/rollouts/")
    assert _flag(step.argv, "--output-path").endswith("/scores/")
    assert step.outputs[0]["uri"].endswith(RESULT_FILENAME)


def test_loop_spec_scores_a_set_and_declares_the_aggregate_report() -> None:
    """The property that made retiring `sim-to-real-loop.yaml` possible."""

    _spec, step = _only_step("vlm-eval-loop.yaml")

    assert step.tool_ref == "workbench.vlm_eval.loop"
    assert step.argv[:4] == ["npa", "workbench", "vlm-eval", "loop"]
    assert _flag(step.argv, "--input-path").endswith("/rollouts/")
    # The coarse gate the template computed with `jq`.
    assert _flag(step.argv, "--success-threshold") == "0.8"
    assert _flag(step.argv, "--frame-selection") == "keyframes"
    assert _flag(step.argv, "--max-frames") == "4"
    assert step.outputs[0]["uri"].endswith(LOOP_REPORT_FILENAME)


def test_benchmark_spec_sweeps_the_same_grid_the_template_did() -> None:
    _spec, step = _only_step("vlm-eval-benchmark.yaml")

    assert step.tool_ref == "workbench.vlm_eval.benchmark"
    assert step.argv[:4] == ["npa", "workbench", "vlm-eval", "benchmark"]
    assert _flag(step.argv, "--models") == DEFAULT_MODEL
    assert _flag(step.argv, "--rubrics") == "default,strict"
    assert _flag(step.argv, "--thresholds") == "0.5,0.8,0.9"
    assert _flag(step.argv, "--format") == "json"


def test_benchmark_dataset_is_an_object_uri_not_a_repo_path() -> None:
    """A stage runs in a pod with no checkout; the template used an S3 URI too."""

    _spec, step = _only_step("vlm-eval-benchmark.yaml")

    dataset = _flag(step.argv, "--dataset")
    assert dataset.startswith("s3://"), dataset
    assert dataset.endswith(".json")
    output = _flag(step.argv, "--output")
    assert output.startswith("s3://")
    # `--output` names a file here, so the tool writes exactly there rather than appending
    # BENCHMARK_RESULT_FILENAME; the declared artifact must agree.
    assert output.endswith(".json")
    assert step.outputs[0]["uri"] == output
    assert BENCHMARK_RESULT_FILENAME.endswith(".json")


def test_the_three_retired_templates_are_gone() -> None:
    """Anchor the retirement so a revert cannot quietly restore a second surface."""

    skypilot = ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot"

    for name in ("vlm-eval.yaml", "vlm-eval-benchmark.yaml", "sim-to-real-loop.yaml"):
        assert not (skypilot / name).exists(), f"{name} came back"
