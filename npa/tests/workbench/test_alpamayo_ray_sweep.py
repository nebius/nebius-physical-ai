"""Exercise sweep statistics and real Ray scheduling with explicit synthetic workers."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from npa.workbench.alpamayo2_super.ray_report import (
    case_grid, matched_comparisons, select_hard_samples,
    summarize_sample, validate_measurements,
)
from npa.workbench.alpamayo2_super.ray_sweep import (
    AlpamayoSweepRequest, _cases_and_baseline, dispatch_cases,
)
from npa.workbench.alpamayo2_super.runtime import DEFAULT_DATASET_REVISION, DEFAULT_MODEL_REVISION


def test_sweep_cli_and_sdk_use_shared_request_and_keep_stdout_json(monkeypatch):
    from typer.testing import CliRunner
    from npa.cli.main import app
    from npa.sdk.workbench.alpamayo2_super import sweep
    from npa.workbench.alpamayo2_super import ray_sweep

    observed = []
    def execute(request):
        observed.append(request)
        print("synthetic progress message")
        return {"status": "complete"}
    monkeypatch.setattr(ray_sweep, "run_sweep", execute)
    result = CliRunner().invoke(app, [
        "workbench", "alpamayo2-super", "sweep", "--output-path", "s3://fixture-bucket/sweep/",
        "--run-id", "fixture", "--sample-indices", "0,2", "--seeds", "42,44",
        "--diffusion-steps", "10,20",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"status": "complete"}
    assert "synthetic progress message" in result.stderr
    assert observed[0].sample_indices == [0, 2]
    assert observed[0].seeds == [42, 44]
    assert sweep(observed[0]) == {"status": "complete"}
    assert observed[0] == observed[1]


def test_public_sweep_cli_rejects_local_handoffs_before_execution(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from npa.cli.main import app
    from npa.workbench.alpamayo2_super import ray_sweep

    def forbidden(request):
        pytest.fail("invalid path reached inference")
    monkeypatch.setattr(ray_sweep, "run_sweep", forbidden)
    result = CliRunner().invoke(app, ["workbench", "alpamayo2-super", "sweep",
        "--output-path", str(tmp_path / "fixture"), "--run-id", "fixture"])
    assert result.exit_code == 1
    assert "S3 handoff contract" in result.output


def _measurement(case):
    return {
        **case, "min_ade_m": float(case["sample_index"] + 1), "min_fde_m": 3.0,
        "elapsed_seconds": 0.1,
        "sample": {"clip_id": f"fixture-{case['sample_index']}", "t0_us": 1, "manifest_sha256": "fixture"},
    }


@pytest.fixture
def baseline():
    cases = case_grid([0, 1], [42, 43], [10])
    return {
        "schema": "npa.alpamayo.ray-sweep.v1", "status": "complete",
        "cases": cases, "measurements": [_measurement(case) for case in cases],
        "revisions": {"model_revision": DEFAULT_MODEL_REVISION, "dataset_revision": DEFAULT_DATASET_REVISION},
    }


@pytest.mark.parametrize("samples,seeds,steps", [([], [1], [1]), ([0, 0], [1], [1]), ([0], [1], [0]), ([-1], [1], [1])])
def test_invalid_experiment_dimensions(samples, seeds, steps):
    with pytest.raises(ValueError):
        case_grid(samples, seeds, steps)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nan", "negative", "none"])
def test_measurements_must_be_complete_and_finite(baseline, damage):
    rows = baseline["measurements"]
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows[0] = copy.deepcopy(rows[-1])
    else:
        rows[0]["min_ade_m"] = {"nan": float("nan"), "negative": -1, "none": None}[damage]
    with pytest.raises(ValueError):
        validate_measurements(rows, baseline["cases"])


def test_seed_variation_is_not_pooled_across_diffusion_settings():
    rows = [_measurement(case) for case in case_grid([0], [42, 43], [10, 20])]
    for row in rows:
        row["min_ade_m"] = {10: {42: 1, 43: 3}, 20: {42: 8, 43: 12}}[row["diffusion_steps"]][row["seed"]]
    summary = summarize_sample(rows)
    assert [row["mean_ade_m"] for row in summary] == [2, 10]
    assert [row["seed_ade_std_m"] for row in summary] == [1, 2]


def test_hard_case_selection_uses_measured_rows_and_accepts_empty_selection(baseline):
    baseline["summaries"] = [{"sample_index": 0, "mean_ade_m": 9999}]
    assert select_hard_samples(baseline, 1.5) == [1]
    assert select_hard_samples(baseline, 2) == []


def test_refinement_inherits_baseline_seeds_and_rejects_revision_drift(baseline, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(baseline))
    request = AlpamayoSweepRequest(output_path=str(tmp_path), run_id="fixture", input_path=str(path), minimum_ade=1.5)
    cases, _ = _cases_and_baseline(replace(request, seeds=[999], diffusion_steps=[20]))
    assert cases == case_grid([1], [42, 43], [20])
    baseline["revisions"]["model_revision"] = "different"
    path.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="different model or dataset"):
        _cases_and_baseline(request)


def test_matched_comparisons_reject_unmatched_seed_and_manifest_change(baseline):
    refined = _measurement({"sample_index": 1, "seed": 42, "diffusion_steps": 20})
    refined["min_ade_m"] = 1.0
    comparisons = matched_comparisons(baseline["measurements"], [refined])
    assert comparisons[0]["ade_change_m"] == -1.0
    with pytest.raises(ValueError, match="no matched baseline"):
        matched_comparisons(baseline["measurements"], [{**refined, "seed": 123}])
    refined["sample"]["manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="manifest identities"):
        matched_comparisons(baseline["measurements"], [refined])


class _SyntheticWorker:
    def infer(self, case):
        return _measurement(case)


class _FailingWorker:
    def infer(self, case):
        raise ValueError("synthetic inference boundary failure")


@pytest.fixture(scope="module")
def local_ray():
    ray = pytest.importorskip("ray")
    ray.init(address="local", num_cpus=2, include_dashboard=False, log_to_driver=False,
             runtime_env={"env_vars": {"PYTHONPATH": str(Path(__file__).parent)}})
    yield ray
    ray.shutdown()


def test_real_ray_actors_cover_every_case_once(local_ray):
    cases = case_grid([0, 1, 2], [42, 43], [10, 20])
    actor = local_ray.remote(num_cpus=1)(_SyntheticWorker)
    actors = [actor.remote() for _ in range(2)]
    try:
        rows = dispatch_cases(cases, actors)
        assert len(rows) == 12
        assert [(row["sample_index"], row["seed"], row["diffusion_steps"]) for row in rows] == [tuple(case.values()) for case in cases]
    finally:
        for worker in actors:
            local_ray.kill(worker)


def test_real_ray_failure_is_not_reported_as_partial_success(local_ray):
    actor = local_ray.remote(num_cpus=1)(_FailingWorker).remote()
    try:
        with pytest.raises(local_ray.exceptions.RayTaskError, match="synthetic inference boundary"):
            dispatch_cases(case_grid([0], [42], [10]), [actor])
    finally:
        local_ray.kill(actor)
