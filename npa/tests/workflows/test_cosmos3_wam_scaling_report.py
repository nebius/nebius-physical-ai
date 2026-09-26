"""Reject invalid timing comparisons and distinguish run variability from step noise."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "workflows/workbench/cosmos3-wam-slurm"


def _module():
    spec = importlib.util.spec_from_file_location(
        "wam_scaling_report", RECIPE / "scaling_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measurement(gpus, duration, series_hash):
    return {
        "schema": "npa.cosmos3.wam-measurement.v1",
        "status": "measured",
        "comparison_contract": {
            "steps": 200,
            "profile": False,
            "seed": 42,
            "global_batch": 2048,
            "samples_per_rank": 64,
            "sources": {"framework": "fixture-revision"},
        },
        "hardware": {"gpu_names": ["NVIDIA B200"] * 8, "torch": "fixture"},
        "gpus": gpus,
        "nodes": gpus // 8,
        "warmup_steps_excluded": 50,
        "measured_steps": 149,
        "step_mean_seconds": duration,
        "step_p50_seconds": duration,
        "step_p95_seconds": duration,
        "iteration_series_sha256": series_hash,
        "work": {
            "measured_tokens": 149_000,
            "mean_tokens_per_optimizer_step": 1000,
            "tokens_per_second": 1000 / duration,
        },
    }


def _run(root, gpus, repeat, duration):
    directory = root / f"{gpus}-{repeat}"
    directory.mkdir(parents=True)
    series = directory / "iteration-series.csv"
    with series.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "iteration_seconds", "tokens", "loss"])
        writer.writerows([step, duration, 1000, 0.5] for step in range(52, 201))
    digest = hashlib.sha256(series.read_bytes()).hexdigest()
    report = _measurement(gpus, duration, digest)
    (directory / "measurement.json").write_text(json.dumps(report))
    return directory


def _campaign(root):
    return [
        _run(root, gpus, repeat, duration)
        for gpus, durations in ((8, (10, 12, 14)), (16, (6, 7, 8)))
        for repeat, duration in enumerate(durations)
    ]


def _change(directory, change):
    path = directory / "measurement.json"
    report = json.loads(path.read_text())
    change(report)
    path.write_text(json.dumps(report))


def test_scaling_uses_run_means_and_reports_token_work(tmp_path):
    result = _module()._summarize(_campaign(tmp_path))
    assert result["status"] == "scaling_measured"
    assert result["groups"]["8"]["measured_steps"] == 447
    spread = result["groups"]["8"]["replicate_step_means_seconds"]
    assert spread["mean"] == 12
    assert spread["sample_standard_deviation"] == 2
    assert result["comparison"]["speedup_vs_8_gpus"] == pytest.approx(12 / 7)
    assert result["comparison"]["scaling_efficiency"] == pytest.approx(6 / 7)
    assert result["comparison"]["token_throughput_speedup"] == pytest.approx(12 / 7)
    assert result["comparison"]["token_work_ratio"] == 1


def test_single_topology_does_not_invent_scaling(tmp_path):
    result = _module()._summarize(_campaign(tmp_path)[:3])
    assert result["status"] == "baseline_measured"
    assert result["comparison"] is None


@pytest.mark.parametrize(
    "mutation", ["source", "batch", "hardware", "profile", "steps"]
)
def test_incompatible_repetition_is_rejected(tmp_path, mutation):
    runs = _campaign(tmp_path)

    def change(report):
        if mutation == "hardware":
            report["hardware"]["torch"] = "different-runtime"
        else:
            key = {"source": "sources", "batch": "global_batch"}.get(mutation, mutation)
            report["comparison_contract"][key] = {
                "source": {"framework": "different-revision"},
                "batch": 4096,
                "profile": True,
                "steps": 2000,
            }[mutation]

    _change(runs[-1], change)
    with pytest.raises(ValueError):
        _module()._summarize(runs)


def test_changed_csv_and_stale_summary_are_rejected(tmp_path):
    runs = _campaign(tmp_path)
    path = runs[0] / "iteration-series.csv"
    path.write_text(path.read_text().replace("10,1000", "11,1000"))
    with pytest.raises(ValueError, match="hash"):
        _module()._summarize(runs)
    _change(
        runs[0],
        lambda report: report.update(
            iteration_series_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    )
    with pytest.raises(ValueError, match="disagrees"):
        _module()._summarize(runs)


def test_missing_and_duplicate_runs_cannot_supply_repetitions(tmp_path):
    runs = _campaign(tmp_path)
    with pytest.raises(ValueError, match="three"):
        _module()._summarize(runs[:2])
    with pytest.raises(ValueError, match="duplicate timing-run"):
        _module()._summarize([runs[0], runs[1], runs[0]])
    (runs[2] / "measurement.json").write_bytes(
        (runs[0] / "measurement.json").read_bytes()
    )
    (runs[2] / "iteration-series.csv").write_bytes(
        (runs[0] / "iteration-series.csv").read_bytes()
    )
    with pytest.raises(ValueError, match="copied reports"):
        _module()._summarize(runs[:3])


@pytest.mark.parametrize(
    "replacement",
    ["53,10,1000,0.5", "52,nan,1000,0.5", "52,-1,1000,0.5", "52,10,0,0.5"],
)
def test_invalid_iteration_rows_cannot_be_aggregated(tmp_path, replacement):
    runs = _campaign(tmp_path)
    path = runs[0] / "iteration-series.csv"
    path.write_text(path.read_text().replace("52,10,1000,0.5", replacement))
    _change(
        runs[0],
        lambda report: report.update(
            iteration_series_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    )
    with pytest.raises(ValueError):
        _module()._summarize(runs)
