"""Unit coverage for the Isaac Lab sweep stages (no GPU, no S3, no Isaac Lab)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from npa.workflows import rl_sweep


@pytest.fixture()
def local_storage(monkeypatch, tmp_path: Path):
    """Route the module's S3 helpers at a local directory."""

    root = tmp_path / "s3"

    def _local(uri: str) -> Path:
        assert uri.startswith("s3://")
        return root / uri[len("s3://") :]

    def fake_upload_json(payload, uri):
        path = _local(uri) if uri.startswith("s3://") else Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return uri

    def fake_upload_file(local, uri):
        path = _local(uri) if uri.startswith("s3://") else Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(local).read_bytes())
        return uri

    def fake_download_json(uri):
        return json.loads(_local(uri).read_text(encoding="utf-8"))

    def fake_list_keys(uri):
        prefix = uri[len("s3://") :]
        bucket = prefix.split("/", 1)[0]
        base = _local(uri if uri.endswith("/") else uri + "/")
        if not base.exists():
            return []
        return [
            str(item.relative_to(root / bucket))
            for item in sorted(base.rglob("*"))
            if item.is_file()
        ]

    monkeypatch.setattr(rl_sweep, "_upload_json", fake_upload_json)
    monkeypatch.setattr(rl_sweep, "_upload_file", fake_upload_file)
    monkeypatch.setattr(rl_sweep, "_download_json", fake_download_json)
    monkeypatch.setattr(rl_sweep, "_list_keys", fake_list_keys)
    return root


def _runner(stdout: str = "", returncode: int = 0):
    captured: dict[str, list[str]] = {}

    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        captured["argv"] = list(argv)
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")

    run.captured = captured  # type: ignore[attr-defined]
    return run


def test_parse_overrides_splits_hydra_string() -> None:
    assert rl_sweep.parse_overrides("agent.save_interval=1 agent.algorithm.lr=1.0e-3") == [
        "agent.save_interval=1",
        "agent.algorithm.lr=1.0e-3",
    ]
    assert rl_sweep.parse_overrides(["a=1"]) == ["a=1"]
    assert rl_sweep.parse_overrides("") == []


def test_train_variant_passes_overrides_and_publishes_metrics(local_storage, tmp_path) -> None:
    runner = _runner(stdout="Mean reward: 12.5\nMean reward: 41.25\n")
    metrics = rl_sweep.train_variant(
        variant="lr-1e-3",
        output_uri="s3://bucket/sweep/lr-1e-3/",
        task="Isaac-Cartpole-v0",
        iterations=10,
        num_envs=64,
        overrides="agent.algorithm.learning_rate=1.0e-3",
        run_id="run-1",
        train_script="/tmp/train.py",
        python_bin="/usr/bin/python3",
        runner=runner,
    )

    argv = runner.captured["argv"]
    assert argv[:2] == ["/usr/bin/python3", "/tmp/train.py"]
    assert "--task" in argv and "Isaac-Cartpole-v0" in argv
    assert "--max_iterations" in argv and "10" in argv
    assert "agent.algorithm.learning_rate=1.0e-3" in argv
    assert "--headless" in argv

    assert metrics["status"] == "success"
    assert metrics["variant"] == "lr-1e-3"
    assert metrics["mean_reward"] == 41.25
    written = json.loads(
        (local_storage / "bucket/sweep/lr-1e-3" / rl_sweep.METRICS_FILENAME).read_text()
    )
    assert written["hydra_overrides"] == "agent.algorithm.learning_rate=1.0e-3"


def test_train_variant_raises_and_still_publishes_on_failure(local_storage) -> None:
    with pytest.raises(RuntimeError, match="training failed"):
        rl_sweep.train_variant(
            variant="bad",
            output_uri="s3://bucket/sweep/bad/",
            run_id="run-1",
            train_script="/tmp/train.py",
            python_bin="/usr/bin/python3",
            runner=_runner(stdout="boom", returncode=3),
        )
    published = json.loads(
        (local_storage / "bucket/sweep/bad" / rl_sweep.METRICS_FILENAME).read_text()
    )
    assert published["status"] == "failed"
    assert published["returncode"] == 3


def test_select_best_ranks_variants(local_storage) -> None:
    for variant, reward in (("a", 1.0), ("b", 7.5), ("c", 3.0)):
        rl_sweep._upload_json(
            {
                "schema": "npa.rl_sweep.variant_metrics.v1",
                "variant": variant,
                "status": "success",
                "mean_reward": reward,
            },
            f"s3://bucket/sweep/{variant}/{rl_sweep.METRICS_FILENAME}",
        )

    report = rl_sweep.select_best(
        sweep_uri="s3://bucket/sweep/",
        report_uri="s3://bucket/report/best.json",
        run_id="run-1",
    )

    assert report["variant_count"] == 3
    assert report["succeeded"] == 3
    assert report["best_variant"] == "b"
    assert report["best_value"] == 7.5
    assert json.loads((local_storage / "bucket/report/best.json").read_text())["best_variant"] == "b"


def test_select_best_handles_missing_metrics(local_storage) -> None:
    report = rl_sweep.select_best(
        sweep_uri="s3://bucket/empty/", report_uri="s3://bucket/report/empty.json"
    )
    assert report["variant_count"] == 0
    assert report["best_variant"] == ""


def test_resolve_python_bin_prefers_existing_candidate(tmp_path: Path) -> None:
    candidate = tmp_path / "python.sh"
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    candidate.chmod(0o755)
    assert rl_sweep.resolve_python_bin([str(candidate)]) == str(candidate)
    assert rl_sweep.resolve_python_bin(["/does/not/exist"]).endswith("python3")
