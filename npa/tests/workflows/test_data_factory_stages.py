"""Unit tests for the real Physical AI Data Factory stage functions."""

from __future__ import annotations

import json
from pathlib import Path

from npa.workflows import data_factory_stages as dfs


def test_generate_configs_writes_real_manifest(tmp_path: Path) -> None:
    out = tmp_path / "configs" / "manifest.json"
    result = dfs.generate_configs(str(out), n_augmentations=3, seed="run-x")
    assert result["n_augmentations"] == 3
    assert len(result["augmentations"]) == 3
    for combo in result["augmentations"]:
        assert combo["weather"] in dfs.APPEARANCE_VARIABLES["weather"]
        assert combo["time_of_day"] in dfs.APPEARANCE_VARIABLES["time_of_day"]
    assert out.is_file()
    assert json.loads(out.read_text())["schema"] == "npa.data_factory.configs.v1"


def test_generate_configs_is_deterministic_by_seed(tmp_path: Path) -> None:
    a = dfs.generate_configs(str(tmp_path / "a.json"), n_augmentations=2, seed="s")
    b = dfs.generate_configs(str(tmp_path / "b.json"), n_augmentations=2, seed="s")
    assert a["augmentations"] == b["augmentations"]


def test_grade_gate_promotes_above_threshold(tmp_path: Path, monkeypatch) -> None:
    scores = tmp_path / "vlm_eval_stub.json"
    scores.write_text(json.dumps({"score": 0.8}))
    captured = {}
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: captured.update(uri=uri, decision=decision),
    )
    decision = dfs.grade_gate(str(scores), str(tmp_path / "decision.json"), threshold=0.5)
    assert decision == "promote_checkpoint"
    assert captured["decision"] == "promote_checkpoint"


def test_grade_gate_loops_below_threshold(tmp_path: Path, monkeypatch) -> None:
    scores = tmp_path / "vlm_eval_stub.json"
    scores.write_text(json.dumps({"score": 0.1}))
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert dfs.grade_gate(str(scores), str(tmp_path / "decision.json"), threshold=0.5) == "loop_back"


def test_curate_counts_augmented_set(tmp_path: Path, monkeypatch) -> None:
    keys = [
        "p/cosmos_augmented/video_0_aug0/augmented_video.mp4",
        "p/cosmos_augmented/video_0_aug0/frame_01.png",
        "p/cosmos_augmented/video_0_aug1/augmented_video.mp4",
        "p/cosmos_augmented/video_1_aug0/frame_01.png",
    ]
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: keys)
    written = {}
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: written.update(payload=payload, uri=uri) or uri)
    report = dfs.curate("s3://b/p/cosmos_augmented/", "s3://b/p/curation/report.json")
    assert report["video_count"] == 2
    assert report["frame_count"] == 2
    assert set(report["clip_ids"]) == {"video_0_aug0", "video_0_aug1", "video_1_aug0"}
    assert report["status"] == "curated"


def test_finalize_aggregates_stage_artifacts(tmp_path: Path, monkeypatch) -> None:
    keys = [
        "physical-ai-data-factory/run1/input/video_0.mp4",
        "physical-ai-data-factory/run1/labeled_original/captions.json",
        "physical-ai-data-factory/run1/reports/sim2real.rrd",
    ]
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: keys)
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    report = dfs.finalize("s3://b/physical-ai-data-factory/run1/", "s3://b/physical-ai-data-factory/run1/reports/final.json")
    assert report["artifact_count"] == 3
    assert report["has_rrd"] is True
    assert report["stages"]["input"] == 1
