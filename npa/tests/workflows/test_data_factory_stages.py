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


def test_download_json_missing_exact_file_does_not_substitute(tmp_path: Path, monkeypatch) -> None:
    """When the requested .json is missing and download falls back to the prefix
    dir, _download_json must raise, not silently return a different JSON."""
    import pytest

    prefix_dir = tmp_path / "grade"
    prefix_dir.mkdir()
    (prefix_dir / "decision.json").write_text(json.dumps({"decision": "loop_back"}))

    class _FakeStorage:
        def download_path(self, uri, dest):  # noqa: ARG002
            return str(prefix_dir)

    monkeypatch.setattr(dfs, "_storage", lambda: _FakeStorage())
    with pytest.raises(FileNotFoundError):
        dfs._download_json("s3://bucket/grade/vlm_eval_stub.json")


def test_grade_gate_missing_eval_loops_not_reads_decision(tmp_path: Path, monkeypatch) -> None:
    """A missing eval result must loop_back, never mis-read decision.json as score."""
    prefix_dir = tmp_path / "grade"
    prefix_dir.mkdir()
    # A promote decision.json is present but the eval result is absent.
    (prefix_dir / "decision.json").write_text(json.dumps({"decision": "promote_checkpoint"}))

    class _FakeStorage:
        def download_path(self, uri, dest):  # noqa: ARG002
            return str(prefix_dir)

    monkeypatch.setattr(dfs, "_storage", lambda: _FakeStorage())
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert dfs.grade_gate("s3://bucket/grade/", "s3://bucket/grade/decision.json", 0.5) == "loop_back"


def test_curate_counts_augmented_set(tmp_path: Path, monkeypatch) -> None:
    # Per-clip layout as emitted by publish_transfer_to_s3 (subdirs + top-level
    # manifest.json which must NOT be counted as a clip).
    keys = [
        "p/cosmos_augmented/manifest.json",
        "p/cosmos_augmented/aug-run/augmented_video.mp4",
        "p/cosmos_augmented/aug-run/frame-00000.png",
        "p/cosmos_augmented/aug-run/frame-00001.png",
        "p/cosmos_augmented/aug-run/metadata.json",
    ]
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: keys)
    written = {}
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: written.update(payload=payload, uri=uri) or uri)
    report = dfs.curate("s3://b/p/cosmos_augmented/", "s3://b/p/curation/report.json")
    assert report["video_count"] == 1
    assert report["frame_count"] == 2
    assert set(report["clip_ids"]) == {"aug-run"}
    assert "manifest.json" not in report["clip_ids"]
    assert report["status"] == "curated"
    # Single-variant limitation surfaced in the machine-readable report.
    assert report["multiply"]["mode"] == "single-variant"


def test_generate_configs_feeds_first_augmentation_to_transfer(tmp_path: Path) -> None:
    """The sampled config manifest must be consumable by the augment stage."""
    from npa.cli.workbench.cosmos2 import _first_augmentation

    configs_uri = str(tmp_path / "configs") + "/"
    manifest = dfs.generate_configs(configs_uri, "3", seed="run-xyz")
    assert manifest["n_augmentations"] == 3

    combo = _first_augmentation(configs_uri)
    assert combo == manifest["augmentations"][0]
    assert set(combo) == {"time_of_day", "weather", "road_condition"}


def test_generate_configs_non_numeric_count_falls_back(tmp_path: Path) -> None:
    manifest = dfs.generate_configs(str(tmp_path / "c") + "/", "not-a-number", seed="s")
    assert manifest["n_augmentations"] == 2


def _png(path: Path) -> Path:
    import pytest

    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), (20, 40, 60)).save(path)
    return path


def test_publish_transfer_layout_interoperates_with_curate_and_viz(tmp_path: Path, monkeypatch) -> None:
    """The real producer's S3 layout must flow through curate + build_run_rrd."""
    import pytest

    pytest.importorskip("rerun")
    from npa.workbench.cosmos import transfer as tx
    from npa.workflows.data_factory_viz import build_run_rrd

    video = tmp_path / "out.mp4"
    video.write_bytes(b"x" * 200_000)

    # Mock frame extraction (no cosmos venv here); write real PNGs into dest.
    def fake_extract(vp, dest, max_frames=8):
        return [_png(Path(dest) / f"frame-{i:05d}.png") for i in range(3)]

    monkeypatch.setattr(tx, "extract_frames", fake_extract)

    # Fake storage: mirror uploaded keys into a local tree so we can (a) collect
    # bucket-relative keys for curate, and (b) run build_run_rrd against the tree.
    mirror = tmp_path / "mirror"
    recorded: list[str] = []

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            key = uri.replace("s3://bkt/", "")
            recorded.append(key)
            out = mirror / key
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(Path(local).read_bytes())
            return uri

    manifest = tx.publish_transfer_to_s3(
        {"video_path": str(video), "video_bytes": 200_000, "spec": "s"},
        "s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        variables={"weather": "rainy", "time_of_day": "night"},
        storage_client=FakeStorage(),
    )
    assert manifest["frame_count"] == 3

    # (a) curate must parse the produced layout correctly.
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: recorded)
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    report = dfs.curate("s3://bkt/run1/cosmos_augmented/", "s3://bkt/run1/curation/report.json")
    assert report["clip_ids"] == ["aug-run1"], report["clip_ids"]
    assert report["video_count"] == 1
    assert report["frame_count"] == 3
    assert "manifest.json" not in report["clip_ids"]

    # (b) build_run_rrd must consume the same per-clip layout (frames + metadata).
    out_rrd = tmp_path / "reports" / "sim2real.rrd"
    result = build_run_rrd(str(mirror / "run1"), str(out_rrd))
    assert result["frames_logged"] >= 3
    assert out_rrd.is_file()


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
    assert report["multiply_mode"] == "single-variant"
