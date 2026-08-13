"""Unit tests for the Physical AI Data Factory Rerun recording builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.workflows.data_factory_viz import DataFactoryVizError, _frame_index, build_run_rrd


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def test_build_run_rrd_from_local_run(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    run = tmp_path / "df-run"
    # Input frames for two clips.
    _write_png(run / "input" / "video_0_frame_01.png", (10, 20, 30))
    _write_png(run / "input" / "video_0_frame_02.png", (40, 50, 60))
    _write_png(run / "input" / "video_1_frame_01.png", (70, 80, 90))
    # One augmented clip with metadata.
    aug = run / "cosmos_augmented" / "video_0_aug0"
    _write_png(aug / "frame_01.png", (11, 22, 33))
    (aug / "metadata.json").write_text('{"variables": {"weather": "rainy", "time_of_day": "night"}}')

    out = tmp_path / "reports" / "sim2real.rrd"
    result = build_run_rrd(str(run), str(out))

    assert result["status"] == "completed"
    assert result["frames_logged"] == 4
    assert result["run_id"] == "df-run"
    assert out.is_file()
    assert out.stat().st_size > 0


def test_frame_index_parses_both_naming_schemes() -> None:
    # Hyphen-delimited producer names (frame-00000) and underscore input names
    # (video_0_frame_01) must both yield distinct, ordered indices.
    assert _frame_index("frame-00000") == 0
    assert _frame_index("frame-00007") == 7
    assert _frame_index("video_0_frame_01") == 1
    assert _frame_index("video_0_frame_02") == 2
    assert _frame_index("noindex") == 0


def test_augmented_frames_get_distinct_time_points(tmp_path: Path, monkeypatch) -> None:
    """Hyphen-named augmented frames must map to distinct Rerun time-sequences."""
    pytest.importorskip("rerun")
    import npa.workflows.data_factory_viz as viz

    run = tmp_path / "df-run"
    aug = run / "cosmos_augmented" / "aug-run"
    for i in range(4):
        _write_png(aug / f"frame-{i:05d}.png", (10 * i, 20, 30))
    (aug / "metadata.json").write_text('{"variables": {"weather": "rainy"}}')

    seen: list[int] = []
    orig = viz._set_frame
    monkeypatch.setattr(viz, "_set_frame", lambda rr, rec, idx: (seen.append(idx), orig(rr, rec, idx))[-1])

    build_run_rrd(str(run), str(tmp_path / "reports" / "sim2real.rrd"))
    assert sorted(seen) == [0, 1, 2, 3], seen


def test_load_stage_docs_covers_all_pipeline_stages(tmp_path: Path) -> None:
    """The Rerun recording must surface every stage — scenarios, hallucination /
    grade, curation, finalize, and a stage log — not just the frames."""
    import json

    from npa.workflows.data_factory_viz import _load_stage_docs

    run = tmp_path / "run"
    (run / "configs").mkdir(parents=True)
    (run / "configs" / "manifest.json").write_text(json.dumps({
        "scene": "robot folding cloth",
        "augmentations": [
            {"cloth_color": "blue", "prompt": "a blue cloth, bright daylight"},
            {"cloth_color": "red", "prompt": "a red cloth, dim evening light"},
        ],
    }))
    (run / "input").mkdir(parents=True)
    (run / "input" / "provenance.json").write_text(
        json.dumps(
            {
                "source_kind": "upstream_sample",
                "input_origin_label": "Upstream real sample",
                "sha256": "a" * 64,
                "derivation": {"kind": "normalized_conditioning_clip"},
            }
        )
    )
    (run / "cosmos_augmented").mkdir(parents=True)
    (run / "cosmos_augmented" / "manifest.json").write_text(json.dumps({
        "mode": "cosmos_transfer2.5_gpu", "variant_count": 2, "input_conditioned": True,
        "clips": ["aug-0", "aug-1"], "variants": [{"clip": "aug-0"}, {"clip": "aug-1"}],
    }))
    (run / "grade").mkdir(parents=True)
    (run / "grade" / "vlm_eval_stub.json").write_text(json.dumps({"score": 0.82, "model": "Qwen/Qwen2.5-VL-72B-Instruct"}))
    (run / "grade" / "decision.json").write_text(json.dumps({"decision": "promote_checkpoint"}))
    (run / "curation").mkdir(parents=True)
    (run / "curation" / "report.json").write_text(json.dumps({"augmented_clips": 2, "multiply": {"mode": "multi-variant"}}))
    (run / "reports").mkdir(parents=True)
    (run / "reports" / "final.json").write_text(json.dumps({"artifact_count": 20, "multiply_mode": "multi-variant"}))

    docs = _load_stage_docs(run)
    assert set(docs) == {
        "pipeline/0_log",
        "pipeline/0_input_provenance",
        "pipeline/1_scenarios",
        "pipeline/2_augment",
        "pipeline/3_grade",
        "pipeline/4_curation",
        "pipeline/5_finalize",
    }
    assert "2 scenario" in docs["pipeline/1_scenarios"] or "Scenarios sampled:** 2" in docs["pipeline/1_scenarios"]
    assert "a red cloth" in docs["pipeline/1_scenarios"]
    assert "Upstream real sample" in docs["pipeline/0_input_provenance"]
    assert "normalized_conditioning_clip" in docs["pipeline/0_input_provenance"]
    # Hallucination / attribute-verify grade + gate decision are both present.
    assert "0.82" in docs["pipeline/3_grade"]
    assert "promote_checkpoint" in docs["pipeline/3_grade"]
    # Stage log lists each stage.
    assert "augment" in docs["pipeline/0_log"]
    assert "grade" in docs["pipeline/0_log"]


def test_build_run_rrd_logs_pipeline_docs(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: a run tree with stage reports logs pipeline/* text docs into the
    recording alongside the input/augmented frames."""
    pytest.importorskip("rerun")
    import json

    run = tmp_path / "run"
    _write_png(run / "input" / "video_0_frame_01.png", (10, 20, 30))
    aug = run / "cosmos_augmented" / "aug-0"
    _write_png(aug / "frame-00000.png", (11, 22, 33))
    (aug / "metadata.json").write_text('{"variables": {"cloth_color": "blue"}}')
    (run / "grade").mkdir(parents=True)
    (run / "grade" / "vlm_eval_stub.json").write_text(json.dumps({"score": 0.9}))

    import rerun as rr

    logged_entities: list[str] = []
    real_log = rr.log

    def spy_log(entity, *args, **kwargs):
        logged_entities.append(entity)
        return real_log(entity, *args, **kwargs)

    monkeypatch.setattr(rr, "log", spy_log)
    build_run_rrd(str(run), str(tmp_path / "reports" / "sim2real.rrd"))
    assert any(e.startswith("pipeline/") for e in logged_entities)
    assert "pipeline/3_grade" in logged_entities


def test_captions_carry_self_identifying_header(tmp_path: Path) -> None:
    """Caption docs must announce they are Token Factory captioning (not the VLM
    eval / hallucination gate) so the two are not confused in the Rerun grid."""
    import json

    from npa.workflows.data_factory_viz import _load_captions

    run = tmp_path / "run"
    (run / "labeled_original").mkdir(parents=True)
    (run / "labeled_original" / "captions.json").write_text(
        json.dumps({"captions": [{"image": "frame_01.png", "caption": "a robot arm folds cloth"}]})
    )
    (run / "labeled_augmented").mkdir(parents=True)
    (run / "labeled_augmented" / "captions.json").write_text(
        json.dumps({"captions": [{"image": "frame_01.png", "caption": "a blue cloth under warm light"}]})
    )

    caps = _load_captions(run)
    assert "Derived conditioning-frame captions" in caps["labeled_original"]
    assert "Augmented-clip captions" in caps["labeled_augmented"]
    # Each caption panel points at the grade panel and says it is NOT the gate.
    for body in caps.values():
        assert "pipeline/3_grade" in body
        assert "not the quality gate" in body
        assert "Token Factory VLM" in body
    # The actual caption text is still present.
    assert "a robot arm folds cloth" in caps["labeled_original"]


def test_build_run_rrd_requires_rrd_output(tmp_path: Path) -> None:
    with pytest.raises(DataFactoryVizError):
        build_run_rrd(str(tmp_path), str(tmp_path / "out.json"))


def test_build_run_rrd_errors_when_no_frames(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    empty = tmp_path / "empty-run"
    empty.mkdir()
    with pytest.raises(DataFactoryVizError):
        build_run_rrd(str(empty), str(tmp_path / "reports" / "sim2real.rrd"))
