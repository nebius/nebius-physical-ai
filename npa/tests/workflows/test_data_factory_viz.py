"""Unit tests for the Physical AI Data Factory Rerun recording builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from npa.workflows.data_factory_viz import (
    DataFactoryVizError,
    _committed_variant_dirs,
    _frame_index,
    build_run_rrd,
)


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
    rerun_cli = Path(sys.executable).with_name("rerun")
    assert rerun_cli.is_file()
    verified = subprocess.run(
        [str(rerun_cli), "rrd", "verify", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr


def test_rejected_rrd_component_stats_include_actual_augmented_media(
    tmp_path: Path,
) -> None:
    """Regression: rejected evidence must contain pixels/video, not URI-only text."""

    pytest.importorskip("rerun")
    run = tmp_path / "rejected-run"
    candidate = run / "cosmos_augmented" / "iteration-1" / "candidate-a"
    _write_png(candidate / "frame-00000.png", (12, 34, 56))
    video = candidate / "augmented_video.mp4"
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            str(candidate / "frame-00000.png"),
            "-t",
            "0.2",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert encoded.returncode == 0, encoded.stderr
    (candidate.parent / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "npa.cosmos2.transfer.v1",
                "mode": "cosmos_transfer2.5_gpu",
                "status": "executed",
                "node_count": 1,
                "variant_count": 1,
                "variants": [
                    {
                        "clip": "candidate-a",
                        "variant_index": 0,
                        "augmented_video_uri": (
                            "s3://test/run/cosmos_augmented/iteration-1/"
                            "candidate-a/augmented_video.mp4"
                        ),
                        "control_uris": {},
                    }
                ],
            }
        )
    )
    grade = run / "grade"
    (grade / "iteration-1" / "ranking").mkdir(parents=True)
    (grade / "iteration-1" / "ranking" / "cosmos_evaluator.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "clips": [
                    {
                        "clip_id": "candidate-a",
                        "score": 0.7,
                        "passed": False,
                        "attribute_verification": {
                            "passed": False,
                            "checks": [
                                {"variable": "lighting", "passed": False}
                            ],
                        },
                        "hallucination": {"passed": True},
                    }
                ],
            }
        )
    )
    (grade / "quality_disposition.json").write_text(
        json.dumps(
            {
                "quality_status": "rejected",
                "score": 0.7,
                "threshold": 0.75,
            }
        )
    )

    out = tmp_path / "reports" / "rejected.rrd"
    result = build_run_rrd(str(run), str(out))

    assert result["augmented_media_entities"] == 1
    assert result["augmented_frame_components"] == 1
    assert result["augmented_video_components"] == 1
    printed = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "print", "-v", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert printed.returncode == 0, printed.stderr
    component_stats = printed.stdout
    assert "augmented/iteration-1/candidate-a" in component_stats
    assert "@EncodedImage:blob" in component_stats
    assert "@AssetVideo:blob" in component_stats
    assert "augmented/iteration-1/candidate-a/disposition" in component_stats
    import npa.workflows.data_factory_viz as viz

    verified = viz._verify_terminal_rrd_media(
        out,
        variant_records=[
            {
                "candidate_id": "iteration-1/candidate-a",
                "video": video,
            }
        ],
        quality_status="REJECTED",
    )
    assert verified == {
        "augmented_video_entities": 1,
        "augmented_disposition_entities": 1,
    }
    video.write_bytes(b"changed")
    with pytest.raises(DataFactoryVizError, match="differs from its canonical"):
        viz._verify_terminal_rrd_media(
            out,
            variant_records=[
                {
                    "candidate_id": "iteration-1/candidate-a",
                    "video": video,
                }
            ],
            quality_status="REJECTED",
        )


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
    (run / "grade" / "cosmos_evaluator.json").write_text(
        json.dumps({"score": 0.72, "status": "completed", "passed": False})
    )
    (run / "grade" / "decision.json").write_text(
        json.dumps({"decision": "loop_back_to_inner_loop"})
    )
    (run / "grade" / "quality_disposition.json").write_text(
        json.dumps(
            {
                "quality_status": "rejected",
                "score": 0.72,
                "threshold": 0.75,
                "reasons": ["aggregate score is below threshold"],
            }
        )
    )
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
    assert "0.72" in docs["pipeline/3_grade"]
    assert "loop_back_to_inner_loop" in docs["pipeline/3_grade"]
    assert "rejected" in docs["pipeline/3_grade"]
    assert "aggregate score is below threshold" in docs["pipeline/3_grade"]


def test_stage_docs_select_latest_append_only_refinement_iteration(
    tmp_path: Path,
) -> None:
    from npa.workflows.data_factory_viz import _load_stage_docs

    run = tmp_path / "run"
    for iteration, score in ((1, 0.2), (2, 0.8)):
        augment = run / "cosmos_augmented" / f"iteration-{iteration}"
        augment.mkdir(parents=True)
        (augment / "manifest.json").write_text(
            json.dumps(
                {
                    "mode": "cosmos_transfer2.5_gpu",
                    "variant_count": iteration,
                    "input_conditioned": True,
                }
            )
        )
        grade = run / "grade" / f"iteration-{iteration}"
        grade.mkdir(parents=True)
        (grade / "cosmos_evaluator.json").write_text(
            json.dumps({"score": score, "status": "completed"})
        )
        (grade / "decision.json").write_text(
            json.dumps(
                {
                    "decision": "promote_checkpoint"
                    if iteration == 2
                    else "loop_back"
                }
            )
        )
    (run / "grade" / "quality_disposition.json").write_text(
        json.dumps({"quality_status": "accepted", "score": 0.8})
    )

    docs = _load_stage_docs(run)

    assert '"variant_count": 2' in docs["pipeline/2_augment"]
    assert '"score": 0.8' in docs["pipeline/3_grade"]
    assert '"score": 0.2' not in docs["pipeline/3_grade"]
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


def test_control_maps_are_logged_beside_the_variants_they_conditioned(
    tmp_path: Path, monkeypatch
) -> None:
    """A seg-conditioned run is only reviewable if the segmentation is visible."""
    pytest.importorskip("rerun")

    run = tmp_path / "run"
    aug = run / "cosmos_augmented" / "aug-0"
    _write_png(aug / "frame-00000.png", (11, 22, 33))
    control = run / "cosmos_control" / "aug-0"
    _write_png(control / "control_seg" / "frame-00000.png", (0, 255, 0))
    _write_png(control / "control_seg" / "frame-00001.png", (0, 200, 0))
    _write_png(control / "mask_seg" / "frame-00000.png", (255, 255, 255))

    import rerun as rr

    logged_entities: list[str] = []
    real_log = rr.log

    def spy_log(entity, *args, **kwargs):
        logged_entities.append(entity)
        return real_log(entity, *args, **kwargs)

    monkeypatch.setattr(rr, "log", spy_log)
    result = build_run_rrd(str(run), str(tmp_path / "reports" / "sim2real.rrd"))

    assert "control/aug-0/control_seg" in logged_entities
    assert "control/aug-0/mask_seg" in logged_entities
    # The three control frames count as logged frames alongside the variant's one.
    assert result["frames_logged"] == 4


def test_the_stage_log_names_what_conditioned_the_augment(tmp_path: Path) -> None:
    import json

    from npa.workflows.data_factory_viz import _load_stage_docs

    run = tmp_path / "run"
    (run / "cosmos_augmented").mkdir(parents=True)
    (run / "cosmos_augmented" / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "cosmos_transfer2.5_gpu",
                "variant_count": 2,
                "input_conditioned": True,
                "control": "seg",
                "control_prompt": "robot arm, conveyor",
                "mask_prompt": "robot arm",
                "clips": ["aug-0", "aug-1"],
            }
        )
    )

    log = _load_stage_docs(run)["pipeline/0_log"]
    assert "control=seg" in log
    assert "robot arm, conveyor" in log
    assert "masked to 'robot arm'" in log


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


def test_viewer_publication_preservation_check_is_additive_and_fail_closed() -> None:
    import npa.workflows.data_factory_viz as viz

    before = [
        {"key": "run/input/source.mp4", "size": 10, "etag": "source"},
        {"key": "run/cosmos_augmented/a/frame.png", "size": 20, "etag": "frame"},
        {"key": "run/npa-workflow/runtime.json", "size": 20, "etag": "old"},
    ]
    after = [
        before[0],
        before[1],
        {"key": "run/npa-workflow/runtime.json", "size": 21, "etag": "new"},
        {"key": "run/reports/sim2real.rrd", "size": 30, "etag": "rrd"},
    ]
    preserved = viz._verify_additive_publication(
        before, after, "run/reports/sim2real.rrd"
    )
    assert preserved == before[:2]
    assert (
        viz._verify_additive_publication(
            after, after, "run/reports/sim2real.rrd"
        )
        == before[:2]
    )

    changed = [
        {**before[0], "etag": "changed"},
        before[1],
        after[2],
        after[-1],
    ]
    with pytest.raises(DataFactoryVizError, match="changed the canonical"):
        viz._verify_additive_publication(
            before, changed, "run/reports/sim2real.rrd"
        )
    changed_rrd = [*after[:-1], {**after[-1], "etag": "changed"}]
    with pytest.raises(DataFactoryVizError, match="changed an existing recording"):
        viz._verify_additive_publication(
            after, changed_rrd, "run/reports/sim2real.rrd"
        )


def test_build_run_rrd_errors_when_no_frames(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    empty = tmp_path / "empty-run"
    empty.mkdir()
    with pytest.raises(DataFactoryVizError):
        build_run_rrd(str(empty), str(tmp_path / "reports" / "sim2real.rrd"))


def test_visualization_follows_only_committed_attempt_directories(tmp_path: Path) -> None:
    current = tmp_path / "cosmos_augmented" / "_attempts" / "current" / "aug-1"
    old = tmp_path / "cosmos_augmented" / "_attempts" / "old" / "aug-1"
    current.mkdir(parents=True)
    old.mkdir(parents=True)
    video = current / "augmented_video.mp4"
    video.write_bytes(b"current")
    (tmp_path / "cosmos_augmented" / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "npa.cosmos2.transfer.v1",
                "mode": "cosmos_transfer2.5_gpu",
                "status": "executed",
                "node_count": 2,
                "attempt_id": "current",
                "scheduler_fence_sequence": 2,
                "scheduler_fence_attempt": 1,
                "scheduler_launch_id": "job",
                "publication_generation": 2,
                "logical_publication": "conditional",
                "logical_wave_id": "grade-loop-2",
                "membership_digest": "current-members",
                "variant_count": 1,
                "variants": [
                    {
                        "clip": "aug-1",
                        "variant_index": 0,
                        "augmented_video_uri": (
                            "s3://bucket/run/cosmos_augmented/_attempts/"
                            "current/aug-1/augmented_video.mp4"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert _committed_variant_dirs(tmp_path) == [current]


def test_visualization_refuses_attempt_layout_without_manifest(tmp_path: Path) -> None:
    (tmp_path / "cosmos_augmented" / "_attempts" / "orphan").mkdir(parents=True)
    with pytest.raises(DataFactoryVizError, match="without a valid canonical"):
        _committed_variant_dirs(tmp_path)
