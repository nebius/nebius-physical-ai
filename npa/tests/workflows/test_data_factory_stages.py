"""Unit tests for the real Physical AI Data Factory stage functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from npa.workflows import data_factory_stages as dfs


def _mock_committed_manifest(
    monkeypatch: pytest.MonkeyPatch, keys: list[str], *, bucket: str = "b"
) -> None:
    """Make listed canonical test objects carry the real committed contract."""

    original = dfs._download_json
    videos = sorted(key for key in keys if key.endswith("/augmented_video.mp4"))
    variants = [
        {
            "clip": key.rsplit("/", 2)[-2],
            "augmented_video_uri": f"s3://{bucket}/{key}",
        }
        for key in videos
    ]

    def load(uri: str):
        if uri.rstrip("/").endswith("cosmos_augmented/manifest.json"):
            return {
                "schema": "npa.cosmos2.transfer.v1",
                "mode": "cosmos_transfer2.5_gpu",
                "status": "executed",
                "node_count": 1,
                "variant_count": len(variants),
                "variants": variants,
            }
        return original(uri)

    monkeypatch.setattr(dfs, "_download_json", load)


def test_attempt_keys_without_canonical_manifest_fail_closed() -> None:
    keys = ["run/cosmos_augmented/_attempts/orphan/clip/augmented_video.mp4"]
    with pytest.raises(RuntimeError, match="without a canonical manifest"):
        dfs._committed_augment_manifest(
            "s3://b/run/cosmos_augmented/", listed_keys=keys
        )


def test_generate_configs_writes_real_manifest(tmp_path: Path) -> None:
    out = tmp_path / "configs" / "manifest.json"
    result = dfs.generate_configs(str(out), n_augmentations=3, seed="run-x")
    assert result["n_augmentations"] == 3
    assert len(result["augmentations"]) == 3
    for combo in result["augmentations"]:
        # Appearance vars work for a replaceable physical-scene input.
        assert combo["color_grade"] in dfs.APPEARANCE_VARIABLES["color_grade"]
        assert combo["lighting"] in dfs.APPEARANCE_VARIABLES["lighting"]
        # Each combo carries the prompt that actually conditions the augmentation.
        assert combo["color_grade"] in combo["prompt"]
        assert "input-conditioned" in combo["prompt"]
    assert out.is_file()
    assert json.loads(out.read_text())["schema"] == "npa.data_factory.configs.v1"


def test_generate_configs_propagates_augmentation_subject_into_real_prompts(tmp_path: Path) -> None:
    result = dfs.generate_configs(
        str(tmp_path / "subject") + "/",
        n_augmentations=2,
        seed="run-subject",
        augment_subject="warehouse picking robot clips",
    )
    assert result["scene"] == "warehouse picking robot clips"
    assert all(
        "warehouse picking robot clips" in item["prompt"]
        for item in result["augmentations"]
    )


def test_prompt_from_combo_is_appearance_only() -> None:
    combo = {
        "color_grade": "warm",
        "surface_finish": "matte",
        "lighting": "dim evening light",
        "background": "plain wall",
    }
    prompt = dfs.prompt_from_combo(combo)
    assert "warm color grade" in prompt
    assert "matte surface finish" in prompt
    assert "change appearance only" in prompt
    assert "Preserve the exact input objects" in prompt
    assert "cloth" not in prompt


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
    decision = dfs.grade_gate(
        str(scores), str(tmp_path / "decision.json"), threshold=0.5
    )
    assert decision == "promote_checkpoint"
    assert captured["decision"] == "promote_checkpoint"


def test_grade_gate_loops_below_threshold(tmp_path: Path, monkeypatch) -> None:
    scores = tmp_path / "vlm_eval_stub.json"
    scores.write_text(json.dumps({"score": 0.1}))
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(scores), str(tmp_path / "decision.json"), threshold=0.5)
        == "loop_back"
    )


def test_grade_gate_accepts_string_threshold(tmp_path: Path, monkeypatch) -> None:
    """The blueprint interpolates a quoted config.grade_threshold; grade_gate must
    cast a str threshold (and fall back to 0.5 on a non-numeric value)."""
    scores = tmp_path / "vlm_eval_stub.json"
    scores.write_text(json.dumps({"score": 0.6}))
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    # "0.5" (str) -> 0.6 >= 0.5 -> promote.
    assert (
        dfs.grade_gate(str(scores), str(tmp_path / "d.json"), threshold="0.5")
        == "promote_checkpoint"
    )
    # non-numeric -> fallback 0.5 -> 0.6 >= 0.5 -> promote.
    assert (
        dfs.grade_gate(str(scores), str(tmp_path / "d.json"), threshold="bogus")
        == "promote_checkpoint"
    )


@pytest.mark.parametrize(
    "report",
    [
        {"score": "n/a"},  # non-numeric score
        {"score": None},
        {"score": {"overall": 0.9}},  # nested where a number is expected
        ["not", "an", "object"],  # JSON root is not a report at all
    ],
    ids=["non-numeric", "null", "nested", "not-an-object"],
)
def test_grade_gate_loops_back_on_a_malformed_report(
    tmp_path: Path, monkeypatch, report
) -> None:
    """A gate exists to make a decision, so a malformed score must not abort the loop.

    The report downloads cleanly here — only its ``score`` is unusable — so the
    download's own error handling never sees it.
    """

    scores = tmp_path / "cosmos_evaluator.json"
    scores.write_text(json.dumps(report))
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(scores), str(tmp_path / "d.json"), threshold=0.5)
        == "loop_back"
    )


def test_grade_gate_will_not_promote_a_degraded_report(
    tmp_path: Path, monkeypatch
) -> None:
    """A high score the evaluator itself flagged as degraded must not promote.

    The evaluator marks a run degraded when it lost object storage part-way, so the
    score reflects the clips it managed to read rather than the batch.
    """

    scores = tmp_path / "cosmos_evaluator.json"
    scores.write_text(json.dumps({"score": 0.95, "status": "degraded"}))
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(scores), str(tmp_path / "d.json"), threshold=0.5)
        == "loop_back"
    )


def test_grade_gate_will_not_promote_when_a_hard_check_failed(
    tmp_path: Path, monkeypatch
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    scores.write_text(
        json.dumps({"score": 0.95, "status": "completed", "passed": False})
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(scores), str(tmp_path / "d.json"), threshold=0.75)
        == "loop_back"
    )


def test_quality_disposition_accepts_only_a_complete_hard_check_pass(
    tmp_path: Path,
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    scores.write_text(
        json.dumps({"score": 0.81, "status": "completed", "passed": True})
    )
    result = dfs.enforce_quality_disposition(
        str(scores), str(disposition), threshold="0.75"
    )
    assert result["quality_status"] == "accepted"
    assert json.loads(disposition.read_text())["quality_status"] == "accepted"


def test_quality_disposition_persists_rejection_before_failing(tmp_path: Path) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    scores.write_text(
        json.dumps({"score": 0.9, "status": "completed", "passed": False})
    )
    with pytest.raises(RuntimeError, match="quality rejected"):
        dfs.enforce_quality_disposition(str(scores), str(disposition), threshold=0.75)
    payload = json.loads(disposition.read_text())
    assert payload["quality_status"] == "rejected"
    assert payload["hard_checks_passed"] is False


@pytest.mark.parametrize(
    "report_contents",
    [None, "not-json", '["valid", "json", "but-not-an-object"]'],
    ids=["missing", "unparseable", "non-object"],
)
def test_quality_disposition_is_written_for_every_unreadable_report(
    tmp_path: Path, report_contents: str | None
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    if report_contents is not None:
        scores.write_text(report_contents)
    with pytest.raises(RuntimeError, match="quality rejected"):
        dfs.enforce_quality_disposition(str(scores), str(disposition), threshold=0.75)
    payload = json.loads(disposition.read_text())
    assert payload["quality_status"] == "rejected"
    assert payload["evaluator_status"] == "missing"
    assert any("unavailable or malformed" in reason for reason in payload["reasons"])


def test_quality_disposition_rejects_a_non_numeric_score_and_persists_reason(
    tmp_path: Path,
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    scores.write_text(
        json.dumps({"score": "n/a", "status": "completed", "passed": True})
    )
    with pytest.raises(RuntimeError, match="quality rejected"):
        dfs.enforce_quality_disposition(str(scores), str(disposition), threshold=0.75)
    payload = json.loads(disposition.read_text())
    assert payload["score"] == 0.0
    assert "evaluator score is not numeric" in payload["reasons"]


def test_quality_disposition_resolves_a_report_prefix(tmp_path: Path) -> None:
    grade = tmp_path / "grade"
    grade.mkdir()
    (grade / "cosmos_evaluator.json").write_text(
        json.dumps({"score": 0.9, "status": "completed", "passed": True})
    )
    disposition = tmp_path / "quality_disposition.json"
    result = dfs.enforce_quality_disposition(
        str(grade), str(disposition), threshold=0.75
    )
    assert result["quality_status"] == "accepted"
    assert result["evaluator_report_uri"].endswith("grade/cosmos_evaluator.json")


def test_quality_disposition_rejects_a_degraded_report(tmp_path: Path) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    scores.write_text(json.dumps({"score": 1.0, "status": "degraded", "passed": True}))
    with pytest.raises(RuntimeError, match="quality rejected"):
        dfs.enforce_quality_disposition(str(scores), str(disposition), threshold=0.75)
    payload = json.loads(disposition.read_text())
    assert payload["quality_status"] == "rejected"
    assert "evaluator status is degraded" in payload["reasons"]


@pytest.mark.parametrize(
    ("report", "expected_status", "expected_decision"),
    [
        (
            {"score": 0.9, "status": "completed", "passed": True},
            "accepted",
            "promote_checkpoint",
        ),
        (
            {"score": 0.9, "status": "incomplete", "passed": True},
            "rejected",
            "loop_back",
        ),
        (
            {"score": 0.9, "status": "completed", "passed": False},
            "rejected",
            "loop_back",
        ),
    ],
)
def test_dynamic_quality_disposition_persists_strict_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report: dict,
    expected_status: str,
    expected_decision: str,
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    scores.write_text(json.dumps(report), encoding="utf-8")
    decisions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: decisions.append((uri, decision)),
    )

    result = dfs.write_quality_disposition(
        str(scores),
        str(disposition),
        "s3://example-bucket/run/decision.json",
        threshold=0.75,
    )

    assert result["quality_status"] == expected_status
    assert result["decision"] == expected_decision
    assert decisions == [
        ("s3://example-bucket/run/decision.json", expected_decision)
    ]
    assert json.loads(disposition.read_text())["quality_status"] == expected_status


@pytest.mark.parametrize("contents", [None, "not-json", "[]"])
def test_dynamic_quality_disposition_rejects_unavailable_or_malformed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str | None,
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    if contents is not None:
        scores.write_text(contents, encoding="utf-8")
    decisions: list[str] = []
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda _uri, decision: decisions.append(decision),
    )

    result = dfs.write_quality_disposition(
        str(scores), str(disposition), str(tmp_path / "decision.json"), 0.75
    )

    assert result["quality_status"] == "rejected"
    assert result["decision"] == "loop_back"
    assert result["reasons"]
    assert decisions == ["loop_back"]


def test_grade_gate_falls_through_a_malformed_report_to_the_older_contract(
    tmp_path: Path, monkeypatch
) -> None:
    """A malformed newest-contract report must not shadow a usable older one."""

    (tmp_path / "cosmos_evaluator.json").write_text(json.dumps({"score": "n/a"}))
    (tmp_path / "vlm_eval_stub.json").write_text(json.dumps({"score": 0.9}))
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert dfs.grade_gate(str(tmp_path), str(tmp_path / "d.json"), threshold=0.5) == (
        "promote_checkpoint"
    )


def test_download_json_missing_exact_file_does_not_substitute(
    tmp_path: Path, monkeypatch
) -> None:
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


def test_grade_gate_missing_eval_loops_not_reads_decision(
    tmp_path: Path, monkeypatch
) -> None:
    """A missing eval result must loop_back, never mis-read decision.json as score."""
    prefix_dir = tmp_path / "grade"
    prefix_dir.mkdir()
    # A promote decision.json is present but the eval result is absent.
    (prefix_dir / "decision.json").write_text(
        json.dumps({"decision": "promote_checkpoint"})
    )

    class _FakeStorage:
        def download_path(self, uri, dest):  # noqa: ARG002
            return str(prefix_dir)

    monkeypatch.setattr(dfs, "_storage", lambda: _FakeStorage())
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate("s3://bucket/grade/", "s3://bucket/grade/decision.json", 0.5)
        == "loop_back"
    )


def test_grade_gate_reads_the_cosmos_evaluator_report(
    tmp_path: Path, monkeypatch
) -> None:
    """The gate must threshold on the Cosmos Evaluator score the evaluate stage writes."""
    from npa.workbench.cosmos_evaluator import RESULT_FILENAME

    prefix_dir = tmp_path / "grade"
    prefix_dir.mkdir()
    (prefix_dir / RESULT_FILENAME).write_text(
        json.dumps({"score": 0.9, "passed": True})
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(prefix_dir), str(tmp_path / "decision.json"), 0.5)
        == "promote_checkpoint"
    )


def test_grade_gate_falls_back_to_the_older_vlm_eval_report(
    tmp_path: Path, monkeypatch
) -> None:
    """Runs started before the evaluate stage existed must still grade."""
    from npa.workbench.vlm_eval import RESULT_FILENAME

    prefix_dir = tmp_path / "grade"
    prefix_dir.mkdir()
    (prefix_dir / RESULT_FILENAME).write_text(json.dumps({"score": 0.2}))
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(prefix_dir), str(tmp_path / "decision.json"), 0.5)
        == "loop_back"
    )


def test_curate_merges_the_cosmos_curator_report(tmp_path: Path, monkeypatch) -> None:
    """The FiftyOne review report must carry the curator stage's summary."""
    curator_report = tmp_path / "cosmos_curator.json"
    curator_report.write_text(
        json.dumps(
            {
                "schema": "npa.cosmos_curate.curation.v1",
                "status": "completed",
                "engine": "cosmos-curator-stages",
                "curated_uri": "s3://b/p/curation/cosmos_curator/",
                "clip_count": 6,
                "filtered_count": 1,
                "variant_count": 2,
                "total_duration_s": 18.0,
                "motion_filter": "score-only",
            }
        )
    )
    keys = [
        "p/cosmos_augmented/manifest.json",
        "p/cosmos_augmented/aug-0/augmented_video.mp4",
        "p/cosmos_augmented/aug-1/augmented_video.mp4",
    ]
    monkeypatch.setattr(
        dfs,
        "_list_keys",
        lambda uri: keys,
    )
    _mock_committed_manifest(monkeypatch, keys)
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    report = dfs.curate(
        "s3://b/p/cosmos_augmented/",
        str(tmp_path / "report.json"),
        curator_report_uri=str(curator_report),
    )
    assert report["cosmos_curator"]["engine"] == "cosmos-curator-stages"
    assert report["cosmos_curator"]["clip_count"] == 6
    assert report["cosmos_curator"]["filtered_count"] == 1
    # The review stage's own findings are untouched by the merge.
    assert report["schema"] == "npa.fiftyone.curation.v1"
    assert report["multiply"]["mode"] == "multi-variant"


def test_curate_records_a_missing_curator_report_without_failing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        dfs, "_list_keys", lambda uri: ["p/cosmos_augmented/aug-0/augmented_video.mp4"]
    )
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    report = dfs.curate(
        "s3://b/p/cosmos_augmented/",
        str(tmp_path / "report.json"),
        curator_report_uri=str(tmp_path / "absent.json"),
    )
    assert report["cosmos_curator"]["status"] == "unavailable"
    assert report["status"] == "curated"


def test_curate_omits_the_curator_block_when_no_report_is_passed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        dfs, "_list_keys", lambda uri: ["p/cosmos_augmented/aug-0/augmented_video.mp4"]
    )
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    report = dfs.curate("s3://b/p/cosmos_augmented/", str(tmp_path / "report.json"))
    assert "cosmos_curator" not in report


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
    _mock_committed_manifest(monkeypatch, keys)
    written = {}
    monkeypatch.setattr(
        dfs,
        "_upload_json",
        lambda payload, uri: written.update(payload=payload, uri=uri) or uri,
    )
    report = dfs.curate("s3://b/p/cosmos_augmented/", "s3://b/p/curation/report.json")
    assert report["video_count"] == 1
    assert report["frame_count"] == 2
    assert set(report["clip_ids"]) == {"aug-run"}
    assert "manifest.json" not in report["clip_ids"]
    assert report["status"] == "curated"
    # Single-variant limitation surfaced in the machine-readable report.
    assert report["multiply"]["mode"] == "single-variant"
    # FiftyOne is not installed in the unit-test env, so curate degrades to the
    # report-only path (real Brain curation only runs inside the npa-fiftyone image).
    assert report["curation_engine"] == "report-only"


def test_generate_configs_feeds_first_augmentation_to_transfer(tmp_path: Path) -> None:
    """The sampled config manifest must be consumable by the augment stage."""
    from npa.cli.workbench.cosmos2 import _first_augmentation

    configs_uri = str(tmp_path / "configs") + "/"
    manifest = dfs.generate_configs(configs_uri, "3", seed="run-xyz")
    assert manifest["n_augmentations"] == 3

    combo = _first_augmentation(configs_uri)
    assert combo == manifest["augmentations"][0]
    assert set(combo) == {
        "lighting",
        "background",
        "color_grade",
        "surface_finish",
        "prompt",
    }
    # The prompt is what the augment stage feeds into Cosmos Transfer.
    assert combo["prompt"]


def test_generate_configs_non_numeric_count_falls_back(tmp_path: Path) -> None:
    manifest = dfs.generate_configs(str(tmp_path / "c") + "/", "not-a-number", seed="s")
    assert manifest["n_augmentations"] == 2


def _png(path: Path) -> Path:
    import pytest

    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), (20, 40, 60)).save(path)
    return path


def test_publish_transfer_layout_interoperates_with_curate_and_viz(
    tmp_path: Path, monkeypatch
) -> None:
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
    _mock_committed_manifest(monkeypatch, recorded, bucket="bkt")
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    report = dfs.curate(
        "s3://bkt/run1/cosmos_augmented/", "s3://bkt/run1/curation/report.json"
    )
    assert report["clip_ids"] == ["aug-run1"], report["clip_ids"]
    assert report["video_count"] == 1
    assert report["frame_count"] == 3
    assert "manifest.json" not in report["clip_ids"]

    # (b) build_run_rrd must consume the same per-clip layout (frames + metadata).
    out_rrd = tmp_path / "reports" / "sim2real.rrd"
    result = build_run_rrd(str(mirror / "run1"), str(out_rrd))
    assert result["frames_logged"] >= 3
    assert out_rrd.is_file()


def test_curate_reports_multi_variant_for_multiple_clips(
    tmp_path: Path, monkeypatch
) -> None:
    """N augmented clip dirs (multiply) must be counted and reported multi-variant."""
    keys = [
        "p/cosmos_augmented/manifest.json",
        "p/cosmos_augmented/aug-run-0/augmented_video.mp4",
        "p/cosmos_augmented/aug-run-0/frame-00000.png",
        "p/cosmos_augmented/aug-run-0/metadata.json",
        "p/cosmos_augmented/aug-run-1/augmented_video.mp4",
        "p/cosmos_augmented/aug-run-1/frame-00000.png",
        "p/cosmos_augmented/aug-run-1/metadata.json",
        "p/cosmos_augmented/aug-run-2/augmented_video.mp4",
        "p/cosmos_augmented/aug-run-2/metadata.json",
    ]
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: keys)
    _mock_committed_manifest(monkeypatch, keys)
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    report = dfs.curate("s3://b/p/cosmos_augmented/", "s3://b/p/curation/report.json")
    assert report["augmented_clips"] == 3
    assert set(report["clip_ids"]) == {"aug-run-0", "aug-run-1", "aug-run-2"}
    assert report["video_count"] == 3
    assert report["multiply"]["mode"] == "multi-variant"
    assert report["multiply"]["variant_count"] == 3


def test_finalize_reports_multi_variant_from_clip_dirs(
    tmp_path: Path, monkeypatch
) -> None:
    keys = [
        "physical-ai-data-factory/run1/input/video_0.mp4",
        "physical-ai-data-factory/run1/cosmos_augmented/manifest.json",
        "physical-ai-data-factory/run1/cosmos_augmented/aug-run1-0/augmented_video.mp4",
        "physical-ai-data-factory/run1/cosmos_augmented/aug-run1-1/augmented_video.mp4",
        "physical-ai-data-factory/run1/reports/sim2real.rrd",
    ]
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: keys)
    _mock_committed_manifest(monkeypatch, keys, bucket="b")
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    report = dfs.finalize(
        "s3://b/physical-ai-data-factory/run1/", "s3://b/.../final.json"
    )
    assert report["multiply_mode"] == "multi-variant"
    assert report["variant_count"] == 2


def test_all_augmentations_reads_every_combo(tmp_path: Path) -> None:
    from npa.cli.workbench.cosmos2 import _all_augmentations, _first_augmentation

    configs_uri = str(tmp_path / "configs") + "/"
    manifest = dfs.generate_configs(configs_uri, "4", seed="multi")
    combos = _all_augmentations(configs_uri)
    assert len(combos) == 4
    assert combos == manifest["augmentations"]
    assert _first_augmentation(configs_uri) == combos[0]


def test_all_augmentations_missing_manifest_fails_closed(tmp_path: Path) -> None:
    from npa.cli.workbench.cosmos2 import _all_augmentations

    with pytest.raises(
        typer.BadParameter,
        match="configured augmentation manifest could not be read",
    ):
        _all_augmentations(str(tmp_path / "nope") + "/")


def test_finalize_aggregates_stage_artifacts(tmp_path: Path, monkeypatch) -> None:
    keys = [
        "physical-ai-data-factory/run1/input/video_0.mp4",
        "physical-ai-data-factory/run1/labeled_original/captions.json",
        "physical-ai-data-factory/run1/reports/sim2real.rrd",
    ]
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: keys)
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    report = dfs.finalize(
        "s3://b/physical-ai-data-factory/run1/",
        "s3://b/physical-ai-data-factory/run1/reports/final.json",
    )
    assert report["artifact_count"] == 3
    assert report["has_rrd"] is True
    assert report["stages"]["input"] == 1
    assert report["multiply_mode"] == "single-variant"


def test_curation_and_final_reports_carry_input_provenance(monkeypatch) -> None:
    source = {
        "schema_version": "npa.paidf.input-provenance.v1",
        "source_kind": "upstream_sample",
        "input_origin": "actual_capture",
        "input_origin_label": "Upstream real sample",
        "sha256": "a" * 64,
        "staged_canonical_s3_uri": "s3://b/physical-ai-data-factory/run/input/",
        "derivation": {"kind": "normalized_conditioning_clip"},
    }
    monkeypatch.setattr(
        dfs,
        "_list_keys",
        lambda _uri: [
            "physical-ai-data-factory/run/cosmos_augmented/aug-run/frame-00000.png"
        ],
    )
    monkeypatch.setattr(dfs, "_download_json", lambda _uri: source)
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)

    curated = dfs.curate(
        "s3://b/physical-ai-data-factory/run/cosmos_augmented/",
        "s3://b/physical-ai-data-factory/run/curation/report.json",
    )
    final = dfs.finalize(
        "s3://b/physical-ai-data-factory/run/",
        "s3://b/physical-ai-data-factory/run/reports/final.json",
    )

    assert curated["input_source"] == source
    assert [group["name"] for group in curated["dataset_groups"]] == [
        "source",
        "conditioning",
        "augmented",
    ]
    assert final["input_source"] == source


def test_is_truthy_matches_common_values() -> None:
    for v in ("1", "true", "TRUE", "yes", "on", True):
        assert dfs._is_truthy(v) is True
    for v in ("", "0", "false", "no", None, "off"):
        assert dfs._is_truthy(v) is False


class _FakeStorage:
    def __init__(self) -> None:
        self.uploads: list[str] = []

    def upload_file(self, local: str, dest: str) -> str:
        assert Path(local).is_file()  # a real PNG was produced
        self.uploads.append(dest)
        return dest


def test_seed_default_input_frames_writes_when_empty(monkeypatch) -> None:
    monkeypatch.setattr(dfs, "_list_keys", lambda _uri: [])
    fake = _FakeStorage()
    monkeypatch.setattr(dfs, "_storage", lambda: fake)

    written = dfs._seed_default_input_frames("s3://b/physical-ai-data-factory/run/input/", count=3, seed="x")

    assert written == 3
    assert len(fake.uploads) == 3
    assert all(dest.endswith(".png") for dest in fake.uploads)
    assert fake.uploads[0].endswith("input/frame_0000.png")


def test_seed_default_input_frames_skips_when_images_exist(monkeypatch) -> None:
    monkeypatch.setattr(dfs, "_list_keys", lambda _uri: ["physical-ai-data-factory/run/input/frame_0000.png"])
    fake = _FakeStorage()
    monkeypatch.setattr(dfs, "_storage", lambda: fake)

    written = dfs._seed_default_input_frames("s3://b/physical-ai-data-factory/run/input/", seed="x")

    assert written == 0
    assert fake.uploads == []


def test_seed_default_input_frames_noop_without_uri() -> None:
    assert dfs._seed_default_input_frames("", seed="x") == 0


def test_generate_configs_seeds_default_input_when_flag_set(tmp_path: Path, monkeypatch) -> None:
    calls: dict[str, str] = {}

    def fake_seed(input_uri: str, seed: str = "") -> int:
        calls["input_uri"] = input_uri
        calls["seed"] = seed
        return 8

    monkeypatch.setattr(dfs, "_seed_default_input_frames", fake_seed)
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    result = dfs.generate_configs(
        str(tmp_path / "c.json"),
        n_augmentations=1,
        seed="run-x",
        input_uri="s3://b/physical-ai-data-factory/run-x/input/",
        seed_default_input="true",
    )
    assert result["seeded_default_input_frames"] == 8
    assert result["input_source"]["kind"] == "npa_seeded_fixture"
    assert result["input_source"]["staged_canonical_s3_uri"] == calls["input_uri"]
    assert result["input_source"]["frame_count"] == 8
    assert calls["input_uri"] == "s3://b/physical-ai-data-factory/run-x/input/"
    assert calls["seed"] == "run-x"


def test_generate_configs_no_seed_when_flag_false(tmp_path: Path, monkeypatch) -> None:
    def boom(*_a, **_k) -> int:  # pragma: no cover - must not run
        raise AssertionError("must not seed default input when the flag is falsy")

    monkeypatch.setattr(dfs, "_seed_default_input_frames", boom)
    result = dfs.generate_configs(
        str(tmp_path / "c.json"),
        n_augmentations=1,
        seed="s",
        input_uri="s3://b/input/",
        seed_default_input="false",
    )
    assert result["seeded_default_input_frames"] == 0
    assert result["input_source"]["kind"] == "operator_provided"


def test_generate_configs_records_the_seeded_count_in_the_written_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """The count has to be in the artifact, not just stdout.

    Regression: the manifest was uploaded before the seeding block ran, so
    configs/manifest.json never said whether the run used synthetic frames.
    """
    monkeypatch.setattr(dfs, "_seed_default_input_frames", lambda input_uri, seed="": 5)
    real_upload = dfs._upload_json
    monkeypatch.setattr(
        dfs,
        "_upload_json",
        lambda payload, uri: uri if uri.startswith("s3://") else real_upload(payload, uri),
    )
    configs = tmp_path / "c.json"

    dfs.generate_configs(
        str(configs),
        n_augmentations=1,
        seed="run-x",
        input_uri="s3://b/physical-ai-data-factory/run-x/input/",
        seed_default_input="true",
    )

    written = json.loads(configs.read_text(encoding="utf-8"))
    assert written["seeded_default_input_frames"] == 5
    assert written["input_source"]["kind"] == "npa_seeded_fixture"


def test_generate_configs_fails_when_requested_seeding_fails(tmp_path: Path, monkeypatch) -> None:
    """Requested-but-failed seeding must not defer the failure to annotate-original."""

    def boom(*_a, **_k) -> int:
        raise RuntimeError("Pillow is not installed")

    monkeypatch.setattr(dfs, "_seed_default_input_frames", boom)

    with pytest.raises(RuntimeError, match="seed_default_input was requested"):
        dfs.generate_configs(
            str(tmp_path / "c.json"),
            n_augmentations=1,
            seed="s",
            input_uri="s3://b/input/",
            seed_default_input="true",
        )


def test_generate_configs_fixture_refuses_existing_user_media(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dfs, "_seed_default_input_frames", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        dfs,
        "_download_json",
        lambda _uri: {
            "schema_version": "npa.paidf.input-provenance.v1",
            "source_kind": "user_supplied",
            "input_origin_label": "User-supplied input",
        },
    )

    with pytest.raises(RuntimeError, match="refusing to silently reuse or overwrite"):
        dfs.generate_configs(
            str(tmp_path / "c.json"),
            n_augmentations=1,
            seed="s",
            input_uri="s3://b/input/",
            seed_fixture="true",
        )
