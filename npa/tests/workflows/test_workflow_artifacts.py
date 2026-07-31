from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from npa.workflows.artifacts import (
    Artifact,
    ArtifactDiscoveryError,
    artifact_media_type,
    build_fiftyone_dataset,
    download_s3_uri,
    find_run_artifacts,
    list_all_runs,
    list_artifacts,
    list_run_categories,
    list_runs,
    render_hint_for_object,
    select_preferred_artifact,
)


def test_build_fiftyone_dataset_groups_variants_and_summarizes() -> None:
    run = "paidf-demo"
    base = f"checkpoints/physical-ai-data-factory/{run}"
    keys = [
        f"{base}/input/video_0_frame_01.png",
        f"{base}/input/video_0_frame_02.png",
        f"{base}/cosmos_augmented/manifest.json",
        f"{base}/cosmos_augmented/aug-{run}-0/augmented_video.mp4",
        f"{base}/cosmos_augmented/aug-{run}-0/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json",
        f"{base}/cosmos_augmented/aug-{run}-1/augmented_video.mp4",
        f"{base}/cosmos_augmented/aug-{run}-1/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-1/metadata.json",
        f"{base}/labeled_augmented/captions.json",
        f"{base}/grade/vlm_eval_stub.json",
        f"{base}/grade/decision.json",
        f"{base}/curation/report.json",
    ]
    payloads = {
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json": {
            "variables": {"cloth_color": "green", "lighting": "warm lamp light", "prompt": "a green cloth"}
        },
        f"{base}/cosmos_augmented/aug-{run}-1/metadata.json": {
            "variables": {"cloth_color": "blue", "lighting": "cool overhead light", "prompt": "a blue cloth"}
        },
        f"{base}/labeled_augmented/captions.json": {
            "captions": [
                {"image": f"aug-{run}-0/frame-00000.png", "caption": "green cloth on a countertop"},
                {"image": f"aug-{run}-1/frame-00000.png", "caption": "blue cloth on a sofa"},
            ]
        },
        f"{base}/grade/vlm_eval_stub.json": {"score": 0.0, "model": "Qwen/Qwen2.5-VL-72B-Instruct"},
        f"{base}/grade/decision.json": {"decision": "loop_back"},
        f"{base}/curation/report.json": {"multiply": {"mode": "multi-variant", "variant_count": 2}},
    }

    dataset = build_fiftyone_dataset(keys, run_id=run, read_json=lambda k: payloads.get(k))

    summary = dataset["summary"]
    assert summary["augmented_count"] == 2
    assert summary["variant_count"] == 2
    assert summary["multiply_mode"] == "multi-variant"
    assert summary["grade_score"] == 0.0
    assert summary["grade_decision"] == "loop_back"
    assert summary["input_count"] == 2
    assert set(dataset["fields"]) == {"cloth_color", "lighting"}

    aug = [s for s in dataset["samples"] if s["group"] == "augmented"]
    inp = [s for s in dataset["samples"] if s["group"] == "input"]
    assert len(aug) == 2 and len(inp) == 2
    first = aug[0]
    assert first["thumbnail_key"].endswith("frame-00000.png")
    assert first["video_key"].endswith("augmented_video.mp4")
    assert "prompt" not in first["tags"]  # prompt is surfaced separately, not a tag
    assert first["caption"] == "green cloth on a countertop"
    # Report-only curation (no fiftyone block) -> empty curation surface.
    assert summary["curation_engine"] == ""
    assert first["uniqueness"] is None
    assert first["curated"] is None


def test_build_fiftyone_dataset_surfaces_real_fiftyone_curation() -> None:
    run = "paidf-fo"
    base = f"checkpoints/physical-ai-data-factory/{run}"
    keys = [
        f"{base}/cosmos_augmented/aug-{run}-0/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json",
        f"{base}/cosmos_augmented/aug-{run}-1/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-1/metadata.json",
        f"{base}/curation/report.json",
    ]
    payloads = {
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json": {"variables": {"cloth_color": "green"}},
        f"{base}/cosmos_augmented/aug-{run}-1/metadata.json": {"variables": {"cloth_color": "blue"}},
        f"{base}/curation/report.json": {
            "multiply": {"mode": "multi-variant", "variant_count": 2},
            "curation_engine": "fiftyone-brain",
            "curated_kept": 1,
            "curated_dropped": 1,
            "fiftyone": {
                "brain": {
                    "uniqueness": {"count": 2, "mean": 0.55, "min": 0.2, "max": 0.9},
                    "near_duplicate_count": 1,
                },
                "selection": {"near_duplicate_count": 1},
                "samples": {
                    f"aug-{run}-0": {"uniqueness": 0.9, "kept": True, "redundant": False},
                    f"aug-{run}-1": {"uniqueness": 0.2, "kept": False, "redundant": True},
                },
            },
        },
    }

    dataset = build_fiftyone_dataset(keys, run_id=run, read_json=lambda k: payloads.get(k))
    summary = dataset["summary"]
    assert summary["curation_engine"] == "fiftyone-brain"
    assert summary["curated_kept"] == 1
    assert summary["curated_dropped"] == 1
    assert summary["near_duplicate_count"] == 1
    assert summary["uniqueness"]["mean"] == 0.55

    aug = sorted(
        (s for s in dataset["samples"] if s["group"] == "augmented"), key=lambda s: s["id"]
    )
    assert aug[0]["uniqueness"] == 0.9
    assert aug[0]["curated"] is True
    assert aug[0]["curation_flags"] == []
    assert aug[1]["uniqueness"] == 0.2
    assert aug[1]["curated"] is False
    assert aug[1]["curation_flags"] == ["redundant"]


def test_build_fiftyone_dataset_surfaces_input_captions_video_and_visualization() -> None:
    run = "paidf-more"
    base = f"checkpoints/physical-ai-data-factory/{run}"
    keys = [
        f"{base}/input/video_0.mp4",
        f"{base}/input/frame_01.png",
        f"{base}/input/frame_02.png",
        f"{base}/labeled_original/captions.json",
        f"{base}/cosmos_augmented/aug-{run}-0/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json",
        f"{base}/cosmos_augmented/aug-{run}-1/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-1/metadata.json",
        f"{base}/curation/report.json",
    ]
    payloads = {
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json": {"variables": {"cloth_color": "green"}},
        f"{base}/cosmos_augmented/aug-{run}-1/metadata.json": {"variables": {"cloth_color": "blue"}},
        f"{base}/labeled_original/captions.json": {
            "captions": [
                {"image": "frame_01.png", "caption": "source: robot arm, plain wall"},
                {"image": "frame_02.png", "caption": "source: robot arm mid-fold"},
            ]
        },
        f"{base}/curation/report.json": {
            "curation_engine": "fiftyone-brain",
            "fiftyone": {
                "visualization": [
                    {"id": f"aug-{run}-0", "point": [0.1, 0.2]},
                    {"id": f"aug-{run}-1", "point": [0.9, -0.3]},
                ],
                "samples": {
                    f"aug-{run}-0": {"uniqueness": 0.9, "kept": True, "redundant": False},
                    f"aug-{run}-1": {"uniqueness": 0.2, "kept": True, "redundant": True},
                },
            },
        },
    }

    dataset = build_fiftyone_dataset(keys, run_id=run, read_json=lambda k: payloads.get(k), bucket="bkt")

    # Visualization surfaced at the top level.
    assert len(dataset["visualization"]) == 2

    inp = [s for s in dataset["samples"] if s["group"] == "input"]
    # Source video is now included as an input sample (with a poster + video_uri).
    videos = [s for s in inp if s["video_uri"]]
    assert len(videos) == 1
    assert videos[0]["video_uri"].endswith("video_0.mp4")
    # Input frames carry their annotate-original captions.
    frame1 = next(s for s in inp if s["id"] == "frame_01.png")
    assert frame1["caption"] == "source: robot arm, plain wall"

    # Augmented samples carry their PCA point.
    aug = sorted((s for s in dataset["samples"] if s["group"] == "augmented"), key=lambda s: s["id"])
    assert aug[0]["point"] == [0.1, 0.2]


class _FakePaginator:
    def __init__(self, pages: list[dict]):
        self._pages = pages

    def paginate(self, **_kwargs):
        for page in self._pages:
            yield page


class _FakeS3:
    def __init__(self, pages: list[dict]):
        self._pages = pages
        self.download_calls: list[tuple[str, str, str]] = []

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return _FakePaginator(self._pages)

    def download_file(self, bucket: str, key: str, dest: str) -> None:
        self.download_calls.append((bucket, key, dest))
        Path(dest).write_text("ok", encoding="utf-8")


def _obj(key: str, size: int = 1, ts: str = "2026-06-30T00:00:00+00:00") -> dict:
    return {
        "Key": key,
        "Size": size,
        "LastModified": datetime.fromisoformat(ts).astimezone(timezone.utc),
    }


def test_list_artifacts_returns_all_objects_including_unknown_extension() -> None:
    s3 = _FakeS3(
        [
            {
                "Contents": [
                    _obj("run-a/reports/sim2real.rrd", 128),
                    _obj("run-a/metrics/report.json", 22),
                    _obj("run-a/raw/new-format.fooz", 19),
                ]
            }
        ]
    )
    artifacts = list_artifacts("bucket", "run-a", s3=s3)
    keys = [item.key for item in artifacts]
    assert "run-a/reports/sim2real.rrd" in keys
    assert "run-a/metrics/report.json" in keys
    assert "run-a/raw/new-format.fooz" in keys
    unknown = next(item for item in artifacts if item.key.endswith(".fooz"))
    assert unknown.render == "download"
    assert unknown.inline is False


@pytest.mark.parametrize("suffix", [".newkind", ".novelblob", ".artifactx"])
def test_new_artifact_type_is_discoverable_without_code_changes(suffix: str) -> None:
    s3 = _FakeS3([{"Contents": [_obj(f"run-b/data/object{suffix}", 7)]}])
    artifacts = list_artifacts("bucket", "run-b", s3=s3)
    assert len(artifacts) == 1
    assert artifacts[0].key.endswith(suffix)
    assert artifacts[0].render == "download"


def test_select_preferred_artifact_ranks_rerun_highest() -> None:
    artifacts = [
        Artifact("run", "run/frame.png", "s3://bucket/run/frame.png", 1, "2026-01-01T00:00:00+00:00", "image", True),
        Artifact("run", "run/trace.rrd", "s3://bucket/run/trace.rrd", 1, "2026-01-01T00:00:00+00:00", "rerun", True),
        Artifact("run", "run/out.mp4", "s3://bucket/run/out.mp4", 1, "2026-01-01T00:00:00+00:00", "video", True),
    ]
    chosen = select_preferred_artifact(artifacts)
    assert chosen is not None
    assert chosen.render == "rerun"


def test_select_preferred_artifact_chooses_run_report_rrd_before_component_images() -> None:
    artifacts = [
        Artifact(
            "run",
            "run/component-io/vlm/input/rollout/camera-001.ppm",
            "s3://bucket/run/component-io/vlm/input/rollout/camera-001.ppm",
            1,
            "2026-01-02T00:00:00+00:00",
            "image",
            True,
        ),
        Artifact(
            "run",
            "run/reports/sim2real.rrd",
            "s3://bucket/run/reports/sim2real.rrd",
            1,
            "2026-01-01T00:00:00+00:00",
            "rerun",
            True,
        ),
    ]
    chosen = select_preferred_artifact(artifacts)
    assert chosen is not None
    assert chosen.key.endswith("reports/sim2real.rrd")


def test_select_preferred_artifact_keeps_unknown_download_selectable() -> None:
    artifacts = [
        Artifact("run", "run/raw.foo", "s3://bucket/run/raw.foo", 1, "2026-01-01T00:00:00+00:00", "download", False)
    ]
    chosen = select_preferred_artifact(artifacts)
    assert chosen is not None
    assert chosen.key.endswith("raw.foo")


def test_list_runs_reports_truncation_explicitly() -> None:
    s3 = _FakeS3(
        [
            {"Contents": [_obj("run-1/a.txt"), _obj("run-2/b.txt"), _obj("run-3/c.txt")]},
        ]
    )
    page = list_runs("bucket", limit=2, s3=s3)
    assert page.total_runs == 3
    assert page.truncated is True
    assert len(page.runs) == 2


def test_download_s3_uri_fetches_explicit_object(tmp_path: Path) -> None:
    s3 = _FakeS3([])
    dest = tmp_path / "artifact.bin"
    output = download_s3_uri("s3://bucket-a/path/to/object.bin", dest, s3=s3)
    assert output == dest
    assert s3.download_calls == [("bucket-a", "path/to/object.bin", str(dest))]


def test_render_hint_detects_text_csv_and_unknown_fallback() -> None:
    assert render_hint_for_object(key="x/table.csv") == "text"
    assert render_hint_for_object(key="x/video.bin", content_type="video/mp4") == "video"
    assert render_hint_for_object(key="x/opaque.new") == "download"


def test_render_hint_maps_mcap_to_lichtblick_render() -> None:
    from npa.workflows.artifacts import is_inline_render

    assert render_hint_for_object(key="run/reports/sim2real.mcap") == "mcap"
    # MCAP is an inline (viewable) render so the artifact browser offers it.
    assert is_inline_render("mcap") is True


def test_render_hint_and_media_type_handle_ppm_as_image() -> None:
    # .ppm (sim2real rollout camera dumps) are classified as images and transcoded to
    # PNG on serve (browsers cannot decode PPM), so they render in the Image pane.
    from npa.workflows.artifacts import artifact_media_type

    for key in ("run/rollout/camera-000.ppm", "run/x.pgm", "run/y.bmp", "run/z.tiff"):
        assert render_hint_for_object(key=key) == "image"
    assert artifact_media_type("camera-000.ppm") == "image/png"
    assert artifact_media_type("x.bmp") == "image/png"


def test_artifact_media_type_prefers_explicit_browser_types() -> None:
    assert artifact_media_type("demo.mp4") == "video/mp4"
    assert artifact_media_type("demo.webm") == "video/webm"
    assert artifact_media_type("shot.png") == "image/png"
    assert artifact_media_type("notes.md").startswith("text/plain")
    assert artifact_media_type("blob.bin") == "application/octet-stream"


def test_list_runs_requires_positive_limit() -> None:
    with pytest.raises(ArtifactDiscoveryError):
        list_runs("bucket", limit=0, s3=_FakeS3([]))


def test_list_runs_started_at_uses_run_start_not_newest_write() -> None:
    """started_at reflects when the run started; last_modified the newest write."""
    s3 = _FakeS3(
        [
            {
                "Contents": [
                    # Run started 2026-07-25 22:26:36Z (encoded in the id); first
                    # artifact a few seconds later, newest artifact two days on.
                    _obj("sim2real-b/s2r-real-0725t222636z/env/data.json", ts="2026-07-25T22:26:39+00:00"),
                    _obj("sim2real-b/s2r-real-0725t222636z/reports/sim2real.rrd", ts="2026-07-27T02:17:31+00:00"),
                ]
            }
        ]
    )
    page = list_runs("bucket", prefix="sim2real-b", limit=50, s3=s3)
    run = next(r for r in page.runs if r.run_id == "s2r-real-0725t222636z")
    # Newest write is July 27; the displayed start is July 25 (id-encoded time).
    assert run.last_modified == "2026-07-27T02:17:31+00:00"
    assert run.started_at == "2026-07-25T22:26:36+00:00"
    assert run.to_dict()["started_at"] == "2026-07-25T22:26:36+00:00"


def test_list_runs_started_at_falls_back_to_earliest_write() -> None:
    """Runs whose id has no embedded timestamp start at the earliest artifact."""
    s3 = _FakeS3(
        [
            {
                "Contents": [
                    _obj("cat/plain-run/a.json", ts="2026-05-10T08:00:00+00:00"),
                    _obj("cat/plain-run/b.json", ts="2026-05-12T09:00:00+00:00"),
                ]
            }
        ]
    )
    page = list_runs("bucket", prefix="cat", limit=50, s3=s3)
    run = next(r for r in page.runs if r.run_id == "plain-run")
    assert run.started_at == "2026-05-10T08:00:00+00:00"


def test_parse_run_id_timestamps_handles_full_and_yearless_forms() -> None:
    from npa.workflows.artifacts import _parse_run_id_timestamps, _run_started_at

    assert _parse_run_id_timestamps("job-20260725T222636Z") == ["2026-07-25T22:26:36+00:00"]
    # Year-less: the hinted year plus the prior year are offered as candidates.
    assert _parse_run_id_timestamps("s2r-real-0725t222636z", year_hint=2026) == [
        "2026-07-25T22:26:36+00:00",
        "2025-07-25T22:26:36+00:00",
    ]
    # No embedded timestamp -> nothing parsed, fall back to the earliest write.
    assert _parse_run_id_timestamps("free-form-run-name") == []
    assert _run_started_at("free-form-run-name", "2026-05-10T08:00:00+00:00") == "2026-05-10T08:00:00+00:00"


def test_run_started_at_distrusts_far_off_id_date() -> None:
    from npa.workflows.artifacts import _run_started_at

    # A far-off id date (not just-before the first write) is distrusted.
    assert (
        _run_started_at("legacy-v20200101t000000-run", "2026-05-10T08:00:00+00:00")
        == "2026-05-10T08:00:00+00:00"
    )


def test_run_started_at_picks_real_start_past_a_red_herring_timestamp() -> None:
    from npa.workflows.artifacts import _run_started_at

    # A leading red-herring timestamp (2020) plus the real year-less start; the
    # latest candidate at/just-before the first write (2026-07-25) is chosen.
    started = _run_started_at("legacy-v20200101t000000-s2r-0725t222636z", "2026-07-25T22:26:39+00:00")
    assert started == "2026-07-25T22:26:36+00:00"


def test_run_started_at_handles_year_boundary() -> None:
    from npa.workflows.artifacts import _run_started_at

    # Started 2025-12-31 23:59:00; first artifact 2026-01-01. The prior-year
    # candidate is selected so the start isn't after the first write.
    started = _run_started_at("run-1231t235900z", "2026-01-01T00:00:05+00:00")
    assert started == "2025-12-31T23:59:00+00:00"


class _PrefixAwareS3:
    """Fake S3 that honors Prefix + Delimiter over an in-memory key store."""

    def __init__(self, keys: list[tuple[str, str]]):
        # keys: list of (key, iso_ts)
        self._keys = keys

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        store = self._keys

        class _P:
            def paginate(self, Bucket=None, Prefix="", Delimiter=None):  # noqa: N803
                if Delimiter:
                    seen: set[str] = set()
                    cps: list[dict] = []
                    for key, _ts in store:
                        if not key.startswith(Prefix):
                            continue
                        rest = key[len(Prefix):]
                        if "/" not in rest:
                            continue
                        seg = rest.split("/", 1)[0]
                        cp = Prefix + seg + "/"
                        if seg and cp not in seen:
                            seen.add(cp)
                            cps.append({"Prefix": cp})
                    yield {"CommonPrefixes": cps}
                else:
                    contents = [
                        _obj(key, ts=ts) for key, ts in store if key.startswith(Prefix)
                    ]
                    yield {"Contents": contents}

        return _P()


_LAYOUT = [
    ("checkpoints/sim2real-b/run-a/reports/sim2real.rrd", "2026-07-01T00:00:00+00:00"),
    ("checkpoints/physical-ai-data-factory/paidf-1/cosmos_augmented/f.png", "2026-07-22T00:00:00+00:00"),
    ("checkpoints/lerobot/default/model.pt", "2026-06-01T00:00:00+00:00"),
]


def test_list_run_categories_enumerates_dynamically() -> None:
    s3 = _PrefixAwareS3(_LAYOUT)
    cats = list_run_categories("bucket", base_prefix="checkpoints", s3=s3)
    assert set(cats) == {
        "checkpoints/sim2real-b",
        "checkpoints/physical-ai-data-factory",
        "checkpoints/lerobot",
    }


def test_list_all_runs_merges_across_categories_latest_first() -> None:
    s3 = _PrefixAwareS3(_LAYOUT)
    page = list_all_runs("bucket", base_prefix="checkpoints", limit=50, s3=s3)
    ids = [r.run_id for r in page.runs]
    # All runs across every category, no hardcoded workflow path; newest first.
    assert ids == ["paidf-1", "run-a", "default"]
    assert page.total_runs == 3


def test_find_run_artifacts_locates_run_in_any_category() -> None:
    s3 = _PrefixAwareS3(_LAYOUT)
    arts = find_run_artifacts("bucket", base_prefix="checkpoints", run_id="paidf-1", s3=s3)
    assert [a.key for a in arts] == [
        "checkpoints/physical-ai-data-factory/paidf-1/cosmos_augmented/f.png"
    ]
    # A run under a different category is also found without a hardcoded prefix.
    assert find_run_artifacts("bucket", base_prefix="checkpoints", run_id="run-a", s3=s3)
    assert find_run_artifacts("bucket", base_prefix="checkpoints", run_id="missing", s3=s3) == []


# Runs also live at the BUCKET ROOT under a category (not under the configured
# base root), e.g. scenario-gen-smoke/<run>/... and physical-ai-data-factory/<run>/...
# Discovery must span both roots so these are visible + openable.
_MULTI_ROOT_LAYOUT = _LAYOUT + [
    ("scenario-gen-smoke/scenario-gen-smoke-1/npa-workflow/manifest.json", "2026-07-23T15:32:22+00:00"),
    ("scenario-gen-smoke/scenario-gen-smoke-1/ranked/ranked.json", "2026-07-23T15:32:20+00:00"),
    ("physical-ai-data-factory/paidf-root-1/reports/final.json", "2026-07-19T00:00:00+00:00"),
]


def test_discovery_categories_spans_base_and_bucket_root() -> None:
    from npa.workflows.artifacts import discovery_categories

    cats = discovery_categories("bucket", base_prefix="checkpoints", s3=_PrefixAwareS3(_MULTI_ROOT_LAYOUT))
    # Base-root categories come first, then root-level categories; the base root
    # itself ("checkpoints") is NOT treated as a run parent (its children are cats).
    assert "checkpoints/sim2real-b" in cats
    assert "checkpoints/physical-ai-data-factory" in cats
    assert "scenario-gen-smoke" in cats
    assert "physical-ai-data-factory" in cats
    assert "checkpoints" not in cats


def test_list_all_runs_surfaces_root_level_runs() -> None:
    s3 = _PrefixAwareS3(_MULTI_ROOT_LAYOUT)
    page = list_all_runs("bucket", base_prefix="checkpoints", limit=50, s3=s3)
    ids = [r.run_id for r in page.runs]
    # Root-level runs are discovered alongside checkpoints runs, newest first.
    assert ids[0] == "scenario-gen-smoke-1"
    assert set(ids) == {"scenario-gen-smoke-1", "paidf-1", "paidf-root-1", "run-a", "default"}


def test_find_run_artifacts_locates_root_level_run() -> None:
    s3 = _PrefixAwareS3(_MULTI_ROOT_LAYOUT)
    arts = find_run_artifacts(
        "bucket", base_prefix="checkpoints", run_id="scenario-gen-smoke-1", s3=s3
    )
    keys = sorted(a.key for a in arts)
    assert keys == [
        "scenario-gen-smoke/scenario-gen-smoke-1/npa-workflow/manifest.json",
        "scenario-gen-smoke/scenario-gen-smoke-1/ranked/ranked.json",
    ]


def test_list_runs_skips_bare_files_not_run_dirs() -> None:
    # A file sitting directly under a category is not a run directory.
    s3 = _PrefixAwareS3([
        ("scenario-gen-smoke/records.json", "2026-07-23T00:00:00+00:00"),
        ("scenario-gen-smoke/real-run-1/npa-workflow/status.json", "2026-07-23T10:00:00+00:00"),
    ])
    page = list_runs("bucket", prefix="scenario-gen-smoke", limit=50, s3=s3)
    ids = [r.run_id for r in page.runs]
    assert ids == ["real-run-1"]
    assert "records.json" not in ids


def test_discovery_categories_excludes_infra_roots() -> None:
    from npa.workflows.artifacts import discovery_categories

    layout = _MULTI_ROOT_LAYOUT + [
        ("npa-agent/session-state/a/b.json", "2026-07-23T00:00:00+00:00"),
        ("npa-agent/tenants/t/chat-sessions/s.json", "2026-07-23T00:00:00+00:00"),
    ]
    cats = discovery_categories(
        "bucket", base_prefix="checkpoints", exclude={"npa-agent"}, s3=_PrefixAwareS3(layout)
    )
    assert "npa-agent" not in cats
    assert "scenario-gen-smoke" in cats


def test_list_all_runs_excludes_infra_roots() -> None:
    layout = _MULTI_ROOT_LAYOUT + [
        ("npa-agent/session-state/a/state.json", "2026-07-23T00:00:00+00:00"),
    ]
    page = list_all_runs(
        "bucket", base_prefix="checkpoints", limit=50, exclude={"npa-agent"}, s3=_PrefixAwareS3(layout)
    )
    ids = [r.run_id for r in page.runs]
    assert "session-state" not in ids
    assert "scenario-gen-smoke-1" in ids


def test_ppm_and_netpbm_are_images_and_need_transcode() -> None:
    from npa.workflows.artifacts import needs_image_transcode, render_hint_for_object

    # Sim-rollout camera frames are saved as .ppm — classified as viewable images.
    assert render_hint_for_object(key="run/actions/rollout/camera-000.ppm") == "image"
    assert render_hint_for_object(key="run/x.bmp") == "image"
    # Browser cannot render these natively → must transcode to PNG on the way out.
    for name in ("camera-000.ppm", "x.pgm", "y.bmp", "z.tiff"):
        assert needs_image_transcode(name) is True
    # Web-native images are served as-is (no transcode).
    for name in ("frame.png", "a.jpg", "b.webp"):
        assert needs_image_transcode(name) is False
    assert render_hint_for_object(key="run/frame.png") == "image"


def test_list_runs_contains_search_survives_limit() -> None:
    # An old run beyond the newest `limit` must still be found by substring search.
    keys = [(f"cat/run-new-{i:03d}/reports/r.json", f"2026-07-{(i%27)+1:02d}T00:00:00+00:00") for i in range(30)]
    keys.append(("cat/rtxpro-staged-2x2-old/actions/train/camera-000.ppm", "2026-06-13T01:13:56+00:00"))
    s3 = _PrefixAwareS3(keys)
    # Without search, limit=5 returns only the 5 newest → the old run is cut off.
    page = list_runs("bucket", prefix="cat", limit=5, s3=s3)
    assert "rtxpro-staged-2x2-old" not in [r.run_id for r in page.runs]
    # With substring search, the old run is found despite the small limit.
    page = list_runs("bucket", prefix="cat", limit=5, contains="rtxpro-staged-2x2", s3=s3)
    assert [r.run_id for r in page.runs] == ["rtxpro-staged-2x2-old"]


def test_list_all_runs_contains_search_across_roots() -> None:
    layout = _MULTI_ROOT_LAYOUT + [
        ("sim2real-b/rtxpro-staged-2x2-old/actions/train/camera-000.ppm", "2026-06-13T01:13:56+00:00"),
    ]
    page = list_all_runs("bucket", base_prefix="checkpoints", limit=3, contains="rtxpro-staged", s3=_PrefixAwareS3(layout))
    assert [r.run_id for r in page.runs] == ["rtxpro-staged-2x2-old"]


def test_list_runs_cached_serves_fresh_then_refreshes_when_stale(monkeypatch) -> None:
    """Fresh hits avoid re-walking S3; a stale entry is refreshed (here inline)."""
    import npa.workflows.artifacts as A

    A._run_list_cache_clear()
    calls = {"n": 0}

    def fake_all(bucket, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        run = A.RunSummary(
            run_id=f"run-{calls['n']}", last_modified="2026-07-24T00:00:00+00:00",
            artifact_count=1, has_viewable=True,
        )
        return A.RunListPage(runs=[run], truncated=False, total_runs=1, limit=kwargs.get("limit", 50))

    monkeypatch.setattr(A, "list_all_runs", fake_all)

    # Cold miss -> computes once.
    p1 = A.list_runs_cached("bucket", all_categories=True, base_prefix="checkpoints", ttl=1000, s3=object())
    assert calls["n"] == 1 and p1.runs[0].run_id == "run-1"
    # Fresh hit -> no recompute, same page.
    p2 = A.list_runs_cached("bucket", all_categories=True, base_prefix="checkpoints", ttl=1000, s3=object())
    assert calls["n"] == 1 and p2.runs[0].run_id == "run-1"
    # Stale (ttl=0) -> inline refresh recomputes and returns fresh.
    p3 = A.list_runs_cached(
        "bucket", all_categories=True, base_prefix="checkpoints", ttl=0, s3=object(), refresh_sync=True
    )
    assert calls["n"] == 2 and p3.runs[0].run_id == "run-2"
    A._run_list_cache_clear()


def test_list_runs_cached_prefix_path_matches_list_runs() -> None:
    """The prefix (single-category) cache path returns the same runs as list_runs."""
    import npa.workflows.artifacts as A

    A._run_list_cache_clear()
    s3 = _PrefixAwareS3(_MULTI_ROOT_LAYOUT)
    direct = list_runs("bucket", prefix="checkpoints/sim2real-b", limit=50, s3=s3)
    cached = A.list_runs_cached("bucket", prefix="checkpoints/sim2real-b", limit=50, s3=s3, ttl=1000)
    assert [r.run_id for r in cached.runs] == [r.run_id for r in direct.runs]
    A._run_list_cache_clear()


# --- Multi-bucket discovery ---------------------------------------------------


def test_list_accessible_buckets_primary_first_deduped() -> None:
    import npa.workflows.artifacts as A

    class _S3:
        def list_buckets(self):
            return {"Buckets": [{"Name": "b2"}, {"Name": "primary"}, {"Name": "b3"}]}

    got = A.list_accessible_buckets(_S3(), primary="primary", extra=["b2"])
    assert got[0] == "primary"
    assert got.count("primary") == 1 and got.count("b2") == 1
    assert set(got) == {"primary", "b2", "b3"}


def test_list_accessible_buckets_survives_no_listbuckets_permission() -> None:
    import npa.workflows.artifacts as A

    class _S3:
        def list_buckets(self):
            raise A.BotoCoreError()

    got = A.list_accessible_buckets(_S3(), primary="primary", extra=["x"])
    assert got == ["primary", "x"]  # falls back to primary/extras only


def test_find_run_artifacts_across_buckets_returns_first_match(monkeypatch) -> None:
    import npa.workflows.artifacts as A

    scanned: list[str] = []

    def fake_find(bucket, *, base_prefix, run_id, s3):
        scanned.append(bucket)
        if bucket == "b2":
            return [A.Artifact(run_id, f"byof/{run_id}/x.json", f"s3://b2/byof/{run_id}/x.json", 1, "t", "json", False)]
        return []

    monkeypatch.setattr(A, "find_run_artifacts", fake_find)
    bkt, arts = A.find_run_artifacts_across_buckets(["b1", "b2", "b3"], base_prefix="", run_id="run-x", s3=object())
    assert bkt == "b2" and len(arts) == 1
    assert arts[0].s3_uri == "s3://b2/byof/run-x/x.json"
    assert scanned == ["b1", "b2"]  # stops at first match


def test_list_all_runs_across_buckets_merges_and_tags(monkeypatch) -> None:
    import npa.workflows.artifacts as A

    def fake_all(bucket, *, base_prefix, limit, exclude, contains, s3):
        return A.RunListPage(
            runs=[A.RunSummary(f"run-{bucket}", "2026-06-30T00:00:00+00:00", 1, True)],
            truncated=False,
            total_runs=1,
            limit=limit,
        )

    monkeypatch.setattr(A, "list_all_runs", fake_all)
    page = A.list_all_runs_across_buckets(["b1", "b2"], base_prefix="", limit=50, exclude=None, contains="", s3=object())
    tagged = {(r.bucket, r.run_id) for r in page.runs}
    assert ("b1", "run-b1") in tagged and ("b2", "run-b2") in tagged
    assert page.total_runs == 2


def test_build_fiftyone_dataset_emits_bucket_qualified_uris() -> None:
    run = "paidf-demo"
    base = f"checkpoints/physical-ai-data-factory/{run}"
    keys = [
        f"{base}/cosmos_augmented/aug-{run}-0/augmented_video.mp4",
        f"{base}/cosmos_augmented/aug-{run}-0/frame-00000.png",
        f"{base}/cosmos_augmented/aug-{run}-0/metadata.json",
    ]
    payloads = {f"{base}/cosmos_augmented/aug-{run}-0/metadata.json": {"variables": {"cloth_color": "green"}}}
    ds = build_fiftyone_dataset(keys, run_id=run, read_json=lambda k: payloads.get(k), bucket="lerobot-d87cf691")
    aug = [s for s in ds["samples"] if s["group"] == "augmented"][0]
    assert aug["thumbnail_uri"].startswith("s3://lerobot-d87cf691/")
    assert aug["thumbnail_uri"].endswith("frame-00000.png")
    assert aug["video_uri"].endswith("augmented_video.mp4")
